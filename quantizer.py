import copy
import gc
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
import torch
import torch.ao.nn.quantized as nnq
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from torcheval.metrics.functional.text import perplexity
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers import AutoTokenizer

SwitchLayerData: OrderedDict[str, nn.Module]


@dataclass
class _SwitchLayer:
    """
    A class that non-destructively fetches the first or last
    element in `data` when called with `fetch`.

    `data` in this case specifically is constructed such that
    module layers are ordered by how much that layer quantized
    affects perplexity compared to a fully quantized model.
    The most substantial layers, where quantizing most hinders
    perplexity, are placed first in descending order.

    The logic implemented here, allows unquantized-quantized swaps
    to ascend the dictionary, while quantized-unquantized swaps can
    descend the dictionary, fetched in highest-priority order for both
    """

    data: SwitchLayerData

    def __post_init__(self):
        self.quant_iter = reversed(self.data)
        self.unquant_iter = iter(self.data)

    def fetch(self, quant_or_unquant: str):
        if quant_or_unquant == "quantized":
            layer = next(self.quant_iter)
            return layer, self.data[layer]["quantized"]
        elif quant_or_unquant == "unquantized":
            # Iterate from first-to-last since first layers are most beneficial
            # to unquantize
            layer = next(self.unquant_iter)
            return layer, self.data[layer]["unquantized"]
        else:
            raise ValueError("quant_or_unquant must be quantized or unquantized.")


@dataclass
class AdaptiveQuantizer:
    """
    Main helper class for adaptive quantization.

    `model` is quantized dynamically using `torch.quantization.quantize_dynamic`
    in place. The original unquantized modules are ephemerally preserved along
    with the quantized model's new `.named_modules()`.

    Mappings are created between the quantized and unquantized layers, and
    perplexities are calculated when each quantized layer is swapped to unquantized.

    Layers are scored based off of their perplexities compared to the fully-quantized
    baseline. When the adaptive quantized model is then called, it will automatically
    swap between unquantized and quantized layers to optimize resource usage.
    """

    model: nn.Module
    tokenizer: PreTrainedTokenizerFast = None
    dataset: Dataset = None
    quant_threshold: int = 60
    unquant_threshold: int = 40
    n: int = 5

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)
        if not self.dataset:
            self.dataset = load_dataset(
                "ag_news", split="train"
            )  # Train split for calibration

        self._model_module_dict = OrderedDict(
            {k: copy.deepcopy(mod) for k, mod in self.model.named_modules()}
        )

        self.quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8, inplace=True
        )

        self._quantized_model_module_dict = OrderedDict(
            {k: mod for k, mod in self.quantized_model.named_modules()}
        )
        differences = set(self._quantized_model_module_dict).difference(
            self._model_module_dict
        )

        self.quant_mappings = {}
        for key in differences:
            prefix = key.rpartition(".")[0]
            self.try_map_to_quantized_layer(prefix)
        self._validate_quant_mappings()

        self.get_swap_sensitivities()

    def try_map_to_quantized_layer(self, prefix):

        if prefix.endswith("._packed_params"):
            # This is a nested _packed_params
            prefix = prefix.replace("._packed_params", "")

        self.quant_mappings[prefix] = {}
        self.quant_mappings[prefix]["unquantized"] = self._model_module_dict[
            f"{prefix}"
        ]
        self.quant_mappings[prefix]["quantized"] = self._quantized_model_module_dict[
            f"{prefix}"
        ]

    def _validate_quant_mappings(self):

        def _validate_layer(sd: dict, layer):
            before_len = len(sd.keys())
            sd.update(layer)
            assert before_len == len(sd.keys())
            for key in layer:
                # Check we're not copying and just holding a ref
                assert id(layer[key]) == id(sd[key])

        for k, v in self.quant_mappings.items():
            for quant_or_unquant, layer in v.items():
                if quant_or_unquant == "quantized":
                    _validate_layer(self._quantized_model_module_dict, {k: layer})
                if quant_or_unquant == "unquantized":
                    _validate_layer(self._model_module_dict, {k: layer})

    def swap_precisions(self, layer_prefix):

        want_quant_or_unquant = get_linear_type(self.quantized_model, layer_prefix)
        module_to_replace = self.quant_mappings[layer_prefix][want_quant_or_unquant]
        prefixes = iter(layer_prefix.split("."))
        replace_module(
            self.quantized_model, module_to_replace, next(prefixes), prefixes
        )

    def get_perplexity(self):
        perp = 0
        for i in range(self.n):
            datum = self.dataset[i]
            full_inp = self.tokenizer(datum["text"], return_tensors="pt")
            input_ids = full_inp["input_ids"]

            logits = self.quantized_model(input_ids).logits[:, :-1, :]

            # Targets are shifted by one, since loss functions here will expect
            # the gold label at token index idx.
            targets = input_ids[:, 1:]

            perp += perplexity(logits, targets)
        return perp.item()

    def _get_dotted_attr(self, model, layer_prefix):
        modules = layer_prefix.split(".")
        parent = model
        for name in modules:
            parent = getattr(parent, name)
        return parent

    def get_swap_sensitivities(self, cutoff_percentage: int = 0.1):
        perplexity_by_swap = OrderedDict()

        # Assumes the model is entirely quantized here
        baseline = self.get_perplexity()

        for layer_prefix in self.quant_mappings:
            original_layer = self._get_dotted_attr(self.quantized_model, layer_prefix)

            # Performs two swaps: one to unquantize a layer and then one to restore it
            self.swap_precisions(layer_prefix)
            swapped_layer = self._get_dotted_attr(self.quantized_model, layer_prefix)
            assert original_layer != swapped_layer
            perplexity_by_swap.update({layer_prefix: baseline - self.get_perplexity()})
            self.swap_precisions(layer_prefix)

            after_layer = self._get_dotted_attr(self.quantized_model, layer_prefix)
            assert original_layer == after_layer

        # Unquantize the model
        for layer_prefix in self.quant_mappings:
            self.swap_precisions(layer_prefix)

        keep_layers = {
            k: v
            for k, v in sorted(
                perplexity_by_swap.items(), key=lambda item: item[1], reverse=True
            )
            if v < cutoff_percentage * baseline
        }

        # This dict will order all quantizable layers in order of "sensitivity"
        # to precision (defined by how much perplexity is improved by unquantizing
        # a layer). Least sensitive (where quantizing will incur the least drawback)
        # is intentionally placed at the end to fetch from when swapping a layer
        switch_layers = OrderedDict()

        # Finally, only retain the kept layers in memory of the higher precision model
        for k, v in keep_layers.items():
            switch_layers[k] = copy.deepcopy(self.quant_mappings[k])
        self.switch_layers = _SwitchLayer(switch_layers)
        del self._model_module_dict
        del self.quant_mappings
        delattr(self, "model")
        gc.collect()

    def __call__(self, *args, **kwargs):
        ## TODO: Additionally, possibly allow perplexity calculations
        ##  to be built off of the forward passes this call method
        ##  performs
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.quant_threshold:
            layer_prefix, layer = self.switch_layers.fetch("quantized")
            prefixes = iter(layer_prefix.split("."))

            # Switch to quantized layer
            replace_module(self.quantized_model, layer, next(prefixes), prefixes)

        if cpu_percent < self.unquant_threshold:
            layer_prefix, layer = self.switch_layers.fetch("unquantized")
            prefixes = iter(layer_prefix.split("."))

            # Switch to unquantized layer
            replace_module(self.quantized_model, layer, next(prefixes), prefixes)

        self.quantized_model(*args, **kwargs)


def quantize_layers(model: nn.Module, key: str):
    raise NotImplementedError


def replace_module(module, new_module, subprefix, layer_prefix):
    """
    Replaces module with new_module. Assumes module represents a torch model
    before this is called recursively.
    """
    is_linear = lambda obj: isinstance(obj, nn.Linear) or isinstance(obj, nnq.Linear)
    if is_linear(module):
        return
    elif is_linear(getattr(module, subprefix, None)):
        setattr(module, subprefix, new_module)
    else:
        new_prefix = next(layer_prefix)
        return replace_module(
            getattr(module, subprefix, None), new_module, new_prefix, layer_prefix
        )


def get_linear_type(quant_or_unquant_model, prefix):
    modules = prefix.split(".")
    parent = quant_or_unquant_model
    for name in modules:
        parent = getattr(parent, name)

    # Want unquantized if quant linear, and vice versa for swapping
    return (
        "unquantized"
        if isinstance(parent, nnq.Linear)
        else "quantized" if isinstance(parent, nn.Linear) else ""
    )


@contextmanager
def adapt_precision(model: nn.Module):
    ad_model = AdaptiveQuantizer(model)
    try:
        yield ad_model
    finally:
        # Is this a good idea?
        pass


if __name__ == "__main__":
    model_ref = "EleutherAI/pythia-160m"
    torch.backends.quantized.engine = "qnnpack"
    model = AutoModelForCausalLM.from_pretrained(model_ref)

    shape = (3, 40)  # batch_size, seq_len
    input_data = torch.randint(2, 30, shape)

    with adapt_precision(model) as ad_model:
        try:
            for i in range(1000000):  # High number of iterations
                with torch.no_grad():  # Disable gradient computation
                    _ = ad_model(input_data)
                if i % 1000 == 0:
                    print(f"Iteration {i}")
        except KeyboardInterrupt:
            print("Test stopped.")
