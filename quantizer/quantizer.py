import copy
import gc
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from torcheval.metrics.functional.text import perplexity
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from utils import get_linear_type, replace_module, SwitchLayer


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

    model: nn.Module  # Model to adaptively quantize
    tokenizer: PreTrainedTokenizerFast = None  # Tokenizer for `dataset`
    dataset: Dataset = None  # Dataset to provide samples for perplexity scores
    quant_threshold: int = 60  # Resource usage threshold for swapping to quantized
    unquant_threshold: int = 40  # Resource usage threshold for swapping to unquantized
    n: int = 5  # Number of dataset samples for perplexity scores

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
            self.model, {nn.Linear}, dtype=torch.qint8, inplace=True
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
        replace_module(self.quantized_model, module_to_replace, layer_prefix)

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

        keep_layers = {
            k: v
            for k, v in sorted(
                perplexity_by_swap.items(), key=lambda item: item[1], reverse=True
            )
            if v > cutoff_percentage * baseline
        }

        # This dict will order all quantizable layers in order of "sensitivity"
        # to precision (defined by how much perplexity is improved by unquantizing
        # a layer). Least sensitive (where quantizing will incur the least drawback)
        # is intentionally placed at the end to fetch from when swapping a layer
        switch_layers = OrderedDict()

        # Finally, only retain the kept layers in memory of the higher precision model
        for k, v in keep_layers.items():
            switch_layers[k] = copy.deepcopy(self.quant_mappings[k])
        self.switch_layers = SwitchLayer(switch_layers)
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

            # Switch to quantized layer
            replace_module(self.quantized_model, layer, layer_prefix)

        if cpu_percent < self.unquant_threshold:
            layer_prefix, layer = self.switch_layers.fetch("unquantized")

            # Switch to unquantized layer
            replace_module(self.quantized_model, layer, layer_prefix)

        self.quantized_model(*args, **kwargs)


@contextmanager
def adapt_precision(model: nn.Module):
    ad_model = AdaptiveQuantizer(model)
    try:
        yield ad_model
    finally:
        # Is this a good idea?
        pass