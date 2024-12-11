from collections import OrderedDict
from dataclasses import dataclass
from itertools import cycle

import torch.ao.nn.quantized as nnq
from torch import nn

SwitchLayerData = OrderedDict[str, nn.Module]


@dataclass
class SwitchLayer:
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
    descend the dictionary, fetched in highest-priority order for both.

    Some source of samples need to be fed to the model to generate
    perplexity scores. A `dataset` can be provided to accomplish this,
    which is chosen arbitrary if not set explicitly.
    """

    data: SwitchLayerData

    def __post_init__(self):
        self.iter = cycle(self.data)

    def fetch(self, quant_or_unquant: str):
        layer = next(self.iter)
        return layer, self.data[layer][quant_or_unquant]


def replace_module(module, new_module, layer_prefix):
    """
    Replaces module with new_module. Assumes module represents a torch model
    before this is called recursively. If the new_module is replacing the same
    type, this is a no-op.
    """
    prefixes = iter(layer_prefix.split("."))

    _replace_module(module, new_module, next(prefixes), prefixes)


def _replace_module(module, new_module, subprefix, layer_prefix):
    is_linear = lambda obj: isinstance(obj, nn.Linear) or isinstance(obj, nnq.Linear)
    attr = getattr(module, subprefix, None)
    if is_linear(module):
        return
    elif is_linear(attr):
        if not type(attr) == type(new_module):
            setattr(module, subprefix, new_module)
        else:
            return
    else:
        new_prefix = next(layer_prefix)
        return _replace_module(attr, new_module, new_prefix, layer_prefix)


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
