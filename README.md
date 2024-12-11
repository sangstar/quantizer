# Quantizer

Adaptive quantization for `torch.nn` modules

## How It Works

`quantizer` can be used simply with the function `adapt_precision` as a context manager for
your `torch` model, as shown in the `adaptive_quantized_model` example script:

```python
import torch
from transformers import AutoModelForCausalLM

from quantizer import adapt_precision

model_ref = "EleutherAI/pythia-160m"
model = AutoModelForCausalLM.from_pretrained(model_ref)

shape = (3, 40)  # batch_size, seq_len
input_data = torch.randint(2, 30, shape)

with adapt_precision(model) as ad_model:
  _ = ad_model(input_data)
```

`ad_model` wraps over your model, quantizes it, and then keeps note of which
original layers contributes the most to perplexity and keeps those. When resources
free up, it swaps the quantized layer with the higher-precision one, and swaps back
to the quantized layer when resources are low.

## Installation

Install the package in editable model with:

```bash
git clone https://github.com/sangstar/quantizer
cd quantizer
pip install -e .
```

## More to come..

- More agnostic resource monitoring (not just CPU)
- Perplexity calculations based on first N encountered
  samples rather than on an arbitrary dataset
- Profiling
- Better type annotations, docstrings
- Smarter swapping decisions rather than simply
  swapping based on resource thresholds