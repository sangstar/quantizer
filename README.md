# Quantizer

Adaptive quantization for `torch.nn` modules

## How It Works

`quantizer` can be used simply with the function `adapt_precision` as a context manager for
your `torch` model:

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

`ad_model` wraps over your model, quantizes it, and then after `num_samples`
forward passes, keeps note of which original layers contributes the most to
perplexity and keeps those. When resources free up, it swaps quantized layers
with the higher-precision ones, and swaps back to the quantized layers when
resources are low.

## Installation

Install the package in editable model with:

```bash
git clone https://github.com/sangstar/quantizer
cd quantizer
pip install -e .
```

## More to come..

- More agnostic resource monitoring (not just CPU)
- Reduce latency burden between n_samples and n_samples + 1 iterations
  (due to the `.get_swap_sensitivities()` call)
- Profiling
- Better type annotations, docstrings
- Smarter swapping decisions rather than simply
  swapping based on resource thresholds