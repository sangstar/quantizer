import torch
from transformers import AutoModelForCausalLM

from quantizer import adapt_precision


def main():
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


if __name__ == "__main__":
    main()
