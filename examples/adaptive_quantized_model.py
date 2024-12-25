import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from quantizer import adapt_precision


def main():
    model_ref = "EleutherAI/pythia-160m"
    model = AutoModelForCausalLM.from_pretrained(model_ref)

    dataset = load_dataset("ag_news", split="train")  # Train split for calibration

    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    with adapt_precision(model, n_samples=100) as ad_model:
        try:
            for i in range(1000000):
                datum = dataset[i]
                full_inp = tokenizer(datum["text"], return_tensors="pt")

                with torch.no_grad():
                    _ = ad_model(full_inp["input_ids"])
                if i % 1000 == 0:
                    print(f"Iteration {i}")
        except KeyboardInterrupt:
            print("Test stopped.")


if __name__ == "__main__":
    main()
