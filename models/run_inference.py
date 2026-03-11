import json
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

MODELS = [
"mistralai/Mistral-7B-Instruct-v0.1",
"meta-llama/Llama-2-7b-chat-hf"
]

DATASETS = [
"dataset/reasoning_prompts.json",
"dataset/factual_prompts.json"
]

def load_dataset(path):
    with open(path) as f:
        return json.load(f)

def run_model(model_name, prompts):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    results = []

    for item in tqdm(prompts):

        prompt = item["question"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

        latency = time.time() - start

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        tokens_used = outputs.shape[1]

        results.append({
            "model": model_name,
            "prompt": prompt,
            "response": response,
            "latency": latency,
            "tokens_used": tokens_used,
            "ground_truth": item["answer"]
        })

    return results


def main():

    all_results = []

    for dataset in DATASETS:

        prompts = load_dataset(dataset)

        for model in MODELS:

            model_results = run_model(model, prompts)

            all_results.extend(model_results)

    df = pd.DataFrame(all_results)

    df.to_csv("results/benchmark_results.csv", index=False)

    print("Results saved to results/benchmark_results.csv")


if __name__ == "__main__":
    main()
