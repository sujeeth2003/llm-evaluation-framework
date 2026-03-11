# LLM Evaluation Framework

A benchmarking pipeline for evaluating open-source large language models
across reasoning, factual recall, and hallucination robustness.

## Features

• Multi-model evaluation  
• Prompt dataset benchmarking  
• Accuracy metrics  
• Hallucination detection  
• Latency tracking  
• Result visualization  

## Models Tested

- Mistral 7B
- LLaMA 2

## Dataset

Two prompt categories:

reasoning_prompts.json  
factual_prompts.json  

Each prompt includes a ground truth answer.

## Metrics

Accuracy

correct predictions / total prompts

Hallucination Rate

Semantic similarity between response and ground truth.

Low similarity → hallucination.

Latency

Time required for generation.

## Pipeline

1. Run inference across models
2. Store outputs in CSV
3. Compute evaluation metrics
4. Generate comparison plots

## Running the Project

Install dependencies

pip install -r requirements.txt

Run model inference

python models/run_inference.py

Analyze results

python analysis/analyze_results.py

## Output

results/

benchmark_results.csv  
model_summary.csv  
accuracy_plot.png
