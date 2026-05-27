# llm-quantization-study# LLM Quantization Comparison Study

Rigorous benchmark of four quantization techniques on LLaMA-3-8B-Instruct.
Same model, four formats — measuring exactly what each costs in accuracy
and buys in memory and speed.

## Headline findings

> charts coming once benchmarks are complete

## Techniques compared

| Variant | Bits | Library |
|---|---|---|
| FP16 | 16 | transformers |
| INT8 | 8 | bitsandbytes |
| GPTQ | 4 | auto-gptq |
| AWQ | 4 | autoawq |

## Accuracy benchmarks

MMLU · HellaSwag · GSM8K · HumanEval · TruthfulQA

## Repo structure

- `quantization/` — export scripts for each variant
- `benchmarks/` — latency, memory, throughput measurement
- `accuracy/` — lm-eval-harness config and run scripts
- `analysis/` — notebooks for charts and synthesis
- `results/` — raw CSVs and eval JSON

## Setup

```bash
bash setup/install.sh
python setup/download_models.py
python setup/download_datasets.py
```

## Status

🔄 In progress — part of a series on LLM inference optimization.
