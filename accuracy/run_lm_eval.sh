#!/bin/bash
accuracy/run_lm_eval.sh
─────────────────────────────────────────────────────────────
Runs lm-evaluation-harness for all four model variants.
Results saved as JSON files under results/accuracy/
#
Usage:
  bash accuracy/run_lm_eval.sh
#
Requirements:
  - lm-eval installed (in requirements.txt)
  - All four model variants saved under ./models/
  - HF_TOKEN environment variable set
─────────────────────────────────────────────────────────────

set -e  # stop immediately if any command fails
export HF_ALLOW_CODE_EVAL=1

── Auth ──────────────────────────────────────────────────────
Some benchmark datasets are gated on HuggingFace
huggingface-cli login --token $HF_TOKEN

── Config ────────────────────────────────────────────────────
RESULTS_DIR="./results/accuracy"
mkdir -p $RESULTS_DIR

shared settings — same for every variant, never change between runs
TASKS="mmlu,hellaswag,gsm8k,humaneval,truthfulqa_mc1"
FEWSHOT=5
BATCH=8

── FP16 ──────────────────────────────────────────────────────
Baseline — reference point for all other variants
echo "=========================================="
# echo "Evaluating FP16 (baseline)..."
# echo "=========================================="
# lm_eval --model hf \
#         --confirm_run_unsafe_code --model_args pretrained=./models/fp16 \
#         --tasks $TASKS \
#         --num_fewshot $FEWSHOT \
#         --batch_size $BATCH \
#         --output_path $RESULTS_DIR/fp16_eval.json
# echo "FP16 eval complete."

── INT8 ──────────────────────────────────────────────────────
bitsandbytes 8-bit
echo "=========================================="
echo "Evaluating INT8..."
echo "=========================================="
lm_eval --model hf \
        --confirm_run_unsafe_code --model_args pretrained=./models/int8 \
        --tasks $TASKS \
        --num_fewshot $FEWSHOT \
        --batch_size $BATCH \
        --output_path $RESULTS_DIR/int8_eval.json
echo "INT8 eval complete."

── GPTQ 4-bit ────────────────────────────────────────────────
Post-training 4-bit
echo "=========================================="
echo "Evaluating GPTQ 4-bit..."
echo "=========================================="
lm_eval --model hf \
        --confirm_run_unsafe_code --model_args pretrained=./models/gptq \
        --tasks $TASKS \
        --num_fewshot $FEWSHOT \
        --batch_size $BATCH \
        --output_path $RESULTS_DIR/gptq_eval.json
echo "GPTQ eval complete."

── AWQ 4-bit ─────────────────────────────────────────────────
Activation-aware 4-bit
echo "=========================================="
echo "Evaluating AWQ 4-bit..."
echo "=========================================="
lm_eval --model hf \
        --confirm_run_unsafe_code --model_args pretrained=./models/awq,dtype=float16 \
        --tasks $TASKS \
        --num_fewshot $FEWSHOT \
        --batch_size $BATCH \
        --output_path $RESULTS_DIR/awq_eval.json
echo "AWQ eval complete."

── Done ──────────────────────────────────────────────────────
echo "=========================================="
echo "All evals complete."
echo "Results saved to $RESULTS_DIR"
echo "Files:"
ls -lh $RESULTS_DIR
echo "=========================================="