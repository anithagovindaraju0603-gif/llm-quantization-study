import os
import statistics
import time
import pandas as pd
import torch
from utils.constants import warmup, make_input
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "throughput_results.csv")

def save_checkpoint(results):
    df = pd.DataFrame(results)
    header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=header, index=False)
    print(f"Checkpoint saved → {CSV_PATH} ({len(results)} rows)")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True)

def benchmark_throughput(model, tokenizer, model_name):
    results = []

    # Part 1 - tokens/sec at batch sizes 1, 4, 16
    BATCH_SIZES = [1, 4, 16]
    MAX_NEW_TOKENS = 128
    PROMPT_LENGTH = 512

    for bs in BATCH_SIZES:
        trial_tokens = []
        warmup(model, tokenizer) # warmup at this batch size
        input_ids = make_input(PROMPT_LENGTH, tokenizer, batch_size=bs)
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            tokens_per_sec = (bs * MAX_NEW_TOKENS) / elapsed
            trial_tokens.append(tokens_per_sec)

            print(f"Batch size {bs} | Time taken: {elapsed:.4f}s | Tokens/sec: {tokens_per_sec:.1f}")

        total_tokens_median = statistics.median(trial_tokens)

        results.append({
            "model": model_name,
            "phase": "throughput",
            "batch_size": bs,
            "prompt_length": PROMPT_LENGTH,
            "tokens_per_sec": total_tokens_median,
            "status": "ok"
        })

    # Part 2 - OOM sweep
    torch.cuda.empty_cache()

    OOM_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    OOM_PROMPT_LENGTH = 2048
    for bs in OOM_BATCH_SIZES:
        try:
            inputs = make_input(OOM_PROMPT_LENGTH, tokenizer, batch_size=bs)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            status = "ok"
        except torch.cuda.OutOfMemoryError:
            status = "OOM"
            torch.cuda.empty_cache()
        
        results.append({
            "model": model_name,
            "phase": "oom_sweep",
            "batch_size": bs,
            "prompt_length": OOM_PROMPT_LENGTH,
            "tokens_per_sec": None,
            "status": status
        })
    
    return results


#---------model : fp16--------------------------------------
model = AutoModelForCausalLM.from_pretrained(f"./models/fp16", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/fp16")

warmup(model, tokenizer)

throughput_results_model   = benchmark_throughput(model, tokenizer, model_name="fp16")
save_checkpoint(throughput_results_model) 

del model
torch.cuda.empty_cache()
#---------model : int8--------------------------------------
model = AutoModelForCausalLM.from_pretrained(f"./models/int8", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/int8")

warmup(model, tokenizer)

throughput_results_model = benchmark_throughput(model, tokenizer, model_name="int8")
save_checkpoint(throughput_results_model) 

del model
torch.cuda.empty_cache()

#---------model : gptq--------------------------------------
model = AutoGPTQForCausalLM.from_quantized(f"./models/gptq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/gptq")

warmup(model, tokenizer)

throughput_results_model = benchmark_throughput(model, tokenizer, model_name="gptq")
save_checkpoint(throughput_results_model) 

del model
torch.cuda.empty_cache()

#---------model : awq--------------------------------------
model = AutoAWQForCausalLM.from_quantized(f"./models/awq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/awq")

warmup(model, tokenizer)
throughput_results_model = benchmark_throughput(model, tokenizer, model_name="awq")
save_checkpoint(throughput_results_model) 

del model
torch.cuda.empty_cache()

print(f"All throughput benchmarks complete. Results in {CSV_PATH}")

