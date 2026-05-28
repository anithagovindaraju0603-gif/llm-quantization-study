import os
import time
import torch
import pandas as pd
import statistics
import nvidia_ml_py as pynvml 
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM   
from utils.constants import make_input, warmup

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True)

CSV_PATH = os.path.join(RESULTS_DIR, "latency_results.csv")
def save_checkpoint(results):
    df = pd.DataFrame(results)
    # append if file exists, write header only on first save
    header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=header, index=False)
    print(f"Checkpoint saved → {CSV_PATH} ({len(results)} rows)")

# ── GPU utilization sampler ───────────────────────────────────────────────────
# Samples SM utilization % in a background thread while the GPU is working.
# Returns the median — same idea as taking median of timing runs.
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
 
def sample_utilization_during(fn):
    """Run fn(), sample GPU util every 1ms in background, return median %."""
    compute_samples = []
    memory_samples = []
    stop = threading.Event()
 
    def sampler():
        while not stop.is_set():
            rates = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            compute_samples.append(rates.gpu)
            memory_samples.append(rates.memory)
            time.sleep(0.001)
 
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    fn()
    torch.cuda.synchronize()
    stop.set()
    t.join()
    compute_median = statistics.median(compute_samples) if compute_samples else 0
    memory_median = statistics.median(memory_samples) if memory_samples else 0
    return compute_median, memory_median

# ── Prefill + Decode measurement ─────────────────────────────────────────────
# Combined loop: prefill is timed first, KV cache is captured and immediately
# reused for the decode step. Cache is freed at end of each iteration.

#-------------latency calculation--------------------------
def benchmark_latency(model, tokenizer, model_name):
    latency_results = []
    for length in PROMPT_LENGTHS:
        inputs = make_input(length, tokenizer)
        actual_length = inputs["input_ids"].shape[1]

        times = []
        for _ in range(5):
            # prefill — timed, use_cache=True captures KV cache for decode reuse
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                prefill_out = model(**inputs, use_cache=True)

            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        prefill_elapsed = statistics.median(times)
        past_key_values = prefill_out.past_key_values

        def run_prefill():
            with torch.no_grad():
                model(**inputs, use_cache=True)
    
        prefill_compute, prefill_memory = sample_utilization_during(run_prefill)
        prefill_mem_gb = torch.cuda.memory_allocated() / 1e9 

        latency_results.append({
            "model": model_name,
            "phase": "prefill",
            "prompt_length": actual_length,
            "time_sec": prefill_elapsed,
            "tokens_per_sec": actual_length / prefill_elapsed,
            "mem_allocated_gb": prefill_mem_gb,
            "gpu_memory_util_pct": prefill_memory
        })
        print(f"prefill | length={actual_length} | time_taken_sec={prefill_elapsed:.4f}s | tokens_per_sec={actual_length / prefill_elapsed:.1f} tok/s | gpu_memory={prefill_memory:.1f}% gpu memory util")

        # TTFT — full prompt in, first token out
        ttft_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            ttft_times.append(time.perf_counter() - start)

        ttft_elapsed = statistics.median(ttft_times)

        latency_results.append({
            "model": model_name,
            "phase": "ttft",
            "prompt_length": actual_length,
            "time_sec": ttft_elapsed,
            "tokens_per_sec": None,
            "mem_allocated_gb": None,
            "gpu_memory_util_pct": None
        })

        # decode — timed, feeds last token with pre-built KV cache
        next_token = inputs["input_ids"][:, -1:]

        decode_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                model(next_token, past_key_values=past_key_values, use_cache=True)

            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - start)

        decode_elapsed = statistics.median(decode_times)

        def run_decode_loop():
            for _ in range(150):  # ~3.5 seconds of pure decode
                with torch.no_grad():
                    model(next_token, past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize()         
        
        time.sleep(1.5)  # let NVML window forget the prefill
        decode_compute, decode_memory = sample_utilization_during(run_decode_loop)
        decode_mem_gb = torch.cuda.memory_allocated() / 1e9 

        latency_results.append({
            "model": model_name,
            "phase": "decode",
            "prompt_length": actual_length,
            "time_sec": decode_elapsed,
            "tokens_per_sec": 1 / decode_elapsed,
            "mem_allocated_gb": decode_mem_gb,
            "gpu_memory_util_pct": decode_memory
        })  
        print(f"decode  | kv_length={actual_length} | time_taken_sec={decode_elapsed:.4f}s | tokens_per_sec={1 / decode_elapsed:.1f} tok/s | gpu_memory={decode_memory:.1f}% gpu memory util")
    
        del past_key_values, prefill_out
        torch.cuda.empty_cache()

    return latency_results
#---------model : fp16--------------------------------------
model = AutoModelForCausalLM.from_pretrained(f"./models/fp16", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/fp16")

warmup(model, tokenizer)

latency_results = benchmark_latency(model, tokenizer, model_name="fp16")
save_checkpoint(latency_results)

del model
torch.cuda.empty_cache()
#---------model : int8--------------------------------------
model = AutoModelForCausalLM.from_pretrained(f"./models/int8", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/int8")

warmup(model, tokenizer)

latency_results = benchmark_latency(model, tokenizer, model_name="int8")
save_checkpoint(latency_results)

del model
torch.cuda.empty_cache()

#---------model : gptq--------------------------------------
model = AutoGPTQForCausalLM.from_quantized(f"./models/gptq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/gptq")

warmup(model, tokenizer)

latency_results = benchmark_latency(model, tokenizer, model_name="gptq")
save_checkpoint(latency_results)

del model
torch.cuda.empty_cache()

#---------model : awq--------------------------------------
model = AutoAWQForCausalLM.from_quantized(f"./models/awq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/awq")

warmup(model, tokenizer)
latency_results = benchmark_latency(model, tokenizer, model_name="awq")
save_checkpoint(latency_results)

del model
torch.cuda.empty_cache()

print(f"All latency benchmarks complete. Results in {CSV_PATH}")

