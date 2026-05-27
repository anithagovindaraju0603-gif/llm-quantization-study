import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM   
from utils.constants import allocated_mem, reserved_mem, make_input, max_mem_allocated, warmup

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True)

memory_results = {}

# -------------model loading and memory tracking----------------
#-----------------------------------------------------
# Model variant fp16: 16-bit floating point precision
#-----------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(f"./models/fp16", 
                                             dtype=torch.float16,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/fp16")

mem_allocated = allocated_mem()
mem_reserved = reserved_mem()

warmup(model, tokenizer)

# reset peak memory usage
torch.cuda.reset_peak_memory_stats()

#GPU memory during generation at 2K context
input_ids = make_input(2048, tokenizer)
with torch.no_grad():
    outputs = model.generate(**input_ids, max_new_tokens=50, do_sample=False)
torch.cuda.synchronize()
mem_allocated_decode = max_mem_allocated()

del model
torch.cuda.empty_cache()

print(f"Model loaded in FP16")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")
print(f"Memory allocated during decoding: {mem_allocated_decode:.2f} GB")

#add results to dict
memory_results["fp16"] = {"allocated": mem_allocated, "reserved": mem_reserved, "memory_allocated_for_2k_generation": mem_allocated_decode}

#-----------------------------------------------------
# Model variant int8: 8-bit integer precision
#-----------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(f"./models/int8", 
                                             quantization_config=bnb_config,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/int8")
mem_allocated = allocated_mem()
mem_reserved = reserved_mem()   

warmup(model, tokenizer)

# reset peak memory usage
torch.cuda.reset_peak_memory_stats()

#GPU memory during generation at 2K context
input_ids = make_input(2048, tokenizer)
with torch.no_grad():
    outputs = model.generate(**input_ids, max_new_tokens=50, do_sample=False)
torch.cuda.synchronize()
mem_allocated_decode = max_mem_allocated()

del model
torch.cuda.empty_cache()

print(f"Model loaded in INT8")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")
print(f"Memory allocated during decoding: {mem_allocated_decode:.2f} GB")

#add results to dict
memory_results["int8"] = {"allocated": mem_allocated, "reserved": mem_reserved, "memory_allocated_for_2k_generation": mem_allocated_decode} 

#-----------------------------------------------------
# Model variant gptq: GPTQ quantization method
#-----------------------------------------------------
model = AutoGPTQForCausalLM.from_quantized(f"./models/gptq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/gptq")
mem_allocated = allocated_mem()
mem_reserved = reserved_mem()

warmup(model, tokenizer)

# reset peak memory usage
torch.cuda.reset_peak_memory_stats()

#GPU memory during generation at 2K context
input_ids = make_input(2048, tokenizer)
with torch.no_grad():
    outputs = model.generate(**input_ids, max_new_tokens=50, do_sample=False)
torch.cuda.synchronize()
mem_allocated_decode = max_mem_allocated()

del model
torch.cuda.empty_cache()

print(f"Model loaded in GPTQ")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")
print(f"Memory allocated during decoding: {mem_allocated_decode:.2f} GB")

#add results to dict
memory_results["gptq"] = {"allocated": mem_allocated, "reserved": mem_reserved, "memory_allocated_for_2k_generation": mem_allocated_decode}

#-----------------------------------------------------
# Model variant awq: AWQ quantization method
#-----------------------------------------------------
model = AutoAWQForCausalLM.from_quantized(f"./models/awq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"./models/awq")

mem_allocated = allocated_mem()
mem_reserved = reserved_mem()

warmup(model, tokenizer)

# reset peak memory usage
torch.cuda.reset_peak_memory_stats()

#GPU memory during generation at 2K context
input_ids = make_input(2048, tokenizer)
with torch.no_grad():
    outputs = model.generate(**input_ids, max_new_tokens=50, do_sample=False)
torch.cuda.synchronize()
mem_allocated_decode = max_mem_allocated()

del model
torch.cuda.empty_cache()

print(f"Model loaded in AWQ")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")
print(f"Memory allocated during decoding: {mem_allocated_decode:.2f} GB")

#add results to dict
memory_results["awq"] = {"allocated": mem_allocated, "reserved": mem_reserved, "memory_allocated_for_2k_generation": mem_allocated_decode}

#save results to csv
df = pd.DataFrame.from_dict(memory_results, orient="index")
df.to_csv(os.path.join(RESULTS_DIR, "memory_results.csv"))