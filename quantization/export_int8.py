import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.constants import allocated_mem, reserved_mem

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")
login(token=hf_token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0) #values with magnitude > 6.0 will be left in higher precision, which can improve accuracy with minimal increase in memory usage
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', 
                                             quantization_config=bnb_config,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

model.save_pretrained("./models/int8")
tokenizer.save_pretrained("./models/int8")

mem_allocated = allocated_mem()
mem_reserved = reserved_mem()
print(f"Model loaded in INT8")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")