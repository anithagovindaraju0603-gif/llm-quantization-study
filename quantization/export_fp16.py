import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import allocated_mem, reserved_mem

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face token.")
login(token=hf_token)

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', 
                                             torch_dtype=torch.float16,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

model.save_pretrained("./models/fp16")
tokenizer.save_pretrained("./models/fp16")

mem_allocated = allocated_mem()
mem_reserved = reserved_mem()
print(f"Model loaded in FP16")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")