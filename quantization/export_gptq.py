#Generative Pre-trained Transformer Quantization - Post training quantization
import os
import itertools
from huggingface_hub import login
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.constants import allocated_mem, reserved_mem

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")
login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# load 128 samples from C4 or wikitext as calibration data
calib = load_dataset("allenai/c4", # dataset has 3 columns : "text", "timestamp", "url"
                     split="train", #gets the training split of the dataset
                     streaming=True) # streaming downloads samples one at a time as you need them
calib_samples = [
    tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=2048)
    for ex in itertools.islice(calib, 128)
]

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128, # group parameters in group of 128, find the scaling factor by taking in the min and max value in the group, quantize each parameter in the group to 4 bits using the scaling factor. smaller group size can lead to better accuracy but slower inference
    desc_act=False,  # False= left to right quantization. if True, most influencial column to least for quantization, slightly better accuracy, slower inference
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantize_config=quantize_config,
    device_map="auto"
)
model.quantize(calib_samples)

model.save_quantized("./models/gptq")
tokenizer.save_pretrained("./models/gptq") 

mem_allocated = allocated_mem()
mem_reserved = reserved_mem()
print(f"Model loaded in GPTQ")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")