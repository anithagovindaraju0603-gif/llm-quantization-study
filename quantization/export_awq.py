import os
import itertools
from huggingface_hub import login
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.constants import allocated_mem, reserved_mem

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")
login(token=hf_token)

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# load 128 samples from C4 or wikitext as calibration data
calib = load_dataset("allenai/c4", # dataset has 3 columns : "text", "timestamp", "url"
                     "en", # english subset
                     split="train", #gets the training split of the dataset
                     streaming=True) # streaming downloads samples one at a time as you need them
calib_samples = [
    ex["text"] for ex in itertools.islice(calib, 128)
]# Raw text. AWQ's .quantize() method handles tokenization internally. When you pass tokenizer as the first argument to .quantize(), you're giving AWQ the tool it needs to tokenize the text itself.

quant_config = {
    "zero_point": True, #asymmetric quantization, uses all 16 values efficiently
    "q_group_size": 128, #128 weights per scaling group, same as GPTQ
    "w_bit": 4, # compress weights to 4 bit
    "version": "GEMM"  # use the matrix-matrix kernel, faster for batch size > 1, GEMV is for very small batches
}
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_samples)

model.save_quantized("./models/awq")
tokenizer.save_pretrained("./models/awq")

mem_allocated = allocated_mem()
mem_reserved = reserved_mem()
print(f"Model loaded in AWQ")
print(f"Memory allocated: {mem_allocated:.2f} GB")
print(f"Memory reserved: {mem_reserved:.2f} GB")

#NOTES:
# zero_point=False → symmetric  → range centered at zero → wastes buckets if weights are skewed
# zero_point=True  → asymmetric → range shifts to fit actual weights → all 16 buckets used efficiently → better precision