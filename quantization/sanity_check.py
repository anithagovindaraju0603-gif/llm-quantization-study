import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM   

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True)

#model_names = ["fp16", "int8", "gptq", "awq"]
model_names = ["fp16", "int8"]
for model_name  in model_names:
    print(f"Loading {model_name }...")
    if model_name == "fp16":
        model = AutoModelForCausalLM.from_pretrained(f"./models/{model_name}", 
                                                     torch_dtype=torch.float16,
                                                    device_map="auto")
    elif model_name == "int8":
        model = AutoModelForCausalLM.from_pretrained(f"./models/{model_name}", 
                                                    quantization_config=bnb_config,
                                                    device_map="auto")
    # elif model_name == "gptq":
    #     model = AutoGPTQForCausalLM.from_quantized(f"./models/{model_name}", device_map="auto")
    # elif model_name == "awq":
    #     model = AutoAWQForCausalLM.from_quantized(f"./models/{model_name}", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(f"./models/{model_name}")

    prompt = "What are the benefits of quantizing large language models?"
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response from {model_name}:\n{response}\n")

    del model
    torch.cuda.empty_cache()