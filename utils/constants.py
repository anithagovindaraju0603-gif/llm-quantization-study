import torch

def allocated_mem():
    return torch.cuda.memory_allocated() / 1e9

def reserved_mem():
    return torch.cuda.memory_reserved() /1e9

def max_mem_allocated():
    return torch.cuda.max_memory_allocated() / 1e9

def make_input(length, tokenizer, batch_size=1):
    prompt = "hello " * length
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=length
    ).to("cuda")
    if batch_size > 1:
        tokens["input_ids"] = tokens["input_ids"].expand(batch_size, -1)
        tokens["attention_mask"] = tokens["attention_mask"].expand(batch_size, -1)
    return tokens

def warmup(model, tokenizer):
    # warmup - dummy run to load models and avoid including loading time in generation trace
    dummy = make_input(512, tokenizer)
    for _ in range(3):
        with torch.no_grad():
            model.generate(**dummy, max_new_tokens=10, do_sample=False) # do_sample=False for deterministic output during warmup, picks single most likely next token instead of sampling from distribution.
    torch.cuda.synchronize()
