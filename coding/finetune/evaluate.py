from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_tokens: dict, query: str) -> str:
    inputs = tokenizer(prompt_tokens["prefix"] + query.replace("<|fim_hole|>", prompt_tokens["middle"]) + prompt_tokens["suffix"], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)