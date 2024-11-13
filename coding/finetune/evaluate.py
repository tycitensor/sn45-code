from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, renderer: callable, query: str) -> str:
    messages = [{ "role": "user", "content": query }]
    inputs = tokenizer(renderer(messages), return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=16000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)