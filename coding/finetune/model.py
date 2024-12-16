import os
import json
import torch
import shutil
from jinja2 import Template
from accelerate.utils import release_memory
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

def get_renderer(template_str: str):
    """
    Renders a Jinja2 template string with a list of dictionaries and additional context.

    Parameters:
        data (list): A list of dictionaries with data to render the template.
        template_str (str): A Jinja2 template string.
        kwargs (dict): Additional context for the template rendering.

    Returns:
        str: The rendered template as a string.
    """
    template = Template(template_str)
    def render(messages):
        return template.render(messages=messages)
    return render

def load_model_and_tokenizer(model_name: str, finetune_gpu_id: int) -> tuple[AutoModelForCausalLM, AutoTokenizer, callable]:
    # Replace any forward slashes with dashes in model name
    safe_model_name = model_name.replace('/', '-')
    
    cache_dir = os.path.join(os.getcwd(), "hf_cache") 
    cache_path = os.path.join(cache_dir, safe_model_name)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
        
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"model_cache_dir/{cache_path}",
    )
    
    input_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        force_download=True,
    )
    output_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        force_download=True,
    )
    if input_tokenizer.pad_token is None:
        input_tokenizer.pad_token = input_tokenizer.eos_token  # add a pad token if not present
        input_tokenizer.pad_token_id = input_tokenizer.eos_token_id
        output_tokenizer.pad_token = output_tokenizer.eos_token  # add a pad token if not present
        output_tokenizer.pad_token_id = output_tokenizer.eos_token_id
    # Load tokenizer config to get chat_template
    
    tokenizer_config = tokenizer.init_kwargs
    # Extract the chat template if it exists
    chat_template = tokenizer_config.get("chat_template")
    renderer = get_renderer(chat_template)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map={"": finetune_gpu_id},
            cache_dir=f"model_cache_dir/{cache_path}",
        )

    except Exception as err:
        try:
            print(
                f"Error loading model in 4 bit quant with flash attention.: {err}. Trying vanilla load. This might cause OOM."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                cache_dir=f"model_cache_dir/{cache_path}",
                # force_download=True
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    return model, tokenizer, renderer

def cleanup(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    # Release VRAM
    with torch.no_grad():
        model.cpu()
        release_memory(model)
        del model
    
    # Release tokenizer resources
    release_memory(tokenizer) 
    del tokenizer
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear model cache
    cache_dir = os.path.join(os.getcwd(), "hf_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # Clear model cache dir
    model_cache = os.path.join(os.getcwd(), "model_cache_dir") 
    if os.path.exists(model_cache):
        shutil.rmtree(model_cache)