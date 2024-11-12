import os
import shutil
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory


def load_model_and_tokenizer(model_name: str, finetune_gpu_id: int) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map={"": finetune_gpu_id},
            cache_dir=f"model_cache_dir/{cache_path}",
        )

    except Exception as e:
        try:
            print(
                f"Error loading model in 4 bit quant with flash attention.: {e}. Trying vanilla load. This might cause OOM."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                cache_dir=f"model_cache_dir/{cache_path}",
                # force_download=True
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    return model, tokenizer

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