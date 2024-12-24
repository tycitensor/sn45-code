import os
import time
import shutil
import psutil
import asyncio
import requests
import bittensor as bt
from langchain_openai import ChatOpenAI
from sglang.utils import terminate_process

from coding.utils.shell import execute_shell_command

MODEL_DIR = "~/.cache/huggingface/hub"

# Delete the model from the huggingface cache when we're done serving it so we don't run out of disk space
def delete_model_from_hf_cache(self, model_name: str):
    # Determine the cache directory
    cache_dir = os.path.expanduser(self.config.validator_hf_cache_dir)
    
    # Format the directory name based on the model name
    model_cache_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    
    # Check if the directory exists and delete it
    if os.path.exists(model_cache_dir):
        try:
            shutil.rmtree(model_cache_dir)
            bt.logging.debug(f"Finetune: Model has been removed from the HF cache.")
        except Exception as e:
            bt.logging.error(f"Finetune: Error deleting model: from HF cache: {e}")
    else:
        bt.logging.debug(f"Finetune: Model not found in the cache, could not delete")

def wait_for_server(base_url: str, server_process, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        server_process: The process to terminate if the server is ready
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    procutil = psutil.Process(int(server_process.pid))
    while True:
        try:
            if timeout and time.time() - start_time > timeout:
                bt.logging.error(f"Finetune: Server did not become ready within timeout period")
                raise TimeoutError("Server did not become ready within timeout period")

            # Use psutil to monitor the process
            if not procutil.is_running():  # Check if process is still running
                bt.logging.error(f"Finetune: Server process terminated unexpectedly, check VRAM usage")
                raise Exception("Server process terminated unexpectedly, potentially VRAM usage issue")
            if server_process.poll() is not None:
                bt.logging.error(f"Finetune: Server process terminated with code {server_process.poll()}")
                raise Exception(f"Server process terminated with code {server_process.poll()}")

            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)   
                break

        except requests.exceptions.RequestException:
            time.sleep(1)


class ModelServer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.server_process = None
        self.start_server()
        self.llm = None

    def invoke(self, messages: list[dict]):
        return self.llm.invoke(messages).content

    def start_server(self):
        self.server_process = execute_shell_command(
            f"""
            {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server \
            --port 12000 \ 
            --host 0.0.0.0 \
            --mem-fraction-static 0.5 \
            --context-length 25000
            """,
            self.model_name
        )

        try:
            wait_for_server(f"http://localhost:12000", self.server_process, timeout=60*10)
        except TimeoutError:
            bt.logging.error(f"Finetune: Server did not become ready within timeout period")
            self.cleanup()
            raise TimeoutError("Server did not become ready within timeout period")
        except Exception as e:
            bt.logging.error(f"Finetune: Error running model: {e}")
            self.cleanup()
            raise Exception(f"Error running model: {e}")

        self.llm = ChatOpenAI(
            api_key="None",
            base_url="http://localhost:12000",
            model=self.model_name,
        )

    def cleanup(self):
        if self.server_process:
            try:
                terminate_process(self.server_process)
            except:
                pass
            self.server_process = None
        delete_model_from_hf_cache(self.model_name)


    def __del__(self):
        self.cleanup()