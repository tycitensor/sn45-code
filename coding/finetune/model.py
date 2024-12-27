import os
import time
import shutil
import psutil
import random
import asyncio
import requests
from tqdm import tqdm   
import bittensor as bt
from langchain_openai import ChatOpenAI
from sglang.utils import terminate_process
from coding.utils.shell import execute_shell_command

MODEL_DIR = "~/.cache/huggingface/hub"

# Delete the model from the huggingface cache when we're done serving it so we don't run out of disk space
def delete_model_from_hf_cache(model_name: str):
    # Determine the cache directory
    cache_dir = os.path.expanduser(MODEL_DIR)
    
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
        self.model_path = f"{model_name}"
        self.model_name = model_name
        self.server_process = None
        self.start_server()
        # random port between 12000 and 15999
        self.port = random.randint(12000, 15999)

    def invoke(self, messages: list[dict]):
        return self.llm.invoke(messages).content

    async def ainvoke(self, messages: list[dict]):
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def invoke_batch(self, message_batches: list[list[dict]], batch_size: int = 10):
        """Run multiple message batches in parallel
        
        Args:
            message_batches: List of message batches, where each batch is a list of message dicts
            batch_size: Number of batches to run in parallel
        
        Returns:
            List of responses in same order as input batches
        """
        results = []
        for i in tqdm(range(0, len(message_batches), batch_size), desc="Processing batches"):
            batch = message_batches[i:i + batch_size]
            # Run batch in parallel using asyncio
            async def run_batch():
                tasks = []
                for messages in batch:
                    tasks.append(self.llm.ainvoke(messages))
                responses = await asyncio.gather(*tasks)
                return [r.content for r in responses]
            
            # Run the async batch and collect results
            batch_results = asyncio.run(run_batch())
            results.extend(batch_results)
        return results

    def start_server(self):
        if "phi" not in self.model_name.lower():
            self.server_process = execute_shell_command(
                f"""
                {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server \
                --model {self.model_name} \
                --model-path {self.model_path} \
                --port {self.port} \ 
                --host 0.0.0.0 \
                --quantization fp8 \ 
                --mem-fraction-static 0.6 \
                --context-length 8096 \
                --disable-cuda-graph
                """,
                self.model_name
            )
        else:
            self.server_process = execute_shell_command(
                f"""
                {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server \
                --model {self.model_name} \
                --model-path {self.model_path} \
                --port {self.port} \ 
                --host 0.0.0.0 \
                --quantization fp8 \ 
                --mem-fraction-static 0.6 \
                --context-length 8096 \
                --attention-backend triton
                """,
                self.model_name
            )
        # Wait for the server to be ready
        try:
            wait_for_server(f"http://localhost:{self.port}", self.server_process, timeout=60*15)
        except Exception as e:
            terminate_process(self.server_process)
            self.server_process.kill()
            bt.logging.error(f"Finetune: Server did not become ready within timeout period")

            if "phi" not in self.model_name.lower():
                self.server_process = execute_shell_command(
                    f"""
                    {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server \
                    --model {self.model_name} \
                    --model-path {self.model_path} \
                    --port {self.port} \ 
                    --host 0.0.0.0 \
                    --mem-fraction-static 0.6 \
                    --context-length 8096 \
                    --disable-cuda-graph
                    """,
                    self.model_name
                )
            else:
                self.server_process = execute_shell_command(
                    f"""
                    {os.getcwd()}/.venvsglang/bin/python -m sglang.launch_server \
                    --model {self.model_name} \
                    --model-path {self.model_path} \
                    --port {self.port} \ 
                    --host 0.0.0.0 \
                    --mem-fraction-static 0.6 \
                    --context-length 8096 \
                    --attention-backend triton
                    """,
                    self.model_name
                )

            try:
                wait_for_server(f"http://localhost:{self.port}", self.server_process, timeout=60*15)
            except TimeoutError:
                bt.logging.error(f"Finetune: Server did not become ready within timeout period")
                self.cleanup()
                raise TimeoutError("Server did not become ready within timeout period")
            except Exception as e:
                bt.logging.error(f"Finetune: Error running model: {e}")
                self.server_process.kill()
                self.cleanup()
                raise Exception(f"Error running model: {e}")

        self.llm = ChatOpenAI(
            api_key="None",
            base_url=f"http://localhost:{self.port}/v1",
            model=self.model_name,
        )

    def cleanup(self):
        try:
            if self.server_process:
                try:
                    terminate_process(self.server_process)
                except:
                    pass
                self.server_process = None
            delete_model_from_hf_cache(self.model_name)
            self.server_process.kill()
        except Exception as e:
            pass

    def __del__(self):
        self.cleanup()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

if __name__ == "__main__":
    # Test the model server with a simple prompt
    model_name = "MistralAI/Mistral-7B-Instruct-v0.1"
    server = ModelServer(model_name)
    
    try:
        # Test basic invoke
        query = "What is 2+2?"
        response = server.invoke(query)
        print("Basic invoke test:")
        print(f"Response: {response}\n")

        # Test batch invoke
        queries = [f"What is {i}+{i}?" for i in range(3)]
        responses = server.invoke_batch(queries, batch_size=2)
        print("Batch invoke test:")
        for i, response in enumerate(responses):
            print(f"Batch {i} response: {response}")

    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        server.cleanup()