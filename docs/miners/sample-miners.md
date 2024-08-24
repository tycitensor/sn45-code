# Sample Miners


## Qwen Mistral Miner

To get started on this miner you are going to want two models, `CodeQwen` and `Mistral`. This setup requires quite a bit of VRAM, I would suggest at a minimum 24gb of VRAM.



### Starting LLM's

```pip install vllm```

Then start the LLM's

```
pm2 start --name "mistral" "vllm serve thesven/Mistral-7B-Instruct-v0.3-GPTQ  --max-model-len 4096 --quantization gptq --dtype half --gpu-memory-utilization 0.40 --port 8001"
```

```
pm2 start --name "qwen" "vllm serve Qwen/CodeQwen1.5-7B-AWQ  --max-model-len 4096 --quantization gptq --dtype half --gpu-memory-utilization 0.40 --port 8000"
```


### Starting the Miner


```
pm2 start neurons/miner.py --interpreter python3 --name miner -- --netuid 45 --subtensor.network finney --wallet.name coldkey --wallet.hotkey hotkey --neuron.model_id Qwen/CodeQwen1.5-7B-AWQ --axon.port 8091 --logging.debug --miner.name qwen_mistral
```