# Quickstart


## Dependencies

You must have the following things:

- System with at least 24gb of VRAM
- Python >=3.10
- Docker with [gpu support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting started


## Installation


This repository requires python3.9 or higher. To install it, simply clone this repository and run the [install.sh](./install.sh) script.
```bash
git clone https://github.com/brokespace/coding
cd coding
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip uninstall uvloop # b/c it causes issues with threading/loops
```


##### Run the LLM image

The following command will run vllm on gpu:0. The `--gpu-memory-utilization` flag signifies how much of the gpu vllm will claim. 

```bash
sudo docker run -d -p 8028:8000  --gpus device=0 --ipc host --name mistral-instruct docker.io/vllm/vllm-openai:latest --model thesven/Mistral-7B-Instruct-v0.3-GPTQ --max-model-len 8912 --quantization gptq --dtype half --gpu-memory-utilization 0.5
```