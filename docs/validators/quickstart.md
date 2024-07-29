# Quickstart


## Dependencies

You must have the following things:

- System with at least 24gb of VRAM
- Python >=3.10
- Docker with [gpu support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting started


## Installation

Ensure that you have Docker with GPU support, you can choose to follow either of the instructions:

- [Official Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 
- [Quick and Dirty Stack Overflow Guide](https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab)



This repository requires python3.9 or higher. To install it, simply clone this repository and run the [install.sh](./install.sh) script.
```bash
git clone https://github.com/brokespace/code
cd code
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip uninstall uvloop # b/c it causes issues with threading/loops
```


##### Run the LLM image

The following command will run vllm on gpu:0. The `--gpu-memory-utilization` flag signifies how much of the gpu vllm will claim. 


```bash
sudo docker run -d -p 8028:8000  --gpus device=0 --ipc host --name mistral-instruct docker.io/vllm/vllm-openai:latest --model thesven/Mistral-7B-Instruct-v0.3-GPTQ --max-model-len 8912 --quantization gptq --dtype half --gpu-memory-utilization 0.5
```


#### Start the validator

```bash
python <SCRIPT_PATH>
    --netuid 1
    --subtensor.network <finney/local/test>
    --neuron.device cuda
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
    --neuron.model_url # OPTIONAL, if you are hosting the model somewhere else other then port 8028
```
