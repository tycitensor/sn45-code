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



This repository requires python3.11, follow the commands below to install it if you do not already have it.

ONLY RUN THE FOLLOWING COMMANDS IF YOU DO NOT HAVE PYTHON INSTALLED
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

Ensure that your python version is 3.11 before continuing:
```bash
python3 --version
```

If the above doesnt return `python3.11` try using the command `python3.11` instead. If the cmd `python3.11` works, use that in place of every python command below. 


After ensuring you have python run the following commands:
```bash
git clone https://github.com/brokespace/code
cd code
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pip uninstall uvloop # b/c it causes issues with threading/loops
```


##### Run the LLM image

The following command will run vllm on gpu:0. The `--gpu-memory-utilization` flag signifies how much of the gpu vllm will claim. 


```bash
sudo docker run -d -p 8028:8000  --gpus device=0 --ipc host --name mistral-instruct docker.io/vllm/vllm-openai:latest --model thesven/Mistral-7B-Instruct-v0.3-GPTQ --max-model-len 8912 --quantization gptq --dtype half --gpu-memory-utilization 0.5
```

#### Setup Wandb 

This is optional, but recommended make sure you login

```bash
wandb login
```


#### Setup your dotenv

Copy `.env.example` to `.env` - `cp .env.example .env`. Then edit the `.env` file with the github token you get below

#### Get a Github Token

We require github tokens, to get one follow the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens), or below.

1. Go to [Github](http://Github.com)
2. Open the top right menu and select `Settings`
3. Go to the bottom left and select `Developer Settings`
4. Go to either `Tokens (classic)` or `Fine-grained tokens`
5. Generate a new token and place it in the .env

#### Start the validator



```bash
source .venv/bin/activate
python3 neurons/validator.py
    --netuid 45
    --subtensor.network <finney/local/test>
    --neuron.device cuda
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
    --neuron.model_url # OPTIONAL, if you are hosting the model somewhere else other then port 8028
    --neuron.vllm_api_key # OPTIONAL, only use if your vllm instance has an api key requirement
    --wandb.on True # default is true but you can disable
```

