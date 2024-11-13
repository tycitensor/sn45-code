# Finetuning

You must submit a model to hugginface to be used for finetuning. Then you must ensure that the code within [`coding/miners/finetune.py`](https://github.com/brokespace/code/blob/main/coding/miners/finetune.py) is updated with the correct model name and competition id. This synapse will be hit at the beginning of each competition, at this time you must ensure that your miner is accessable.

The model you submit must follow the guidlines for the current competition. 

The model must contain a tokenizer_config.json with the key "chat_template" and the value being a jinja template representing the chat template. An example can be found [here](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/tokenizer_config.json).