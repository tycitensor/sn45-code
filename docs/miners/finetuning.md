# Finetuning

You must submit a model to huggingface to be used for finetuning. Then you must ensure that the code within [`coding/miners/finetune.py`](https://github.com/brokespace/code/blob/finetune/coding/miners/finetune.py) is updated with the correct model name and competition id. The code in `finetune.py` will be hit at the beginning of each competition. At this time you must ensure that your miner is accessible.

The model you submit must follow the guidelines for the current competition. The guidelines will be posted in the discord channel for the competition.

The model must contain a `tokenizer_config.json` with a key `chat_template` and the value of the key being a jinja template representing the chat template. An example can be found [here](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/tokenizer_config.json).
