from coding.protocol import HFModelSynapse


def miner_init(self):
    pass
    
def miner_process(self, synapse: HFModelSynapse) -> HFModelSynapse:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    synapse.model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    synapse.prompt_tokens = {
        "prefix": "<｜fim▁begin｜>",
        "middle": "<｜fim▁hole｜>",
        "suffix": "<｜fim▁end｜>",
    }

    return synapse
