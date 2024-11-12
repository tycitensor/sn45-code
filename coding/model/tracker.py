from typing import List

from coding.model.schema import Model
from coding.utils.uids import get_miner_hotkeys
from coding.model.storage.chain import ChainModelMetadataStore

def gather_all_models(validator) -> List[Model]:
    hotkeys = get_miner_hotkeys(validator.metagraph)
    metadata_store = ChainModelMetadataStore(validator.subtensor, validator.config.netuid)
    models = []
    for hotkey in hotkeys:
        try:
            model = metadata_store.retrieve_model_metadata(hotkey)
        except Exception as e:
            print(f"Error retrieving model metadata for hotkey {hotkey}: {e}") # TODO make logging
            continue
        models.append(model)
    return models
