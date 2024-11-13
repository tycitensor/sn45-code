from typing import List

from coding.schemas.model import Model
from coding.protocol import HFModelSynapse
from coding.schemas.tracking import TrackingInfo
from coding.utils.uids import get_miner_uids, get_hotkey_from_uid

def gather_all_models(validator) -> List[TrackingInfo]:
    uids = get_miner_uids(validator)
    axons = [validator.metagraph.axons[uid] for uid in uids]
    synapse = HFModelSynapse()
    responses = []
    for axon in axons:
        try:
            responses.append(validator.dendrite.query(axons=[axon], synapse=synapse, timeout=45, deserialize=False)[0])
        except Exception as e:
            print("Error querying axon", axon, e)
            responses.append(synapse)
    return [
        TrackingInfo(
            model=Model(model_name=synapse.model_name, competition_id=synapse.competition_id, block=validator.metagraph.block),
            block=validator.metagraph.block,
            hotkey=get_hotkey_from_uid(validator, uids[i]),
            uid=uids[i],
        )
        for i, synapse in enumerate(responses) if synapse.model_name is not None
    ]