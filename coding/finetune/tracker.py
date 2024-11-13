from typing import List

from coding.protocol import HFModelSynapse
from coding.schemas.tracking import TrackingInfo
from coding.utils.uids import get_miner_uids, get_hotkey_from_uid


def gather_all_models(validator) -> List[TrackingInfo]:
    uids = get_miner_uids(validator)
    axons = [validator.metagraph.axons[uid] for uid in uids]
    synapse = HFModelSynapse()
    responses = validator.dendrite.foward(axons=axons, synapse=synapse, timeout=45, deserialize=False)
    return [
        TrackingInfo(
            model=response.synapse.model,
            block=validator.metagraph.block,
            hotkey=get_hotkey_from_uid(validator.metagraph, uids[i]),
            uid=uids[i],
        )
        for i, response in enumerate(responses)
    ]