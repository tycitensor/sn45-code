from typing import List

from coding.protocol import LogicSynapse
from coding.schemas.tracking import TrackingInfo
from coding.utils.uids import get_miner_uids, get_hotkey_from_uid

def gather_all_logics(validator) -> List[TrackingInfo]:
    uids = get_miner_uids(validator)
    axons = [validator.metagraph.axons[uid] for uid in uids]
    synapse = LogicSynapse()
    responses = validator.dendrite.query(axons=axons, synapse=synapse, timeout=45, deserialize=False)
    # for axon in axons:
    #     try:
    #         responses.append(validator.dendrite.query(axons=[axon], synapse=synapse, timeout=45, deserialize=False)[0])
    #     except Exception as e:
    #         print("Error querying axon", axon, e)
    #         responses.append(synapse)
    return [
        TrackingInfo(
            logic=synapse.logic,
            block=validator.metagraph.block,
            hotkey=get_hotkey_from_uid(validator, uids[i]),
            uid=uids[i],
            score=0.0,
            score_timestamps=[],
        )
        for i, synapse in enumerate(responses)
    ]

def regrab_tracker(tracker: TrackingInfo, validator) -> TrackingInfo:
    uid = tracker.uid
    if tracker.hotkey != get_hotkey_from_uid(validator, uid):
        return tracker
    synapse = LogicSynapse()
    logic_synapse = validator.dendrite.query(axons=[validator.metagraph.axons[uid]], synapse=synapse, timeout=45, deserialize=False)[0]
    return TrackingInfo(
        logic=logic_synapse.logic,
        block=validator.metagraph.block,
        hotkey=get_hotkey_from_uid(validator, uid),
        uid=uid,
        score=0.0,
    )
