from typing import List
from pydantic import BaseModel


class TrackingInfo(BaseModel):
    logic: dict
    block: int # deprecated
    hotkey: str
    uid: int
    score: float = 0.0
    score_timestamps: List[int] = [] # timestamp is the block number
