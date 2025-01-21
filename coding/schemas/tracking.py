from typing import List
from pydantic import BaseModel

from .model import Model

class TrackingInfo(BaseModel):
    logic: dict
    block: int
    hotkey: str
    uid: int
    score: float = 0.0
