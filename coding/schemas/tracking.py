from typing import List
from pydantic import BaseModel

from .model import Model

class TaskResult(BaseModel):
    score: float
    logic: dict
    commit_hash: str

class TrackingInfo(BaseModel):
    logic: dict
    block: int
    hotkey: str
    uid: int
    results: List[TaskResult] = []
