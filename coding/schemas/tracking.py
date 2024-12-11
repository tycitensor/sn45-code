from typing import List
from pydantic import BaseModel

from .model import Model

class TaskResult(BaseModel):
    score: float
    repo_name: str
    commit_hash: str

class TrackingInfo(BaseModel):
    logic: dict
    block: int
    hotkey: str
    uid: int
    llm_name: str
    results: List[TaskResult] = []
