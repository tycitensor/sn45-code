from pydantic import BaseModel

from .model import Model

class TrackingInfo(BaseModel):
    model: Model
    block: int
    hotkey: str
    uid: int