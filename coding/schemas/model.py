from pydantic import BaseModel

class Model(BaseModel):
    model_name: str
    prompt_tokens: dict
    hash: str
    competition_id: str
    block: int
    
    
    def to_compressed_str(self) -> str:
        return f"{self.model_name}-{self.hash}-{self.competition_id}-{self.block}-{self.hotkey}"
    
    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "Model":
        model_name, hash, competition_id, block, hotkey = compressed_str.split("-")
        return cls(model_name=model_name, hash=hash, competition_id=competition_id, block=block, hotkey=hotkey)
    
    
