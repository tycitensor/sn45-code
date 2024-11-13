from pydantic import BaseModel

class Model(BaseModel):
    model_name: str
    # prompt_tokens: dict
    # hash: str
    competition_id: str
    block: int
    
    
    def to_compressed_str(self) -> str:
        return f"{self.model_name}-{self.competition_id}-{self.block}"
    
    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "Model":
        model_name, competition_id, block = compressed_str.split("-")
        return cls(model_name=model_name, competition_id=competition_id, block=block)
    
    
