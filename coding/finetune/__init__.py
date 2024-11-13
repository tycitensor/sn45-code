import bittensor as bt
from pydantic import BaseModel
from typing import List, Tuple

from .tracker import gather_all_models

from coding.tasks.schema import Task
from coding.schemas.model import Model
from coding.finetune.score import score
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo



class FinetuneResult(BaseModel):
    score: float
    tracking_info: TrackingInfo



class FinetunePipeline():
    def __init__(self, validator, tasks: List[Task], code_sim_model: CodeSimModel):
        self.validator = validator
        self.tasks = tasks
        self.code_sim_model = code_sim_model
        self.scores = []
        self.tracking_models: List[TrackingInfo] = []
    
    @property
    def results(self) -> List[FinetuneResult]:
        return [FinetuneResult(score=score, tracking_info=tracking_info) for score, tracking_info in zip(self.scores, self.tracking_models)]
    
    def evaluate(self) -> List[FinetuneResult]:
        # gather all models
        self.tracking_models = gather_all_models(self.validator)
        # evaluate all models
        scores = []
        for tracking_info in self.tracking_models:
            scores.append(score(self.validator, tracking_info.model, self.tasks, self.code_sim_model))
        
        self.scores = scores
        return self.results
    
    def get_top_model(self) -> TrackingInfo:
        # Zip scores with models and sort by score
        return sorted(self.results, key=lambda x: x.score, reverse=True)[0]
    
    def __str__(self):
        return f"{self.__class__.__name__}(scores={self.scores!r}, models={self.tracking_models!r})"
    
    def __repr__(self):
        return self.__str__()

    
    
    
    
    
            
