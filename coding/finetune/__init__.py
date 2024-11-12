import bittensor as bt
from pydantic import BaseModel
from typing import List, Tuple


from coding.tasks.schema import Task
from coding.model.schema import Model
from coding.finetune.score import score
from coding.rewards.codesim import CodeSimModel
from coding.model.tracker import gather_all_models

class FinetuneResult(BaseModel):
    score: float
    model: Model



class FinetunePipeline():
    def __init__(self, validator, tasks: List[Task], code_sim_model: CodeSimModel):
        self.validator = validator
        self.tasks = tasks
        self.code_sim_model = code_sim_model
        self.scores = []
        self.models = []
    
    @property
    def results(self) -> List[FinetuneResult]:
        return [FinetuneResult(score=score, model=model) for score, model in zip(self.scores, self.models)]
    
    def evaluate(self) -> List[FinetuneResult]:
        # gather all models
        models = gather_all_models(self.validator)
        # evaluate all models
        scores = []
        for model in models:
            scores.append(score(self.validator, model, self.tasks, self.code_sim_model))
        
        self.scores = scores
        self.models = models
        return self.results
    
    def get_top_model(self) -> Model:
        return sorted(self.models, key=lambda x: x.score, reverse=True)[0]
    
    def __str__(self):
        return f"{self.__class__.__name__}(scores={self.scores!r}, models={self.models!r})"
    
    def __repr__(self):
        return self.__str__()

    
    
    
    
    
            
