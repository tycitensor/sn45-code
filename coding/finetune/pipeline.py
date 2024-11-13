import bittensor as bt
from typing import List
from pydantic import BaseModel

from coding.constants import COMPETITION_ID
from .tracker import gather_all_models

from coding.finetune.score import score
from coding.schemas.context import Context
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo

from coding.tasks.bigcodebench import BigCodeBenchTask
from coding.datasets.bigcodebench import BigcodeBenchDataset

class FinetuneEventResults(BaseModel):
    scores: List[float]
    tracking_infos: List[TrackingInfo]
    
    def __state_dict__(self):
        return {
            "scores": self.scores,
            "tracking_infos": [model.model_dump() for model in self.tracking_infos],
        }


def generate_bigcode_tasks(ds: BigcodeBenchDataset, n: int = 1) -> List[BigCodeBenchTask]:
    tasks = []
    for _ in range(n):
        tasks.append(BigCodeBenchTask(context=Context(**ds.get())))
    return tasks

def bittensor_injector(self):
    self.wallet = bt.wallet(config=self.config)
    self.dendrite = bt.dendrite(wallet=self.wallet)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)

class FinetunePipeline:
    def __init__(self, config, code_sim_model: CodeSimModel = CodeSimModel()):
        self.config = config
        bittensor_injector(self)
        self.code_sim_model = code_sim_model
        self.scores = []
        self.tracking_models: List[TrackingInfo] = []
        self.dataset = BigcodeBenchDataset()
        self.tasks = generate_bigcode_tasks(self.dataset, self.config.finetune_test_size)
    
    
    @property
    def results(self) -> FinetuneEventResults:
        return FinetuneEventResults(scores=self.scores, tracking_infos=self.tracking_models)

    def evaluate(self) -> FinetuneEventResults:
        # gather all models
        bt.logging.info("Gathering all models...")
        self.tracking_models = gather_all_models(self)
        bt.logging.info(f"Gathered {len(self.tracking_models)} models.")
        
        # evaluate all models
        scores = []
        for tracking_info in self.tracking_models:
            bt.logging.info(f"Evaluating model: {tracking_info.model}")
            # ensure competition_id is equal to current competition_id
            if tracking_info.model.competition_id != COMPETITION_ID:
                scores.append(0.0)
                continue
            model_score = score(
                self, tracking_info.model.model_name, self.tasks, self.code_sim_model
            )
            bt.logging.info(f"Model score: {model_score}")
            scores.append(model_score)

        self.scores = scores
        bt.logging.info(f"All scores: {self.scores}")
        return self.results
    
    def get_top_model(self) -> TrackingInfo:
        # Zip scores with models and sort by score
        return sorted(self.results, key=lambda x: x.score, reverse=True)[0]

    def __str__(self):
        return f"{self.__class__.__name__}(scores={self.scores!r}, models={self.tracking_models!r})"

    def __repr__(self):
        return self.__str__()

    def __state_dict__(self):
        return {
            "scores": self.scores,
            "tracking_models": [model.model_dump() for model in self.tracking_models],
        }
    
    @staticmethod
    def start(config, code_sim_model: CodeSimModel = CodeSimModel()) -> FinetuneEventResults:
        pipeline = FinetunePipeline(config, code_sim_model)
        return pipeline.evaluate()
        