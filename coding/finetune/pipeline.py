import os
import pickle
import atexit
import weakref
import requests
import bittensor as bt
from typing import List
from pydantic import BaseModel

from coding.constants import COMPETITION_ID
from .tracker import gather_all_logics
from .dockerutil import build_docker_container, run_docker_container
from .git import GitRepo

from coding.schemas import Patch
from coding.finetune.score import score
from coding.schemas.context import Context
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo, TaskResult

from coding.tasks.swe import SWEBenchTask
from coding.datasets.swe import SWEBenchDataset
class FinetuneEventResults(BaseModel):
    scores: List[float]
    tracking_infos: List[TrackingInfo]
    
    def __state_dict__(self):
        return {
            "scores": self.scores,
            "tracking_infos": [model.model_dump() for model in self.tracking_infos],
        }


def generate_swe_tasks(ds: SWEBenchDataset, n: int = 1000) -> List[SWEBenchTask]:
    tasks = []
    for _ in range(n):
        tasks.append(SWEBenchTask(context=Context(**ds.get())))
    return tasks

def bittensor_injector(self):
    self.wallet = bt.wallet(config=self.config)
    self.dendrite = bt.dendrite(wallet=self.wallet)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)


class FinetunePipeline:
    def __init__(self, config, competition_id: str, code_sim_model: CodeSimModel = CodeSimModel()):
        self.config = config
        bittensor_injector(self)
        self.competition_id = competition_id
        self.code_sim_model = code_sim_model
        self.scores = []
        self.tracking_logics: List[TrackingInfo] = []
        self.dataset = SWEBenchDataset()
        self.load_tasks()
        self.load_results()

        # Register cleanup to be called when the object is deleted
        self._finalizer = weakref.finalize(self, self.cleanup)

    def load_tasks(self):
        if os.path.exists(f"tasks_{self.competition_id}.pkl"):
            with open(f"tasks_{self.competition_id}.pkl", "rb") as f:
                self.tasks = pickle.load(f)
        else:
            self.tasks = generate_swe_tasks(self.dataset, self.config.finetune_test_size)
            self.store_tasks()
            
    def load_results(self):
        results_file = f"results_{self.competition_id}.pkl"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                saved_results = pickle.load(f)
                self.scores = saved_results.get("scores", [])
                self.tracking_logics = saved_results.get("tracking_logics", [])
            
    @property
    def results(self) -> FinetuneEventResults:
        return FinetuneEventResults(scores=self.scores, tracking_infos=self.tracking_logics)

    # first need to gather all logics
    # build all logic containers 
    # go through each task and get the repo, copy into container, run the logic, add to logic scores
    def evaluate(self) -> FinetuneEventResults:
        # gather all logics
        bt.logging.info("Gathering all logics...")
        self.tracking_logics = gather_all_logics(self)
        bt.logging.info(f"Gathered {len(self.tracking_logics)} logics.")
        for tracker in self.tracking_logics:
            build_docker_container(tracker.logic, tracker.hotkey)
        
        for task in self.tasks:
            repo = GitRepo(task.repo, task.base_commit)
            for tracker in self.tracking_logics:
                container = run_docker_container(repo, tracker.hotkey, tracker.llm_name)
                response = requests.post(f"http://localhost:3000/call", json={"repo_location": "/app/repo", "issue_description": task.issue})
                try:
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    bt.logging.error(f"Request failed: {e}")
                    continue
                result = response.json()["result"]
                patch = Patch(**result)
                score = task.score(patch)
                tracker.results.append(TaskResult(repo_name=tracker.logic, commit_hash=task.base_commit, score=score))
                bt.logging.info(f"Result from SWE: {result}")
                container.stop()
                container.remove()
        # average the scores for each tracker
        scores = []
        for tracker in self.tracking_logics:
            scores.append(sum(tracker.results.score) / len(tracker.results.score))
        self.scores = scores
        self.store_results()
                

    def __str__(self):
        return f"{self.__class__.__name__}(scores={self.scores!r}, models={self.tracking_logics!r})"

    def __repr__(self):
        return self.__str__()

    def __state_dict__(self):
        return {
            "scores": self.scores,
            "tracking_logics": [model.model_dump() for model in self.tracking_logics],
        }
    
    @staticmethod
    def start(config, code_sim_model: CodeSimModel = CodeSimModel()) -> FinetuneEventResults:
        pipeline = FinetunePipeline(config, code_sim_model)
        result = pipeline.evaluate()
        pipeline.cleanup()  # Ensure cleanup is called after evaluation
        return result

    def store_tasks(self):
        with open(f"tasks_{self.competition_id}.pkl", "wb") as f:
            pickle.dump(self.tasks, f)
            
    def store_results(self):
        with open(f"results_{self.competition_id}.pkl", "wb") as f:
            pickle.dump({
                "scores": self.scores,
                "tracking_logics": self.tracking_logics
            }, f)
    
    def cleanup(self):
        """
        Delete the tasks file and any other task files
        """
        os.remove(f"tasks_{self.competition_id}.pkl")
        # check if tasks_*.pkl exists and delete it if it does
        for file in os.listdir("."):
            if file.startswith("tasks_") and file.endswith(".pkl"):
                os.remove(file)
            if file.startswith("results_") and file.endswith(".pkl"):
                os.remove(file)

# Register cleanup to be called when the process exits
atexit.register(FinetunePipeline.cleanup)