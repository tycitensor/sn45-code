import os
import copy
import pickle
import atexit
import weakref
import traceback
import bittensor as bt
from typing import List
from pydantic import BaseModel

from .tracker import gather_all_trackers

from coding.finetune.score import score
from coding.schemas.context import Context
from coding.constants import COMPETITION_ID
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo

from coding.tasks.bigcodebench import BigCodeBenchTask
from coding.datasets.bigcodebench import BigCodeBenchDataset


    
    
class FinetuneEventResults(BaseModel):
    trackers: List[TrackingInfo]
    competition_id: int = COMPETITION_ID
    
    def __state_dict__(self):
        return {
            "trackers": [tracker.model_dump() for tracker in self.trackers],
            "competition_id": self.competition_id,
        }


def generate_bigcode_tasks(ds: BigCodeBenchDataset, n: int = 1) -> List[BigCodeBenchTask]:
    tasks = []
    while len(tasks) < n:
        try:
            tasks.append(BigCodeBenchTask(context=Context(**ds.get())))
            bt.logging.info(f"Finetune: Task progress: {len(tasks)}/{n}")
        except Exception as e:
            bt.logging.error(f"Finetune: Error generating task: {e}")
            continue
    return tasks

def bittensor_injector(self):
    self.wallet = bt.wallet(config=self.config)
    self.dendrite = bt.dendrite(wallet=self.wallet)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)


class FinetunePipeline:
    def __init__(self, config, competition_id: int):
        self.config = config
        bittensor_injector(self)
        self.competition_id = competition_id
        self.code_sim_model = None
        self.trackers: List[TrackingInfo] = []
        self.dataset = BigCodeBenchDataset(config=self.config)
        
        new_trackers = self.load_unfinished_trackers()
        if len(new_trackers) > 0:
            bt.logging.info(f"Finetune: Found {len(new_trackers)} unfinished trackers.")
        else:
            new_trackers = gather_all_trackers(self)
            bt.logging.info(f"Finetune: Gathered {len(new_trackers)} trackers.")
            self.store_unfinished_trackers(new_trackers)
        
        self.load_tasks()
        self.load_results()
        
        # Register cleanup to be called when the object is deleted
        # self._finalizer = weakref.finalize(self, lambda: FinetunePipeline.cleanup())

    def load_tasks(self):
        if os.path.exists(f"{self.config.neuron.full_path}/tasks_{self.competition_id}.pkl"):
            with open(f"{self.config.neuron.full_path}/tasks_{self.competition_id}.pkl", "rb") as f:
                self.tasks = pickle.load(f)[:self.config.neuron.finetune_test_size]
        else:
            self.tasks = generate_bigcode_tasks(self.dataset, self.config.neuron.finetune_test_size)
            self.store_tasks()
            
    def load_results(self):
        results_file = f"{self.config.neuron.full_path}/results_{self.competition_id}.pkl"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                saved_results = pickle.load(f)
                self.trackers = saved_results.get("trackers", [])
            
    @property
    def results(self) -> FinetuneEventResults:
        return FinetuneEventResults(
            trackers=self.trackers,
            competition_id=self.competition_id
        ).__state_dict__()
    
    def evaluate(self) -> FinetuneEventResults:
        # gather all models
        bt.logging.info("Finetune: Gathering all models...")
        new_trackers = self.load_unfinished_trackers()
        if len(new_trackers) > 0:
            bt.logging.info(f"Finetune: Found {len(new_trackers)} unfinished trackers.")
        else:
            new_trackers = gather_all_trackers(self)
            bt.logging.info(f"Finetune: Gathered {len(new_trackers)} trackers.")
            self.store_unfinished_trackers(new_trackers)
        
        for tracking_info in new_trackers:
            
            # Check if the model has already been scored
            previous_score = next((tracker.score for tracker in self.trackers if tracker.model.model_name == tracking_info.model.model_name), None)
            if previous_score is not None:
                bt.logging.info(f"Finetune: Using previously evaluated score for hotkey: {tracking_info.hotkey}")
                tracking_info.score = previous_score
                self.trackers.append(tracking_info)
                self.store_results()
                continue
            
            # if tracking_info in self.trackers:
            #     # bt.logging.info(f"Finetune: Skipping already evaluated model: {tracking_info.model}")
            #     continue
            
            bt.logging.info(f"Finetune: Evaluating hotkey: {tracking_info.hotkey}")
            
            # ensure competition_id is equal to current competition_id
            if tracking_info.model.competition_id != COMPETITION_ID:
                tracking_info.score = 0.0
                continue
            
            try:
                model_score = score(
                    self, tracking_info.model.model_name, self.tasks)
                bt.logging.info(f"Finetune: Model score from FinetunePipeline: {model_score}")
                tracking_info.score = model_score
            except Exception as e:
                bt.logging.error(f"Finetune: Error scoring model: {e}")
                bt.logging.error(f"Finetune: Error traceback: {traceback.format_exc()}")
                tracking_info.score = 0.0
            
            # Save intermediate results after each model evaluation
            self.trackers.append(tracking_info)
            self.store_results()

        bt.logging.info(f"Finetune: All scores from FinetunePipeline: {[tracker.score for tracker in self.trackers]}")
        return self.results
    
    def get_top_model(self) -> TrackingInfo:
        # Zip scores with models and sort by score
        return sorted(self.results, key=lambda x: x.score, reverse=True)[0]

    def __str__(self):
        return f"{self.__class__.__name__}(models={self.trackers!r})"

    def __repr__(self):
        return self.__str__()

    def __state_dict__(self):
        return {
            "trackers": [tracker.model_dump() for tracker in self.trackers],
        }
    
    def store_tasks(self):
        with open(f"{self.config.neuron.full_path}/tasks_{self.competition_id}.pkl", "wb") as f:
            pickle.dump(self.tasks, f)
    
    def load_unfinished_trackers(self):
        try:
            with open(f"{self.config.neuron.full_path}/trackers_{self.competition_id}.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []
    
    def store_unfinished_trackers(self, trackers: List[TrackingInfo]):
        with open(f"{self.config.neuron.full_path}/trackers_{self.competition_id}.pkl", "wb") as f:
            pickle.dump(trackers, f)
    
    def store_results(self):
        with open(f"{self.config.neuron.full_path}/results_{self.competition_id}.pkl", "wb") as f:
            pickle.dump({
                "trackers": self.trackers
            }, f)
    
    # def cleanup():
    #     """
    #     Delete the tasks file and any other task files
    #     """
    #     try:
    #         os.remove(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl")
    #     except FileNotFoundError:
    #         pass
            
    #     # check if tasks_*.pkl exists and delete it if it does
    #     for file in os.listdir(self.config.neuron.full_path):
    #         if file.startswith("tasks_") and file.endswith(".pkl"):
    #             try:
    #                 os.remove(file)
    #             except FileNotFoundError:
    #                 pass
    #         if file.startswith("results_") and file.endswith(".pkl"):
    #             try:
    #                 os.remove(file)
    #             except FileNotFoundError:
    #                 pass
                
    @staticmethod
    def start(config, code_sim_model) -> FinetuneEventResults:
        pipeline = FinetunePipeline(config, code_sim_model)
        result = pipeline.evaluate()
        # pipeline.cleanup()  # Ensure cleanup is called after evaluation
        return result
    
# Register cleanup to be called when the process exits
# atexit.register(FinetunePipeline.cleanup)