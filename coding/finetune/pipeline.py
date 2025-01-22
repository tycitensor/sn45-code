import os
import copy
import pickle
import argparse
import requests
import traceback
from time import sleep
import bittensor as bt
from typing import List
from pydantic import BaseModel
from .tracker import gather_all_logics
from .dockerutil import build_docker_container, run_docker_container, run_docker_container_from_base
from ..helpers.git import GitRepo

from coding.schemas import Patch
from coding.schemas.context import Context
from coding.constants import COMPETITION_ID
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo
from coding.constants import COMPETITION_ID, ALLOWED_MODULES, NUM_ALLOWED_CHARACTERS, ALLOWED_IMPORTS

from coding.tasks.swe import SWEBenchTask
from coding.datasets.swe import SWEBenchDataset
from coding.finetune.llm.manager import LLMManager
from coding.helpers.codeanal import verify_code_usage
from coding.utils.config import config as util_config
from coding.utils.config import add_validator_args



    
    
class FinetuneEventResults(BaseModel):
    trackers: List[TrackingInfo]
    competition_id: int = COMPETITION_ID
    
    def __state_dict__(self):
        return {
            "trackers": [tracker.model_dump() for tracker in self.trackers],
            "competition_id": COMPETITION_ID,
        }
    
    def public_state_dict(self):
        trackers = [tracker.model_dump() for tracker in self.trackers]
        for tracker in trackers:
            tracker["model"] = None
        return {
            "trackers": trackers,
            "competition_id": COMPETITION_ID,
        }



def generate_swe_tasks(ds: SWEBenchDataset, n: int = 1000) -> List[SWEBenchTask]:
    tasks = []
    while len(tasks) < n:
        try:
            tasks.append(SWEBenchTask(llm=None, context=Context(**ds.get())))
        except Exception as e:
            bt.logging.error(f"Error generating task: {e}")
            print(traceback.format_exc())
    return tasks


def bittensor_injector(self):
    self.wallet = bt.wallet(config=self.config)
    self.dendrite = bt.dendrite(wallet=self.wallet)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)


def verify_logic(logic: dict) -> tuple[bool, str]:
    # Dictionary mapping modules to allowed functions/imports
    allowed_modules = ALLOWED_MODULES.copy()
    
    for module in logic:
        # Handle folder paths by taking first component
        module_name = module.split("/")[0].split(".")[0]
        if module_name not in allowed_modules:
            allowed_modules.append(module_name)
            
    for key, value in logic.items():
        if value:
            # Create expanded allowed modules list that includes submodules and specific imports
            expanded_allowed = set()
            for mod in allowed_modules:
                expanded_allowed.add(mod)
                # If module is allowed, all its submodules are allowed
                for used_mod in value.split():
                    if used_mod.startswith(f"{mod}."):
                        expanded_allowed.add(used_mod)
                    # Check for specific allowed imports like "from os import getenv"
            usage_pass, usage_msg = verify_code_usage(value, list(expanded_allowed), ALLOWED_IMPORTS)
            if not usage_pass:
                return False, usage_msg
                
    total_chars = 0
    for key, value in logic.items():
        # Include full folder path in character count
        total_chars += len(key) + len(value)
        
    if total_chars > NUM_ALLOWED_CHARACTERS:
        return (
            False,
            f"Total characters: {total_chars} exceeds the limit of {NUM_ALLOWED_CHARACTERS}",
        )
        
    return True, "Logic is valid"

class FinetunePipeline:
    def __init__(
        self, config, tracking_logics: List[TrackingInfo] = None,
    ):
        self.config = config
        bittensor_injector(self)
        self.code_sim_model = CodeSimModel()
        self.trackers = []
        self.dataset = SWEBenchDataset()
        self.load_tasks()
        self.load_results()
        self.llm_manager = LLMManager()
        
        if tracking_logics is None:
            self.load_logics()
        else:
            self.tracking_logics = tracking_logics

        # Register cleanup to be called when the object is deleted
        # self._finalizer = weakref.finalize(self, self.cleanup)

    def load_tasks(self):
        if os.path.exists(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl"):
            with open(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "rb") as f:
                self.tasks = pickle.load(f)[:self.config.neuron.finetune_test_size]
        else:
            self.tasks = generate_swe_tasks(self.dataset, self.config.neuron.finetune_test_size)
            self.store_tasks()

    def load_results(self):
        results_file = f"{self.config.neuron.full_path}/results_{COMPETITION_ID}.pkl"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                saved_results = pickle.load(f)
                self.trackers = saved_results.get("trackers", [])

    def store_logics(self):
        with open(f"{self.config.neuron.full_path}/logics_{COMPETITION_ID}.pkl", "wb") as f:
            pickle.dump(self.tracking_logics, f)
    
    def load_logics(self):
        if os.path.exists(f"{self.config.neuron.full_path}/logics_{COMPETITION_ID}.pkl"):
            with open(f"{self.config.neuron.full_path}/logics_{COMPETITION_ID}.pkl", "rb") as f:
                self.tracking_logics = pickle.load(f)
        else:
            self.tracking_logics = gather_all_logics(self)
            self.store_logics()
    
    @property
    def results(self) -> FinetuneEventResults:
        return FinetuneEventResults(
            trackers=self.trackers
        )

    # TODO add time taken
    def evaluate(self) -> FinetuneEventResults:
        # gather all logics
        bt.logging.info("Gathering all logics...")
        bt.logging.info(f"Gathered {len(self.tracking_logics)} logics.")

        bt.logging.info("Verifying and building docker containers for each logic...")
        for tracker in self.tracking_logics:
            bt.logging.info(f"Verifying logic for hotkey {tracker.hotkey}...")
            pass_logic, pass_msg = verify_logic(tracker.logic)
            if not pass_logic:
                bt.logging.info(
                    f"Logic failed verification: {pass_msg} on tracker {tracker.hotkey}"
                )
                tracker.logic = {}
                continue
        
        bt.logging.info(f"Beginning evaluation of {len(self.tasks)} tasks...")
        for tracker_idx, tracking_logic in enumerate(self.tracking_logics):
            bt.logging.info(f"Processing tracker {tracker_idx + 1}/{len(self.tracking_logics)}")
            # Skip if no logic provided
            if not tracking_logic.logic:
                bt.logging.info(f"No logic provided for tracker {tracking_logic.hotkey}, skipping...")
                tracking_logic.score = 0
                self.trackers.append(tracking_logic)
                continue
            
            previous_tracker = next((tracker for tracker in self.trackers if str(tracker.logic) == str(tracking_logic.logic)), None)
            if previous_tracker is not None:
                bt.logging.info(f"Finetune: Using previously evaluated score for hotkey: {tracking_logic.hotkey}")
                tracking_logic.score = previous_tracker.score
                if tracking_logic.hotkey != previous_tracker.hotkey:
                    self.trackers.append(tracking_logic)
                    self.store_results()
                continue


            # Otherwise, evaluate the logic
            bt.logging.info(f"Initializing LLM key for hotkey {tracking_logic.hotkey}...")
            self.llm_manager.init_key(tracking_logic.hotkey)
            bt.logging.info(f"Starting docker container for hotkey {tracking_logic.hotkey}...")
            scores = []
            for task in self.tasks:
                build_docker_container(tracking_logic.logic, tracking_logic.hotkey, task.repo.files)
                # sleep(20)
                try:
                    bt.logging.info(f"Making request to container for hotkey {tracking_logic.hotkey}...")
                    result = run_docker_container_from_base( 
                        f"swe-logic-{str(tracking_logic.hotkey)}-{COMPETITION_ID}".lower(),
                        task.repo,
                        tracking_logic.hotkey,
                        task.query,
                        tracking_logic.logic
                    )
                    patch = Patch(**result)
                    bt.logging.info(f"Scoring response for hotkey {tracking_logic.hotkey}...")
                    # TODO in the next comp uncomment the below
                    # score = task.score(patch, self.llm_manager.get_count())
                    score = task.score(patch, 1)
                    self.llm_manager.reset_count()
                    bt.logging.info(f"Score for hotkey {tracking_logic.hotkey}: {score}")
                    scores.append(score)
                except Exception as e:
                    bt.logging.error(f"Request failed for hotkey {tracking_logic.hotkey}: {e}")
                    print(traceback.format_exc())
                    scores.append(0)
                print(f"Average score for hotkey {tracking_logic.hotkey}: {sum(scores) / len(scores)}")

            tracking_logic.score = sum(scores) / len(scores)
            self.trackers.append(tracking_logic)
            self.store_results()
            
            bt.logging.info(f"Cleaning up container for hotkey {tracking_logic.hotkey}...")
            bt.logging.info(f"Final score for hotkey {tracking_logic.hotkey}: {tracking_logic.score}")
            
        bt.logging.info("Evaluation complete!")

        return self.results
    
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
    def start(
        config, code_sim_model: CodeSimModel = None
    ) -> FinetuneEventResults:
        if code_sim_model is None:
            code_sim_model = CodeSimModel()
        pipeline = FinetunePipeline(config, code_sim_model)
        result = pipeline.evaluate()
        pipeline.cleanup()  # Ensure cleanup is called after evaluation
        return result

    def store_tasks(self):
        with open(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "wb") as f:
            pickle.dump(self.tasks, f)

    def store_results(self):
        with open(f"{self.config.neuron.full_path}/results_{COMPETITION_ID}.pkl", "wb") as f:
            pickle.dump({
                "trackers": self.trackers
            }, f)

    @staticmethod
    def generate_tasks(config) -> List[SWEBenchTask]:
        dataset = SWEBenchDataset()
        tasks = generate_swe_tasks(dataset, config.neuron.finetune_test_size)
        with open(f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "wb") as f:
            pickle.dump(tasks, f)
    
    @staticmethod
    def tasks_exist(config):
        return os.path.exists(f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl")
    
    def cleanup(self):
        """
        Delete the tasks file and any other task files
        """
        os.remove(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl")
        # check if tasks_*.pkl exists and delete it if it does
        for file in os.listdir(self.config.neuron.full_path):
            if file.startswith("tasks_") and file.endswith(".pkl"):
                os.remove(os.path.join(self.config.neuron.full_path, file))
            if file.startswith("results_") and file.endswith(".pkl"):
                os.remove(os.path.join(self.config.neuron.full_path, file))

if __name__ == "__main__":
    config = util_config(None)
    parser = argparse.ArgumentParser()
    add_validator_args(config, parser)
    config.netuid = 1
    config.neuron = type('Neuron', (), {'full_path': "tests", "finetune_test_size": 100})()
    test_submission_dir = "notebooks/test-submission"
    logic = {}
    # Read all files in test-submission directory
    for root, dirs, files in os.walk(test_submission_dir):
        # Skip __pycache__ directories
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        # Get relative path from test_submission_dir
        rel_path = os.path.relpath(root, test_submission_dir)
        
        # Process all files in current directory
        for filename in files:
            # Skip __pycache__ files
            if '__pycache__' in filename:
                continue
                
            file_path = os.path.join(root, filename)
            # Get the relative path for the logic dict key
            if rel_path == '.':
                logic_key = filename
            else:
                logic_key = os.path.join(rel_path, filename)
                
            with open(file_path, 'r', encoding='latin-1') as f:
                logic[logic_key] = f.read()
    tracking_logics = [TrackingInfo(logic=logic, hotkey="hotkey1", uid=1, score=0.0, block=0)]
    pipeline = FinetunePipeline(config, tracking_logics)
    pipeline.evaluate()
