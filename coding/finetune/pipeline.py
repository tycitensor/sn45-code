import os
import pickle
import atexit
import weakref
import requests
import traceback
from time import sleep
import bittensor as bt
from typing import List
from pydantic import BaseModel
from .tracker import gather_all_logics
from .dockerutil import build_docker_container, run_docker_container
from ..helpers.git import GitRepo

from coding.schemas import Patch
from coding.schemas.context import Context
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo, TaskResult
from coding.constants import COMPETITION_ID, ALLOWED_MODULES, NUM_ALLOWED_CHARACTERS, ALLOWED_IMPORTS

from coding.tasks.swe import SWEBenchTask
from coding.datasets.swe import SWEBenchDataset
from coding.finetune.llm.manager import LLMManager
from coding.helpers.codeanal import verify_code_usage


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
    while len(tasks) < n:
        try:
            tasks.append(SWEBenchTask(llm=None, context=Context(**ds.get())))
        except Exception as e:
            bt.logging.error(f"Error generating task: {e}")
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
        self, config
    ):
        self.config = config
        # TODO uncomment
        # bittensor_injector(self)
        self.code_sim_model = CodeSimModel()
        self.scores = []
        self.tracking_logics: List[TrackingInfo] = []
        self.dataset = SWEBenchDataset()
        self.load_tasks()
        self.load_results()
        self.llm_manager = LLMManager()

        # Register cleanup to be called when the object is deleted
        # self._finalizer = weakref.finalize(self, self.cleanup)

    def load_tasks(self):
        if os.path.exists(f"tasks_{COMPETITION_ID}.pkl"):
            with open(f"tasks_{COMPETITION_ID}.pkl", "rb") as f:
                self.tasks = pickle.load(f)
        else:
            self.tasks = generate_swe_tasks(
                self.dataset, self.config.finetune_test_size
            )
            self.store_tasks()

    def load_results(self):
        results_file = f"results_{COMPETITION_ID}.pkl"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                saved_results = pickle.load(f)
                self.scores = saved_results.get("scores", [])
                self.tracking_logics = saved_results.get("tracking_logics", [])

    @property
    def results(self) -> FinetuneEventResults:
        return FinetuneEventResults(
            scores=self.scores, tracking_infos=self.tracking_logics
        )

    # TODO save progress to file
    # may need to create custom LLM class that just requests.post to a custom server. should be a clone of the ChatOpenAI class kinda
    # TODO add time taken
    def evaluate(self) -> FinetuneEventResults:
        # gather all logics
        bt.logging.info("Gathering all logics...")
        self.tracking_logics = gather_all_logics(self)
        bt.logging.info(f"Gathered {len(self.tracking_logics)} logics.")

        for tracker in self.tracking_logics:
            pass_logic, pass_msg = verify_logic(tracker.logic)
            if not pass_logic:
                bt.logging.info(
                    f"Logic failed verification: {pass_msg} on tracker {tracker.hotkey}"
                )
                tracker.logic = {}
                continue
            build_docker_container(tracker.logic, tracker.hotkey)

        for task in self.tasks:
            # Track which logics we've already scored for this task
            scored_logics = {}

            for tracker in self.tracking_logics:
                # Skip if no logic provided
                if not tracker.logic:
                    tracker.results.append(
                        TaskResult(
                            logic=tracker.logic,
                            commit_hash=task.base_commit,
                            score=0,
                        )
                    )
                    continue

                # Convert logic dict to string for hashing
                logic_str = str(tracker.logic)

                # If we've already scored this logic, use the previous score
                if logic_str in scored_logics:
                    tracker.results.append(
                        TaskResult(
                            logic=tracker.logic,
                            commit_hash=task.base_commit,
                            score=scored_logics[logic_str],
                        )
                    )
                    continue

                # Otherwise, evaluate the logic
                self.llm_manager.init_key(tracker.hotkey)
                container = run_docker_container(
                    f"swe-logic-{str(tracker.hotkey)}-{COMPETITION_ID}",
                    task.repo,
                    tracker.hotkey,
                )
                sleep(2)
                try:
                    response = requests.post(
                        f"http://{os.environ['DOCKER_HOST_IP']}:3000/call",
                        json={
                            "repo_location": "/app/repo",
                            "issue_description": task.query,
                        },
                        timeout=360,
                    )
                    response.raise_for_status()
                    result = response.json()["result"]
                    patch = Patch(**result)
                    score = task.score(patch, self.llm_manager.get_count())
                    self.llm_manager.reset_count()
                except Exception as e:
                    bt.logging.error(f"Request failed: {e}")
                    score = 0

                # Store the score for this logic
                scored_logics[logic_str] = score

                tracker.results.append(
                    TaskResult(
                        logic=tracker.logic,
                        commit_hash=task.base_commit,
                        score=score,
                    )
                )
                bt.logging.info(f"Result from SWE: {result}")
                container.stop()
                container.remove()
            del container
            del repo
        # average the scores for each tracker
        scores = []
        for tracker in self.tracking_logics:
            scores.append(sum(tracker.results.score) / len(tracker.results.score))
        self.scores = scores
        self.store_results()

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
        config, code_sim_model: CodeSimModel = CodeSimModel()
    ) -> FinetuneEventResults:
        pipeline = FinetunePipeline(config, code_sim_model)
        result = pipeline.evaluate()
        pipeline.cleanup()  # Ensure cleanup is called after evaluation
        return result

    def store_tasks(self):
        with open(f"tasks_{COMPETITION_ID}.pkl", "wb") as f:
            pickle.dump(self.tasks, f)

    def store_results(self):
        with open(f"results_{COMPETITION_ID}.pkl", "wb") as f:
            pickle.dump(
                {"scores": self.scores, "tracking_logics": self.tracking_logics}, f
            )

    def cleanup(self):
        """
        Delete the tasks file and any other task files
        """
        os.remove(f"tasks_{COMPETITION_ID}.pkl")
        # check if tasks_*.pkl exists and delete it if it does
        for file in os.listdir("."):
            if file.startswith("tasks_") and file.endswith(".pkl"):
                os.remove(file)
            if file.startswith("results_") and file.endswith(".pkl"):
                os.remove(file)
