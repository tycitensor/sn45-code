import random
from typing import Callable

from .task import Task
from .swe import SWETask
# from .debug import DebugTask
from .fim import FillInMiddleTask
from .repofile import RepoFileTask
from .repo import RepoCompletionTask
from .completion import CompletionTask
from .organic_convo import OrganicConvoTask

TASKS = {
    RepoCompletionTask.name: RepoCompletionTask,
    FillInMiddleTask.name: FillInMiddleTask,
    CompletionTask.name: CompletionTask,
    RepoFileTask.name: RepoFileTask,
    # DebugTask.name: DebugTask,
    SWETask.name: SWETask,
}

from coding.repl import REPLClient
from coding.schemas import Context
from coding.helpers import Selector
from coding.datasets import DATASET_MANAGER
from coding.protocol import StreamCodeSynapse
from coding.datasets import GithubDataset, PipDataset, SWEDataset

TASK_REGISTRY = {
    RepoCompletionTask.name: [GithubDataset.name],
    FillInMiddleTask.name: [GithubDataset.name],
    CompletionTask.name: [GithubDataset.name],
    RepoFileTask.name: [GithubDataset.name],
    # DebugTask.name: [PipDataset.name],
    SWETask.name: [SWEDataset.name],
}


def create_task(
    llm,
    task_name: str,
    selector: Selector = random.choice,
    repl: REPLClient = REPLClient(),
    code_scorer: Callable = None
) -> Task:
    """Create a task from the given task name and LLM pipeline.

    Args:
        llm (Pipeline): Pipeline to use for text generation
        task_name (str): Name of the task to create
        selector (Selector, optional): Selector function to choose a dataset. Defaults to random.choice.

    Raises:
        ValueError: If task_name is not a valid alias for a task, or if the task is not a subclass of Task
        ValueError: If no datasets are available for the given task
        ValueError: If the dataset for the given task is not found

    Returns:
        Task: Task instance
    """
    task = TASKS.get(task_name, None)
    if task is None or not issubclass(task, Task):
        raise ValueError(f"Task {task_name} not found")

    dataset_choices = TASK_REGISTRY.get(task_name, None)
    if len(dataset_choices) == 0:
        raise ValueError(f"No datasets available for task {task_name}")
    dataset_name = selector(dataset_choices)
    dataset = DATASET_MANAGER.datasets.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} not found")
    return task(llm=llm, context=dataset.next(**dict(task.dataset_options)), repl=repl, code_scorer=code_scorer)


def create_organic_task(
    llm,
    synapse: StreamCodeSynapse,
    repl: REPLClient = REPLClient(),
) -> Task:
    """Create a task from the given synapse and LLM pipeline."""

    return OrganicConvoTask(
        llm=llm,
        context=Context(messages=synapse.messages, files=synapse.files),
        repl=repl,
    )
 