from abc import ABC
from dataclasses import dataclass, field
from typing import List, Union, Any, Dict, Callable

from coding.repl import REPLClient
from coding.schemas import Context, File


@dataclass
class Task(ABC):
    name: str
    desc: str
    goal: str
    query: str
    topic: str
    subtopic: str
    tags: List[str]
    context: Context
    reward_definition: List[dict]
    attachments: List[Any] = field(default_factory=[])
    files: List[File] = field(default_factory=[])
    penalty_definition: List[dict] = None
    dataset_options: Dict = field(default_factory=dict)
    reward_threshold: float = 0.0
    reference: Union[str, List[str], Dict] = ""
    criteria: str = ("",)
    delimiter: str = ""
    complete: bool = False
    static_reference: bool = False
    static_query: bool = False
    reference_prompt: str = ""
    query_system_prompt: str = ""
    query_prompt: str = ""
    llm: Callable = None
    repl: REPLClient = None
    extra_info: Dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, desc={self.desc!r}, goal={self.goal!r}, query={self.query!r}, reference={self.reference!r}, topic={self.topic!r}, subtopic={self.subtopic!r}, tags={self.tags!r})"

    def __repr__(self):
        return str(self)

    def __state_dict__(self, full=False):
        state = {
            "task": self.name,
            "desc": self.desc,
            "goal": self.goal,
            "query": self.query, 
            "query_time": getattr(self, "query_time", 0),
            "reference": self.reference,
            "reference_time": getattr(self, "reference_time", 0),
            "topic": self.topic,
            "subtopic": self.subtopic,
            "context_time": self.context.stats.get("fetch_time", 0.0),
        }
        if full:
            state.update(dict(self.context))

        return state
