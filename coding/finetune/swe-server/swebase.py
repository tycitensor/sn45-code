from schemas.swe import Patch
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI

class SWE(ABC):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @abstractmethod
    def __call__(self, repo_location: str, issue_description: str) -> Patch:
        pass
