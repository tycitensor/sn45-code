from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    def __call__(self, query: str):
        pass