import os
import requests
from pydantic import BaseModel
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI

class Edit(BaseModel):
    file_name: str
    line_number: int
    line_content: str
    new_line_content: str

class Patch(BaseModel):
    edits: list[Edit]

class LLMClient:
    def __init__(self, base_url: str = f"http://{os.getenv('HOST_IP', 'localhost')}:25000"):
        """Initialize LLM client with API server URL"""
        self.base_url = base_url.rstrip("/")
        self.use_server = True
        try:
            # Test connection to server
            requests.get(self.base_url)
        except requests.exceptions.RequestException:
            # If server not available, fall back to local ChatOpenAI
            self.use_server = False
            from langchain_openai import ChatOpenAI
            self.chat_models = {}

    def __call__(self, query: str, llm_name: str) -> tuple[str, int]:
        """
        Call LLM API endpoint or local ChatOpenAI

        Args:
            query (str): The prompt/query to send to the LLM
            llm_name (str): Name of LLM model to use (e.g. "gpt-4", "claude-3-sonnet")

        Returns:
            tuple[str, int]: (Generated response text, Total tokens used for this key)

        Raises:
            requests.exceptions.RequestException: If API call fails when using server
        """
        if self.use_server:
            payload = {"query": query, "llm_name": llm_name}
            response = requests.post(f"{self.base_url}/call", json=payload)
            response.raise_for_status()
            result = response.json()
            return result["result"], result["total_tokens"]
        else:
            # Use local ChatOpenAI
            if llm_name not in self.chat_models:
                self.chat_models[llm_name] = ChatOpenAI(model_name=llm_name)
            response = self.chat_models[llm_name].invoke(query)
            # ChatOpenAI doesn't provide token count, so return -1
            return response.content, -1
    
    def embed(self, query: str) -> list[float]:
        """
        Get embeddings for text using the embedding API endpoint or local embeddings

        Args:
            query (str): The text to get embeddings for

        Returns:
            list[float]: Vector embedding of the input text

        Raises:
            requests.exceptions.RequestException: If API call fails when using server
        """
        if self.use_server:
            payload = {"query": query}
            response = requests.post(f"{self.base_url}/embed", json=payload)
            response.raise_for_status()
            result = response.json()
            return result["vector"]
        else:
            # Use local embeddings
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            return embeddings.embed_query(query)
        
class SWEBase(ABC):
    def __init__(self):
        self.llm = LLMClient()

    @abstractmethod
    def __call__(self, repo_location: str, issue_description: str) -> Patch:
        pass
