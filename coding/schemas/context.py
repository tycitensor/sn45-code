from typing import List, Any, Dict
from pydantic import BaseModel

from .file import File
from .chat import ChatMessage

class Context(BaseModel):
    title: str = ""
    topic: str = ""
    content: str = ""
    internal_links: List[str] = []
    external_links: List[str] = []
    source: str = ""
    tags: List[str] = None
    extras: Dict[str, Any] = None
    files: List[File] = None
    messages: List[ChatMessage] = []
