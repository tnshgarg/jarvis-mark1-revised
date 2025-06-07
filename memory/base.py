# memory/base.py
from abc import ABC, abstractmethod

class BaseMemory(ABC):
    @abstractmethod
    def add_memory(self, task_id: str, content: str):
        pass

    @abstractmethod
    def search_memory(self, query: str, top_k: int = 5):
        pass

    @abstractmethod
    def clear_memory(self):
        pass
