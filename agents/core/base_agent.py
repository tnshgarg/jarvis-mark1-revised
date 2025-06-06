from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, description: str, capabilities: list):
        self.name = name
        self.description = description
        self.capabilities = capabilities

    @abstractmethod
    def run(self, task: dict) -> dict:
        pass
