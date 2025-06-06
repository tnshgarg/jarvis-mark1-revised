from dataclasses import dataclass

@dataclass
class Task:
    id: str
    input: str
    capability: str
