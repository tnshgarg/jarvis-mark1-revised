# memory/memory_router.py
from memory.vector_store import VectorStore
from memory.agent_memory import AgentMemory

class MemoryRouter:
    def __init__(self):
        self.vector = VectorStore()
        self.agent_memory_map = {}

    def get_agent_memory(self, agent_name):
        if agent_name not in self.agent_memory_map:
            self.agent_memory_map[agent_name] = AgentMemory(agent_name)
        return self.agent_memory_map[agent_name]
