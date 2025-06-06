from agents.registry.registry_store import AgentRegistry
from memory.vector_store import VectorStore

class Orchestrator:
    def __init__(self, registry: AgentRegistry, memory: VectorStore):
        self.registry = registry
        self.memory = memory

    def execute_task(self, task: dict) -> dict:
        capability = task.get("capability")
        suitable_agents = self.registry.get_agent_by_capability(capability)
        if not suitable_agents:
            return {"error": "No suitable agent found."}
        agent = suitable_agents[0]

        result = agent.run(task)
        self.memory.add_memory(task_id=task.get("input", "unknown"), content=result["result"])
        return result
