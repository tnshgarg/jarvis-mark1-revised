# orchestrator/orchestrator.py

import datetime
import json
import os
from agents.registry.registry_store import AgentRegistry
from memory.memory_router import MemoryRouter
from memory.utils.summary import summarize_chunks

class Orchestrator:
    def __init__(self, registry: AgentRegistry, memory: MemoryRouter):
        self.registry = registry
        self.memory = memory

    def execute_task(self, task: dict) -> dict:
        capability = task.get("capability")
        input_text = task.get("input")

        suitable_agents = self.registry.get_agent_by_capability(capability)
        if not suitable_agents:
            return {"error": "No suitable agent found."}

        agent = suitable_agents[0]
        agent_memory = self.memory.get_agent_memory(agent.name)

        # 🔍 Retrieve relevant memory chunks for context
        related_memories = self.memory.vector.search_memory(input_text, top_k=3)

        if len(related_memories) > 3:
            context = summarize_chunks(related_memories)
        else:
            context = "\n".join(related_memories)

        # 🧠 Build context
        context = "\n".join(related_memories)

        # 📦 Inject context into task
        enhanced_task = {
            **task,
            "context": context
        }

        result = agent.run(enhanced_task)

        # 📝 Store result in long-term and agent memory
        self.memory.vector.add_memory(input_text, result.get("result", "unknown"))
        agent_memory.add_memory(input_text, result.get("result", "unknown"))

        return result

    def log_task(task, result):
        os.makedirs("./data/tasks", exist_ok=True)
        with open("./data/tasks/history.json", "a") as f:
            f.write(json.dumps({
                "task": task,
                "result": result,
                "timestamp": str(datetime.now())
            }) + "\n")
