# 1. Add new adapters: Continue.dev and Devika
adapter_files = {
    "agents/adapters/continue_adapter/adapter.py": '''\
from agents.core.base_agent import BaseAgent

class ContinueDevAdapter(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ContinueDevAdapter",
            description="Adapter for Continue.dev integration",
            capabilities=["code_refactor", "contextual_assist"]
        )

    def run(self, task: dict) -> dict:
        return {"result": f"Continue.dev handled: {task['input']}"}
''',

    "agents/adapters/devika_adapter/adapter.py": '''\
from agents.core.base_agent import BaseAgent

class DevikaAdapter(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DevikaAdapter",
            description="Adapter for Devika AI integration",
            capabilities=["natural_language_interface", "interactive_debugging"]
        )

    def run(self, task: dict) -> dict:
        return {"result": f"Devika handled: {task['input']}"}
'''
}

# 2. Hook vector memory into orchestrator
orchestrator_with_memory = '''\
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
'''

# 3. Self-description parsing from plug-and-play folders
plugin_loader = '''\
import importlib.util
import os

def load_agent_plugins(agent_dir="agents/adapters"):
    agents = []
    for adapter_dir in os.listdir(agent_dir):
        path = os.path.join(agent_dir, adapter_dir, "adapter.py")
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location(f"{adapter_dir}.adapter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type):
                try:
                    instance = obj()
                    if hasattr(instance, "run") and hasattr(instance, "capabilities"):
                        agents.append(instance)
                except Exception:
                    pass
    return agents
'''

# Write files
adapter_files["orchestrator/orchestrator.py"] = orchestrator_with_memory
adapter_files["agents/registry/plugin_loader.py"] = plugin_loader

for path, content in adapter_files.items():
    with open(path, "w") as f:
        f.write(content)

"✅ Added Continue.dev & Devika adapters, updated Orchestrator to use VectorStore, and built auto-loader for plug-and-play agents."
