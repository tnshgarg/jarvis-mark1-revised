from agents.registry.registry_store import AgentRegistry
from agents.registry.plugin_loader import load_agent_plugins
from orchestrator.orchestrator import Orchestrator
from memory.vector_store import VectorStore

if __name__ == "__main__":
    registry = AgentRegistry()
    memory = VectorStore()

    # Auto-load all plug-and-play agents
    agents = load_agent_plugins()
    for agent in agents:
        registry.register(agent)
        print(f"[REGISTERED] {agent.name} with capabilities {agent.capabilities}")

    orchestrator = Orchestrator(registry, memory)

    # Example task (can be modified)
    task = {
        "input": "Build a collaborative editor",
        "capability": "code_generation"
    }

    result = orchestrator.execute_task(task)
    print("Output:", result)
