import os


class AgentRegistry:
    def __init__(self):
        self.agents = []
        self.registry_file = "./data/registry/agents.json"
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self.load_registry()

    def register_agent(self, agent):
        self.agents.append(agent)
        self.save_registry()

    def get_agent_by_capability(self, capability):
        return [agent for agent in self.agents if capability in agent.capabilities]

    def save_registry(self):
        data = [{
            "name": agent.name,
            "capabilities": agent.capabilities
        } for agent in self.agents]
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file) as f:
                registry_data = json.load(f)
                for entry in registry_data:
                    # You could map name → adapter dynamically later
                    pass
