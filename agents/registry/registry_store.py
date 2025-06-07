import os
import json
from agents.core.base_agent import BaseAgent

class AgentRegistry:
    def __init__(self):
        self.agents = []
        self.agent_map = {}  # name -> agent instance
        self.registry_file = "./data/registry/agents.json"
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self.load_registry()

    def register_agent(self, agent):
        self.agents.append(agent)
        self.agent_map[agent.name] = agent
        self.save_registry()

    def register(self, agent):
        """Alias for register_agent"""
        self.register_agent(agent)

    def get_agent_by_capability(self, capability):
        return [agent for agent in self.agents if capability in agent.capabilities]

    def get_agent_by_name(self, name):
        return self.agent_map.get(name)

    def list_agents(self):
        return [(agent.name, agent.capabilities, agent.description) for agent in self.agents]

    def save_registry(self):
        data = [{
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities
        } for agent in self.agents]
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file) as f:
                registry_data = json.load(f)
                # Registry data is loaded but agents need to be instantiated separately
                # This is handled by the plugin loader