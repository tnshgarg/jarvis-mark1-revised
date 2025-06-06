class AgentRegistry:
    def __init__(self):
        self.agents = {}

    def register(self, agent):
        self.agents[agent.name] = agent

    def get_agent_by_capability(self, capability: str):
        return [agent for agent in self.agents.values() if capability in agent.capabilities]

    def list_agents(self):
        return list(self.agents.keys())
