# core/agent_messenger.py

class AgentMessenger:
    def __init__(self, registry):
        self.registry = registry

    def send(self, sender_name, receiver_name, message):
        receiver = self.registry.get_agent_by_name(receiver_name)
        if receiver:
            return receiver.receive_message(sender_name, message)
        return {"error": f"No agent named {receiver_name}"}
