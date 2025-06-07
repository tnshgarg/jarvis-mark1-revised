import json, os

class AgentMemory:
    def __init__(self, agent_name: str):
        self.file_path = f"./data/agent_memory/{agent_name}.json"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return []

    def add_memory(self, input_text, result_text):
        entry = {"input": input_text, "result": result_text}
        self.memory.append(entry)
        self.save()

    def get_memory(self):
        return self.memory

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
