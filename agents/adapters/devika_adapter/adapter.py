# agents/adapters/devika_adapter.py

from agents.base import BaseAgent
import subprocess

class DevikaAdapter(BaseAgent):
    def __init__(self):
        self.name = "Devika"
        self.capabilities = ["natural_language_interface", "code_execution"]

    def run(self, task: dict) -> dict:
        prompt = task.get("input")
        context = task.get("context", "")
        combined_prompt = f"[CONTEXT]\n{context}\n\n[INSTRUCTION]\n{prompt}"

        try:
            output = subprocess.check_output(["python3", "main.py", "--task", combined_prompt], cwd="agents/adapters/devika_adapter/devika")
            return {"result": output.decode("utf-8")}
        except Exception as e:
            return {"error": f"[EXCEPTION] Running Devika failed: {str(e)}"}
