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
