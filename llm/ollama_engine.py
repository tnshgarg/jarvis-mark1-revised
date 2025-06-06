import subprocess

class OllamaEngine:
    def __init__(self, model_name="mistral"):
        self.model = model_name

    def generate(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
