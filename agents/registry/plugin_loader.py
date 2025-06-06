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
