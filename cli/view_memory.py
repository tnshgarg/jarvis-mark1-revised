# cli/view_memory.py
import json
import sys

def load_agent_memory(agent_name):
    file_path = f"./data/agent_memory/{agent_name}.json"
    try:
        with open(file_path) as f:
            memory = json.load(f)
            print(f"\n🧠 Memory for agent: {agent_name}")
            for i, entry in enumerate(memory):
                print(f"\n--- Memory {i+1} ---")
                print(f"Input: {entry['input']}")
                print(f"Result: {entry['result']}")
    except FileNotFoundError:
        print(f"[!] No memory found for {agent_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cli/view_memory.py <agent_name>")
    else:
        load_agent_memory(sys.argv[1])
