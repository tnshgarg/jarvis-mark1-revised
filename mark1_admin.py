# mark1_admin.py
import os
import json
import shutil
import click
from datetime import datetime

BASE_DIR = "./data"
MEMORY_DIR = os.path.join(BASE_DIR, "agent_memory")
TASKS_FILE = os.path.join(BASE_DIR, "tasks", "history.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "summary.json")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

@click.group()
def cli():
    """Mark-1 Admin Tool: Inspect and manage agent memory, logs, and backups."""
    pass

@cli.command()
@click.argument('agent_name')
def view_memory(agent_name):
    """View memory for a specific agent."""
    file_path = os.path.join(MEMORY_DIR, f"{agent_name}.json")
    if os.path.exists(file_path):
        with open(file_path) as f:
            memory = json.load(f)
            click.echo(f"\n🧠 Memory for agent: {agent_name}")
            for i, entry in enumerate(memory):
                click.echo(f"\n--- Memory {i+1} ---")
                click.echo(f"Input: {entry['input']}")
                click.echo(f"Result: {entry['result']}")
    else:
        click.echo(f"[!] No memory found for {agent_name}")

@cli.command()
def rotate_logs():
    """Rotate task logs and store a timestamped backup."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"{timestamp}_history.json"
    dst_path = os.path.join(BACKUP_DIR, backup_name)
    if os.path.exists(TASKS_FILE):
        shutil.copy(TASKS_FILE, dst_path)
        open(TASKS_FILE, "w").close()
        click.echo(f"[✓] Rotated log to {dst_path}")
    else:
        click.echo("[!] No history.json file found to rotate.")

@cli.command()
def summarize():
    """Create auto-summaries for all agent memories."""
    summary = {}
    for file in os.listdir(MEMORY_DIR):
        if file.endswith(".json"):
            agent_name = file.replace(".json", "")
            file_path = os.path.join(MEMORY_DIR, file)
            with open(file_path) as f:
                memory = json.load(f)
                summary[agent_name] = {
                    "total_entries": len(memory),
                    "last_entry": memory[-1] if memory else {}
                }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    click.echo(f"[✓] Summary written to {SUMMARY_FILE}")

if __name__ == "__main__":
    cli()
