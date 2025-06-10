# MARK-1 Commands and Usage Guide

This document provides a comprehensive list of commands and usage instructions for the MARK-1 AI Orchestration System.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/JARVIS-Mark1-revised.git
cd JARVIS-Mark1-revised/mark1

# Install dependencies
pip install -r requirements.txt
```

### Basic Commands

```bash
# Run the unified workflow demo (complete process visualization)
python unified_workflow_demo.py

# Run the simplified demo (basic agent discovery)
python real_ai_demo.py

# Check available agents
python check_agents.py

# Run the launcher to start the system
python mark1_launcher.py
```

## Core Demo Scripts

| Script                          | Description                                                                                                                                                 | Command                                |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `unified_workflow_demo.py`      | **[Recommended]** Complete end-to-end demo showing agent discovery, adapter generation, task orchestration, and context management with full CLI visibility | `python unified_workflow_demo.py`      |
| `real_ai_demo.py`               | Simplified demo focusing on agent discovery and database integration                                                                                        | `python real_ai_demo.py`               |
| `check_agents.py`               | Quick utility to check available agents                                                                                                                     | `python check_agents.py`               |
| `mark1_launcher.py`             | System launcher for production use                                                                                                                          | `python mark1_launcher.py`             |
| `demo_universal_integration.py` | Demo for universal agent integration                                                                                                                        | `python demo_universal_integration.py` |

## Advanced Usage

### Customizing Agent Discovery

By default, the system searches for agents in the `test_agents` and `agents` directories. You can modify the discovery path in the demo scripts:

```python
# Example:
agents_data = await AgentDiscovery.discover_agents("custom/path/to/agents")
```

### Running with Different Database Settings

```bash
# Using a custom database path
DATABASE_URL="sqlite:///path/to/custom.sqlite" python unified_workflow_demo.py

# Using PostgreSQL (if configured)
DATABASE_URL="postgresql://user:password@localhost:5432/mark1" python unified_workflow_demo.py
```

### Running with Specific Agents

```bash
# Example:
AGENT_FILTER="langchain,autogpt" python unified_workflow_demo.py
```

## Workflow Commands

The system's workflow process follows these steps:

1. **Agent Discovery**: Scan repositories for potential AI agents

   ```bash
   python -c "import asyncio; from unified_workflow_demo import AgentDiscovery; asyncio.run(AgentDiscovery.discover_agents('test_agents'))"
   ```

2. **Adapter Generation**: Generate adapters for discovered agents

   ```bash
   # Generate adapters for all agents in a directory
   python -c "import asyncio; from unified_workflow_demo import AgentDiscovery, AgentAdapterGenerator; agents = asyncio.run(AgentDiscovery.discover_agents('test_agents')); adapters = [asyncio.run(AgentAdapterGenerator.generate_adapter(agent)) for agent in agents]"
   ```

3. **Task Planning**: Create a workflow plan for a specific task

   ```bash
   # This step is handled by the orchestrator in unified_workflow_demo.py
   # See the Orchestrator.plan_workflow method
   ```

4. **Task Execution**: Execute the planned workflow
   ```bash
   # This step is handled by the orchestrator in unified_workflow_demo.py
   # See the Orchestrator.execute_workflow method
   ```

## Viewing Workflow Results

After running the unified workflow demo, you can examine the generated outputs:

```bash
# View generated adapters
ls -la generated_output/
cat generated_output/autogpt_agent_adapter.py  # View a specific adapter

# View task contexts (shows the data flow between agents)
ls -la data/contexts/
cat data/contexts/task_*.json  # View task context

# View task results
ls -la data/results/
cat data/results/task_*.json  # View final results
```

## Debugging Commands

```bash
# Run with debug logging
DEBUG=1 python unified_workflow_demo.py

# Check database contents
python -c "import sqlite3; conn = sqlite3.connect('data/unified_workflow_demo.sqlite'); cursor = conn.cursor(); cursor.execute('SELECT * FROM agents'); print(cursor.fetchall()); conn.close()"

# Clear all data and start fresh
rm -rf data/unified_workflow_demo.sqlite data/contexts/* data/results/* generated_output/*
```

## Environment Variables

| Variable            | Description                                    | Default                          |
| ------------------- | ---------------------------------------------- | -------------------------------- |
| `DATABASE_URL`      | Database connection URL                        | `sqlite:///data/mark1_db.sqlite` |
| `DEBUG`             | Enable debug logging                           | `0` (disabled)                   |
| `AGENT_FILTER`      | Comma-separated list of agent types to include | None (all agents)                |
| `FORCE_RECREATE_DB` | Force recreation of database tables            | `0` (disabled)                   |

## Project Structure

```
mark1/
├── data/                  # Database and output files
│   ├── contexts/          # Task context storage
│   └── results/           # Task results storage
├── generated_output/      # Generated adapter files
├── src/                   # Source code
│   └── mark1/             # Core modules
├── test_agents/           # Test agent files
├── unified_workflow_demo.py    # Unified workflow demo
├── real_ai_demo.py        # Simplified demo
└── mark1_launcher.py      # System launcher
```

## Example: Running a Complete Workflow

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Run the unified workflow demo
python unified_workflow_demo.py

# 3. Check the generated output
ls -la generated_output/
ls -la data/contexts/
ls -la data/results/

# 4. View a specific context file to see data flow between agents
cat data/contexts/task_<TIMESTAMP>.json
```

## Troubleshooting

### Common Issues

1. **Database Errors**:

   ```bash
   # Reset the database
   rm -f data/unified_workflow_demo.sqlite
   python unified_workflow_demo.py
   ```

2. **Import Errors**:

   ```bash
   # Check Python path
   PYTHONPATH=. python unified_workflow_demo.py
   ```

3. **No Agents Found**:
   ```bash
   # Create sample agents manually
   mkdir -p test_agents/sample
   # Add sample agent files to this directory
   ```

### Getting Help

For more information or assistance, refer to the main project documentation:

- `README.md` - Project overview
- `project.md` - Detailed project specifications
- `UNIVERSAL_INTEGRATION_SUMMARY.md` - Integration capabilities
