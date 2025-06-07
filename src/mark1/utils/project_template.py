"""
Project Template Generator for Mark-1 Orchestrator

This module provides functionality to create new Mark-1 projects with
proper directory structure and configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


def create_project_template(project_path: Path, template: str = "basic") -> None:
    """
    Create a new Mark-1 project from template
    
    Args:
        project_path: Path where to create the project
        template: Template type to use (basic, advanced, etc.)
    """
    logger.info("Creating Mark-1 project", path=str(project_path), template=template)
    
    # Ensure project path exists
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    _create_directory_structure(project_path)
    
    # Create configuration files
    _create_config_files(project_path, template)
    
    # Create example files
    _create_example_files(project_path, template)
    
    logger.info("Mark-1 project created successfully", path=str(project_path))


def _create_directory_structure(project_path: Path) -> None:
    """Create the basic directory structure"""
    directories = [
        "data",
        "data/logs",
        "data/models",
        "data/cache",
        "agents",
        "agents/custom",
        "config",
        "scripts",
        "docs"
    ]
    
    for directory in directories:
        dir_path = project_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory", path=str(dir_path))


def _create_config_files(project_path: Path, template: str) -> None:
    """Create configuration files"""
    
    # Create .env file
    env_content = """# Mark-1 Orchestrator Configuration

# Application Settings
MARK1_APP_NAME="Mark-1 Orchestrator"
MARK1_VERSION="0.1.0"
MARK1_ENVIRONMENT="development"
MARK1_DEBUG="true"
MARK1_LOG_LEVEL="INFO"

# Database Configuration (SQLite for development)
MARK1_DATABASE__URL="sqlite:///./data/mark1.db"
MARK1_DATABASE__POOL_SIZE=10
MARK1_DATABASE__ECHO=false

# Redis Configuration (optional)
# MARK1_REDIS__HOST="localhost"
# MARK1_REDIS__PORT=6379
# MARK1_REDIS__DB=0

# ChromaDB Configuration (optional)
# MARK1_CHROMADB__HOST="localhost"
# MARK1_CHROMADB__PORT=8000

# Ollama Configuration
MARK1_OLLAMA__BASE_URL="http://localhost:11434"
MARK1_OLLAMA__DEFAULT_MODEL="llama2"
MARK1_OLLAMA__TIMEOUT=300.0

# Security Configuration
MARK1_SECURITY__SECRET_KEY="dev-secret-key-change-in-production"
MARK1_SECURITY__ALGORITHM="HS256"

# API Configuration
MARK1_API__HOST="0.0.0.0"
MARK1_API__PORT=8000
MARK1_API__RELOAD=true

# Agent Configuration
MARK1_AGENTS__MAX_CONCURRENT_AGENTS=10
MARK1_AGENTS__ENABLE_SANDBOXING=true
MARK1_AGENTS__AUTO_DISCOVERY_ENABLED=true

# Scanning Configuration
MARK1_SCANNING__MAX_FILE_SIZE_MB=10
MARK1_SCANNING__ENABLE_ML_PATTERN_DETECTION=true

# Monitoring Configuration
MARK1_MONITORING__ENABLE_PROMETHEUS=true
MARK1_MONITORING__PROMETHEUS_PORT=8090
MARK1_MONITORING__ENABLE_HEALTH_CHECKS=true
"""
    
    env_file = project_path / ".env"
    env_file.write_text(env_content)
    logger.debug("Created .env file", path=str(env_file))
    
    # Create .gitignore
    gitignore_content = """# Mark-1 Orchestrator
.env
.env.local
.env.production

# Data directories
data/logs/
data/cache/
data/models/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    gitignore_file = project_path / ".gitignore"
    gitignore_file.write_text(gitignore_content)
    logger.debug("Created .gitignore file", path=str(gitignore_file))


def _create_example_files(project_path: Path, template: str) -> None:
    """Create example files and documentation"""
    
    # Create README.md
    readme_content = f"""# Mark-1 Orchestrator Project

This is a Mark-1 AI Agent Orchestration project created from the `{template}` template.

## Getting Started

1. **Configure Environment**
   ```bash
   # Edit .env file with your settings
   cp .env.example .env
   ```

2. **Discover Agents**
   ```bash
   # Scan for existing AI agents in your codebase
   mark1 scan .
   ```

3. **Start the API Server**
   ```bash
   # Start the Mark-1 orchestration server
   mark1 serve
   ```

4. **Check System Status**
   ```bash
   # Verify everything is working
   mark1 status
   ```

## Directory Structure

```
.
├── agents/          # Discovered and custom agents
├── config/          # Configuration files
├── data/            # Data storage
│   ├── logs/        # Log files
│   ├── models/      # Local model storage
│   └── cache/       # System cache
├── docs/            # Documentation
├── scripts/         # Utility scripts
├── .env             # Environment configuration
└── README.md        # This file
```

## Configuration

The system is configured through environment variables in the `.env` file:

- **Database**: SQLite by default, PostgreSQL for production
- **LLM Provider**: Ollama (local) by default
- **Agent Discovery**: Automatic scanning enabled
- **API Server**: Runs on port 8000 by default

## Usage Examples

### Orchestrate a Task
```bash
mark1 orchestrate "Analyze this codebase and suggest improvements"
```

### Interactive Mode
```bash
mark1 interactive
```

### Scan for Agents
```bash
mark1 scan ./my-project --frameworks langchain,autogpt
```

## Documentation

- [Mark-1 Documentation](https://docs.mark1.ai)
- [Agent Integration Guide](./docs/agent-integration.md)
- [API Reference](./docs/api-reference.md)

## Support

For support and questions, please visit the [Mark-1 GitHub repository](https://github.com/mark1-ai/orchestrator).
"""
    
    readme_file = project_path / "README.md"
    readme_file.write_text(readme_content)
    logger.debug("Created README.md file", path=str(readme_file))
    
    # Create example agent
    example_agent_content = '''"""
Example Custom Agent for Mark-1 Orchestrator

This is a simple example of how to create a custom agent
that can be discovered and orchestrated by Mark-1.
"""

import asyncio
from typing import Dict, Any, List


class ExampleAgent:
    """
    Example agent that demonstrates basic Mark-1 integration
    """
    
    def __init__(self):
        self.name = "Example Agent"
        self.description = "A simple example agent for demonstration"
        self.capabilities = ["text_processing", "data_analysis"]
        self.version = "1.0.0"
    
    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task with the given context
        
        Args:
            task: Task description
            context: Additional context for the task
            
        Returns:
            Dict containing the task result
        """
        # Simulate some processing
        await asyncio.sleep(1)
        
        return {
            "status": "completed",
            "result": f"Processed task: {task}",
            "agent": self.name,
            "processing_time": 1.0
        }
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return self.capabilities
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "version": self.version,
            "type": "custom"
        }


# Mark-1 will automatically discover this agent
agent = ExampleAgent()
'''
    
    example_agent_file = project_path / "agents" / "custom" / "example_agent.py"
    example_agent_file.write_text(example_agent_content)
    logger.debug("Created example agent", path=str(example_agent_file))
    
    # Create startup script
    startup_script_content = '''#!/bin/bash
# Mark-1 Orchestrator Startup Script

echo "Starting Mark-1 Orchestrator..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Check if Mark-1 is installed
if ! command -v mark1 &> /dev/null; then
    echo "Error: mark1 command not found. Please install Mark-1 first."
    echo "pip install mark1-orchestrator"
    exit 1
fi

# Start the server
echo "Starting Mark-1 API server..."
mark1 serve --host 0.0.0.0 --port 8000 --reload
'''
    
    startup_script_file = project_path / "scripts" / "start.sh"
    startup_script_file.write_text(startup_script_content)
    startup_script_file.chmod(0o755)  # Make executable
    logger.debug("Created startup script", path=str(startup_script_file)) 