# Mark-1 Orchestrator Project

This is a Mark-1 AI Agent Orchestration project created from the `basic` template.

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
