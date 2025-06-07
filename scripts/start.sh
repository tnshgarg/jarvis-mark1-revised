#!/bin/bash
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
