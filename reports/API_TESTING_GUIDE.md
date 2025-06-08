# Mark-1 API Testing Guide

## 🚀 Starting the Server

### ⭐ **RECOMMENDED: Smart Startup (Auto-finds available port)**

```bash
# From project root - automatically finds available port
python start_api_smart.py
```

### Quick Start

```bash
# From project root - fixed port 8000
python start_api.py
```

### Alternative Methods

```bash
# Option 1: Direct execution with uvicorn
uvicorn src.mark1.api.rest_api:create_app --factory --host 127.0.0.1 --port 8001 --reload

# Option 2: Manual port selection
python -c "
from src.mark1.api.rest_api import create_app
import uvicorn
app = create_app(enable_auth=False)
uvicorn.run(app, host='127.0.0.1', port=8001)
"
```

## 🧪 **Quick API Testing**

### **Automatic Test Suite**

```bash
# Auto-detects running server and tests all endpoints
python test_api_quick.py

# Or test specific URL
python test_api_quick.py http://127.0.0.1:8001
```

### **Manual Testing Commands**

## 📍 API Endpoints Overview

Once running, your API will be available at:

- **Base URL**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs (Swagger UI)
- **ReDoc**: http://127.0.0.1:8000/redoc
- **OpenAPI Schema**: http://127.0.0.1:8000/openapi.json

## 🧪 Testing with cURL

### 1. System Endpoints

#### Health Check

```bash
curl -X GET "http://127.0.0.1:8000/health" -H "accept: application/json"
```

#### System Status

```bash
curl -X GET "http://127.0.0.1:8000/" -H "accept: application/json"
```

#### System Metrics

```bash
curl -X GET "http://127.0.0.1:8000/metrics" -H "accept: application/json"
```

### 2. Agent Management

#### List All Agents

```bash
curl -X GET "http://127.0.0.1:8000/agents" -H "accept: application/json"
```

#### Create a New Agent

```bash
curl -X POST "http://127.0.0.1:8000/agents" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_test_agent",
    "display_name": "My Test Agent",
    "description": "A test agent for API testing",
    "framework": "langchain",
    "version": "1.0.0",
    "capabilities": [
      {
        "name": "text_processing",
        "category": "nlp",
        "description": "Process and analyze text",
        "confidence": 0.9
      }
    ],
    "endpoint": {
      "url": "http://localhost:8001/agent",
      "protocol": "http",
      "timeout": 30
    },
    "metadata": {
      "tags": ["test", "demo"],
      "labels": {"environment": "development"}
    },
    "max_concurrent_tasks": 5
  }'
```

#### Get Specific Agent

```bash
curl -X GET "http://127.0.0.1:8000/agents/agent_123" -H "accept: application/json"
```

#### Test Agent

```bash
curl -X POST "http://127.0.0.1:8000/agents/agent_123/test" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "connectivity",
    "test_data": {"sample": "data"},
    "timeout": 30
  }'
```

### 3. Task Management

#### List All Tasks

```bash
curl -X GET "http://127.0.0.1:8000/tasks" -H "accept: application/json"
```

#### Create a New Task

```bash
curl -X POST "http://127.0.0.1:8000/tasks" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Process customer data file",
    "requirements": [
      {
        "capability": "data_processing",
        "parameters": {"format": "csv", "size_limit": "10MB"},
        "priority": "high"
      }
    ],
    "priority": "high",
    "input_data": {"file_path": "/data/customers.csv"},
    "metadata": {"project": "demo", "urgent": true},
    "timeout_seconds": 300,
    "auto_execute": false
  }'
```

#### Execute a Task

```bash
curl -X POST "http://127.0.0.1:8000/tasks/task_123/execute" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_123",
    "async_execution": true,
    "parameters": {"verbose": true}
  }'
```

#### Get Task Logs

```bash
curl -X GET "http://127.0.0.1:8000/tasks/task_123/logs?limit=50" -H "accept: application/json"
```

### 4. Context Management

#### List All Contexts

```bash
curl -X GET "http://127.0.0.1:8000/contexts" -H "accept: application/json"
```

#### Create a New Context

```bash
curl -X POST "http://127.0.0.1:8000/contexts" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "user_session_data",
    "content": {
      "user_id": "12345",
      "preferences": {"theme": "dark", "language": "en"},
      "session_data": {"login_time": "2024-01-15T10:30:00Z"}
    },
    "context_type": "session",
    "scope": "agent",
    "priority": "medium",
    "tags": ["session", "user_data"],
    "expires_in_hours": 24
  }'
```

#### Update Context

```bash
curl -X PUT "http://127.0.0.1:8000/contexts/context_123" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "updated_preferences": {"theme": "light"},
    "last_activity": "2024-01-15T11:00:00Z"
  }'
```

### 5. Orchestration

#### Start Orchestration Workflow

```bash
curl -X POST "http://127.0.0.1:8000/orchestrate" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Complete data processing workflow",
    "requirements": {
      "data_processing": {"format": "csv", "size": "large"},
      "analysis": {"type": "statistical"},
      "reporting": {"format": "pdf"}
    },
    "context": {
      "project": "quarterly_report",
      "deadline": "2024-01-20T00:00:00Z"
    },
    "priority": "high",
    "async_execution": true,
    "timeout_seconds": 3600
  }'
```

#### Get Orchestration Status

```bash
curl -X GET "http://127.0.0.1:8000/orchestrations/orch_123" -H "accept: application/json"
```

## 🐍 Testing with Python

### Using requests library

```python
import requests
import json

# Base URL
BASE_URL = "http://127.0.0.1:8000"

# 1. Health check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Create an agent
agent_data = {
    "name": "python_test_agent",
    "framework": "langchain",
    "capabilities": [
        {
            "name": "python_execution",
            "category": "programming",
            "confidence": 0.95
        }
    ]
}

response = requests.post(f"{BASE_URL}/agents", json=agent_data)
print("Agent created:", response.json())

# 3. Create a task
task_data = {
    "description": "Execute Python script for data analysis",
    "requirements": [
        {
            "capability": "python_execution",
            "priority": "high"
        }
    ],
    "priority": "medium",
    "auto_execute": True
}

response = requests.post(f"{BASE_URL}/tasks", json=task_data)
print("Task created:", response.json())

# 4. Get system metrics
response = requests.get(f"{BASE_URL}/metrics")
print("Metrics:", response.json())
```

### Using httpx (async)

```python
import httpx
import asyncio

async def test_api():
    async with httpx.AsyncClient() as client:
        # Test endpoints
        health = await client.get("http://127.0.0.1:8000/health")
        agents = await client.get("http://127.0.0.1:8000/agents")
        metrics = await client.get("http://127.0.0.1:8000/metrics")

        print("Health:", health.json())
        print("Agents:", agents.json())
        print("Metrics:", metrics.json())

# Run async test
asyncio.run(test_api())
```

## 🔒 Testing with Authentication

### Enable Authentication

Start server with auth enabled:

```python
from src.mark1.api.rest_api import create_app
import uvicorn

app = create_app(enable_auth=True)
uvicorn.run(app, host="127.0.0.1", port=8000)
```

### Get Authentication Token

```python
from src.mark1.api.auth import create_test_token

# Get test token
token = create_test_token("admin")
print(f"Test token: {token}")
```

### Use Token in Requests

```bash
# Get token first
TOKEN="your_jwt_token_here"

# Use in requests
curl -X GET "http://127.0.0.1:8000/agents" \
  -H "accept: application/json" \
  -H "Authorization: Bearer $TOKEN"
```

## 🧪 Validation Testing

### Test Schema Validation

#### Invalid Agent Framework

```bash
curl -X POST "http://127.0.0.1:8000/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_agent",
    "framework": "invalid_framework"
  }'
# Should return 422 Validation Error
```

#### Empty Task Description

```bash
curl -X POST "http://127.0.0.1:8000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "",
    "requirements": []
  }'
# Should return 422 Validation Error
```

#### Invalid Context Type

```bash
curl -X POST "http://127.0.0.1:8000/contexts" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "test",
    "content": {},
    "context_type": "invalid_type"
  }'
# Should return 422 Validation Error
```

## 📊 Expected Responses

### Successful Agent Creation

```json
{
  "id": "agent_12345",
  "name": "my_test_agent",
  "display_name": "My Test Agent",
  "framework": "langchain",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "capabilities": [...],
  "performance": {...}
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "checks": {
    "orchestrator": "healthy",
    "agent_selector": "healthy",
    "context_manager": "healthy"
  },
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

### Validation Error Response

```json
{
  "detail": [
    {
      "loc": ["body", "framework"],
      "msg": "Framework must be one of: ['langchain', 'autogpt', 'crewai', 'custom', 'openai', 'anthropic']",
      "type": "value_error"
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root
2. **Port Already in Use**: Kill existing processes or use a different port
3. **Schema Validation**: Check the API docs at `/docs` for correct formats
4. **Authentication**: Use `/admin/docs` if auth is enabled

### Debugging Commands

```bash
# Check if server is running
curl -X GET "http://127.0.0.1:8000/health"

# View OpenAPI schema
curl -X GET "http://127.0.0.1:8000/openapi.json" | jq

# Test with verbose output
curl -v -X GET "http://127.0.0.1:8000/"
```

Happy testing! 🚀
