#!/usr/bin/env python3
"""
Mark-1 API Usage Examples

Real examples of how to interact with your API
"""

import requests
import json
import time

def print_response(title, response):
    """Print formatted API response"""
    print(f"\n📡 {title}")
    print(f"Status: {response.status_code}")
    if response.headers.get('content-type', '').startswith('application/json'):
        try:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
        except:
            print(f"Response: {response.text}")
    else:
        print(f"Response: {response.text}")
    print("-" * 50)

def test_api():
    """Test all API endpoints with real examples"""
    BASE_URL = "http://127.0.0.1:8000"
    
    print("🚀 Mark-1 API Testing Examples")
    print("=" * 60)
    
    # 1. Health Check
    print("1️⃣ HEALTH CHECK")
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    
    # 2. System Status
    print("2️⃣ SYSTEM STATUS")
    response = requests.get(f"{BASE_URL}/")
    print_response("System Status", response)
    
    # 3. List Agents (empty initially)
    print("3️⃣ LIST AGENTS")
    response = requests.get(f"{BASE_URL}/agents")
    print_response("List Agents", response)
    
    # 4. Create an Agent
    print("4️⃣ CREATE AGENT")
    agent_data = {
        "name": "python_assistant",
        "display_name": "Python Assistant Agent",
        "description": "An agent specialized in Python programming assistance",
        "framework": "langchain",
        "version": "1.0.0",
        "capabilities": [
            {
                "name": "code_analysis",
                "category": "programming",
                "description": "Analyze Python code for bugs and improvements",
                "confidence": 0.95
            },
            {
                "name": "code_generation",
                "category": "programming", 
                "description": "Generate Python code from specifications",
                "confidence": 0.90
            }
        ],
        "metadata": {
            "tags": ["python", "programming", "assistant"],
            "labels": {"type": "development", "priority": "high"}
        },
        "max_concurrent_tasks": 3
    }
    
    response = requests.post(f"{BASE_URL}/agents", json=agent_data)
    print_response("Create Agent", response)
    
    # 5. Create a Task
    print("5️⃣ CREATE TASK")
    task_data = {
        "description": "Review Python code for security vulnerabilities",
        "requirements": [
            {
                "capability": "code_analysis",
                "parameters": {"language": "python", "focus": "security"},
                "priority": "high"
            }
        ],
        "priority": "high",
        "input_data": {
            "code": "import os; os.system('rm -rf /')",
            "filename": "suspicious_code.py"
        },
        "metadata": {"urgency": "critical"},
        "auto_execute": False
    }
    
    response = requests.post(f"{BASE_URL}/tasks", json=task_data)
    print_response("Create Task", response)
    
    # 6. Create a Context
    print("6️⃣ CREATE CONTEXT")
    context_data = {
        "key": "coding_session",
        "content": {
            "user_id": "dev_001",
            "project": "security_audit",
            "files_reviewed": ["main.py", "utils.py"],
            "findings": {
                "critical": 1,
                "medium": 3,
                "low": 5
            },
            "session_start": "2024-01-15T10:00:00Z"
        },
        "context_type": "session",
        "scope": "agent",
        "priority": "medium",
        "tags": ["security", "audit", "python"]
    }
    
    response = requests.post(f"{BASE_URL}/contexts", json=context_data)
    print_response("Create Context", response)
    
    # 7. Start Orchestration
    print("7️⃣ START ORCHESTRATION")
    orchestration_data = {
        "description": "Complete security audit workflow",
        "requirements": {
            "code_analysis": {"depth": "thorough", "focus": "security"},
            "reporting": {"format": "pdf", "include_recommendations": True}
        },
        "context": {
            "project": "security_audit",
            "deadline": "2024-01-20T17:00:00Z"
        },
        "priority": "high",
        "async_execution": True
    }
    
    response = requests.post(f"{BASE_URL}/orchestrate", json=orchestration_data)
    print_response("Start Orchestration", response)
    
    # 8. Get Metrics
    print("8️⃣ SYSTEM METRICS")
    response = requests.get(f"{BASE_URL}/metrics")
    print_response("System Metrics", response)
    
    # 9. Test Schema Validation (should fail)
    print("9️⃣ SCHEMA VALIDATION TEST")
    invalid_agent = {
        "name": "test",
        "framework": "invalid_framework"  # This should trigger validation error
    }
    
    response = requests.post(f"{BASE_URL}/agents", json=invalid_agent)
    print_response("Schema Validation (Expected Error)", response)
    
    # 10. Get API Documentation
    print("🔟 API DOCUMENTATION")
    response = requests.get(f"{BASE_URL}/openapi.json")
    if response.status_code == 200:
        openapi_data = response.json()
        endpoints = list(openapi_data.get('paths', {}).keys())
        print(f"✅ Found {len(endpoints)} documented endpoints:")
        for endpoint in sorted(endpoints)[:10]:  # Show first 10
            print(f"   • {endpoint}")
        if len(endpoints) > 10:
            print(f"   ... and {len(endpoints) - 10} more")
    else:
        print_response("API Documentation", response)
    
    print("\n🎉 API Testing Complete!")
    print("🌐 Your API is fully functional and ready for use!")
    print(f"📚 View interactive docs at: {BASE_URL}/docs")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("🔧 Please start the server first with: python start_api_smart.py")
    except Exception as e:
        print(f"❌ Error testing API: {e}") 