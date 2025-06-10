#!/usr/bin/env python3
"""
Test script for Mark-1 AI Orchestrator
Demonstrates prompt-based AI orchestration with different agent types
"""

import requests
import json
import time
from typing import Dict, Any

class Mark1OrchestratorTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_connection(self) -> bool:
        """Test if the orchestrator is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Orchestrator is healthy!")
                print(f"   - Service: {health_data['service']}")
                print(f"   - Version: {health_data['version']}")
                print(f"   - CrewAI Available: {health_data['crewai_available']}")
                print(f"   - Active Agents: {health_data['active_agents']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def send_prompt(self, prompt: str, task_type: str = "general") -> Dict[str, Any]:
        """Send a prompt to the orchestrator"""
        payload = {
            "prompt": prompt,
            "task_type": task_type
        }
        
        try:
            print(f"\n🤖 Sending prompt ({task_type}): {prompt}")
            response = requests.post(f"{self.base_url}/prompt", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Task completed!")
                print(f"   - Task ID: {result['task_id']}")
                print(f"   - Agent Used: {result['agent_used']}")
                print(f"   - Execution Time: {result['execution_time']:.2f}s")
                print(f"   - Status: {result['status']}")
                print(f"   - Result Preview: {result['result'][:200]}...")
                return result
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Error sending prompt: {e}")
            return {}
    
    def get_agents(self) -> Dict[str, Any]:
        """Get list of available agents"""
        try:
            response = requests.get(f"{self.base_url}/agents")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get agents: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error getting agents: {e}")
            return {}
    
    def run_demo(self) -> None:
        """Run demonstration of various prompt types"""
        try:
            response = requests.post(f"{self.base_url}/demo")
            if response.status_code == 200:
                demo_data = response.json()
                print(f"\n🎭 Demo Results:")
                for result in demo_data['demo_results']:
                    print(f"   📝 Prompt: {result['prompt']}")
                    print(f"   🤖 Agent: {result['agent_used']}")
                    print(f"   ⏱️  Time: {result['execution_time']:.2f}s")
                    print(f"   📄 Preview: {result['result_preview']}")
                    print()
            else:
                print(f"❌ Demo failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error running demo: {e}")

def main():
    print("🚀 Mark-1 AI Orchestrator Test Suite")
    print("=" * 50)
    
    tester = Mark1OrchestratorTester()
    
    # Test connection
    if not tester.test_connection():
        print("❌ Cannot proceed - orchestrator is not responding")
        return
    
    # Get available agents
    print(f"\n📋 Available Agents:")
    agents_data = tester.get_agents()
    if agents_data:
        for agent in agents_data.get('agents', []):
            print(f"   🤖 {agent['name']}: {agent['role']}")
            print(f"      Goal: {agent['goal']}")
            print(f"      Tools: {', '.join(agent['tools'])}")
            print()
    
    # Test various prompt types
    test_prompts = [
        {
            "prompt": "Research the current trends in quantum computing and provide a summary",
            "task_type": "research"
        },
        {
            "prompt": "Write a Python function that implements a binary search algorithm",
            "task_type": "code"
        },
        {
            "prompt": "Analyze the pros and cons of remote work vs office work",
            "task_type": "analysis"
        },
        {
            "prompt": "Help me create a daily routine for productivity",
            "task_type": "general"
        },
        {
            "prompt": "Find information about sustainable energy solutions",
            "task_type": "research"
        }
    ]
    
    print(f"\n🧪 Testing Individual Prompts:")
    print("=" * 30)
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}]")
        result = tester.send_prompt(test["prompt"], test["task_type"])
        time.sleep(1)  # Brief pause between tests
    
    # Run built-in demo
    print(f"\n🎭 Running Built-in Demo:")
    print("=" * 25)
    tester.run_demo()
    
    print(f"\n✅ Test suite completed!")
    print("💡 You can now use the orchestrator as a service by sending prompts to the /prompt endpoint")

if __name__ == "__main__":
    main() 