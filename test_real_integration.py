#!/usr/bin/env python3
"""
Test Real Integration of Weather Agent with Mark-1 Orchestrator

This script demonstrates how to use the weather agent integrated with the
Mark-1 orchestrator to process a weather-related query.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import time

# Add the project root to sys.path to make imports work
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Now we can import the adapter
from test_agents.simple_ai_agent.mark1_adapter import Mark1WeatherAgentAdapter

async def test_weather_agent_integration():
    """Test the integration of the weather agent with Mark-1."""
    
    print("\n" + "="*70)
    print("🌤️  TESTING WEATHER AGENT INTEGRATION WITH MARK-1")
    print("="*70)
    
    try:
        # Load test query
        with open("test_data/weather_query.json", "r") as f:
            test_data = json.load(f)
        
        print(f"\n📝 Test Query: {test_data['query']}")
        
        # Create and initialize the adapter
        print("\n🔧 Initializing Weather Agent Adapter...")
        adapter = Mark1WeatherAgentAdapter()
        
        # Initialize the adapter
        init_success = await adapter.initialize()
        if not init_success:
            print("❌ Weather Agent Adapter initialization failed")
            return False
        
        print("✅ Weather Agent Adapter initialized successfully")
        
        # Get agent info
        agent_info = adapter.get_info()
        print(f"\n📋 Agent Information:")
        print(f"   Name: {agent_info['name']}")
        print(f"   Version: {agent_info['version']}")
        print(f"   Capabilities: {', '.join(agent_info['capabilities'])}")
        
        # Execute the query
        print("\n🔍 Executing weather query...")
        start_time = time.time()
        
        result = await adapter.execute(test_data)
        
        execution_time = time.time() - start_time
        print(f"✅ Query executed in {execution_time:.2f} seconds")
        
        # Display results
        print("\n📊 RESULTS:")
        print(f"   Status: {result['status']}")
        if result['status'] == 'success':
            print(f"   City: {result['city']}")
            print(f"   Weather: {result['weather']['weather'][0]['main']} ({result['weather']['weather'][0]['description']})")
            print(f"   Temperature: {result['weather']['main']['temp']}°C")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"   Error: {result.get('message', 'Unknown error')}")
        
        print("\n🏁 Weather Agent Integration Test Completed Successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing weather agent integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_weather_agent_integration())
    sys.exit(0 if success else 1) 