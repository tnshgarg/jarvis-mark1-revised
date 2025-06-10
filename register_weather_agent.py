#!/usr/bin/env python3
"""
Register Weather Agent with Mark-1 Database

This script registers the Weather Agent with the Mark-1 database system
to make it available for the orchestrator.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Import simplified database implementation
from database_simplified import (
    init_database, get_db_session, AgentRepository
)

async def register_weather_agent():
    """Register the Weather Agent with the Mark-1 database."""
    try:
        print("Initializing database...")
        await init_database()
        print("Database initialized successfully.")
        
        # Prepare agent metadata (compatible with the existing schema)
        agent_metadata = {
            "author": "Demo Developer",
            "created_at": datetime.now().isoformat(),
            "description": "A simple AI agent that provides weather information and recommendations",
            "version": "1.0.0",
            "capabilities": ["weather_forecast", "weather_analysis", "recommendation"]
        }
        
        # Register the agent
        async with get_db_session() as session:
            repo = AgentRepository()
            agent = await repo.create_agent(
                session=session,
                name="Weather Agent",
                agent_type="custom",
                framework="python",
                file_path="test_agents/simple_ai_agent/weather_agent.py",
                metadata=agent_metadata
            )
            print(f"Agent registered successfully with ID: {agent.id}")
            print(f"Agent details: {agent.name} ({agent.agent_type})")
            
            # Update additional fields
            query = f"""
            UPDATE agents 
            SET display_name = 'Weather Information Agent', 
                description = 'Provides weather information and recommendations based on location',
                integration_type = 'direct',
                tags = '{json.dumps(["weather", "recommendations", "forecast"])}',
                security_level = 'standard',
                is_integrated = 1,
                success_rate = 100,
                total_executions = 0,
                failed_executions = 0,
                scan_count = 1,
                is_sandboxed = 1
            WHERE id = '{agent.id}'
            """
            await session.execute(query)
            await session.commit()
            
            print("Agent updated with additional fields")
            print(f"Agent {agent.name} is now registered and ready to use")
            
    except Exception as e:
        print(f"Error registering agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(register_weather_agent())
    sys.exit(0 if success else 1) 