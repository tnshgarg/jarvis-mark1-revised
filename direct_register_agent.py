#!/usr/bin/env python3
"""
Direct Agent Registration Script for Mark-1

This script bypasses the ORM layer and directly executes SQL to register
our weather agent with the Mark-1 database.
"""

import asyncio
import sys
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# Database path
DB_PATH = "data/mark1_db.sqlite"

def register_weather_agent_direct():
    """Register the Weather Agent directly using SQL."""
    try:
        print(f"Connecting to database at {DB_PATH}...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Generate a UUID for the agent
        agent_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Tags as JSON
        tags = json.dumps(["weather", "recommendations", "forecast"])
        
        # Extra metadata as JSON
        extra_metadata = json.dumps({
            "author": "Demo Developer",
            "version": "1.0.0",
            "capabilities": ["weather_forecast", "weather_analysis", "recommendation"],
            "requires_api_key": True,
            "api_key_env": "WEATHER_API_KEY"
        })
        
        # Insert the agent record using the actual columns from the schema
        # Including all required NOT NULL fields
        sql = """
        INSERT INTO agents (
            id, name, display_name, description, agent_type, 
            integration_type, status, health_score, error_count, file_path, 
            success_rate, total_executions, failed_executions, is_integrated,
            llm_calls_replaced, scan_count, is_sandboxed, security_level,
            tags, extra_metadata, created_at, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        
        cursor.execute(sql, (
            agent_id,                                     # id
            "Weather Agent",                              # name
            "Weather Information Agent",                  # display_name
            "Provides weather information and recommendations based on location", # description
            "custom",                                     # agent_type
            "direct",                                     # integration_type
            "ready",                                      # status
            100,                                          # health_score
            0,                                            # error_count
            "test_agents/simple_ai_agent/weather_agent.py", # file_path
            100,                                          # success_rate
            0,                                            # total_executions
            0,                                            # failed_executions
            1,                                            # is_integrated
            0,                                            # llm_calls_replaced
            1,                                            # scan_count
            1,                                            # is_sandboxed
            "standard",                                   # security_level
            tags,                                         # tags
            extra_metadata,                               # extra_metadata
            now,                                          # created_at
            now                                           # updated_at
        ))
        
        # Commit the transaction
        conn.commit()
        
        print(f"Agent registered successfully with ID: {agent_id}")
        print(f"Agent details: Weather Agent (custom)")
        print("Agent is now registered and ready to use")
        
        # Verify the agent was registered
        cursor.execute("SELECT id, name, agent_type FROM agents WHERE id = ?", (agent_id,))
        agent = cursor.fetchone()
        print(f"Verification: {agent}")
        
        # Close the connection
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error registering agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = register_weather_agent_direct()
    sys.exit(0 if success else 1) 