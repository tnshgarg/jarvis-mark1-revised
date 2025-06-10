#!/usr/bin/env python3
"""
REAL AI DEMO - Mark-1 Orchestrator with Actual AI Agents (Simplified Version)

This demo uses the simplified database implementation with SQLite compatibility.
It focuses on agent discovery and registration without the full orchestration.
"""

import asyncio
import sys
import os
import traceback
import json
from pathlib import Path
from datetime import datetime

# Import simplified database implementation
from database_simplified import (
    init_database, get_db_session, Agent, AgentType, AgentStatus, 
    AgentRepository, DatabaseError
)


async def discover_agents(directory_path):
    """Simple agent discovery function"""
    agents = []
    
    # Define agent patterns to look for
    agent_patterns = {
        "langchain": ["langchain", "chain", "llm"],
        "autogpt": ["autogpt", "autonomous", "agent"],
        "crewai": ["crewai", "crew", "agent"],
        "custom": ["agent", "task", "ai"]
    }
    
    try:
        directory = Path(directory_path)
        if not directory.exists():
            return []
            
        for file_path in directory.glob("**/*.py"):
            try:
                # Skip __init__.py and similar files
                if file_path.name.startswith("_"):
                    continue
                    
                # Read file content
                content = file_path.read_text()
                
                # Check for agent patterns
                agent_type = AgentType.UNKNOWN.value
                for pattern_type, keywords in agent_patterns.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        agent_type = pattern_type
                        break
                
                # If any agent pattern was found
                if agent_type != AgentType.UNKNOWN.value:
                    # Extract name from filename
                    name = file_path.stem.replace("_", " ").title()
                    
                    # Determine capabilities based on content
                    capabilities = []
                    if "text" in content.lower():
                        capabilities.append("text_processing")
                    if "analyze" in content.lower():
                        capabilities.append("analysis")
                    if "generate" in content.lower():
                        capabilities.append("generation")
                    
                    # Create agent data
                    agent_data = {
                        "name": name,
                        "type": agent_type,
                        "file_path": str(file_path),
                        "capabilities": capabilities,
                        "metadata": {
                            "discovered_at": datetime.now().isoformat(),
                            "size_bytes": file_path.stat().st_size
                        }
                    }
                    
                    agents.append(agent_data)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                
        return agents
    except Exception as e:
        print(f"Error discovering agents: {e}")
        return []


async def simplified_demo():
    """
    Simplified AI demo using the modified database
    """
    print("🤖 SIMPLIFIED REAL AI DEMO - Mark-1")
    print("=" * 50)
    print("This demo demonstrates agent discovery and database integration.")
    
    # Step 1: Initialize database
    print("\n🔧 Initializing database...")
    
    # Ensure aiosqlite is installed
    try:
        import aiosqlite
        print("✅ aiosqlite is installed")
    except ImportError:
        print("⚠️ aiosqlite is not installed. Installing now...")
        os.system("pip install aiosqlite")
        import aiosqlite
        print("✅ aiosqlite installed successfully")
    
    try:
        # Define settings override for a clean test database
        db_path = "data/simplified_demo.sqlite"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        settings_override = {
            "database_url": f"sqlite:///{db_path}",
            "database_echo": False
        }
        
        # Use force_recreate=True to ensure tables use the updated schema
        await init_database(settings_override=settings_override, force_recreate=True)
        print("✅ Database initialized successfully")
    except DatabaseError as e:
        print(f"❌ Database initialization failed: {e}")
        traceback.print_exc()
        return
    
    # Step 2: Discover agents
    print("\n🔍 Discovering AI agents...")
    
    # Look in test_agents directory
    agents_data = await discover_agents("test_agents")
    
    if not agents_data:
        # Try agents directory if test_agents is empty
        agents_data = await discover_agents("agents")
    
    if not agents_data:
        print("⚠️ No agent directories found. Creating sample agents...")
        # Create a sample agent directory and file for testing
        os.makedirs("test_agents/sample", exist_ok=True)
        sample_agent = """
        '''
        Sample AI Agent for testing
        
        This is a langchain-compatible agent that can analyze text and generate responses.
        '''
        
        def process_text(text):
            '''Analyze and process text input'''
            return "Processed: " + text
            
        def generate_response(prompt):
            '''Generate AI response to the prompt'''
            return "AI response to: " + prompt
        """
        
        with open("test_agents/sample/text_processor_agent.py", "w") as f:
            f.write(sample_agent)
            
        print("✅ Created sample agent in test_agents/sample/text_processor_agent.py")
        
        # Try again with the new sample
        agents_data = await discover_agents("test_agents")
    
    print(f"✅ Found {len(agents_data)} potential AI agents")
    
    # Step 3: Register agents in database
    print("\n📝 Registering agents in database...")
    
    registered_agents = []
    async with get_db_session() as session:
        agent_repo = AgentRepository()
        
        for agent_data in agents_data:
            try:
                agent = await agent_repo.create_agent(
                    session=session,
                    name=agent_data["name"],
                    agent_type=agent_data["type"],
                    framework="python",
                    file_path=agent_data["file_path"],
                    capabilities=agent_data["capabilities"],
                    metadata=agent_data["metadata"]
                )
                registered_agents.append(agent)
                print(f"  ✅ Registered: {agent.name} ({agent.agent_type})")
            except Exception as e:
                print(f"  ❌ Failed to register {agent_data['name']}: {e}")
    
    # Step 4: List all registered agents
    print("\n📋 All registered agents:")
    
    async with get_db_session() as session:
        agent_repo = AgentRepository()
        all_agents = await agent_repo.list_all(session)
        
        for i, agent in enumerate(all_agents, 1):
            print(f"  {i}. {agent.name} ({agent.agent_type})")
            print(f"     - ID: {agent.id}")
            print(f"     - File: {agent.file_path}")
            print(f"     - Capabilities: {agent.capabilities}")
            print(f"     - Status: {agent.status}")
            print(f"     - Created: {agent.created_at}")
            print()
    
    print("✅ Demo completed!")


async def main():
    await simplified_demo()


if __name__ == "__main__":
    asyncio.run(main()) 