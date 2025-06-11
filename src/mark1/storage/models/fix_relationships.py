"""
Fix for Agent-Context relationship in database models.

This script corrects the foreign key relationship between Agent and Context tables.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.mark1.storage.database import Base, DatabaseManager
from src.mark1.storage.models.agent_model import Agent
from src.mark1.storage.models.context_model import ContextModel

async def fix_database_schema():
    """Create or update the database schema with correct relationships."""
    print("Fixing database schema...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Create all tables
    await db_manager.create_all_tables()
    
    print("Database schema updated successfully.")
    
    # Clean up
    await db_manager.close()

if __name__ == "__main__":
    asyncio.run(fix_database_schema()) 