#!/usr/bin/env python3
"""
Basic setup test for Mark-1 Orchestrator

Tests that core components can be imported and initialized without errors.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_imports():
    """Test that core modules can be imported"""
    print("Testing basic imports...")
    
    try:
        # Test configuration
        from mark1.config.settings import get_settings
        settings = get_settings()
        print(f"✓ Settings loaded: {settings.app_name} v{settings.version}")
        
        # Test exceptions
        from mark1.utils.exceptions import Mark1BaseException, DatabaseError, ConfigurationError
        print("✓ Exception classes imported successfully")
        
        # Test database
        from mark1.storage.database import DatabaseManager
        db_manager = DatabaseManager()
        print("✓ Database manager created successfully")
        
        # Test agent registry
        from mark1.agents.registry import AgentRegistry
        registry = AgentRegistry()
        print("✓ Agent registry created successfully")
        
        # Test agent pool
        from mark1.agents.pool import AgentPool
        pool = AgentPool()
        print("✓ Agent pool created successfully")
        
        # Test orchestrator
        from mark1.core.orchestrator import Mark1Orchestrator
        orchestrator = Mark1Orchestrator()
        print("✓ Orchestrator created successfully")
        
        print("\n✅ All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_initialization():
    """Test basic initialization of components"""
    print("\nTesting basic initialization...")
    
    try:
        from mark1.config.settings import get_settings
        from mark1.agents.registry import AgentRegistry
        
        settings = get_settings()
        print(f"✓ Settings initialized: environment={settings.environment}")
        
        # Test directory creation
        settings.create_directories()
        print(f"✓ Directories created: {settings.data_dir}, {settings.log_dir}")
        
        # Test agent registry initialization
        registry = AgentRegistry()
        # Note: We can't fully initialize without database setup
        print("✓ Agent registry basic initialization successful")
        
        print("\n✅ Basic initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all basic tests"""
    print("🚀 Mark-1 Orchestrator Basic Setup Test")
    print("=" * 50)
    
    # Run tests
    imports_ok = await test_basic_imports()
    if not imports_ok:
        sys.exit(1)
    
    init_ok = await test_basic_initialization()
    if not init_ok:
        sys.exit(1)
    
    print("\n🎉 All basic tests passed! Mark-1 setup is working correctly.")
    print("\nNext steps:")
    print("1. Set up database (PostgreSQL recommended)")
    print("2. Configure .env file with database URL")
    print("3. Run: mark1 init")
    print("4. Run: mark1 status")

if __name__ == "__main__":
    asyncio.run(main()) 