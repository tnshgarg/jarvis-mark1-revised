#!/usr/bin/env python3
"""
Simple orchestration test to verify the system works without AI-generated invalid plugins.
This test uses only existing plugins and verifies that the orchestration system works correctly.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mark1.core.intelligent_orchestrator import IntelligentOrchestrator
from mark1.plugins import PluginManager
from mark1.core.context_manager import ContextManager
from mark1.core.workflow_engine import WorkflowEngine
from mark1.storage.database import init_database

async def test_simple_orchestration():
    """Test simple orchestration with existing plugins only"""
    print("🧪 Simple Orchestration Test")
    print("=" * 50)
    
    try:
        # Initialize database
        print("🗄️  Initializing database...")
        await init_database()
        print("✅ Database initialized")
        
        # Initialize components
        print("🔧 Initializing components...")
        from pathlib import Path
        plugins_dir = Path.home() / ".mark1" / "plugins"
        plugin_manager = PluginManager(plugins_directory=plugins_dir)
        context_manager = ContextManager()
        workflow_engine = WorkflowEngine()
        
        # Install example plugins if needed
        plugins = await plugin_manager.list_installed_plugins()
        if not plugins:
            print("📦 Installing example plugins...")
            await plugin_manager.install_plugin_from_local_directory(Path("/Users/tanish/.mark1/example_plugins/text_analyzer_plugin"))
            await plugin_manager.install_plugin_from_local_directory(Path("/Users/tanish/.mark1/example_plugins/file_processor_plugin"))
            await plugin_manager.install_plugin_from_local_directory(Path("/Users/tanish/.mark1/example_plugins/data_converter_plugin"))
            plugins = await plugin_manager.list_installed_plugins()
        
        print(f"📦 Found {len(plugins)} plugins:")
        for plugin in plugins:
            print(f"  • {plugin.name} ({plugin.plugin_id})")
        
        # Initialize context manager and workflow engine
        await context_manager.initialize()
        await workflow_engine.initialize()

        # Initialize orchestrator
        OLLAMA_URL = "https://f6da-103-167-213-208.ngrok-free.app"
        orchestrator = IntelligentOrchestrator(ollama_url=OLLAMA_URL)
        await orchestrator.initialize(plugin_manager, context_manager, workflow_engine)
        
        print("✅ All components initialized")
        
        # Test simple text analysis
        print("\n🎯 Test 1: Simple Text Analysis")
        print("-" * 30)
        
        result = await orchestrator.orchestrate_from_prompt(
            user_prompt="Analyze this text: 'Hello world! This is a test message.'",
            context={"test": "simple_analysis"},
            max_plugins=1
        )
        
        print(f"📊 Results:")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
        print(f"  Execution time: {result['execution_time']:.2f}s")
        print(f"  Total steps: {result.get('total_steps', 0)}")
        print(f"  Successful steps: {result.get('successful_steps', 0)}")
        
        if result.get('errors'):
            print(f"  Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"    - {error}")
        
        # Test file processing
        print("\n🎯 Test 2: File Processing")
        print("-" * 30)
        
        result2 = await orchestrator.orchestrate_from_prompt(
            user_prompt="List the files in the current directory",
            context={"test": "file_processing"},
            max_plugins=1
        )
        
        print(f"📊 Results:")
        print(f"  Success: {'✅' if result2['success'] else '❌'}")
        print(f"  Execution time: {result2['execution_time']:.2f}s")
        print(f"  Total steps: {result2.get('total_steps', 0)}")
        print(f"  Successful steps: {result2.get('successful_steps', 0)}")
        
        if result2.get('errors'):
            print(f"  Errors: {len(result2['errors'])}")
            for error in result2['errors']:
                print(f"    - {error}")
        
        # Test data conversion
        print("\n🎯 Test 3: Data Conversion")
        print("-" * 30)
        
        result3 = await orchestrator.orchestrate_from_prompt(
            user_prompt="Convert this JSON to CSV: {'name': 'test', 'value': 123}",
            context={"test": "data_conversion"},
            max_plugins=1
        )
        
        print(f"📊 Results:")
        print(f"  Success: {'✅' if result3['success'] else '❌'}")
        print(f"  Execution time: {result3['execution_time']:.2f}s")
        print(f"  Total steps: {result3.get('total_steps', 0)}")
        print(f"  Successful steps: {result3.get('successful_steps', 0)}")
        
        if result3.get('errors'):
            print(f"  Errors: {len(result3['errors'])}")
            for error in result3['errors']:
                print(f"    - {error}")
        
        # Summary
        print("\n🎉 Test Summary")
        print("=" * 50)
        
        all_successful = all([
            result['success'],
            result2['success'], 
            result3['success']
        ])
        
        print(f"Overall Success: {'✅' if all_successful else '❌'}")
        print(f"Test 1 (Text Analysis): {'✅' if result['success'] else '❌'}")
        print(f"Test 2 (File Processing): {'✅' if result2['success'] else '❌'}")
        print(f"Test 3 (Data Conversion): {'✅' if result3['success'] else '❌'}")
        
        if all_successful:
            print("\n🚀 ALL TESTS PASSED! The orchestration system is working perfectly!")
        else:
            print("\n⚠️  Some tests failed. Check the logs above for details.")
        
        return all_successful
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_orchestration())
    sys.exit(0 if success else 1)
