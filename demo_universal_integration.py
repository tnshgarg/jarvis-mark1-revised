#!/usr/bin/env python3
"""
Simple Demo: Universal AI Agent Integration

This demonstrates the Mark-1 Universal Agent Integration System
with a streamlined example.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from src.mark1.agents.universal_integrator import UniversalAgentIntegrator

async def demo_universal_integration():
    """Demonstrate universal integration capabilities"""
    print("🚀 Mark-1 Universal AI Agent Integration Demo")
    print("=" * 50)
    
    # Initialize integrator
    integrator = UniversalAgentIntegrator(Path.cwd())
    
    try:
        # Demo 1: Analyze a repository
        print("\n📊 Demo 1: Repository Analysis")
        print("-" * 30)
        
        repo_url = "https://github.com/joaomdmoura/crewAI.git"
        print(f"Analyzing: {repo_url}")
        
        # Clone and analyze
        clone_path = await integrator._clone_repository(repo_url, "demo_crewai")
        metadata = await integrator._analyze_repository(clone_path)
        
        print(f"✅ Framework Detected: {metadata.framework.value}")
        print(f"📋 Capabilities: {len(metadata.capabilities)}")
        print(f"📦 Dependencies: {len(metadata.dependencies)}")
        print(f"🎯 Entry Points: {len(metadata.entry_points)}")
        
        # Show some capabilities
        if metadata.capabilities:
            print(f"🔧 Key Capabilities: {', '.join([cap.value for cap in metadata.capabilities[:5]])}")
        
        # Demo 2: Create Integration Plan
        print("\n🔧 Demo 2: Integration Planning")
        print("-" * 30)
        
        plan = await integrator._create_integration_plan(metadata, clone_path)
        print(f"Strategy: {plan.integration_strategy}")
        print(f"Wrapper: {plan.wrapper_class}")
        print(f"API Adapter: {plan.api_adapter}")
        print(f"Health Check: {plan.health_check}")
        
        # Demo 3: Simulate Integration Steps
        print("\n⚙️ Demo 3: Integration Execution")
        print("-" * 30)
        
        # Create wrapper (this actually works)
        await integrator._create_agent_wrapper(plan, clone_path)
        print("✅ Agent wrapper created")
        
        # Create API adapter if needed
        await integrator._create_api_adapter(plan, clone_path)
        print("✅ API adapter created")
        
        # Register with Mark-1
        await integrator._register_with_mark1(plan)
        print("✅ Registered with Mark-1")
        
        # Demo 4: List Integrated Agents
        print("\n📋 Demo 4: Agent Management")
        print("-" * 30)
        
        agents = await integrator.list_integrated_agents()
        print(f"Total integrated agents: {len(agents)}")
        
        for agent in agents:
            print(f"  • {agent['name']} ({agent['framework']})")
            capabilities = agent.get('capabilities', [])
            if capabilities:
                print(f"    Capabilities: {', '.join(capabilities[:3])}")
        
        # Demo 5: Show Integration Details
        print("\n📊 Demo 5: Integration Summary")
        print("-" * 30)
        
        if agents:
            demo_agent = agents[-1]  # Get the most recently added agent
            print(f"Agent: {demo_agent['name']}")
            print(f"Framework: {demo_agent['framework']}")
            print(f"Wrapper Class: {demo_agent['wrapper_class']}")
            print(f"Integration Strategy: {demo_agent['integration_strategy']}")
            
            metadata_info = demo_agent.get('metadata', {})
            print(f"Version: {metadata_info.get('version', 'unknown')}")
            print(f"Dependencies: {len(metadata_info.get('dependencies', []))}")
            print(f"Entry Points: {len(metadata_info.get('entry_points', []))}")
        
        print("\n🎉 Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Automatic framework detection")
        print("  ✅ Capability analysis")
        print("  ✅ Dependency extraction")
        print("  ✅ Integration planning")
        print("  ✅ Wrapper generation")
        print("  ✅ Agent registration")
        print("  ✅ Management interface")
        
        print(f"\n📁 Agent cloned to: {clone_path}")
        print(f"🎭 Wrapper file: {clone_path / f'{plan.wrapper_class.lower()}.py'}")
        print(f"📝 Registry: {integrator.mark1_root / 'config' / 'agent_registry.json'}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        integrator.cleanup()

async def show_usage_examples():
    """Show usage examples"""
    print("\n" + "=" * 60)
    print("📚 USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n💻 Command Line Usage:")
    print("# Integrate any AI agent repository:")
    print("mark1 agent integrate https://github.com/microsoft/semantic-kernel.git")
    print("")
    print("# List all integrated agents:")
    print("mark1 agent list")
    print("")
    print("# Test an integrated agent:")
    print("mark1 agent test crewai --prompt 'Generate a Python script'")
    print("")
    print("# Remove an agent:")
    print("mark1 agent remove crewai --force")
    print("")
    print("# Analyze without integrating:")
    print("mark1 agent analyze https://github.com/langchain-ai/langchain.git")
    
    print("\n🐍 Python API Usage:")
    print("""
from src.mark1.agents.universal_integrator import UniversalAgentIntegrator

# Initialize integrator
integrator = UniversalAgentIntegrator(Path.cwd())

# Integrate any repository
plan = await integrator.integrate_repository(
    "https://github.com/joaomdmoura/crewAI.git",
    custom_name="my_crewai"
)

# List integrated agents
agents = await integrator.list_integrated_agents()

# Remove an agent
await integrator.remove_agent("my_crewai")
    """)
    
    print("\n🔄 Supported Workflows:")
    print("1. 🔍 Analyze → 📋 Plan → 🔧 Integrate → 🧪 Test")
    print("2. 📊 Bulk Analysis → 🎯 Selective Integration")
    print("3. 🔄 Continuous Integration → 📈 Monitoring")
    print("4. 🧹 Management → ⚖️ Resource Optimization")

async def test_quick_integration():
    """Test quick integration with BabyAGI"""
    print("\n🚀 Testing Quick Integration with BabyAGI")
    print("=" * 50)
    
    integrator = UniversalAgentIntegrator(Path.cwd())
    
    try:
        repo_url = "https://github.com/yoheinakajima/babyagi.git"
        print(f"Quick integrating: {repo_url}")
        
        # Quick integration steps
        print("🔗 Cloning repository...")
        clone_path = await integrator._clone_repository(repo_url, "babyagi_quick_test")
        print("✅ Cloned successfully")
        
        print("📊 Analyzing repository...")
        metadata = await integrator._analyze_repository(clone_path)
        print(f"✅ Framework: {metadata.framework.value}")
        
        print("📋 Creating plan...")
        plan = await integrator._create_integration_plan(metadata, clone_path)
        print(f"✅ Strategy: {plan.integration_strategy}")
        
        print("🎭 Generating wrapper...")
        await integrator._create_agent_wrapper(plan, clone_path)
        await integrator._create_api_adapter(plan, clone_path)
        await integrator._register_with_mark1(plan)
        print("✅ Quick integration complete!")
        
        print(f"\n📊 Results:")
        print(f"  Name: {metadata.name}")
        print(f"  Framework: {metadata.framework.value}")
        print(f"  Capabilities: {len(metadata.capabilities)}")
        print(f"  Dependencies: {len(metadata.dependencies)}")
        
    except Exception as e:
        print(f"❌ Quick integration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        integrator.cleanup()

if __name__ == "__main__":
    asyncio.run(demo_universal_integration())
    asyncio.run(test_quick_integration())
    asyncio.run(show_usage_examples()) 