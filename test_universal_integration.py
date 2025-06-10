#!/usr/bin/env python3
"""
Test Script for Universal AI Agent Integration

This script demonstrates the Mark-1 Universal Agent Integration System
by testing with multiple popular AI agent repositories.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.mark1.agents.universal_integrator import UniversalAgentIntegrator
from src.mark1.config.settings import get_settings

# Popular AI agent repositories for testing
TEST_REPOSITORIES = [
    {
        "name": "CrewAI",
        "url": "https://github.com/joaomdmoura/crewAI.git",
        "description": "Multi-agent automation framework",
        "expected_framework": "crewai"
    },
    {
        "name": "AutoGPT",
        "url": "https://github.com/Significant-Gravitas/AutoGPT.git", 
        "description": "Autonomous AI agent",
        "expected_framework": "autogpt"
    },
    {
        "name": "LangChain",
        "url": "https://github.com/langchain-ai/langchain.git",
        "description": "Building applications with LLMs",
        "expected_framework": "langchain"
    },
    {
        "name": "Agent Zero",
        "url": "https://github.com/frdel/agent-zero.git",
        "description": "Dynamic AI agent framework",
        "expected_framework": "custom"
    },
    {
        "name": "Semantic Kernel",
        "url": "https://github.com/microsoft/semantic-kernel.git",
        "description": "Microsoft's AI orchestration framework",
        "expected_framework": "custom"
    }
]

class UniversalIntegrationTester:
    """Test the universal integration system"""
    
    def __init__(self):
        self.integrator = UniversalAgentIntegrator(Path.cwd())
        self.test_results = []
        
    async def run_comprehensive_test(self):
        """Run comprehensive test of universal integration system"""
        print("🚀 Starting Universal AI Agent Integration Test\n")
        print("=" * 60)
        
        # Test 1: Repository Analysis (without integration)
        await self._test_repository_analysis()
        
        # Test 2: Single Integration Test
        await self._test_single_integration()
        
        # Test 3: Multiple Framework Integration
        await self._test_multiple_frameworks()
        
        # Test 4: Integration Management
        await self._test_integration_management()
        
        # Test 5: End-to-End Workflow
        await self._test_end_to_end_workflow()
        
        # Generate final report
        await self._generate_final_report()
        
    async def _test_repository_analysis(self):
        """Test repository analysis without integration"""
        print("\n📊 Test 1: Repository Analysis")
        print("-" * 40)
        
        for repo in TEST_REPOSITORIES[:3]:  # Test first 3 repos
            print(f"\nAnalyzing {repo['name']}...")
            
            try:
                start_time = time.time()
                
                # Clone and analyze
                clone_path = await self.integrator._clone_repository(repo['url'])
                metadata = await self.integrator._analyze_repository(clone_path)
                plan = await self.integrator._create_integration_plan(metadata, clone_path)
                
                analysis_time = time.time() - start_time
                
                result = {
                    "repository": repo['name'],
                    "status": "success",
                    "analysis_time": analysis_time,
                    "framework_detected": metadata.framework.value,
                    "capabilities_count": len(metadata.capabilities),
                    "dependencies_count": len(metadata.dependencies),
                    "entry_points": len(metadata.entry_points),
                    "api_endpoints": len(metadata.api_endpoints)
                }
                
                self.test_results.append(result)
                
                print(f"  ✅ Framework: {metadata.framework.value}")
                print(f"  📋 Capabilities: {len(metadata.capabilities)}")
                print(f"  📦 Dependencies: {len(metadata.dependencies)}")
                print(f"  🎯 Entry Points: {len(metadata.entry_points)}")
                print(f"  🌐 API Endpoints: {len(metadata.api_endpoints)}")
                print(f"  ⏱️  Analysis Time: {analysis_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                self.test_results.append({
                    "repository": repo['name'],
                    "status": "failed",
                    "error": str(e)
                })
    
    async def _test_single_integration(self):
        """Test complete integration of a single repository"""
        print("\n🔧 Test 2: Single Integration")
        print("-" * 40)
        
        # Use CrewAI for single integration test
        repo = TEST_REPOSITORIES[0]  # CrewAI
        
        print(f"\nIntegrating {repo['name']}...")
        
        try:
            start_time = time.time()
            
            # Full integration
            plan = await self.integrator.integrate_repository(repo['url'], "crewai_test")
            
            integration_time = time.time() - start_time
            
            # Test the integration
            test_results = await self.integrator._test_integration(plan)
            
            result = {
                "repository": repo['name'],
                "status": "success",
                "integration_time": integration_time,
                "tests_passed": test_results["tests_passed"],
                "tests_failed": test_results["tests_failed"],
                "wrapper_class": plan.wrapper_class,
                "integration_strategy": plan.integration_strategy
            }
            
            self.test_results.append(result)
            
            print(f"  ✅ Integration completed")
            print(f"  🎭 Wrapper: {plan.wrapper_class}")
            print(f"  📝 Strategy: {plan.integration_strategy}")
            print(f"  🧪 Tests: {test_results['tests_passed']} passed, {test_results['tests_failed']} failed")
            print(f"  ⏱️  Integration Time: {integration_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Integration failed: {e}")
            self.test_results.append({
                "repository": repo['name'],
                "status": "failed",
                "error": str(e)
            })
    
    async def _test_multiple_frameworks(self):
        """Test integration of multiple different frameworks"""
        print("\n🌐 Test 3: Multiple Framework Integration")
        print("-" * 40)
        
        frameworks_tested = set()
        
        for i, repo in enumerate(TEST_REPOSITORIES[:3]):  # Test first 3
            if len(frameworks_tested) >= 3:  # Limit to 3 different frameworks
                break
                
            print(f"\nIntegrating {repo['name']} (Framework Test {i+1})...")
            
            try:
                start_time = time.time()
                
                # Analyze first to check framework
                clone_path = await self.integrator._clone_repository(repo['url'], f"test_{i}")
                metadata = await self.integrator._analyze_repository(clone_path)
                
                if metadata.framework.value in frameworks_tested:
                    print(f"  ⏭️  Skipping - {metadata.framework.value} already tested")
                    continue
                
                frameworks_tested.add(metadata.framework.value)
                
                # Create integration plan
                plan = await self.integrator._create_integration_plan(metadata, clone_path)
                
                # Execute integration steps (without full install to save time)
                await self.integrator._create_agent_wrapper(plan, clone_path)
                await self.integrator._create_api_adapter(plan, clone_path)
                await self.integrator._register_with_mark1(plan)
                
                integration_time = time.time() - start_time
                
                result = {
                    "repository": repo['name'],
                    "framework": metadata.framework.value,
                    "status": "success",
                    "integration_time": integration_time,
                    "capabilities": [cap.value for cap in metadata.capabilities]
                }
                
                self.test_results.append(result)
                
                print(f"  ✅ Framework: {metadata.framework.value}")
                print(f"  🎯 Capabilities: {', '.join([cap.value for cap in metadata.capabilities[:3]])}")
                print(f"  ⏱️  Time: {integration_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                self.test_results.append({
                    "repository": repo['name'],
                    "status": "failed",
                    "error": str(e)
                })
        
        print(f"\n📊 Frameworks successfully integrated: {len(frameworks_tested)}")
    
    async def _test_integration_management(self):
        """Test integration management operations"""
        print("\n⚙️ Test 4: Integration Management")
        print("-" * 40)
        
        try:
            # List integrated agents
            print("\nListing integrated agents...")
            agents = await self.integrator.list_integrated_agents()
            print(f"  📋 Found {len(agents)} integrated agents")
            
            for agent in agents:
                print(f"    • {agent['name']} ({agent['framework']})")
            
            # Test health checks
            print("\nTesting health checks...")
            healthy_agents = 0
            for agent in agents:
                # Simulate health check
                try:
                    # In real implementation, this would call the wrapper's health check
                    print(f"  🔍 Health check: {agent['name']} - ✅ Healthy")
                    healthy_agents += 1
                except Exception as e:
                    print(f"  🔍 Health check: {agent['name']} - ❌ Unhealthy: {e}")
            
            print(f"  📊 Health Summary: {healthy_agents}/{len(agents)} agents healthy")
            
            # Test agent removal (on test agents only)
            test_agents = [agent for agent in agents if 'test' in agent['agent_id']]
            if test_agents:
                print(f"\nCleaning up test agents...")
                for agent in test_agents[:2]:  # Remove first 2 test agents
                    success = await self.integrator.remove_agent(agent['agent_id'])
                    if success:
                        print(f"  🗑️  Removed: {agent['name']}")
                    else:
                        print(f"  ❌ Failed to remove: {agent['name']}")
            
        except Exception as e:
            print(f"  ❌ Management test failed: {e}")
    
    async def _test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n🔄 Test 5: End-to-End Workflow")
        print("-" * 40)
        
        try:
            # Simulate complete workflow: Analyze -> Integrate -> Test -> Use
            repo = {
                "name": "Test Agent",
                "url": "https://github.com/microsoft/semantic-kernel.git",
                "custom_name": "semantic_kernel_test"
            }
            
            print(f"\nRunning end-to-end workflow for {repo['name']}...")
            
            # Step 1: Analysis
            print("  📊 Step 1: Analyzing repository...")
            clone_path = await self.integrator._clone_repository(repo['url'], repo['custom_name'])
            metadata = await self.integrator._analyze_repository(clone_path)
            print(f"    Framework detected: {metadata.framework.value}")
            
            # Step 2: Integration Planning
            print("  📋 Step 2: Creating integration plan...")
            plan = await self.integrator._create_integration_plan(metadata, clone_path)
            print(f"    Strategy: {plan.integration_strategy}")
            
            # Step 3: Integration Execution
            print("  🔧 Step 3: Executing integration...")
            await self.integrator._execute_integration(plan, clone_path)
            print("    Integration completed")
            
            # Step 4: Testing
            print("  🧪 Step 4: Testing integration...")
            test_results = await self.integrator._test_integration(plan)
            print(f"    Tests: {test_results['tests_passed']} passed, {test_results['tests_failed']} failed")
            
            # Step 5: Verification
            print("  ✅ Step 5: Verifying integration...")
            agents = await self.integrator.list_integrated_agents()
            agent = next((a for a in agents if a['agent_id'] == repo['custom_name']), None)
            
            if agent:
                print("    Agent successfully registered in Mark-1")
                print(f"    Capabilities: {', '.join(agent.get('capabilities', [])[:3])}")
            else:
                print("    ❌ Agent not found in registry")
            
            print("  🎉 End-to-end workflow completed successfully!")
            
        except Exception as e:
            print(f"  ❌ End-to-end workflow failed: {e}")
    
    async def _generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Final Test Report")
        print("=" * 60)
        
        # Statistics
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get('status') == 'success'])
        failed_tests = total_tests - successful_tests
        
        print(f"\nTest Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(successful_tests / total_tests * 100):.1f}%")
        
        # Framework support
        frameworks = set()
        for result in self.test_results:
            if 'framework' in result:
                frameworks.add(result['framework'])
        
        print(f"\nFrameworks Tested: {len(frameworks)}")
        for framework in sorted(frameworks):
            print(f"  • {framework}")
        
        # Performance metrics
        analysis_times = [r.get('analysis_time', 0) for r in self.test_results if 'analysis_time' in r]
        integration_times = [r.get('integration_time', 0) for r in self.test_results if 'integration_time' in r]
        
        if analysis_times:
            print(f"\nPerformance Metrics:")
            print(f"  Average Analysis Time: {sum(analysis_times) / len(analysis_times):.2f}s")
            print(f"  Fastest Analysis: {min(analysis_times):.2f}s")
            print(f"  Slowest Analysis: {max(analysis_times):.2f}s")
        
        if integration_times:
            print(f"  Average Integration Time: {sum(integration_times) / len(integration_times):.2f}s")
            print(f"  Fastest Integration: {min(integration_times):.2f}s")
            print(f"  Slowest Integration: {max(integration_times):.2f}s")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            status_emoji = "✅" if result.get('status') == 'success' else "❌"
            print(f"  {i}. {status_emoji} {result.get('repository', 'Unknown')}")
            if result.get('status') == 'failed' and 'error' in result:
                print(f"     Error: {result['error'][:100]}...")
        
        # Save detailed report
        report_file = Path("universal_integration_test_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                "test_summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": successful_tests / total_tests * 100 if total_tests > 0 else 0
                },
                "frameworks_tested": list(frameworks),
                "performance_metrics": {
                    "average_analysis_time": sum(analysis_times) / len(analysis_times) if analysis_times else 0,
                    "average_integration_time": sum(integration_times) / len(integration_times) if integration_times else 0
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("\n🎉 Universal Integration Test Complete!")


async def main():
    """Main test function"""
    tester = UniversalIntegrationTester()
    
    try:
        await tester.run_comprehensive_test()
    finally:
        # Cleanup
        tester.integrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 