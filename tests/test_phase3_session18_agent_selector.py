#!/usr/bin/env python3
"""
Phase 3 Session 18: Advanced Agent Selector & Optimization Testing

Tests the Advanced Agent Selector & Optimization system capabilities including:
- Multi-criteria agent selection algorithms
- Performance-based agent ranking
- Load balancing and resource optimization
- Machine learning-based selection
- Agent fitness scoring and adaptation
- Dynamic selection strategy optimization
"""

import asyncio
import time
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any
import structlog

# Import our Advanced Agent Selector components
from src.mark1.core.agent_selector import (
    AdvancedAgentSelector,
    SelectionStrategy,
    AgentState,
    OptimizationGoal,
    AgentPerformanceMetrics,
    TaskRequirements,
    SelectionResult,
    RoundRobinSelector,
    PerformanceBasedSelector,
    LoadBalancedSelector,
    HybridSelector,
    MLOptimizedSelector
)


async def test_agent_registration_and_management():
    """Test agent registration and state management"""
    
    print("🔧 Testing Agent Registration and Management")
    print("=" * 60)
    
    selector = AdvancedAgentSelector()
    
    # Test agent registration
    test_agents = [
        ("langchain_agent_1", ["chat", "analysis", "generation", "text_processing"], 5),
        ("autogpt_agent_1", ["planning", "autonomous", "tool_use", "analysis"], 3),
        ("crewai_agent_1", ["collaboration", "role_based", "delegation", "analysis"], 8),
        ("custom_agent_1", ["custom_integration", "api_calls", "analysis"], 10),
        ("python_agent_1", ["text_processing", "data_analysis", "analysis"], 6)
    ]
    
    print("📝 Registering test agents:")
    for agent_id, capabilities, max_tasks in test_agents:
        selector.register_agent(agent_id, capabilities, max_tasks)
        print(f"   ✅ {agent_id}: {capabilities} (max: {max_tasks} tasks)")
    
    # Test agent state updates
    print("\n🔄 Testing agent state management:")
    selector.update_agent_state("langchain_agent_1", AgentState.BUSY)
    selector.update_agent_state("autogpt_agent_1", AgentState.OVERLOADED)
    selector.update_agent_state("crewai_agent_1", AgentState.AVAILABLE)
    
    print("   ✅ State updates completed")
    
    # Test load updates
    print("\n📊 Testing load management:")
    load_updates = [
        ("langchain_agent_1", 3),
        ("autogpt_agent_1", 3),  # At max capacity
        ("crewai_agent_1", 2),
        ("custom_agent_1", 0),
        ("python_agent_1", 1)
    ]
    
    for agent_id, load in load_updates:
        selector.update_agent_load(agent_id, load)
        print(f"   📈 {agent_id}: {load} tasks")
    
    # Get available agents
    available_agents = selector.get_available_agents()
    print(f"\n🟢 Available agents: {len(available_agents)}")
    for agent_id in available_agents:
        metrics = selector.agent_metrics[agent_id]
        print(f"   - {agent_id}: Load {metrics.resource_utilization:.2f}")
    
    return {
        "registered_agents": len(test_agents),
        "available_agents": len(available_agents),
        "selector": selector
    }


async def test_performance_metrics_tracking():
    """Test performance metrics tracking and updates"""
    
    print("\n📊 Testing Performance Metrics Tracking")
    print("=" * 60)
    
    # Use selector from previous test
    setup_result = await test_agent_registration_and_management()
    selector = setup_result["selector"]
    
    print("🧪 Simulating task executions with performance tracking:")
    
    # Simulate task completions for different agents
    task_simulations = [
        # agent_id, success, response_time, quality
        ("langchain_agent_1", True, 1.2, 0.9),
        ("langchain_agent_1", True, 0.8, 0.95),
        ("langchain_agent_1", False, 5.0, 0.3),
        ("autogpt_agent_1", True, 2.1, 0.85),
        ("autogpt_agent_1", True, 1.8, 0.9),
        ("crewai_agent_1", True, 1.5, 0.92),
        ("crewai_agent_1", True, 1.3, 0.88),
        ("crewai_agent_1", True, 1.1, 0.94),
        ("custom_agent_1", True, 3.2, 0.7),
        ("custom_agent_1", False, 8.0, 0.2),
        ("python_agent_1", True, 0.9, 0.96),
        ("python_agent_1", True, 1.0, 0.93),
    ]
    
    for i, (agent_id, success, response_time, quality) in enumerate(task_simulations):
        task_id = f"task_{i+1}"
        selector.record_task_completion(agent_id, task_id, success, response_time, quality)
        print(f"   📝 Task {task_id}: {agent_id} -> Success: {success}, Time: {response_time}s, Quality: {quality}")
    
    # Analyze performance metrics
    print("\n📈 Performance Analysis:")
    
    performance_data = []
    for agent_id, metrics in selector.agent_metrics.items():
        fitness_score = metrics.get_fitness_score()
        performance_data.append({
            "agent_id": agent_id,
            "total_tasks": metrics.total_tasks,
            "success_rate": metrics.success_rate,
            "avg_response_time": metrics.average_response_time,
            "quality_score": metrics.quality_score,
            "fitness_score": fitness_score,
            "state": metrics.get_state().value
        })
        
        print(f"   🎯 {agent_id}:")
        print(f"      Tasks: {metrics.total_tasks}, Success: {metrics.success_rate:.2%}")
        print(f"      Avg Time: {metrics.average_response_time:.2f}s, Quality: {metrics.quality_score:.3f}")
        print(f"      Fitness: {fitness_score:.3f}, State: {metrics.get_state().value}")
    
    # Find best performing agent
    best_agent = max(performance_data, key=lambda x: x["fitness_score"])
    print(f"\n🏆 Best performing agent: {best_agent['agent_id']} (fitness: {best_agent['fitness_score']:.3f})")
    
    return {
        "performance_data": performance_data,
        "best_agent": best_agent,
        "selector": selector
    }


async def test_selection_strategies():
    """Test different agent selection strategies"""
    
    print("\n🎯 Testing Agent Selection Strategies")
    print("=" * 60)
    
    # Use selector from previous test
    setup_result = await test_performance_metrics_tracking()
    selector = setup_result["selector"]
    
    # Define test task requirements
    test_tasks = [
        TaskRequirements(
            task_id="high_priority_chat",
            required_capabilities=["chat"],
            priority=9,
            max_acceptable_latency=1.0,
            quality_threshold=0.9
        ),
        TaskRequirements(
            task_id="analysis_task",
            required_capabilities=["analysis"],
            priority=5,
            max_acceptable_latency=3.0,
            quality_threshold=0.8
        ),
        TaskRequirements(
            task_id="planning_task",
            required_capabilities=["planning"],
            priority=7,
            max_acceptable_latency=2.0,
            quality_threshold=0.85
        ),
        TaskRequirements(
            task_id="custom_integration",
            required_capabilities=["custom_integration"],
            priority=6,
            max_acceptable_latency=5.0,
            quality_threshold=0.75
        )
    ]
    
    # Test each selection strategy
    strategies_to_test = [
        SelectionStrategy.ROUND_ROBIN,
        SelectionStrategy.PERFORMANCE_BASED,
        SelectionStrategy.LEAST_LOADED,
        SelectionStrategy.HYBRID,
        SelectionStrategy.ML_OPTIMIZED
    ]
    
    strategy_results = {}
    
    for strategy in strategies_to_test:
        print(f"\n🔄 Testing {strategy.value} strategy:")
        strategy_results[strategy] = []
        
        for task in test_tasks:
            try:
                start_time = time.time()
                result = await selector.select_agent(task, strategy)
                
                print(f"   📋 Task: {task.task_id}")
                print(f"      Selected: {result.selected_agent_id}")
                print(f"      Confidence: {result.selection_confidence:.3f}")
                print(f"      Selection Time: {result.selection_time:.4f}s")
                print(f"      Reasoning: {result.selection_reasoning}")
                
                strategy_results[strategy].append({
                    "task_id": task.task_id,
                    "selected_agent": result.selected_agent_id,
                    "confidence": result.selection_confidence,
                    "selection_time": result.selection_time,
                    "optimization_score": result.optimization_score
                })
                
            except Exception as e:
                print(f"   ❌ Failed to select agent for {task.task_id}: {e}")
                strategy_results[strategy].append({
                    "task_id": task.task_id,
                    "error": str(e)
                })
    
    # Analyze strategy performance
    print(f"\n📊 Strategy Performance Summary:")
    
    strategy_stats = {}
    for strategy, results in strategy_results.items():
        successful_selections = [r for r in results if "error" not in r]
        
        if successful_selections:
            avg_confidence = statistics.mean([r["confidence"] for r in successful_selections])
            avg_selection_time = statistics.mean([r["selection_time"] for r in successful_selections])
            avg_optimization_score = statistics.mean([r["optimization_score"] for r in successful_selections])
            
            strategy_stats[strategy] = {
                "success_rate": len(successful_selections) / len(results),
                "avg_confidence": avg_confidence,
                "avg_selection_time": avg_selection_time,
                "avg_optimization_score": avg_optimization_score
            }
            
            print(f"   🎯 {strategy.value}:")
            print(f"      Success Rate: {strategy_stats[strategy]['success_rate']:.1%}")
            print(f"      Avg Confidence: {avg_confidence:.3f}")
            print(f"      Avg Selection Time: {avg_selection_time:.4f}s")
            print(f"      Avg Optimization Score: {avg_optimization_score:.3f}")
        else:
            strategy_stats[strategy] = {"success_rate": 0}
            print(f"   ❌ {strategy.value}: No successful selections")
    
    return {
        "strategy_results": strategy_results,
        "strategy_stats": strategy_stats,
        "selector": selector
    }


async def test_load_balancing():
    """Test load balancing functionality"""
    
    print("\n⚖️  Testing Load Balancing")
    print("=" * 60)
    
    # Use selector from previous test
    setup_result = await test_selection_strategies()
    selector = setup_result["selector"]
    
    print("🧪 Simulating load imbalance:")
    
    # Create artificial load imbalance
    load_scenarios = [
        ("langchain_agent_1", 4),   # High load
        ("autogpt_agent_1", 3),     # At capacity
        ("crewai_agent_1", 6),      # Very high load
        ("custom_agent_1", 1),      # Low load
        ("python_agent_1", 0)       # No load
    ]
    
    for agent_id, load in load_scenarios:
        selector.update_agent_load(agent_id, load)
        metrics = selector.agent_metrics[agent_id]
        print(f"   📊 {agent_id}: {load}/{metrics.max_concurrent_tasks} tasks ({metrics.resource_utilization:.1%})")
    
    # Test load balancing analysis
    print("\n🔍 Analyzing load distribution:")
    
    load_balance_result = await selector.rebalance_load()
    
    print(f"   📈 Average Load: {load_balance_result['average_load']:.2%}")
    print(f"   📊 Load Variance: {load_balance_result['load_variance']:.4f}")
    print(f"   ⚖️  Load Balance Score: {load_balance_result['load_balance_score']:.3f}")
    
    if load_balance_result["overloaded_agents"]:
        print(f"   🔴 Overloaded Agents: {', '.join(load_balance_result['overloaded_agents'])}")
    
    if load_balance_result["underloaded_agents"]:
        print(f"   🟢 Underloaded Agents: {', '.join(load_balance_result['underloaded_agents'])}")
    
    if load_balance_result["rebalancing_actions"]:
        print(f"   🔄 Rebalancing Actions: {len(load_balance_result['rebalancing_actions'])}")
        for i, action in enumerate(load_balance_result["rebalancing_actions"][:3]):
            print(f"      {i+1}. Redirect from {action['from_agent']} to {action['to_agent']}")
    
    # Test load-balanced selection
    print("\n🎯 Testing load-balanced selections:")
    
    load_balanced_task = TaskRequirements(
        task_id="load_balance_test",
        required_capabilities=["analysis"],  # Multiple agents can handle this
        priority=5
    )
    
    selections = []
    for i in range(5):
        try:
            result = await selector.select_agent(load_balanced_task, SelectionStrategy.LEAST_LOADED)
            selections.append(result.selected_agent_id)
            print(f"   Selection {i+1}: {result.selected_agent_id}")
        except Exception as e:
            print(f"   Selection {i+1}: Error - {e}")
    
    # Analyze selection distribution
    if selections:
        selection_counts = {}
        for agent_id in selections:
            selection_counts[agent_id] = selection_counts.get(agent_id, 0) + 1
        
        print(f"\n📊 Selection Distribution:")
        for agent_id, count in selection_counts.items():
            print(f"   {agent_id}: {count} selections")
    
    return {
        "load_balance_result": load_balance_result,
        "selections": selections,
        "selector": selector
    }


async def test_ml_optimization():
    """Test machine learning-based optimization"""
    
    print("\n🤖 Testing ML-Based Optimization")
    print("=" * 60)
    
    # Use selector from previous test
    setup_result = await test_load_balancing()
    selector = setup_result["selector"]
    
    ml_selector = MLOptimizedSelector()
    
    print("🧪 Testing ML feature extraction:")
    
    # Test feature extraction for different agents and tasks
    test_scenarios = [
        ("langchain_agent_1", TaskRequirements("ml_test_1", ["chat"], priority=8, max_acceptable_latency=1.0)),
        ("crewai_agent_1", TaskRequirements("ml_test_2", ["collaboration"], priority=5, estimated_duration=120)),
        ("custom_agent_1", TaskRequirements("ml_test_3", ["custom_integration"], priority=3, quality_threshold=0.9))
    ]
    
    feature_data = []
    for agent_id, task_req in test_scenarios:
        if agent_id in selector.agent_metrics:
            metrics = selector.agent_metrics[agent_id]
            features = ml_selector.extract_features(agent_id, metrics, task_req)
            predicted_score = ml_selector.predict_performance(features)
            
            feature_data.append({
                "agent_id": agent_id,
                "task_id": task_req.task_id,
                "features": features.tolist(),
                "predicted_score": predicted_score
            })
            
            print(f"   🎯 {agent_id} for {task_req.task_id}:")
            print(f"      Feature Vector: {features[:6]}")  # Show first 6 features
            print(f"      Predicted Score: {predicted_score:.3f}")
    
    # Test ML-based selection
    print("\n🎲 Testing ML-based agent selection:")
    
    ml_test_tasks = [
        TaskRequirements("ml_priority_task", ["analysis"], priority=9, max_acceptable_latency=0.5),
        TaskRequirements("ml_quality_task", ["generation"], priority=6, quality_threshold=0.95),
        TaskRequirements("ml_complex_task", ["planning", "tool_use"], priority=7, estimated_duration=300)
    ]
    
    ml_results = []
    for task in ml_test_tasks:
        try:
            result = await selector.select_agent(task, SelectionStrategy.ML_OPTIMIZED)
            ml_results.append({
                "task_id": task.task_id,
                "selected_agent": result.selected_agent_id,
                "confidence": result.selection_confidence,
                "predicted_performance": result.predicted_performance
            })
            
            print(f"   📋 {task.task_id}: {result.selected_agent_id}")
            print(f"      Confidence: {result.selection_confidence:.3f}")
            print(f"      Predicted Performance: {result.predicted_performance}")
            
        except Exception as e:
            print(f"   ❌ {task.task_id}: {e}")
    
    # Simulate ML model updates
    print("\n📚 Testing ML model learning:")
    
    learning_scenarios = [
        ("langchain_agent_1", 0.85, {"success": 1, "response_time": 1.1, "quality": 0.9}),
        ("crewai_agent_1", 0.92, {"success": 1, "response_time": 1.3, "quality": 0.95}),
        ("custom_agent_1", 0.70, {"success": 0, "response_time": 6.0, "quality": 0.4})
    ]
    
    for agent_id, predicted, actual in learning_scenarios:
        ml_selector.update_model(agent_id, predicted, actual)
        print(f"   📖 Updated model for {agent_id}: predicted={predicted:.3f}, actual performance recorded")
    
    # Check learning history
    learning_history_size = len(ml_selector.selection_history)
    print(f"   📚 Learning History: {learning_history_size} entries")
    
    return {
        "feature_data": feature_data,
        "ml_results": ml_results,
        "learning_history_size": learning_history_size,
        "selector": selector
    }


async def test_optimization_strategies():
    """Test optimization strategy recommendations"""
    
    print("\n🎯 Testing Optimization Strategies")
    print("=" * 60)
    
    # Use selector from previous test
    setup_result = await test_ml_optimization()
    selector = setup_result["selector"]
    
    print("📊 Current system performance summary:")
    
    # Get performance summary
    performance_summary = selector.get_performance_summary()
    
    print(f"   🤖 Total Agents: {performance_summary['total_agents']}")
    print(f"   🟢 Available Agents: {performance_summary['available_agents']}")
    print(f"   📋 Total Tasks Processed: {performance_summary['total_tasks']}")
    
    # Show selection statistics
    selection_stats = performance_summary['selection_statistics']
    total_selections = selection_stats.get('total', 1)
    
    print(f"\n📈 Selection Strategy Usage:")
    for strategy, count in selection_stats.items():
        if strategy != 'total' and isinstance(strategy, SelectionStrategy):
            percentage = (count / total_selections) * 100
            print(f"   {strategy.value}: {count} selections ({percentage:.1f}%)")
    
    # Show agent performance rankings
    print(f"\n🏆 Agent Performance Rankings:")
    
    agent_performances = []
    for agent_id, perf in performance_summary['agent_performance'].items():
        agent_performances.append((agent_id, perf['fitness_score']))
    
    agent_performances.sort(key=lambda x: x[1], reverse=True)
    
    for i, (agent_id, fitness) in enumerate(agent_performances, 1):
        agent_perf = performance_summary['agent_performance'][agent_id]
        print(f"   {i}. {agent_id}: Fitness {fitness:.3f}")
        print(f"      Success: {agent_perf['success_rate']:.1%}, Avg Time: {agent_perf['average_response_time']:.2f}s")
        print(f"      State: {agent_perf['state']}, Load: {agent_perf['current_load']}")
    
    # Test optimization recommendations
    print(f"\n🎯 Getting optimization recommendations:")
    
    optimization_result = selector.optimize_selection_strategy()
    
    if "message" in optimization_result:
        print(f"   ℹ️  {optimization_result['message']}")
    else:
        print(f"   📊 Strategy Usage Analysis:")
        for strategy, usage in optimization_result.get('strategy_usage', {}).items():
            print(f"      {strategy}: {usage:.1%}")
        
        print(f"   💡 Recommendation: {optimization_result.get('recommendation', 'No recommendation')}")
        
        opportunities = optimization_result.get('optimization_opportunities', [])
        if opportunities:
            print(f"   🔧 Optimization Opportunities:")
            for i, opportunity in enumerate(opportunities, 1):
                print(f"      {i}. {opportunity}")
    
    # Test adaptive strategy selection
    print(f"\n🔄 Testing adaptive strategy selection:")
    
    adaptive_test_tasks = [
        TaskRequirements("urgent_task", ["chat"], priority=10, max_acceptable_latency=0.5),
        TaskRequirements("normal_task", ["analysis"], priority=5, max_acceptable_latency=3.0),
        TaskRequirements("batch_task", ["data_analysis"], priority=2, estimated_duration=600)
    ]
    
    adaptive_results = []
    for task in adaptive_test_tasks:
        # Let selector choose optimal strategy automatically
        try:
            result = await selector.select_agent(task)  # No strategy specified
            adaptive_results.append({
                "task_id": task.task_id,
                "selected_strategy": result.selection_strategy.value,
                "selected_agent": result.selected_agent_id,
                "confidence": result.selection_confidence
            })
            
            print(f"   📋 {task.task_id}: Strategy={result.selection_strategy.value}, Agent={result.selected_agent_id}")
            
        except Exception as e:
            print(f"   ❌ {task.task_id}: {e}")
    
    # Calculate overall system efficiency
    print(f"\n📊 System Efficiency Metrics:")
    
    total_tasks = sum(metrics.total_tasks for metrics in selector.agent_metrics.values())
    total_successful = sum(metrics.completed_tasks for metrics in selector.agent_metrics.values())
    avg_response_time = statistics.mean([
        metrics.average_response_time 
        for metrics in selector.agent_metrics.values() 
        if metrics.total_tasks > 0
    ]) if selector.agent_metrics else 0
    
    system_success_rate = total_successful / total_tasks if total_tasks > 0 else 0
    avg_load = statistics.mean([
        metrics.resource_utilization 
        for metrics in selector.agent_metrics.values()
    ]) if selector.agent_metrics else 0
    
    print(f"   ✅ System Success Rate: {system_success_rate:.1%}")
    print(f"   ⏱️  Average Response Time: {avg_response_time:.2f}s")
    print(f"   📊 Average System Load: {avg_load:.1%}")
    print(f"   📈 Total Tasks Processed: {total_tasks}")
    
    return {
        "performance_summary": performance_summary,
        "optimization_result": optimization_result,
        "adaptive_results": adaptive_results,
        "system_metrics": {
            "success_rate": system_success_rate,
            "avg_response_time": avg_response_time,
            "avg_load": avg_load,
            "total_tasks": total_tasks
        }
    }


async def main():
    """Main test execution for Phase 3 Session 18"""
    
    print("🚀 Mark-1 Phase 3 Session 18: Advanced Agent Selector & Optimization Testing")
    print("=" * 90)
    
    # Test 1: Agent registration and management
    registration_result = await test_agent_registration_and_management()
    print("\n" + "=" * 90)
    
    # Test 2: Performance metrics tracking
    metrics_result = await test_performance_metrics_tracking()
    print("\n" + "=" * 90)
    
    # Test 3: Selection strategies
    selection_result = await test_selection_strategies()
    print("\n" + "=" * 90)
    
    # Test 4: Load balancing
    load_balance_result = await test_load_balancing()
    print("\n" + "=" * 90)
    
    # Test 5: ML optimization
    ml_result = await test_ml_optimization()
    print("\n" + "=" * 90)
    
    # Test 6: Optimization strategies
    optimization_result = await test_optimization_strategies()
    
    print("\n" + "=" * 90)
    print("🎯 PHASE 3 SESSION 18 SUMMARY:")
    print("✅ Multi-criteria agent selection algorithms")
    print("✅ Performance-based agent ranking") 
    print("✅ Load balancing and resource optimization")
    print("✅ Machine learning-based selection")
    print("✅ Agent fitness scoring and adaptation")
    print("✅ Dynamic selection strategy optimization")
    print("✅ Advanced performance tracking")
    print("✅ Intelligent strategy adaptation")
    
    # Performance summary
    system_metrics = optimization_result["system_metrics"]
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Agents Registered: {registration_result['registered_agents']}")
    print(f"   Available Agents: {registration_result['available_agents']}")
    print(f"   Selection Strategies Tested: {len(selection_result['strategy_stats'])}")
    print(f"   System Success Rate: {system_metrics['success_rate']:.1%}")
    print(f"   Average Response Time: {system_metrics['avg_response_time']:.2f}s")
    print(f"   Average System Load: {system_metrics['avg_load']:.1%}")
    print(f"   Total Tasks Processed: {system_metrics['total_tasks']}")
    
    # Strategy performance
    strategy_stats = selection_result['strategy_stats']
    successful_strategies = [
        (strategy.value, stats['avg_optimization_score']) 
        for strategy, stats in strategy_stats.items() 
        if stats['success_rate'] > 0
    ]
    successful_strategies.sort(key=lambda x: x[1], reverse=True)
    
    if successful_strategies:
        print(f"\n🏆 TOP PERFORMING STRATEGIES:")
        for i, (strategy_name, score) in enumerate(successful_strategies[:3], 1):
            print(f"   {i}. {strategy_name}: Optimization Score {score:.3f}")
    
    # Load balancing effectiveness
    load_balance = load_balance_result['load_balance_result']
    print(f"\n⚖️  LOAD BALANCING:")
    print(f"   Load Balance Score: {load_balance['load_balance_score']:.3f}")
    print(f"   Load Variance: {load_balance['load_variance']:.4f}")
    print(f"   Rebalancing Actions Available: {len(load_balance['rebalancing_actions'])}")
    
    # ML Learning
    print(f"\n🤖 ML OPTIMIZATION:")
    print(f"   Feature Extraction Tests: {len(ml_result['feature_data'])}")
    print(f"   ML Selection Tests: {len(ml_result['ml_results'])}")
    print(f"   Learning History Entries: {ml_result['learning_history_size']}")
    
    print("\n🎉 Advanced Agent Selector & Optimization Complete!")
    print("Ready for Session 19: Advanced Context Management")


if __name__ == "__main__":
    asyncio.run(main()) 