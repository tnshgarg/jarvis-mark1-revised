"""
Simple AutoGPT Agent - Basic Autonomous Patterns

Demonstrates fundamental AutoGPT capabilities:
- Basic goal management
- Simple memory system
- Iterative task execution
- Self-directed behavior
- Basic planning and execution
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class MemoryStore:
    """Simple memory storage for AutoGPT agent"""
    
    def __init__(self, max_memories: int = 100):
        self.max_memories = max_memories
        self.memories: List[Dict[str, Any]] = []
        self.working_memory: Dict[str, Any] = {}
    
    def add_memory(self, memory: Dict[str, Any]):
        """Add a memory to the store"""
        memory["timestamp"] = datetime.now().isoformat()
        memory["id"] = f"mem_{len(self.memories)}"
        
        self.memories.append(memory)
        
        # Keep memory within limits
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)
    
    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """Simple memory search"""
        query_words = set(query.lower().split())
        relevant_memories = []
        
        for memory in self.memories:
            memory_text = str(memory.get("content", "")).lower()
            memory_words = set(memory_text.split())
            
            # Simple overlap scoring
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0:
                memory["relevance"] = overlap
                relevant_memories.append(memory)
        
        # Return most relevant memories
        relevant_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return relevant_memories[:5]
    
    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get most recent memories"""
        return self.memories[-count:] if self.memories else []
    
    def update_working_memory(self, key: str, value: Any):
        """Update working memory"""
        self.working_memory[key] = value
    
    def get_working_memory(self, key: str) -> Any:
        """Get working memory item"""
        return self.working_memory.get(key)


class SimpleAutoGPTAgent:
    """
    Simple AutoGPT agent with basic autonomous capabilities
    
    Demonstrates:
    - Goal-driven behavior
    - Memory management
    - Iterative execution
    - Basic planning
    - Self-directed operation
    """
    
    def __init__(self, name: str = "SimpleAutoGPT"):
        self.name = name
        self.goals: List[str] = []
        self.memory = MemoryStore()
        self.current_task = None
        self.execution_history: List[Dict[str, Any]] = []
        self.resources = ["search", "analyze", "write", "plan", "execute"]
        self.constraints = [
            "Use only reliable sources",
            "Maintain accuracy",
            "Be efficient with resources",
            "Provide clear outputs"
        ]
        
        print(f"🤖 SimpleAutoGPT Agent '{name}' initialized")
        print(f"   Resources: {', '.join(self.resources)}")
        print(f"   Constraints: {len(self.constraints)} active")
    
    def add_goal(self, goal: str):
        """Add a goal for the agent"""
        self.goals.append(goal)
        
        # Store goal in memory
        self.memory.add_memory({
            "type": "goal",
            "content": goal,
            "status": "pending"
        })
        
        print(f"🎯 Goal added: {goal}")
    
    async def autonomous_execute(self, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Execute goals autonomously with iterative approach
        """
        print(f"🚀 Starting autonomous execution (max {max_iterations} iterations)")
        
        if not self.goals:
            return {
                "success": False,
                "message": "No goals to execute",
                "iterations": 0
            }
        
        execution_results = []
        
        for iteration in range(max_iterations):
            print(f"\n📍 Iteration {iteration + 1}/{max_iterations}")
            
            # Select next goal or task
            current_goal = self._select_next_goal()
            if not current_goal:
                print("✅ All goals completed")
                break
            
            self.current_task = current_goal
            print(f"🎯 Working on: {current_goal}")
            
            # Plan execution
            plan = await self._create_plan(current_goal)
            
            # Execute plan
            execution_result = await self._execute_plan(plan)
            
            # Store results
            execution_results.append({
                "iteration": iteration + 1,
                "goal": current_goal,
                "plan": plan,
                "result": execution_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store in memory
            self.memory.add_memory({
                "type": "execution",
                "goal": current_goal,
                "result": execution_result,
                "iteration": iteration + 1
            })
            
            # Self-assessment
            if await self._assess_completion(current_goal, execution_result):
                self._mark_goal_completed(current_goal)
                print(f"✅ Goal completed: {current_goal}")
            else:
                print(f"🔄 Goal needs more work: {current_goal}")
            
            # Brief pause for demonstration
            await asyncio.sleep(0.1)
        
        # Final summary
        completed_goals = [goal for goal in self.goals if self._is_goal_completed(goal)]
        
        summary = {
            "success": len(completed_goals) > 0,
            "total_goals": len(self.goals),
            "completed_goals": len(completed_goals),
            "iterations_used": len(execution_results),
            "execution_results": execution_results,
            "final_status": "autonomous_execution_complete"
        }
        
        print(f"\n🎉 Autonomous execution completed!")
        print(f"   Goals completed: {len(completed_goals)}/{len(self.goals)}")
        print(f"   Iterations used: {len(execution_results)}")
        
        return summary
    
    def _select_next_goal(self) -> Optional[str]:
        """Select the next goal to work on"""
        # Simple strategy: first incomplete goal
        for goal in self.goals:
            if not self._is_goal_completed(goal):
                return goal
        return None
    
    async def _create_plan(self, goal: str) -> Dict[str, Any]:
        """Create a plan for achieving the goal"""
        print(f"📋 Creating plan for: {goal}")
        
        # Check memory for similar goals
        relevant_memories = self.memory.search_memories(goal)
        
        # Basic planning logic
        steps = []
        
        if "research" in goal.lower():
            steps = [
                "Define research scope",
                "Gather information from reliable sources",
                "Analyze collected data",
                "Synthesize findings",
                "Create summary report"
            ]
        elif "analyze" in goal.lower():
            steps = [
                "Identify key components",
                "Break down into manageable parts",
                "Apply analytical frameworks",
                "Draw conclusions",
                "Present analysis results"
            ]
        elif "create" in goal.lower() or "write" in goal.lower():
            steps = [
                "Define requirements and objectives",
                "Gather necessary information",
                "Create outline or structure",
                "Develop content",
                "Review and refine output"
            ]
        else:
            # Generic plan
            steps = [
                "Understand the requirement",
                "Identify needed resources",
                "Execute core actions",
                "Validate results",
                "Complete task"
            ]
        
        plan = {
            "goal": goal,
            "strategy": "step_by_step",
            "steps": steps,
            "estimated_duration": len(steps) * 2,  # 2 minutes per step
            "resources_needed": self._identify_resources_for_goal(goal),
            "relevant_experience": len(relevant_memories),
            "constraints_considered": self.constraints
        }
        
        print(f"   Plan created: {len(steps)} steps")
        return plan
    
    def _identify_resources_for_goal(self, goal: str) -> List[str]:
        """Identify which resources are needed for a goal"""
        needed_resources = []
        goal_lower = goal.lower()
        
        if "research" in goal_lower or "find" in goal_lower:
            needed_resources.append("search")
        if "analyze" in goal_lower or "study" in goal_lower:
            needed_resources.append("analyze")
        if "create" in goal_lower or "write" in goal_lower:
            needed_resources.append("write")
        if "plan" in goal_lower:
            needed_resources.append("plan")
        
        # Always need execute
        needed_resources.append("execute")
        
        return needed_resources
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan"""
        print(f"⚡ Executing plan with {len(plan['steps'])} steps")
        
        step_results = []
        overall_success = True
        
        for i, step in enumerate(plan["steps"]):
            print(f"   Step {i+1}: {step}")
            
            # Simulate step execution
            await asyncio.sleep(0.05)  # Brief delay for simulation
            
            # Mock step execution
            step_result = {
                "step": step,
                "status": "completed",
                "output": f"Successfully executed: {step}",
                "quality_score": 0.8 + (i * 0.02),  # Slightly increasing quality
                "duration": 0.05
            }
            
            step_results.append(step_result)
            print(f"      ✅ Completed: {step}")
        
        execution_result = {
            "plan_id": plan.get("goal", "unknown"),
            "strategy": plan["strategy"],
            "steps_completed": len(step_results),
            "overall_success": overall_success,
            "step_results": step_results,
            "final_output": f"Plan execution completed for: {plan['goal']}",
            "quality_metrics": {
                "average_step_quality": sum(sr["quality_score"] for sr in step_results) / len(step_results),
                "completion_rate": 1.0,
                "efficiency": len(step_results) / plan.get("estimated_duration", len(step_results))
            }
        }
        
        print(f"   ✅ Plan execution completed")
        return execution_result
    
    async def _assess_completion(self, goal: str, execution_result: Dict[str, Any]) -> bool:
        """Assess if the goal has been completed satisfactorily"""
        # Simple completion assessment
        success = execution_result.get("overall_success", False)
        quality = execution_result.get("quality_metrics", {}).get("average_step_quality", 0)
        
        # Goal is complete if execution succeeded and quality is acceptable
        return success and quality >= 0.7
    
    def _mark_goal_completed(self, goal: str):
        """Mark a goal as completed"""
        self.memory.add_memory({
            "type": "goal_completion",
            "content": goal,
            "status": "completed",
            "completion_time": datetime.now().isoformat()
        })
        
        # Update working memory
        self.memory.update_working_memory("last_completed_goal", goal)
    
    def _is_goal_completed(self, goal: str) -> bool:
        """Check if a goal has been completed"""
        completion_memories = [
            m for m in self.memory.memories
            if m.get("type") == "goal_completion" and m.get("content") == goal
        ]
        return len(completion_memories) > 0
    
    async def run_task(self, task: str) -> Dict[str, Any]:
        """Run a single task (non-autonomous mode)"""
        print(f"🎯 Running single task: {task}")
        
        # Add as temporary goal
        self.add_goal(task)
        
        # Execute autonomously with single iteration
        result = await self.autonomous_execute(max_iterations=1)
        
        return {
            "task": task,
            "execution_mode": "single_task",
            "result": result,
            "success": result.get("success", False)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        completed_goals = [goal for goal in self.goals if self._is_goal_completed(goal)]
        
        return {
            "agent_name": self.name,
            "autonomous_agent": True,
            "goals": {
                "total": len(self.goals),
                "completed": len(completed_goals),
                "pending": len(self.goals) - len(completed_goals),
                "all_goals": self.goals
            },
            "memory": {
                "total_memories": len(self.memory.memories),
                "recent_memories": len(self.memory.get_recent_memories()),
                "working_memory_items": len(self.memory.working_memory)
            },
            "execution": {
                "current_task": self.current_task,
                "execution_history": len(self.execution_history)
            },
            "capabilities": {
                "resources": self.resources,
                "constraints": self.constraints,
                "autonomous_execution": True,
                "memory_management": True,
                "goal_oriented": True
            }
        }
    
    def show_memory_summary(self) -> Dict[str, Any]:
        """Show a summary of the agent's memory"""
        memory_types = {}
        for memory in self.memory.memories:
            mem_type = memory.get("type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        return {
            "total_memories": len(self.memory.memories),
            "memory_types": memory_types,
            "recent_memories": [
                {
                    "type": m.get("type"),
                    "content": str(m.get("content", ""))[:50] + "..." if len(str(m.get("content", ""))) > 50 else str(m.get("content", "")),
                    "timestamp": m.get("timestamp")
                }
                for m in self.memory.get_recent_memories(3)
            ],
            "working_memory": dict(self.memory.working_memory)
        }


# Example usage and testing
async def test_simple_autogpt():
    """Test the simple AutoGPT agent"""
    agent = SimpleAutoGPTAgent("TestAgent-01")
    
    # Add some goals
    test_goals = [
        "Research current trends in artificial intelligence",
        "Analyze the benefits of autonomous systems",
        "Create a summary of machine learning applications"
    ]
    
    print("🧪 Testing Simple AutoGPT Agent")
    print("=" * 50)
    
    # Add goals
    print("\n📝 Adding goals:")
    for goal in test_goals:
        agent.add_goal(goal)
    
    # Show initial status
    print(f"\n📊 Initial Status:")
    status = agent.get_status()
    print(f"   Goals: {status['goals']['total']} total")
    print(f"   Memories: {status['memory']['total_memories']}")
    
    # Run autonomous execution
    print(f"\n🚀 Starting autonomous execution:")
    execution_result = await agent.autonomous_execute(max_iterations=5)
    
    # Show results
    print(f"\n📈 Execution Results:")
    print(f"   Success: {execution_result['success']}")
    print(f"   Completed: {execution_result['completed_goals']}/{execution_result['total_goals']} goals")
    print(f"   Iterations: {execution_result['iterations_used']}")
    
    # Show final status
    print(f"\n📊 Final Status:")
    final_status = agent.get_status()
    print(f"   Completed Goals: {final_status['goals']['completed']}")
    print(f"   Pending Goals: {final_status['goals']['pending']}")
    print(f"   Total Memories: {final_status['memory']['total_memories']}")
    
    # Show memory summary
    print(f"\n🧠 Memory Summary:")
    memory_summary = agent.show_memory_summary()
    print(f"   Memory Types: {memory_summary['memory_types']}")
    print(f"   Recent Memories: {len(memory_summary['recent_memories'])}")
    
    # Test single task execution
    print(f"\n🎯 Testing Single Task Execution:")
    single_task_result = await agent.run_task("Plan a research project on robotics")
    print(f"   Task Success: {single_task_result['success']}")
    
    return {
        "agent_tested": True,
        "autonomous_execution": execution_result,
        "single_task_execution": single_task_result,
        "final_status": final_status,
        "memory_summary": memory_summary
    }


if __name__ == "__main__":
    asyncio.run(test_simple_autogpt()) 