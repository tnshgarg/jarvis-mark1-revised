"""
Autonomous Research Agent - AutoGPT Integration Test

This agent demonstrates advanced autonomous capabilities:
- Goal-oriented behavior
- Memory system integration
- Self-directing execution
- Adaptive planning
- Autonomous tool usage
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ResearchGoal:
    """Research-specific goal structure"""
    goal_id: str
    topic: str
    depth: str  # surface, detailed, comprehensive
    deadline: Optional[datetime]
    success_criteria: List[str]
    sub_goals: List[str] = None
    status: str = "pending"
    progress: float = 0.0
    
    def __post_init__(self):
        if self.sub_goals is None:
            self.sub_goals = []


class EpisodicMemory:
    """Agent's episodic memory system for experiences"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.experiences: List[Dict[str, Any]] = []
        self.retention_policy = "fifo"  # first in, first out
    
    def store_experience(self, experience: Dict[str, Any]):
        """Store a research experience"""
        experience["timestamp"] = datetime.now().isoformat()
        experience["experience_id"] = f"exp_{len(self.experiences)}"
        
        self.experiences.append(experience)
        
        # Maintain capacity
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)
    
    def retrieve_relevant(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a topic"""
        relevant = []
        topic_words = set(topic.lower().split())
        
        for exp in reversed(self.experiences):  # Most recent first
            exp_text = str(exp.get("topic", "")) + str(exp.get("findings", ""))
            exp_words = set(exp_text.lower().split())
            
            # Simple relevance scoring
            overlap = len(topic_words.intersection(exp_words))
            if overlap > 0:
                exp["relevance_score"] = overlap / len(topic_words)
                relevant.append(exp)
        
        # Sort by relevance and return top results
        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant[:limit]
    
    def get_success_patterns(self) -> Dict[str, Any]:
        """Analyze successful research patterns"""
        successful = [exp for exp in self.experiences if exp.get("success", False)]
        
        if not successful:
            return {"patterns": [], "strategies": []}
        
        strategies = [exp.get("strategy", "unknown") for exp in successful]
        topics = [exp.get("topic", "unknown") for exp in successful]
        
        return {
            "successful_strategies": list(set(strategies)),
            "successful_topics": list(set(topics)),
            "success_rate": len(successful) / len(self.experiences),
            "avg_quality": sum(exp.get("quality_score", 0) for exp in successful) / len(successful)
        }


class SemanticMemory:
    """Agent's semantic memory for learned knowledge"""
    
    def __init__(self):
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.concept_relationships: Dict[str, List[str]] = {}
        self.expertise_areas: Dict[str, float] = {}  # topic -> expertise level
    
    def store_knowledge(self, topic: str, knowledge: Dict[str, Any]):
        """Store semantic knowledge about a topic"""
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = {
                "facts": [],
                "concepts": [],
                "relationships": [],
                "confidence": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        
        # Update knowledge
        self.knowledge_base[topic].update(knowledge)
        self.knowledge_base[topic]["last_updated"] = datetime.now().isoformat()
        
        # Update expertise
        if topic not in self.expertise_areas:
            self.expertise_areas[topic] = 0.1
        else:
            self.expertise_areas[topic] = min(1.0, self.expertise_areas[topic] + 0.1)
    
    def get_knowledge(self, topic: str) -> Dict[str, Any]:
        """Retrieve knowledge about a topic"""
        return self.knowledge_base.get(topic, {})
    
    def find_related_topics(self, topic: str) -> List[str]:
        """Find topics related to the given topic"""
        related = self.concept_relationships.get(topic, [])
        
        # Add topics with similar keywords
        topic_words = set(topic.lower().split())
        for known_topic in self.knowledge_base.keys():
            known_words = set(known_topic.lower().split())
            if len(topic_words.intersection(known_words)) > 0 and known_topic not in related:
                related.append(known_topic)
        
        return related[:5]  # Limit to top 5


class GoalManager:
    """Manages autonomous agent goals"""
    
    def __init__(self):
        self.active_goals: List[ResearchGoal] = []
        self.completed_goals: List[ResearchGoal] = []
        self.goal_hierarchy: Dict[str, List[str]] = {}  # parent -> children
    
    def add_goal(self, goal: ResearchGoal):
        """Add a new research goal"""
        self.active_goals.append(goal)
        
        # Decompose complex goals
        if goal.depth == "comprehensive":
            sub_goals = self._decompose_goal(goal)
            goal.sub_goals = [sg.goal_id for sg in sub_goals]
            self.active_goals.extend(sub_goals)
    
    def _decompose_goal(self, goal: ResearchGoal) -> List[ResearchGoal]:
        """Decompose a complex goal into sub-goals"""
        sub_goals = []
        
        # Create sub-goals for comprehensive research
        phases = [
            ("overview", "Get overview and background"),
            ("detailed", "Conduct detailed research"),
            ("analysis", "Analyze findings"),
            ("synthesis", "Synthesize conclusions")
        ]
        
        for i, (phase, description) in enumerate(phases):
            sub_goal = ResearchGoal(
                goal_id=f"{goal.goal_id}_{phase}",
                topic=f"{goal.topic} - {description}",
                depth="detailed",
                deadline=goal.deadline,
                success_criteria=[f"Complete {phase} phase for {goal.topic}"]
            )
            sub_goals.append(sub_goal)
        
        return sub_goals
    
    def get_next_goal(self) -> Optional[ResearchGoal]:
        """Get the next goal to work on"""
        if not self.active_goals:
            return None
        
        # Sort by priority (deadline urgency)
        now = datetime.now()
        urgent_goals = [
            goal for goal in self.active_goals 
            if goal.deadline and goal.deadline < now + timedelta(hours=24)
        ]
        
        if urgent_goals:
            return min(urgent_goals, key=lambda g: g.deadline)
        
        # Return oldest pending goal
        pending = [g for g in self.active_goals if g.status == "pending"]
        return pending[0] if pending else None
    
    def update_progress(self, goal_id: str, progress: float):
        """Update goal progress"""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.progress = progress
                if progress >= 1.0:
                    goal.status = "completed"
                    self.completed_goals.append(goal)
                    self.active_goals.remove(goal)
                break
    
    def get_active_count(self) -> int:
        """Get number of active goals"""
        return len([g for g in self.active_goals if g.status in ["pending", "in_progress"]])


class ResearchPlanner:
    """Autonomous research planning system"""
    
    def __init__(self, memory: EpisodicMemory, knowledge: SemanticMemory):
        self.memory = memory
        self.knowledge = knowledge
        self.planning_strategies = {
            "surface": self._plan_surface_research,
            "detailed": self._plan_detailed_research,
            "comprehensive": self._plan_comprehensive_research
        }
    
    async def create_research_plan(self, goal: ResearchGoal) -> Dict[str, Any]:
        """Create an autonomous research plan"""
        strategy = self.planning_strategies.get(goal.depth, self._plan_detailed_research)
        
        # Check memory for relevant experiences
        relevant_experiences = self.memory.retrieve_relevant(goal.topic)
        
        # Get related knowledge
        related_topics = self.knowledge.find_related_topics(goal.topic)
        existing_knowledge = self.knowledge.get_knowledge(goal.topic)
        
        # Create adaptive plan
        plan = await strategy(goal, relevant_experiences, related_topics, existing_knowledge)
        
        return plan
    
    async def _plan_surface_research(self, goal, experiences, related, knowledge):
        """Plan surface-level research"""
        return {
            "strategy": "surface_research",
            "estimated_duration": "30 minutes",
            "phases": [
                {
                    "phase": "quick_overview",
                    "actions": ["search_overview", "collect_key_facts"],
                    "tools": ["web_search", "summarizer"],
                    "duration": "15 minutes"
                },
                {
                    "phase": "synthesis",
                    "actions": ["create_summary", "identify_key_points"],
                    "tools": ["analyzer", "writer"],
                    "duration": "15 minutes"
                }
            ],
            "success_criteria": goal.success_criteria,
            "adaptive_elements": {
                "use_experiences": len(experiences) > 0,
                "leverage_knowledge": bool(knowledge),
                "explore_related": len(related) > 0
            }
        }
    
    async def _plan_detailed_research(self, goal, experiences, related, knowledge):
        """Plan detailed research"""
        return {
            "strategy": "detailed_research",
            "estimated_duration": "2 hours",
            "phases": [
                {
                    "phase": "background_research",
                    "actions": ["gather_background", "understand_context"],
                    "tools": ["web_search", "academic_search", "analyzer"],
                    "duration": "45 minutes"
                },
                {
                    "phase": "deep_dive",
                    "actions": ["detailed_analysis", "expert_sources"],
                    "tools": ["specialist_search", "data_analyzer", "validator"],
                    "duration": "60 minutes"
                },
                {
                    "phase": "synthesis",
                    "actions": ["analyze_findings", "create_report"],
                    "tools": ["analyzer", "writer", "formatter"],
                    "duration": "35 minutes"
                }
            ],
            "success_criteria": goal.success_criteria,
            "adaptive_elements": {
                "build_on_experiences": experiences,
                "connect_knowledge": knowledge,
                "explore_relationships": related,
                "quality_threshold": 0.8
            }
        }
    
    async def _plan_comprehensive_research(self, goal, experiences, related, knowledge):
        """Plan comprehensive research"""
        return {
            "strategy": "comprehensive_research",
            "estimated_duration": "8 hours",
            "phases": [
                {
                    "phase": "landscape_analysis",
                    "actions": ["map_research_space", "identify_gaps"],
                    "tools": ["meta_search", "gap_analyzer", "mapper"],
                    "duration": "120 minutes"
                },
                {
                    "phase": "multi_source_research",
                    "actions": ["academic_research", "industry_research", "expert_interviews"],
                    "tools": ["academic_db", "industry_reports", "expert_network"],
                    "duration": "240 minutes"
                },
                {
                    "phase": "analysis_synthesis",
                    "actions": ["comparative_analysis", "trend_identification", "future_implications"],
                    "tools": ["advanced_analyzer", "trend_detector", "predictor"],
                    "duration": "120 minutes"
                },
                {
                    "phase": "comprehensive_report",
                    "actions": ["create_detailed_report", "validate_findings", "peer_review"],
                    "tools": ["report_generator", "validator", "reviewer"],
                    "duration": "120 minutes"
                }
            ],
            "success_criteria": goal.success_criteria + ["comprehensive_coverage", "validated_findings"],
            "adaptive_elements": {
                "iterative_refinement": True,
                "quality_gates": [0.7, 0.8, 0.9],
                "peer_validation": True,
                "meta_analysis": True
            }
        }


class AutonomousExecutionEngine:
    """Autonomous execution engine for research tasks"""
    
    def __init__(self, planner: ResearchPlanner):
        self.planner = planner
        self.execution_state = {
            "current_goal": None,
            "current_phase": None,
            "execution_trace": [],
            "adaptive_adjustments": []
        }
    
    async def autonomous_execute(self, goal: ResearchGoal) -> Dict[str, Any]:
        """Execute research goal autonomously"""
        self.execution_state["current_goal"] = goal
        self.execution_state["execution_trace"] = []
        
        try:
            # Create adaptive plan
            plan = await self.planner.create_research_plan(goal)
            
            # Execute phases autonomously
            results = []
            for phase in plan["phases"]:
                phase_result = await self._execute_phase(phase, plan)
                results.append(phase_result)
                
                # Adaptive adjustment based on results
                if phase_result.get("quality_score", 0) < 0.6:
                    adjustment = await self._adapt_execution(phase, phase_result)
                    self.execution_state["adaptive_adjustments"].append(adjustment)
            
            # Synthesize final result
            final_result = await self._synthesize_results(goal, plan, results)
            
            return {
                "goal_id": goal.goal_id,
                "success": True,
                "plan": plan,
                "results": results,
                "final_result": final_result,
                "execution_trace": self.execution_state["execution_trace"],
                "adaptive_adjustments": self.execution_state["adaptive_adjustments"],
                "autonomous_execution": True
            }
            
        except Exception as e:
            return {
                "goal_id": goal.goal_id,
                "success": False,
                "error": str(e),
                "execution_trace": self.execution_state["execution_trace"],
                "autonomous_execution": True
            }
    
    async def _execute_phase(self, phase: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single research phase"""
        self.execution_state["current_phase"] = phase["phase"]
        
        phase_start = datetime.now()
        
        # Simulate autonomous execution
        await asyncio.sleep(0.1)  # Simulated work
        
        # Mock phase execution results
        mock_results = {
            "quick_overview": {
                "findings": ["Key concept identified", "Main applications found", "Primary challenges noted"],
                "quality_score": 0.8,
                "completeness": 0.9
            },
            "background_research": {
                "findings": ["Historical context established", "Current state analyzed", "Key players identified"],
                "quality_score": 0.85,
                "completeness": 0.8
            },
            "deep_dive": {
                "findings": ["Technical details uncovered", "Expert insights gathered", "Data patterns identified"],
                "quality_score": 0.9,
                "completeness": 0.95
            },
            "synthesis": {
                "findings": ["Comprehensive analysis completed", "Key insights extracted", "Actionable recommendations created"],
                "quality_score": 0.88,
                "completeness": 0.92
            }
        }
        
        result = mock_results.get(phase["phase"], {
            "findings": [f"Completed {phase['phase']}"],
            "quality_score": 0.75,
            "completeness": 0.8
        })
        
        phase_duration = datetime.now() - phase_start
        
        result.update({
            "phase": phase["phase"],
            "actions_completed": phase["actions"],
            "tools_used": phase["tools"],
            "duration": str(phase_duration),
            "autonomous_decisions": self._record_autonomous_decisions(phase)
        })
        
        self.execution_state["execution_trace"].append({
            "phase": phase["phase"],
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "quality": result["quality_score"]
        })
        
        return result
    
    def _record_autonomous_decisions(self, phase: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Record autonomous decisions made during execution"""
        decisions = [
            {
                "decision_type": "tool_selection",
                "decision": f"Selected {len(phase['tools'])} tools for {phase['phase']}",
                "reasoning": "Based on phase requirements and available capabilities"
            },
            {
                "decision_type": "execution_order",
                "decision": f"Executed actions in optimal sequence for {phase['phase']}",
                "reasoning": "Dependency analysis and efficiency optimization"
            }
        ]
        
        return decisions
    
    async def _adapt_execution(self, phase: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Make adaptive adjustments based on execution results"""
        adjustment = {
            "trigger": f"Low quality score: {result.get('quality_score', 0)}",
            "adjustment_type": "quality_improvement",
            "actions": [
                "Allocated additional time for quality improvement",
                "Activated additional validation tools",
                "Implemented iterative refinement"
            ],
            "expected_improvement": 0.2
        }
        
        return adjustment
    
    async def _synthesize_results(self, goal: ResearchGoal, plan: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final research results"""
        all_findings = []
        for result in results:
            all_findings.extend(result.get("findings", []))
        
        avg_quality = sum(r.get("quality_score", 0) for r in results) / len(results)
        avg_completeness = sum(r.get("completeness", 0) for r in results) / len(results)
        
        synthesis = {
            "research_topic": goal.topic,
            "research_depth": goal.depth,
            "total_findings": len(all_findings),
            "key_insights": all_findings[:5],  # Top 5 findings
            "quality_metrics": {
                "average_quality": avg_quality,
                "average_completeness": avg_completeness,
                "overall_success": avg_quality > 0.7 and avg_completeness > 0.8
            },
            "research_summary": f"Completed {goal.depth} research on {goal.topic} with {len(all_findings)} key findings",
            "recommendations": [
                "Continue monitoring developments in this area",
                "Consider follow-up research on related topics",
                "Apply insights to relevant projects"
            ],
            "autonomous_insights": [
                "Research strategy was effective for this topic type",
                "Quality threshold was met through adaptive execution",
                "Future research could benefit from similar approach"
            ]
        }
        
        return synthesis


class AutonomousResearchAgent:
    """
    Complete Autonomous Research Agent demonstrating AutoGPT capabilities
    
    Features:
    - Goal-oriented autonomous behavior
    - Episodic and semantic memory systems
    - Adaptive planning and execution
    - Self-improvement through experience
    - Autonomous decision making
    """
    
    def __init__(self, agent_name: str = "AutonomousResearcher"):
        self.agent_name = agent_name
        self.autonomous_behavior = True
        self.self_improvement = True
        
        # Memory systems
        self.episodic_memory = EpisodicMemory(capacity=1000)
        self.semantic_memory = SemanticMemory()
        
        # Goal management
        self.goal_manager = GoalManager()
        
        # Planning and execution
        self.planner = ResearchPlanner(self.episodic_memory, self.semantic_memory)
        self.execution_engine = AutonomousExecutionEngine(self.planner)
        
        # Agent configuration
        self.config = {
            "autonomy_level": "fully_autonomous",
            "planning_strategy": "adaptive_hierarchical",
            "decision_framework": "utility_based",
            "self_improvement": True,
            "memory_integration": True,
            "goal_oriented": True
        }
        
        # Initialize with some default research capabilities
        self.constraints = [
            "Maintain high quality standards",
            "Respect time constraints",
            "Use reliable sources",
            "Provide actionable insights"
        ]
        
        self.resources = [
            "web_search", "academic_databases", "expert_networks",
            "analysis_tools", "validation_systems", "synthesis_engines"
        ]
        
        print(f"🤖 Autonomous Research Agent '{agent_name}' initialized")
        print(f"   Autonomy Level: {self.config['autonomy_level']}")
        print(f"   Self-Improvement: {self.config['self_improvement']}")
        print(f"   Memory Systems: Episodic + Semantic")
        print(f"   Resources: {len(self.resources)} available")
    
    async def autonomous_research(self, topic: str, depth: str = "detailed", deadline: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform autonomous research on a topic
        
        Args:
            topic: Research topic
            depth: Research depth (surface, detailed, comprehensive)
            deadline: Optional deadline (ISO format)
        
        Returns:
            Comprehensive research results
        """
        # Create research goal
        goal = ResearchGoal(
            goal_id=f"research_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            topic=topic,
            depth=depth,
            deadline=datetime.fromisoformat(deadline) if deadline else None,
            success_criteria=[
                f"Complete {depth} research on {topic}",
                "Achieve quality score > 0.8",
                "Provide actionable insights"
            ]
        )
        
        # Add goal to manager
        self.goal_manager.add_goal(goal)
        
        print(f"🎯 Starting autonomous research: {topic}")
        print(f"   Depth: {depth}")
        print(f"   Goal ID: {goal.goal_id}")
        
        try:
            # Execute autonomous research
            result = await self.execution_engine.autonomous_execute(goal)
            
            # Store experience in memory
            experience = {
                "goal": goal.goal_id,
                "topic": topic,
                "depth": depth,
                "strategy": result.get("plan", {}).get("strategy", "unknown"),
                "success": result["success"],
                "quality_score": result.get("final_result", {}).get("quality_metrics", {}).get("average_quality", 0),
                "findings": result.get("final_result", {}).get("total_findings", 0),
                "adaptive_adjustments": len(result.get("adaptive_adjustments", []))
            }
            
            self.episodic_memory.store_experience(experience)
            
            # Update semantic knowledge
            if result["success"]:
                knowledge = {
                    "research_completed": True,
                    "depth_achieved": depth,
                    "quality_score": experience["quality_score"],
                    "key_insights": result.get("final_result", {}).get("key_insights", [])
                }
                self.semantic_memory.store_knowledge(topic, knowledge)
            
            # Update goal progress
            self.goal_manager.update_progress(goal.goal_id, 1.0 if result["success"] else 0.5)
            
            print(f"✅ Research completed: {topic}")
            print(f"   Success: {result['success']}")
            print(f"   Quality: {experience['quality_score']:.2f}")
            print(f"   Findings: {experience['findings']}")
            
            return {
                "agent": self.agent_name,
                "autonomous_execution": True,
                "research_result": result,
                "experience_stored": True,
                "knowledge_updated": result["success"],
                "goal_completed": result["success"]
            }
            
        except Exception as e:
            print(f"❌ Research failed: {str(e)}")
            
            # Store failure experience
            failure_experience = {
                "goal": goal.goal_id,
                "topic": topic,
                "depth": depth,
                "success": False,
                "error": str(e),
                "quality_score": 0.0
            }
            
            self.episodic_memory.store_experience(failure_experience)
            
            return {
                "agent": self.agent_name,
                "autonomous_execution": True,
                "success": False,
                "error": str(e),
                "experience_stored": True
            }
    
    async def continuous_operation(self, duration_hours: float = 1.0):
        """
        Demonstrate continuous autonomous operation
        """
        print(f"🔄 Starting continuous autonomous operation for {duration_hours} hours")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        operation_count = 0
        
        while datetime.now() < end_time:
            # Check for pending goals
            next_goal = self.goal_manager.get_next_goal()
            
            if next_goal:
                print(f"🎯 Processing goal: {next_goal.topic}")
                await self.execution_engine.autonomous_execute(next_goal)
                operation_count += 1
            else:
                # Generate autonomous research goal based on memory
                topic = await self._generate_autonomous_goal()
                if topic:
                    await self.autonomous_research(topic, "surface")
                    operation_count += 1
            
            # Brief pause between operations
            await asyncio.sleep(10)  # 10 seconds in real implementation
        
        print(f"🏁 Continuous operation completed: {operation_count} autonomous operations")
        
        return {
            "operations_completed": operation_count,
            "duration": duration_hours,
            "autonomous_behavior": True,
            "memory_experiences": len(self.episodic_memory.experiences),
            "knowledge_topics": len(self.semantic_memory.knowledge_base)
        }
    
    async def _generate_autonomous_goal(self) -> Optional[str]:
        """Generate autonomous research goals based on memory and knowledge"""
        # Analyze memory for patterns
        success_patterns = self.episodic_memory.get_success_patterns()
        
        # Find knowledge gaps or areas for expansion
        expertise_areas = self.semantic_memory.expertise_areas
        
        if not expertise_areas:
            # Start with a general topic
            return "artificial intelligence fundamentals"
        
        # Find areas with low expertise for improvement
        low_expertise = [topic for topic, level in expertise_areas.items() if level < 0.5]
        
        if low_expertise:
            return f"advanced {low_expertise[0]}"
        
        # Generate related topic
        topics = list(expertise_areas.keys())
        if topics:
            base_topic = topics[0]
            related = self.semantic_memory.find_related_topics(base_topic)
            return related[0] if related else f"applications of {base_topic}"
        
        return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_name": self.agent_name,
            "autonomous_behavior": self.autonomous_behavior,
            "self_improvement": self.self_improvement,
            "config": self.config,
            "memory_status": {
                "episodic_experiences": len(self.episodic_memory.experiences),
                "semantic_knowledge": len(self.semantic_memory.knowledge_base),
                "expertise_areas": len(self.semantic_memory.expertise_areas)
            },
            "goal_status": {
                "active_goals": self.goal_manager.get_active_count(),
                "completed_goals": len(self.goal_manager.completed_goals)
            },
            "constraints": self.constraints,
            "resources": self.resources,
            "success_patterns": self.episodic_memory.get_success_patterns()
        }


# Example usage and testing
async def test_autonomous_agent():
    """Test the autonomous research agent"""
    agent = AutonomousResearchAgent("AdvancedResearcher-01")
    
    # Test autonomous research
    test_topics = [
        ("machine learning applications", "detailed"),
        ("quantum computing", "surface"),
        ("sustainable technology", "comprehensive")
    ]
    
    print("🧪 Testing Autonomous Research Agent")
    print("=" * 60)
    
    for topic, depth in test_topics:
        print(f"\n🔬 Research Task: {topic} ({depth})")
        result = await agent.autonomous_research(topic, depth)
        
        if result.get("success", False):
            print(f"✅ Success: Research completed autonomously")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Display agent status
    status = agent.get_agent_status()
    print(f"\n📊 Agent Status:")
    print(f"   Experiences: {status['memory_status']['episodic_experiences']}")
    print(f"   Knowledge Topics: {status['memory_status']['semantic_knowledge']}")
    print(f"   Active Goals: {status['goal_status']['active_goals']}")
    print(f"   Success Rate: {status['success_patterns'].get('success_rate', 0):.2%}")
    
    # Test continuous operation (brief demo)
    print(f"\n🔄 Testing Continuous Autonomous Operation")
    continuous_result = await agent.continuous_operation(0.01)  # 36 seconds demo
    print(f"   Operations: {continuous_result['operations_completed']}")


if __name__ == "__main__":
    asyncio.run(test_autonomous_agent()) 