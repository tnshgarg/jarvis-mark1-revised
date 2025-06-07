"""
AutoGPT Integration for Mark-1 Agent Orchestrator

Session 15: AutoGPT & Autonomous Agent Integration
Provides comprehensive integration with AutoGPT and autonomous agent patterns:
- AutoGPT agent detection and adaptation
- Goal-oriented task management
- Autonomous behavior preservation
- Memory system integration
- Self-directing agent capabilities
"""

import ast
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict

from mark1.agents.integrations.base_integration import (
    BaseIntegration, BaseAgentAdapter, IntegrationType, AgentCapability,
    IntegratedAgent, IntegrationError
)
from mark1.agents.discovery import DiscoveredAgent


class AutonomyLevel(Enum):
    """Different levels of agent autonomy"""
    REACTIVE = "reactive"           # Responds to commands only
    PROACTIVE = "proactive"         # Takes initiative within bounds  
    AUTONOMOUS = "autonomous"       # Self-directing with goals
    FULLY_AUTONOMOUS = "fully_autonomous"  # Complete autonomy with self-improvement


class GoalType(Enum):
    """Types of autonomous goals"""
    TASK_COMPLETION = "task_completion"
    PROBLEM_SOLVING = "problem_solving"
    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"


@dataclass
class AutonomousGoal:
    """Represents an autonomous goal for an agent"""
    goal_id: str
    description: str
    goal_type: GoalType
    priority: int = 1
    status: str = "pending"
    progress: float = 0.0
    success_criteria: List[str] = field(default_factory=list)
    sub_goals: List[str] = field(default_factory=list)
    deadline: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MemorySystem:
    """Represents a memory system component"""
    memory_type: str  # episodic, semantic, working, long_term
    storage_backend: str  # vector_db, graph, file, memory
    retention_policy: str  # permanent, sliding_window, decay
    capacity: Optional[int] = None
    compression_strategy: Optional[str] = None


@dataclass
class AutoGPTAgentInfo:
    """Comprehensive information about an AutoGPT agent"""
    agent_name: str
    autonomy_level: AutonomyLevel
    goals: List[AutonomousGoal] = field(default_factory=list)
    memory_systems: List[MemorySystem] = field(default_factory=list)
    self_improvement: bool = False
    planning_strategy: str = "simple"
    decision_framework: str = "rule_based"
    execution_engine: str = "sequential"
    learning_capabilities: List[str] = field(default_factory=list)


class GoalDetector:
    """Detects goals and objectives in agent code"""
    
    def __init__(self):
        self.goal_patterns = [
            r'goals?\s*[=:]\s*\[(.*?)\]',
            r'objectives?\s*[=:]\s*\[(.*?)\]',
            r'tasks?\s*[=:]\s*\[(.*?)\]',
            r'add_goal\(["\']([^"\']+)["\']',
            r'set_goal\(["\']([^"\']+)["\']',
            r'create_goal\(["\']([^"\']+)["\']'
        ]
    
    def detect_goals(self, code: str) -> List[AutonomousGoal]:
        """Detect goals from agent code"""
        goals = []
        goal_id_counter = 1
        
        for pattern in self.goal_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                goal_text = match.group(1)
                
                # Handle list format
                if '[' in goal_text or '"' in goal_text:
                    # Parse as list
                    goal_items = re.findall(r'["\']([^"\']+)["\']', goal_text)
                    for item in goal_items:
                        goal = AutonomousGoal(
                            goal_id=f"goal_{goal_id_counter}",
                            description=item.strip(),
                            goal_type=self._classify_goal_type(item),
                            priority=goal_id_counter
                        )
                        goals.append(goal)
                        goal_id_counter += 1
                else:
                    # Single goal
                    goal = AutonomousGoal(
                        goal_id=f"goal_{goal_id_counter}",
                        description=goal_text.strip(),
                        goal_type=self._classify_goal_type(goal_text),
                        priority=goal_id_counter
                    )
                    goals.append(goal)
                    goal_id_counter += 1
        
        return goals
    
    def _classify_goal_type(self, goal_text: str) -> GoalType:
        """Classify the type of goal based on content"""
        goal_lower = goal_text.lower()
        
        if any(word in goal_lower for word in ['research', 'analyze', 'study', 'investigate']):
            return GoalType.EXPLORATION
        elif any(word in goal_lower for word in ['solve', 'fix', 'resolve', 'address']):
            return GoalType.PROBLEM_SOLVING
        elif any(word in goal_lower for word in ['optimize', 'improve', 'enhance', 'maximize']):
            return GoalType.OPTIMIZATION
        elif any(word in goal_lower for word in ['learn', 'understand', 'master', 'acquire']):
            return GoalType.LEARNING
        elif any(word in goal_lower for word in ['maintain', 'monitor', 'check', 'ensure']):
            return GoalType.MAINTENANCE
        else:
            return GoalType.TASK_COMPLETION


class MemorySystemAnalyzer:
    """Analyzes memory systems in agent code"""
    
    def __init__(self):
        self.memory_patterns = {
            'episodic': [r'episodic[_\s]*memory', r'experience[_\s]*store', r'episode[_\s]*buffer'],
            'semantic': [r'semantic[_\s]*memory', r'knowledge[_\s]*base', r'concept[_\s]*store'],
            'working': [r'working[_\s]*memory', r'active[_\s]*memory', r'short[_\s]*term'],
            'long_term': [r'long[_\s]*term[_\s]*memory', r'persistent[_\s]*store', r'permanent[_\s]*memory']
        }
        
        self.storage_patterns = {
            'vector_db': [r'vector', r'embedding', r'chroma', r'pinecone', r'faiss'],
            'graph': [r'graph', r'neo4j', r'networkx', r'knowledge[_\s]*graph'],
            'file': [r'file', r'disk', r'pickle', r'json', r'csv'],
            'memory': [r'dict', r'list', r'memory', r'ram']
        }
    
    def analyze_memory(self, code: str) -> List[MemorySystem]:
        """Analyze memory systems in code"""
        memory_systems = []
        
        # Detect memory system types
        for mem_type, patterns in self.memory_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    # Determine storage backend
                    storage_backend = self._detect_storage_backend(code)
                    retention_policy = self._detect_retention_policy(code)
                    
                    memory_system = MemorySystem(
                        memory_type=mem_type,
                        storage_backend=storage_backend,
                        retention_policy=retention_policy,
                        capacity=self._extract_capacity(code),
                        compression_strategy=self._detect_compression(code)
                    )
                    memory_systems.append(memory_system)
                    break
        
        return memory_systems
    
    def _detect_storage_backend(self, code: str) -> str:
        """Detect storage backend type"""
        for backend, patterns in self.storage_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return backend
        return "memory"  # default
    
    def _detect_retention_policy(self, code: str) -> str:
        """Detect memory retention policy"""
        if re.search(r'sliding[_\s]*window|fifo|lru', code, re.IGNORECASE):
            return "sliding_window"
        elif re.search(r'permanent|persistent|forever', code, re.IGNORECASE):
            return "permanent"
        elif re.search(r'decay|expire|ttl', code, re.IGNORECASE):
            return "decay"
        else:
            return "sliding_window"  # default
    
    def _extract_capacity(self, code: str) -> Optional[int]:
        """Extract memory capacity if specified"""
        capacity_match = re.search(r'capacity[_\s]*[=:]\s*(\d+)', code, re.IGNORECASE)
        if capacity_match:
            return int(capacity_match.group(1))
        
        max_match = re.search(r'max[_\s]*memories?[_\s]*[=:]\s*(\d+)', code, re.IGNORECASE)
        if max_match:
            return int(max_match.group(1))
        
        return None
    
    def _detect_compression(self, code: str) -> Optional[str]:
        """Detect compression strategy"""
        if re.search(r'summary|summarize|compress', code, re.IGNORECASE):
            return "summary"
        elif re.search(r'embed|vector|encode', code, re.IGNORECASE):
            return "embedding"
        return None


class AutonomousGoalManager:
    """Manages autonomous goals for agents"""
    
    def __init__(self):
        self.goals: Dict[str, AutonomousGoal] = {}
        self.goal_hierarchy: Dict[str, List[str]] = {}
        self.active_goals: Set[str] = set()
    
    async def add_goal(self, goal: AutonomousGoal):
        """Add a goal to the manager"""
        self.goals[goal.goal_id] = goal
        if goal.status == "active":
            self.active_goals.add(goal.goal_id)
    
    async def decompose_goal(self, goal: AutonomousGoal) -> List[AutonomousGoal]:
        """Decompose a complex goal into sub-goals"""
        sub_goals = []
        
        # Simple goal decomposition logic
        if goal.goal_type == GoalType.PROBLEM_SOLVING:
            sub_goals = [
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_analysis",
                    description=f"Analyze the problem: {goal.description}",
                    goal_type=GoalType.EXPLORATION,
                    priority=goal.priority + 0.1
                ),
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_solution",
                    description=f"Develop solution for: {goal.description}",
                    goal_type=GoalType.TASK_COMPLETION,
                    priority=goal.priority + 0.2
                ),
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_validation",
                    description=f"Validate solution for: {goal.description}",
                    goal_type=GoalType.TASK_COMPLETION,
                    priority=goal.priority + 0.3
                )
            ]
        elif goal.goal_type == GoalType.EXPLORATION:
            sub_goals = [
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_research",
                    description=f"Research phase: {goal.description}",
                    goal_type=GoalType.TASK_COMPLETION,
                    priority=goal.priority + 0.1
                ),
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_synthesis",
                    description=f"Synthesize findings: {goal.description}",
                    goal_type=GoalType.TASK_COMPLETION,
                    priority=goal.priority + 0.2
                )
            ]
        else:
            # Generic decomposition
            sub_goals = [
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_step1",
                    description=f"Prepare for: {goal.description}",
                    goal_type=GoalType.TASK_COMPLETION,
                    priority=goal.priority + 0.1
                ),
                AutonomousGoal(
                    goal_id=f"{goal.goal_id}_step2",
                    description=f"Execute: {goal.description}",
                    goal_type=GoalType.TASK_COMPLETION,
                    priority=goal.priority + 0.2
                )
            ]
        
        # Add sub-goals to manager
        for sub_goal in sub_goals:
            await self.add_goal(sub_goal)
        
        # Update hierarchy
        self.goal_hierarchy[goal.goal_id] = [sg.goal_id for sg in sub_goals]
        goal.sub_goals = [sg.goal_id for sg in sub_goals]
        
        return sub_goals
    
    async def get_active_goals(self) -> List[AutonomousGoal]:
        """Get currently active goals"""
        return [self.goals[goal_id] for goal_id in self.active_goals if goal_id in self.goals]
    
    async def update_goal_status(self, goal_id: str, status: str):
        """Update goal status"""
        if goal_id in self.goals:
            self.goals[goal_id].status = status
            if status == "completed":
                self.active_goals.discard(goal_id)
            elif status == "active":
                self.active_goals.add(goal_id)


class MemoryManager:
    """Manages memory systems for autonomous agents"""
    
    def __init__(self, memory_systems: List[MemorySystem]):
        self.memory_systems = {ms.memory_type: ms for ms in memory_systems}
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Any] = {}
    
    async def store_experience(self, experience: Dict[str, Any], outcome: str, plan: Dict[str, Any]):
        """Store an experience in episodic memory"""
        memory_record = {
            "experience": experience,
            "outcome": outcome,
            "plan": plan,
            "timestamp": asyncio.get_event_loop().time(),
            "quality_score": experience.get("quality", 0.5)
        }
        
        self.episodic_memory.append(memory_record)
        
        # Limit memory size
        if len(self.episodic_memory) > 1000:
            self.episodic_memory.pop(0)
    
    async def retrieve_relevant(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on context"""
        relevant_memories = []
        
        # Simple relevance scoring
        context_words = set(str(context).lower().split())
        
        for memory in self.episodic_memory:
            memory_words = set(str(memory["experience"]).lower().split())
            overlap = len(context_words.intersection(memory_words))
            
            if overlap > 0:
                memory["relevance"] = overlap
                relevant_memories.append(memory)
        
        # Sort by relevance and return top memories
        relevant_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return relevant_memories[:5]
    
    async def consolidate_memories(self):
        """Consolidate episodic memories into semantic memory"""
        # Simple consolidation: extract patterns from experiences
        task_patterns = defaultdict(list)
        
        for memory in self.episodic_memory:
            task_type = memory["experience"].get("task", "unknown")
            outcome = memory["outcome"]
            quality = memory.get("quality_score", 0.5)
            
            task_patterns[task_type].append({
                "outcome": outcome,
                "quality": quality,
                "plan": memory["plan"]
            })
        
        # Update semantic memory with patterns
        for task_type, experiences in task_patterns.items():
            if len(experiences) >= 3:  # Need multiple experiences for pattern
                avg_quality = sum(exp["quality"] for exp in experiences) / len(experiences)
                successful_rate = sum(1 for exp in experiences if exp["outcome"] == "Success") / len(experiences)
                
                self.semantic_memory[task_type] = {
                    "average_quality": avg_quality,
                    "success_rate": successful_rate,
                    "total_experiences": len(experiences),
                    "best_practices": [exp["plan"] for exp in experiences if exp["quality"] > 0.8]
                }


class AutoGPTAgentAdapter(BaseAgentAdapter):
    """
    Adapter for AutoGPT agents that preserves autonomous behavior
    
    Maintains:
    - Goal-oriented execution
    - Memory-based learning
    - Self-directing behavior
    - Adaptive planning
    """
    
    def __init__(self, agent: Any, metadata: Dict[str, Any]):
        super().__init__(agent, metadata)
        self.autogpt_info = metadata.get('autogpt_info')
        self.autonomy_level = metadata.get('autonomy_level', AutonomyLevel.REACTIVE)
        
        # Initialize managers if needed
        if self.autogpt_info:
            self.goal_manager = AutonomousGoalManager()
            self.memory_manager = MemoryManager(self.autogpt_info.memory_systems)
        else:
            self.goal_manager = None
            self.memory_manager = None
    
    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agent with autonomous behavior preservation"""
        try:
            # Handle autonomous execution
            if self.autonomy_level in [AutonomyLevel.AUTONOMOUS, AutonomyLevel.FULLY_AUTONOMOUS]:
                return await self._autonomous_invoke(input_data)
            else:
                return await self._standard_invoke(input_data)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_type": "autogpt",
                "autonomy_level": self.autonomy_level.value if hasattr(self.autonomy_level, 'value') else str(self.autonomy_level)
            }
    
    async def _autonomous_invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle autonomous agent invocation"""
        task = input_data.get("task", input_data.get("input", ""))
        
        # Process goals if provided
        goals_processed = 0
        if "goals" in input_data and self.goal_manager:
            for goal_data in input_data["goals"]:
                goal = AutonomousGoal(
                    goal_id=goal_data.get("id", f"goal_{goals_processed}"),
                    description=goal_data.get("description", ""),
                    goal_type=GoalType(goal_data.get("type", "task_completion")),
                    priority=goal_data.get("priority", 1)
                )
                await self.goal_manager.add_goal(goal)
                goals_processed += 1
        
        # Execute with agent
        if hasattr(self.agent, 'run'):
            result = await self.agent.run(input_data)
        else:
            result = str(input_data)  # Mock execution
        
        # Store experience in memory
        memory_updated = False
        if self.memory_manager:
            experience = {
                "task": task,
                "input": input_data,
                "output": result,
                "quality": 0.8  # Mock quality score
            }
            plan = {"strategy": "autonomous_execution", "steps": ["analyze", "execute", "validate"]}
            await self.memory_manager.store_experience(experience, "Success", plan)
            memory_updated = True
        
        return {
            "success": True,
            "result": result,
            "agent_type": "autogpt",
            "autonomy_level": self.autonomy_level.value if hasattr(self.autonomy_level, 'value') else str(self.autonomy_level),
            "goals_processed": goals_processed,
            "memory_updated": memory_updated,
            "autonomous_execution": True
        }
    
    async def _standard_invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard agent invocation"""
        if hasattr(self.agent, 'run'):
            result = await self.agent.run(input_data)
        else:
            result = str(input_data)  # Mock execution
        
        return {
            "success": True,
            "result": result,
            "agent_type": "autogpt",
            "autonomy_level": self.autonomy_level.value if hasattr(self.autonomy_level, 'value') else str(self.autonomy_level),
            "autonomous_execution": False
        }
    
    async def stream(self, input_data: Dict[str, Any]):
        """Stream agent responses"""
        result = await self.invoke(input_data)
        
        # Simulate streaming by chunking the response
        chunks = [
            {"status": "processing", "step": "goal_analysis"},
            {"status": "processing", "step": "memory_retrieval"},
            {"status": "processing", "step": "execution"},
            {"status": "complete", "result": result, "final": True}
        ]
        
        for chunk in chunks:
            yield chunk
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        capabilities = ["autonomous_execution", "goal_management"]
        
        if self.autogpt_info:
            if self.autogpt_info.memory_systems:
                capabilities.append("memory_management")
            if self.autogpt_info.self_improvement:
                capabilities.append("self_improvement")
            capabilities.append(f"planning_{self.autogpt_info.planning_strategy}")
            capabilities.append(f"decision_{self.autogpt_info.decision_framework}")
        
        return capabilities
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to this agent"""
        tools = []
        
        # Get tools from metadata
        metadata_tools = self.metadata.get("tools", [])
        for tool in metadata_tools:
            if isinstance(tool, dict):
                tools.append(tool)
            else:
                tools.append({
                    "name": str(tool),
                    "type": "function",
                    "description": f"AutoGPT tool: {tool}"
                })
        
        # Add default autonomous agent tools
        default_tools = [
            {
                "name": "goal_manager",
                "type": "system",
                "description": "Manages autonomous goals and objectives"
            },
            {
                "name": "memory_manager", 
                "type": "system",
                "description": "Manages episodic and semantic memory"
            },
            {
                "name": "planning_engine",
                "type": "system", 
                "description": "Creates and executes autonomous plans"
            }
        ]
        
        tools.extend(default_tools)
        return tools
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model used by this agent"""
        model_info = {
            "framework": "autogpt",
            "agent_type": "autonomous",
            "autonomy_level": self.autonomy_level.value if hasattr(self.autonomy_level, 'value') else str(self.autonomy_level)
        }
        
        if self.autogpt_info:
            model_info.update({
                "planning_strategy": self.autogpt_info.planning_strategy,
                "decision_framework": self.autogpt_info.decision_framework,
                "self_improvement": self.autogpt_info.self_improvement,
                "memory_systems": len(self.autogpt_info.memory_systems),
                "active_goals": len(self.autogpt_info.goals)
            })
        
        return model_info
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and responsive"""
        try:
            # Test basic agent responsiveness
            if hasattr(self.agent, 'run'):
                test_result = await self.agent.run({"test": "health_check"})
                return test_result is not None
            else:
                # Agent exists and has basic attributes
                return hasattr(self.agent, 'name') or hasattr(self.agent, 'metadata')
        except Exception as e:
            return False
    
    def get_autonomy_info(self) -> Dict[str, Any]:
        """Get autonomy-specific information"""
        info = {
            "autonomy_level": self.autonomy_level.value if hasattr(self.autonomy_level, 'value') else str(self.autonomy_level),
            "autonomous_agent": True
        }
        
        if self.autogpt_info:
            info.update({
                "active_goals": len(self.autogpt_info.goals),
                "memory_systems": len(self.autogpt_info.memory_systems),
                "self_improvement": self.autogpt_info.self_improvement,
                "planning_strategy": self.autogpt_info.planning_strategy,
                "decision_framework": self.autogpt_info.decision_framework
            })
        
        return info


class AutoGPTIntegration(BaseIntegration):
    """
    Integration for AutoGPT and autonomous agent frameworks
    
    Handles:
    - Autonomous agent detection
    - Goal-oriented behavior preservation
    - Memory system integration
    - Self-directing capability adaptation
    """
    
    def __init__(self):
        super().__init__()
        self.logger = structlog.get_logger(__name__)
        self.goal_detector = GoalDetector()
        self.memory_analyzer = MemorySystemAnalyzer()
        
        # AutoGPT-specific patterns
        self.autogpt_patterns = [
            r'class.*AutoGPT.*Agent',
            r'from\s+autogpt',
            r'import.*autogpt',
            r'autonomous.*agent',
            r'goal.*driven',
            r'self.*directing',
            r'memory.*system',
            r'planning.*engine'
        ]
        
        # Autonomy indicators
        self.autonomy_patterns = {
            AutonomyLevel.REACTIVE: [
                r'respond_to_command', r'execute_command', r'reactive'
            ],
            AutonomyLevel.PROACTIVE: [
                r'take_initiative', r'proactive', r'anticipatory', r'predictive'
            ],
            AutonomyLevel.AUTONOMOUS: [
                r'autonomous', r'goal_driven', r'self_directing', r'independent'
            ],
            AutonomyLevel.FULLY_AUTONOMOUS: [
                r'fully_autonomous', r'complete_autonomy', r'self_managing', r'self_improvement'
            ]
        }
    
    def get_integration_type(self) -> IntegrationType:
        """Return the integration type"""
        return IntegrationType.AUTOGPT
    
    def _detect_framework_markers(self, code: str) -> bool:
        """Detect AutoGPT framework markers in code"""
        for pattern in self.autogpt_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False
    
    def extract_capabilities(self, code: str) -> List[AgentCapability]:
        """Extract capabilities from AutoGPT agent code"""
        capabilities = []
        
        # Basic capabilities based on patterns
        capability_patterns = {
            AgentCapability.PLANNING: [r'plan', r'strategy', r'schedule'],
            AgentCapability.REASONING: [r'reason', r'logic', r'decide', r'think'],
            AgentCapability.LEARNING: [r'learn', r'adapt', r'improve', r'train'],
            AgentCapability.MEMORY: [r'memory', r'remember', r'recall', r'store'],
            AgentCapability.TOOL_USE: [r'tool', r'function', r'execute', r'call'],
            AgentCapability.COMMUNICATION: [r'communicate', r'message', r'chat', r'speak']
        }
        
        for capability, patterns in capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    capabilities.append(capability)
                    break
        
        return capabilities
    
    def extract_tools(self, code: str) -> List[Dict[str, Any]]:
        """Extract tools from AutoGPT agent code"""
        tools = []
        
        # Tool detection patterns
        tool_patterns = [
            r'def\s+(\w+_tool)\s*\(',
            r'class\s+(\w+Tool)',
            r'tools\s*[=:]\s*\[(.*?)\]',
            r'functions\s*[=:]\s*\[(.*?)\]'
        ]
        
        for pattern in tool_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                tool_name = match.group(1)
                tools.append({
                    "name": tool_name,
                    "type": "function",
                    "description": f"AutoGPT tool: {tool_name}",
                    "capabilities": []
                })
        
        return tools
    
    def _determine_autonomy_level(self, code: str) -> AutonomyLevel:
        """Determine the autonomy level of an agent"""
        # Check in order of increasing autonomy
        for level in [AutonomyLevel.FULLY_AUTONOMOUS, AutonomyLevel.AUTONOMOUS, 
                      AutonomyLevel.PROACTIVE, AutonomyLevel.REACTIVE]:
            patterns = self.autonomy_patterns[level]
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return level
        
        return AutonomyLevel.REACTIVE  # default
    
    async def detect_agents(self, scan_path: Path) -> List[DiscoveredAgent]:
        """Detect AutoGPT and autonomous agents"""
        discovered_agents = []
        
        for py_file in scan_path.rglob("*.py"):
            try:
                agent_info = await self._analyze_agent_file(py_file)
                if agent_info:
                    discovered_agents.append(agent_info)
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze file {py_file}: {e}")
        
        return discovered_agents
    
    async def _analyze_agent_file(self, file_path: Path) -> Optional[DiscoveredAgent]:
        """Analyze a single agent file for AutoGPT patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Check if this is an AutoGPT agent
            if not self._detect_framework_markers(code):
                return None
            
            # Extract basic information
            capabilities = self.extract_capabilities(code)
            tools = self.extract_tools(code)
            
            # AutoGPT-specific analysis
            goals = self.goal_detector.detect_goals(code)
            memory_systems = self.memory_analyzer.analyze_memory(code)
            autonomy_level = self._determine_autonomy_level(code)
            
            # Check for self-improvement capabilities
            self_improvement = bool(re.search(r'self[_\s]*improve|continuous[_\s]*learning|adaptive', code, re.IGNORECASE))
            
            # Determine planning strategy
            planning_strategy = "simple"
            if re.search(r'hierarchical.*plan|multi.*level.*plan', code, re.IGNORECASE):
                planning_strategy = "hierarchical"
            elif re.search(r'adaptive.*plan|dynamic.*plan', code, re.IGNORECASE):
                planning_strategy = "adaptive"
            
            # Determine decision framework
            decision_framework = "rule_based"
            if re.search(r'utility.*based|cost.*benefit', code, re.IGNORECASE):
                decision_framework = "utility_based"
            elif re.search(r'neural.*network|machine.*learning|ai.*model', code, re.IGNORECASE):
                decision_framework = "ml_based"
            
            # Create AutoGPT agent info
            autogpt_info = AutoGPTAgentInfo(
                agent_name=file_path.stem,
                autonomy_level=autonomy_level,
                goals=goals,
                memory_systems=memory_systems,
                self_improvement=self_improvement,
                planning_strategy=planning_strategy,
                decision_framework=decision_framework
            )
            
            # Confidence scoring
            confidence = 0.5
            if len(goals) > 0:
                confidence += 0.1
            if len(memory_systems) > 0:
                confidence += 0.1
            if autonomy_level in [AutonomyLevel.AUTONOMOUS, AutonomyLevel.FULLY_AUTONOMOUS]:
                confidence += 0.2
            if self_improvement:
                confidence += 0.1
            
            return DiscoveredAgent(
                name=file_path.stem,
                file_path=file_path,
                framework="autogpt",
                confidence=min(confidence, 1.0),
                capabilities=[cap.value for cap in capabilities],
                metadata={
                    "autogpt_info": autogpt_info,
                    "autonomy_level": autonomy_level,
                    "goals": len(goals),
                    "memory_systems": len(memory_systems),
                    "self_improvement": self_improvement,
                    "tools": tools
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing AutoGPT agent {file_path}: {e}")
            return None
    
    async def integrate_agent(self, discovered_agent: DiscoveredAgent) -> IntegratedAgent:
        """Integrate an AutoGPT agent"""
        try:
            # Create mock agent instance for testing
            agent_instance = self._create_mock_agent(discovered_agent)
            
            # Create adapter
            adapter = self.create_adapter(agent_instance, discovered_agent.metadata)
            
            # Create integrated agent
            integrated_agent = IntegratedAgent(
                id=f"autogpt_{discovered_agent.name}_{hash(str(discovered_agent.file_path))}",
                name=discovered_agent.name,
                framework=IntegrationType.AUTOGPT,
                original_path=discovered_agent.file_path,
                adapter=adapter,
                capabilities=[AgentCapability(cap) for cap in discovered_agent.capabilities if cap in [c.value for c in AgentCapability]],
                metadata=discovered_agent.metadata,
                tools=discovered_agent.metadata.get("tools", []),
                model_info={"framework": "autogpt", "autonomy_level": discovered_agent.metadata.get("autonomy_level", "reactive")}
            )
            
            return integrated_agent
            
        except Exception as e:
            raise IntegrationError(f"AutoGPT integration failed: {str(e)}")
    
    def create_adapter(self, agent_instance: Any, metadata: Dict[str, Any]) -> AutoGPTAgentAdapter:
        """Create AutoGPT agent adapter"""
        return AutoGPTAgentAdapter(agent_instance, metadata)
    
    def _create_mock_agent(self, discovered_agent: DiscoveredAgent) -> Any:
        """Create a mock agent instance for testing"""
        class MockAutoGPTAgent:
            def __init__(self, metadata):
                self.metadata = metadata
                self.name = discovered_agent.name
            
            async def run(self, input_data):
                return f"Mock AutoGPT agent {self.name} executed with: {input_data}"
            
            async def execute(self, input_data):
                return await self.run(input_data)
        
        return MockAutoGPTAgent(discovered_agent.metadata)
