"""
CrewAI Integration for Mark-1 Agent Orchestrator

Session 16: CrewAI & Multi-Agent Systems
Provides comprehensive integration with CrewAI and multi-agent collaborative frameworks:
- CrewAI agent detection and adaptation
- Role-based agent coordination
- Crew collaboration mechanisms
- Multi-agent workflow orchestration
- Inter-agent communication protocols
- Collaborative task delegation
"""

import ast
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict

from mark1.agents.integrations.base_integration import (
    BaseIntegration, BaseAgentAdapter, IntegrationType, AgentCapability,
    IntegratedAgent, IntegrationError
)
from mark1.agents.discovery import DiscoveredAgent


class CrewRole(Enum):
    """Different crew member roles"""
    LEADER = "leader"                    # Coordinates and delegates
    RESEARCHER = "researcher"            # Gathers information
    ANALYST = "analyst"                  # Analyzes data and patterns
    WRITER = "writer"                    # Creates content and documentation
    REVIEWER = "reviewer"                # Quality assurance and validation
    SPECIALIST = "specialist"            # Domain-specific expertise
    COORDINATOR = "coordinator"          # Inter-team coordination
    EXECUTOR = "executor"                # Task execution and implementation


class CollaborationPattern(Enum):
    """Collaboration patterns for multi-agent systems"""
    HIERARCHICAL = "hierarchical"       # Top-down delegation
    PEER_TO_PEER = "peer_to_peer"      # Equal collaboration
    PIPELINE = "pipeline"               # Sequential processing
    DEMOCRATIC = "democratic"           # Consensus-based decisions
    EXPERT_NETWORK = "expert_network"   # Expertise-based routing
    SWARM = "swarm"                     # Decentralized coordination


class TaskDelegationStrategy(Enum):
    """Task delegation strategies"""
    CAPABILITY_BASED = "capability_based"   # Based on agent capabilities
    WORKLOAD_BALANCED = "workload_balanced" # Based on current workload
    EXPERTISE_MATCHED = "expertise_matched" # Based on domain expertise
    RANDOM_ASSIGNMENT = "random_assignment" # Random distribution
    AUCTION_BASED = "auction_based"         # Agents bid for tasks
    PRIORITY_BASED = "priority_based"       # Based on task priority


@dataclass
class CrewMember:
    """Represents a crew member with role and capabilities"""
    agent_id: str
    role: CrewRole
    backstory: str
    goal: str
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    collaboration_style: str = "cooperative"
    communication_protocols: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CrewTask:
    """Represents a task for crew execution"""
    task_id: str
    description: str
    assigned_role: Optional[CrewRole] = None
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    priority: int = 1
    estimated_duration: Optional[str] = None
    collaboration_required: bool = False
    tools_required: List[str] = field(default_factory=list)
    status: str = "pending"
    progress: float = 0.0


@dataclass
class CrewConfiguration:
    """Configuration for a crew of agents"""
    crew_id: str
    crew_name: str
    members: List[CrewMember] = field(default_factory=list)
    collaboration_pattern: CollaborationPattern = CollaborationPattern.HIERARCHICAL
    delegation_strategy: TaskDelegationStrategy = TaskDelegationStrategy.CAPABILITY_BASED
    communication_channels: List[str] = field(default_factory=list)
    shared_memory: bool = True
    consensus_mechanism: str = "majority_vote"
    coordination_protocols: List[str] = field(default_factory=list)


class CrewAIAgentDetector:
    """Detects CrewAI agents and crew configurations"""
    
    def __init__(self):
        self.crewai_patterns = [
            r'from\s+crewai\s+import\s+Agent',
            r'from\s+crewai\s+import\s+Task',
            r'from\s+crewai\s+import\s+Crew',
            r'Agent\s*\(',
            r'Task\s*\(',
            r'Crew\s*\(',
            r'@agent',
            r'@task',
            r'@crew'
        ]
        
        self.role_patterns = {
            CrewRole.LEADER: [r'leader', r'manager', r'coordinator', r'director'],
            CrewRole.RESEARCHER: [r'researcher', r'investigator', r'analyst', r'data_collector'],
            CrewRole.ANALYST: [r'analyst', r'evaluator', r'assessor', r'examiner'],
            CrewRole.WRITER: [r'writer', r'author', r'content_creator', r'documenter'],
            CrewRole.REVIEWER: [r'reviewer', r'validator', r'quality_assurance', r'checker'],
            CrewRole.SPECIALIST: [r'specialist', r'expert', r'consultant', r'advisor'],
            CrewRole.COORDINATOR: [r'coordinator', r'facilitator', r'liaison', r'mediator'],
            CrewRole.EXECUTOR: [r'executor', r'implementer', r'operator', r'worker']
        }
        
        self.collaboration_indicators = {
            CollaborationPattern.HIERARCHICAL: [r'hierarchy', r'delegate', r'supervise', r'command'],
            CollaborationPattern.PEER_TO_PEER: [r'peer', r'equal', r'collaborate', r'partner'],
            CollaborationPattern.PIPELINE: [r'pipeline', r'sequential', r'workflow', r'chain'],
            CollaborationPattern.DEMOCRATIC: [r'democratic', r'vote', r'consensus', r'majority'],
            CollaborationPattern.EXPERT_NETWORK: [r'expert', r'specialist', r'domain', r'expertise'],
            CollaborationPattern.SWARM: [r'swarm', r'decentralized', r'distributed', r'autonomous']
        }
    
    def detect_crewai_patterns(self, code: str) -> bool:
        """Detect CrewAI framework patterns in code"""
        for pattern in self.crewai_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False
    
    def extract_crew_roles(self, code: str) -> List[CrewRole]:
        """Extract crew roles from code"""
        detected_roles = []
        
        for role, patterns in self.role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    if role not in detected_roles:
                        detected_roles.append(role)
                    break
        
        return detected_roles
    
    def detect_collaboration_pattern(self, code: str) -> CollaborationPattern:
        """Detect collaboration pattern from code"""
        for pattern, indicators in self.collaboration_indicators.items():
            for indicator in indicators:
                if re.search(indicator, code, re.IGNORECASE):
                    return pattern
        
        return CollaborationPattern.PEER_TO_PEER  # default


class RoleBasedTaskDelegator:
    """Handles role-based task delegation for crews"""
    
    def __init__(self, delegation_strategy: TaskDelegationStrategy = TaskDelegationStrategy.CAPABILITY_BASED):
        self.delegation_strategy = delegation_strategy
        self.role_capabilities = {
            CrewRole.LEADER: ["coordination", "decision_making", "planning", "delegation"],
            CrewRole.RESEARCHER: ["information_gathering", "data_collection", "analysis", "investigation"],
            CrewRole.ANALYST: ["data_analysis", "pattern_recognition", "evaluation", "assessment"],
            CrewRole.WRITER: ["content_creation", "documentation", "communication", "storytelling"],
            CrewRole.REVIEWER: ["quality_assurance", "validation", "verification", "testing"],
            CrewRole.SPECIALIST: ["domain_expertise", "consultation", "specialized_knowledge", "guidance"],
            CrewRole.COORDINATOR: ["coordination", "facilitation", "communication", "mediation"],
            CrewRole.EXECUTOR: ["implementation", "execution", "operation", "task_completion"]
        }
    
    async def delegate_task(self, task: CrewTask, crew: CrewConfiguration) -> Optional[CrewMember]:
        """Delegate a task to the most suitable crew member"""
        if not crew.members:
            return None
        
        if self.delegation_strategy == TaskDelegationStrategy.CAPABILITY_BASED:
            return await self._delegate_by_capability(task, crew)
        elif self.delegation_strategy == TaskDelegationStrategy.EXPERTISE_MATCHED:
            return await self._delegate_by_expertise(task, crew)
        elif self.delegation_strategy == TaskDelegationStrategy.WORKLOAD_BALANCED:
            return await self._delegate_by_workload(task, crew)
        else:
            # Default to capability-based
            return await self._delegate_by_capability(task, crew)
    
    async def _delegate_by_capability(self, task: CrewTask, crew: CrewConfiguration) -> Optional[CrewMember]:
        """Delegate based on capability matching"""
        task_requirements = self._extract_task_requirements(task.description)
        
        best_match = None
        best_score = 0
        
        for member in crew.members:
            role_capabilities = self.role_capabilities.get(member.role, [])
            member_capabilities = role_capabilities + member.capabilities
            
            # Calculate capability match score
            score = len(set(task_requirements).intersection(set(member_capabilities)))
            
            if score > best_score:
                best_score = score
                best_match = member
        
        return best_match
    
    async def _delegate_by_expertise(self, task: CrewTask, crew: CrewConfiguration) -> Optional[CrewMember]:
        """Delegate based on domain expertise"""
        task_domain = self._extract_task_domain(task.description)
        
        for member in crew.members:
            if task_domain in member.expertise_areas:
                return member
        
        # Fallback to capability-based
        return await self._delegate_by_capability(task, crew)
    
    async def _delegate_by_workload(self, task: CrewTask, crew: CrewConfiguration) -> Optional[CrewMember]:
        """Delegate based on current workload"""
        # For now, return member with lowest performance metric (indicating availability)
        available_members = [m for m in crew.members if m.performance_metrics.get("current_workload", 0) < 0.8]
        
        if available_members:
            return min(available_members, key=lambda m: m.performance_metrics.get("current_workload", 0))
        
        # Fallback to capability-based
        return await self._delegate_by_capability(task, crew)
    
    def _extract_task_requirements(self, description: str) -> List[str]:
        """Extract required capabilities from task description"""
        requirements = []
        desc_lower = description.lower()
        
        capability_keywords = {
            "research": ["information_gathering", "data_collection"],
            "analyze": ["data_analysis", "pattern_recognition"],
            "write": ["content_creation", "documentation"],
            "review": ["quality_assurance", "validation"],
            "coordinate": ["coordination", "facilitation"],
            "implement": ["implementation", "execution"],
            "plan": ["planning", "strategy"],
            "decide": ["decision_making", "evaluation"]
        }
        
        for keyword, caps in capability_keywords.items():
            if keyword in desc_lower:
                requirements.extend(caps)
        
        return list(set(requirements))
    
    def _extract_task_domain(self, description: str) -> str:
        """Extract domain from task description"""
        domains = ["ai", "machine_learning", "data_science", "software", "business", "research", "marketing"]
        desc_lower = description.lower()
        
        for domain in domains:
            if domain in desc_lower:
                return domain
        
        return "general"


class InterAgentCommunicator:
    """Handles communication between crew members"""
    
    def __init__(self):
        self.communication_channels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.message_queue: List[Dict[str, Any]] = []
        self.communication_protocols = {
            "direct_message": self._direct_message,
            "broadcast": self._broadcast_message,
            "role_specific": self._role_specific_message,
            "request_response": self._request_response
        }
    
    async def send_message(self, sender_id: str, receiver_id: Optional[str], message: Dict[str, Any], protocol: str = "direct_message") -> bool:
        """Send a message between agents"""
        try:
            handler = self.communication_protocols.get(protocol, self._direct_message)
            await handler(sender_id, receiver_id, message)
            return True
        except Exception as e:
            return False
    
    async def _direct_message(self, sender_id: str, receiver_id: str, message: Dict[str, Any]):
        """Send direct message to specific agent"""
        formatted_message = {
            "id": f"msg_{len(self.message_queue)}",
            "sender": sender_id,
            "receiver": receiver_id,
            "type": "direct",
            "content": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.message_queue.append(formatted_message)
        self.communication_channels[receiver_id].append(formatted_message)
    
    async def _broadcast_message(self, sender_id: str, receiver_id: Optional[str], message: Dict[str, Any]):
        """Broadcast message to all agents"""
        formatted_message = {
            "id": f"msg_{len(self.message_queue)}",
            "sender": sender_id,
            "receiver": "broadcast",
            "type": "broadcast",
            "content": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.message_queue.append(formatted_message)
        
        # Add to all channels
        for channel in self.communication_channels.keys():
            if channel != sender_id:  # Don't send to sender
                self.communication_channels[channel].append(formatted_message)
    
    async def _role_specific_message(self, sender_id: str, receiver_role: str, message: Dict[str, Any]):
        """Send message to agents with specific role"""
        formatted_message = {
            "id": f"msg_{len(self.message_queue)}",
            "sender": sender_id,
            "receiver": f"role:{receiver_role}",
            "type": "role_specific",
            "content": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.message_queue.append(formatted_message)
        if receiver_role in self.communication_channels:
            self.communication_channels[receiver_role].append(formatted_message)
    
    async def _request_response(self, sender_id: str, receiver_id: str, message: Dict[str, Any]):
        """Send request expecting response"""
        request_id = f"req_{len(self.message_queue)}"
        formatted_message = {
            "id": request_id,
            "sender": sender_id,
            "receiver": receiver_id,
            "type": "request",
            "content": message,
            "timestamp": asyncio.get_event_loop().time(),
            "expects_response": True
        }
        
        self.message_queue.append(formatted_message)
        self.communication_channels[receiver_id].append(formatted_message)
    
    async def get_messages(self, agent_id: str, unread_only: bool = True) -> List[Dict[str, Any]]:
        """Get messages for an agent"""
        messages = self.communication_channels[agent_id]
        
        if unread_only:
            # Return unread messages (not marked as read)
            return [msg for msg in messages if not msg.get("read", False)]
        
        return messages
    
    async def mark_message_read(self, agent_id: str, message_id: str):
        """Mark a message as read"""
        for message in self.communication_channels[agent_id]:
            if message["id"] == message_id:
                message["read"] = True
                break


class CollaborativeWorkflowEngine:
    """Manages collaborative workflows for crews"""
    
    def __init__(self):
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    async def create_collaborative_workflow(self, workflow_id: str, tasks: List[CrewTask], crew: CrewConfiguration) -> Dict[str, Any]:
        """Create a collaborative workflow"""
        workflow = {
            "id": workflow_id,
            "tasks": {task.task_id: task for task in tasks},
            "crew": crew,
            "status": "created",
            "progress": 0.0,
            "start_time": None,
            "end_time": None,
            "task_execution_order": [],
            "collaboration_points": [],
            "synchronization_points": []
        }
        
        # Analyze task dependencies
        await self._analyze_task_dependencies(workflow)
        
        # Identify collaboration points
        await self._identify_collaboration_points(workflow)
        
        # Create execution plan
        await self._create_execution_plan(workflow)
        
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    async def execute_collaborative_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a collaborative workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        workflow["status"] = "executing"
        workflow["start_time"] = asyncio.get_event_loop().time()
        
        try:
            # Execute tasks in dependency order
            for task_id in workflow["task_execution_order"]:
                task = workflow["tasks"][task_id]
                
                # Execute task
                task_result = await self._execute_collaborative_task(task, workflow)
                
                # Record execution
                self.execution_history.append({
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "result": task_result,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # Update workflow progress
                completed_tasks = sum(1 for t in workflow["tasks"].values() if t.status == "completed")
                workflow["progress"] = completed_tasks / len(workflow["tasks"])
            
            workflow["status"] = "completed"
            workflow["end_time"] = asyncio.get_event_loop().time()
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "tasks_completed": len(workflow["tasks"]),
                "execution_time": workflow["end_time"] - workflow["start_time"],
                "final_progress": workflow["progress"]
            }
            
        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }
    
    async def _analyze_task_dependencies(self, workflow: Dict[str, Any]):
        """Analyze dependencies between tasks"""
        tasks = workflow["tasks"]
        dependencies = {}
        
        for task_id, task in tasks.items():
            dependencies[task_id] = task.dependencies
        
        self.task_dependencies[workflow["id"]] = dependencies
        workflow["dependencies"] = dependencies
    
    async def _identify_collaboration_points(self, workflow: Dict[str, Any]):
        """Identify points where collaboration is required"""
        collaboration_points = []
        
        for task_id, task in workflow["tasks"].items():
            if task.collaboration_required:
                collaboration_points.append({
                    "task_id": task_id,
                    "type": "task_collaboration",
                    "description": f"Collaboration required for: {task.description}"
                })
            
            # Check for tasks that require output from multiple other tasks
            dependent_tasks = [tid for tid, deps in workflow.get("dependencies", {}).items() if task_id in deps]
            if len(dependent_tasks) > 1:
                collaboration_points.append({
                    "task_id": task_id,
                    "type": "output_synthesis",
                    "description": f"Synthesis of outputs from {len(dependent_tasks)} tasks"
                })
        
        workflow["collaboration_points"] = collaboration_points
    
    async def _create_execution_plan(self, workflow: Dict[str, Any]):
        """Create execution plan respecting dependencies"""
        tasks = workflow["tasks"]
        dependencies = workflow.get("dependencies", {})
        
        # Simple topological sort for execution order
        execution_order = []
        completed = set()
        
        while len(execution_order) < len(tasks):
            for task_id, task_deps in dependencies.items():
                if task_id not in completed and all(dep in completed for dep in task_deps):
                    execution_order.append(task_id)
                    completed.add(task_id)
                    break
        
        workflow["task_execution_order"] = execution_order
    
    async def _execute_collaborative_task(self, task: CrewTask, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single collaborative task"""
        # Simulate task execution
        await asyncio.sleep(0.1)  # Simulated work
        
        task.status = "completed"
        task.progress = 1.0
        
        # Mock execution result
        result = {
            "task_id": task.task_id,
            "status": "completed",
            "output": f"Completed collaborative task: {task.description}",
            "assigned_agent": task.assigned_agent,
            "collaboration_used": task.collaboration_required,
            "execution_time": 0.1
        }
        
        return result


class CrewAIAgentAdapter(BaseAgentAdapter):
    """
    Adapter for CrewAI agents that preserves collaborative behavior
    
    Maintains:
    - Role-based coordination
    - Crew collaboration patterns
    - Inter-agent communication
    - Collaborative task execution
    """
    
    def __init__(self, agent: Any, metadata: Dict[str, Any]):
        super().__init__(agent, metadata)
        self.crew_info = metadata.get('crew_info')
        self.role = metadata.get('role', CrewRole.EXECUTOR)
        self.crew_id = metadata.get('crew_id')
        
        # Initialize collaboration components
        self.delegator = RoleBasedTaskDelegator()
        self.communicator = InterAgentCommunicator()
        self.workflow_engine = CollaborativeWorkflowEngine()
    
    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agent with crew collaboration capabilities"""
        try:
            # Handle crew-based execution
            if self.crew_info and input_data.get("crew_task", False):
                return await self._crew_invoke(input_data)
            else:
                return await self._standard_invoke(input_data)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_type": "crewai",
                "role": self.role.value if hasattr(self.role, 'value') else str(self.role)
            }
    
    async def _crew_invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle crew-based task execution"""
        task_description = input_data.get("task", input_data.get("input", ""))
        crew_task_data = input_data.get("crew_task", {})
        
        # Create crew task
        crew_task = CrewTask(
            task_id=crew_task_data.get("id", f"task_{hash(task_description)}"),
            description=task_description,
            assigned_role=self.role,
            collaboration_required=crew_task_data.get("collaboration_required", False),
            tools_required=crew_task_data.get("tools", [])
        )
        
        # Check for collaboration requirements
        collaboration_used = False
        if crew_task.collaboration_required:
            # Simulate collaboration
            collaboration_result = await self._handle_collaboration(crew_task, input_data)
            collaboration_used = True
        
        # Execute with agent
        if hasattr(self.agent, 'run'):
            result = await self.agent_instance.run(input_data)
        else:
            result = f"CrewAI agent executed: {task_description}"  # Mock execution
        
        return {
            "success": True,
            "result": result,
            "agent_type": "crewai",
            "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
            "crew_id": self.crew_id,
            "task_id": crew_task.task_id,
            "collaboration_used": collaboration_used,
            "crew_execution": True
        }
    
    async def _standard_invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard agent invocation"""
        if hasattr(self.agent, 'run'):
            result = await self.agent_instance.run(input_data)
        else:
            result = str(input_data)  # Mock execution
        
        return {
            "success": True,
            "result": result,
            "agent_type": "crewai",
            "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
            "crew_execution": False
        }
    
    async def _handle_collaboration(self, task: CrewTask, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaborative aspects of task execution"""
        # Simulate collaboration process
        collaboration_result = {
            "collaboration_type": "peer_review",
            "participants": [self.role.value],
            "consensus_reached": True,
            "collaboration_quality": 0.85
        }
        
        # Send collaboration messages
        await self.communicator.send_message(
            sender_id=str(self.role.value),
            receiver_id="crew_broadcast",
            message={
                "type": "collaboration_request",
                "task": task.description,
                "input_data": input_data
            },
            protocol="broadcast"
        )
        
        return collaboration_result
    
    async def stream(self, input_data: Dict[str, Any]):
        """Stream agent responses with crew coordination"""
        result = await self.invoke(input_data)
        
        # Simulate streaming with crew coordination steps
        chunks = [
            {"status": "processing", "step": "role_assignment"},
            {"status": "processing", "step": "task_delegation"},
            {"status": "processing", "step": "crew_coordination"},
            {"status": "processing", "step": "collaborative_execution"},
            {"status": "complete", "result": result, "final": True}
        ]
        
        for chunk in chunks:
            yield chunk
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        capabilities = ["crew_collaboration", "role_based_execution"]
        
        if self.crew_info:
            if self.crew_info.shared_memory:
                capabilities.append("shared_memory")
            if self.crew_info.collaboration_pattern:
                capabilities.append(f"collaboration_{self.crew_info.collaboration_pattern.value}")
            capabilities.append(f"role_{self.role.value}")
        
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
                    "description": f"CrewAI tool: {tool}"
                })
        
        # Add default crew collaboration tools
        default_tools = [
            {
                "name": "task_delegator",
                "type": "system",
                "description": "Delegates tasks to appropriate crew members"
            },
            {
                "name": "crew_communicator",
                "type": "system",
                "description": "Facilitates inter-agent communication"
            },
            {
                "name": "workflow_coordinator",
                "type": "system",
                "description": "Coordinates collaborative workflows"
            }
        ]
        
        tools.extend(default_tools)
        return tools
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model used by this agent"""
        model_info = {
            "framework": "crewai",
            "agent_type": "collaborative",
            "role": self.role.value if hasattr(self.role, 'value') else str(self.role)
        }
        
        if self.crew_info:
            model_info.update({
                "crew_id": self.crew_info.crew_id,
                "collaboration_pattern": self.crew_info.collaboration_pattern.value,
                "delegation_strategy": self.crew_info.delegation_strategy.value,
                "shared_memory": self.crew_info.shared_memory,
                "crew_members": len(self.crew_info.members)
            })
        
        return model_info
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and responsive"""
        try:
            # Test basic agent responsiveness
            if hasattr(self.agent, 'run'):
                test_result = await self.agent_instance.run({"test": "health_check"})
                return test_result is not None
            elif hasattr(self.agent, 'execute'):
                test_result = await self.agent_instance.execute({"test": "health_check"})
                return test_result is not None
            else:
                # Agent exists and has basic attributes
                return (hasattr(self.agent, 'name') or 
                       hasattr(self.agent, 'role') or 
                       hasattr(self.agent, 'metadata'))
        except Exception as e:
            return False
    
    def get_crew_info(self) -> Dict[str, Any]:
        """Get crew-specific information"""
        info = {
            "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
            "crew_agent": True
        }
        
        if self.crew_info:
            info.update({
                "crew_id": self.crew_info.crew_id,
                "crew_name": self.crew_info.crew_name,
                "collaboration_pattern": self.crew_info.collaboration_pattern.value,
                "delegation_strategy": self.crew_info.delegation_strategy.value,
                "shared_memory": self.crew_info.shared_memory,
                "crew_members": len(self.crew_info.members),
                "communication_channels": len(self.crew_info.communication_channels)
            })
        
        return info


class CrewAIIntegration(BaseIntegration):
    """
    Integration for CrewAI and multi-agent collaborative frameworks
    
    Handles:
    - CrewAI agent detection
    - Role-based agent coordination
    - Crew collaboration mechanisms
    - Multi-agent workflow orchestration
    """
    
    def __init__(self):
        super().__init__()
        self.logger = structlog.get_logger(__name__)
        self.agent_detector = CrewAIAgentDetector()
        self.delegator = RoleBasedTaskDelegator()
        self.communicator = InterAgentCommunicator()
        self.workflow_engine = CollaborativeWorkflowEngine()
    
    def get_integration_type(self) -> IntegrationType:
        """Return the integration type"""
        return IntegrationType.CREWAI
    
    def _detect_framework_markers(self, code: str) -> bool:
        """Detect CrewAI framework markers in code"""
        return self.agent_detector.detect_crewai_patterns(code)
    
    def extract_capabilities(self, code: str) -> List[AgentCapability]:
        """Extract capabilities from CrewAI agent code"""
        capabilities = []
        
        # Basic capabilities based on patterns
        capability_patterns = {
            AgentCapability.PLANNING: [r'plan', r'strategy', r'coordinate'],
            AgentCapability.ANALYSIS: [r'reason', r'analyze', r'decide', r'evaluate'],
            AgentCapability.GENERATION: [r'learn', r'adapt', r'improve', r'feedback'],
            AgentCapability.MEMORY: [r'memory', r'remember', r'history', r'context'],
            AgentCapability.TOOL_USE: [r'tool', r'function', r'execute', r'action'],
            AgentCapability.CHAT: [r'communicate', r'collaborate', r'message', r'coordinate']
        }
        
        for capability, patterns in capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    capabilities.append(capability)
                    break
        
        return capabilities
    
    def extract_tools(self, code: str) -> List[Dict[str, Any]]:
        """Extract tools from CrewAI agent code"""
        tools = []
        
        # Tool detection patterns for CrewAI
        tool_patterns = [
            r'tools\s*=\s*\[(.*?)\]',
            r'@tool\s*\ndef\s+(\w+)',
            r'Tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']',
            r'def\s+(\w+_tool)\s*\('
        ]
        
        for pattern in tool_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                tool_name = match.group(1) if match.groups() else "unknown_tool"
                tools.append({
                    "name": tool_name,
                    "type": "function",
                    "description": f"CrewAI tool: {tool_name}",
                    "crew_accessible": True
                })
        
        return tools
    
    async def detect_agents(self, scan_path: Path) -> List[DiscoveredAgent]:
        """Detect CrewAI and multi-agent systems"""
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
        """Analyze a single agent file for CrewAI patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Check if this is a CrewAI agent
            if not self._detect_framework_markers(code):
                return None
            
            # Extract basic information
            capabilities = self.extract_capabilities(code)
            tools = self.extract_tools(code)
            
            # CrewAI-specific analysis
            roles = self.agent_detector.extract_crew_roles(code)
            collaboration_pattern = self.agent_detector.detect_collaboration_pattern(code)
            
            # Extract crew member information
            crew_members = self._extract_crew_members(code)
            
            # Extract collaboration indicators
            shared_memory = bool(re.search(r'shared[_\s]*memory|memory[_\s]*sharing', code, re.IGNORECASE))
            delegation_strategy = self._detect_delegation_strategy(code)
            
            # Create crew configuration
            crew_config = CrewConfiguration(
                crew_id=f"crew_{file_path.stem}",
                crew_name=file_path.stem,
                members=crew_members,
                collaboration_pattern=collaboration_pattern,
                delegation_strategy=delegation_strategy,
                shared_memory=shared_memory
            )
            
            # Determine primary role
            primary_role = roles[0] if roles else CrewRole.EXECUTOR
            
            # Confidence scoring
            confidence = 0.5
            if len(roles) > 0:
                confidence += 0.15
            if len(crew_members) > 1:
                confidence += 0.15
            if shared_memory:
                confidence += 0.1
            if collaboration_pattern != CollaborationPattern.PEER_TO_PEER:
                confidence += 0.1
            
            return DiscoveredAgent(
                name=file_path.stem,
                file_path=file_path,
                framework="crewai",
                confidence=min(confidence, 1.0),
                capabilities=[cap.value for cap in capabilities],
                metadata={
                    "crew_info": crew_config,
                    "role": primary_role,
                    "crew_id": crew_config.crew_id,
                    "roles": [role.value for role in roles],
                    "collaboration_pattern": collaboration_pattern.value,
                    "delegation_strategy": delegation_strategy.value,
                    "shared_memory": shared_memory,
                    "crew_members": len(crew_members),
                    "tools": tools
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing CrewAI agent {file_path}: {e}")
            return None
    
    def _extract_crew_members(self, code: str) -> List[CrewMember]:
        """Extract crew member definitions from code"""
        members = []
        
        # Look for Agent definitions
        agent_pattern = r'Agent\s*\(\s*role\s*=\s*["\']([^"\']+)["\'].*?goal\s*=\s*["\']([^"\']+)["\'].*?backstory\s*=\s*["\']([^"\']+)["\']'
        matches = re.finditer(agent_pattern, code, re.IGNORECASE | re.DOTALL)
        
        for i, match in enumerate(matches):
            role_str = match.group(1)
            goal = match.group(2)
            backstory = match.group(3)
            
            # Map role string to enum
            role = self._map_role_string(role_str)
            
            member = CrewMember(
                agent_id=f"agent_{i}",
                role=role,
                backstory=backstory,
                goal=goal,
                collaboration_style="cooperative"
            )
            members.append(member)
        
        return members
    
    def _map_role_string(self, role_str: str) -> CrewRole:
        """Map role string to CrewRole enum"""
        role_lower = role_str.lower()
        
        for role, patterns in self.agent_detector.role_patterns.items():
            for pattern in patterns:
                if pattern in role_lower:
                    return role
        
        return CrewRole.EXECUTOR  # default
    
    def _detect_delegation_strategy(self, code: str) -> TaskDelegationStrategy:
        """Detect task delegation strategy from code"""
        strategy_patterns = {
            TaskDelegationStrategy.CAPABILITY_BASED: [r'capability', r'skill', r'expertise'],
            TaskDelegationStrategy.WORKLOAD_BALANCED: [r'workload', r'balance', r'distribute'],
            TaskDelegationStrategy.EXPERTISE_MATCHED: [r'expert', r'specialist', r'domain'],
            TaskDelegationStrategy.PRIORITY_BASED: [r'priority', r'urgent', r'important'],
            TaskDelegationStrategy.AUCTION_BASED: [r'auction', r'bid', r'compete']
        }
        
        for strategy, patterns in strategy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return strategy
        
        return TaskDelegationStrategy.CAPABILITY_BASED  # default
    
    async def integrate_agent(self, discovered_agent: DiscoveredAgent) -> IntegratedAgent:
        """Integrate a CrewAI agent"""
        try:
            # Create mock agent instance for testing
            agent_instance = self._create_mock_agent(discovered_agent)
            
            # Create adapter
            adapter = self.create_adapter(agent_instance, discovered_agent.metadata)
            
            # Create integrated agent
            integrated_agent = IntegratedAgent(
                id=f"crewai_{discovered_agent.name}_{hash(str(discovered_agent.file_path))}",
                name=discovered_agent.name,
                framework=IntegrationType.CREWAI,
                original_path=discovered_agent.file_path,
                adapter=adapter,
                capabilities=[AgentCapability(cap) for cap in discovered_agent.capabilities if cap in [c.value for c in AgentCapability]],
                metadata=discovered_agent.metadata,
                tools=discovered_agent.metadata.get("tools", []),
                model_info={"framework": "crewai", "role": discovered_agent.metadata.get("role", "executor")}
            )
            
            return integrated_agent
            
        except Exception as e:
            raise IntegrationError(f"CrewAI integration failed: {str(e)}")
    
    def create_adapter(self, agent_instance: Any, metadata: Dict[str, Any]) -> CrewAIAgentAdapter:
        """Create CrewAI agent adapter"""
        return CrewAIAgentAdapter(agent_instance, metadata)
    
    def _create_mock_agent(self, discovered_agent: DiscoveredAgent) -> Any:
        """Create a mock agent instance for testing"""
        class MockCrewAIAgent:
            def __init__(self, metadata):
                self.metadata = metadata
                self.name = discovered_agent.name
                self.role = metadata.get('role', CrewRole.EXECUTOR)
            
            async def run(self, input_data):
                return f"Mock CrewAI agent {self.name} ({self.role.value}) executed with: {input_data}"
            
            async def execute(self, input_data):
                return await self.run(input_data)
        
        return MockCrewAIAgent(discovered_agent.metadata)
