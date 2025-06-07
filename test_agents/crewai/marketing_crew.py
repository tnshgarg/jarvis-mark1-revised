"""
Marketing Crew - CrewAI Multi-Agent System

Demonstrates comprehensive CrewAI capabilities:
- Role-based agent collaboration
- Hierarchical task delegation
- Shared memory systems
- Inter-agent communication
- Collaborative workflow execution
- Multi-agent coordination patterns
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketingTask:
    """Marketing-specific task structure"""
    task_id: str
    description: str
    task_type: str  # research, analysis, content, review
    priority: int
    assigned_role: str
    dependencies: List[str]
    deliverables: List[str]
    collaboration_required: bool = True
    status: str = "pending"
    progress: float = 0.0


class MarketingMemoryStore:
    """Shared memory system for marketing crew"""
    
    def __init__(self):
        self.shared_knowledge: Dict[str, Any] = {}
        self.campaign_data: Dict[str, Any] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        self.role_insights: Dict[str, List[str]] = {
            "researcher": [],
            "analyst": [],
            "writer": [],
            "reviewer": []
        }
    
    def store_insight(self, role: str, insight: str, data: Dict[str, Any]):
        """Store insights from different roles"""
        insight_record = {
            "role": role,
            "insight": insight,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if role in self.role_insights:
            self.role_insights[role].append(insight_record)
        
        # Update shared knowledge
        if insight not in self.shared_knowledge:
            self.shared_knowledge[insight] = []
        self.shared_knowledge[insight].append(insight_record)
    
    def get_role_insights(self, role: str) -> List[Dict[str, Any]]:
        """Get insights specific to a role"""
        return self.role_insights.get(role, [])
    
    def get_shared_knowledge(self, topic: str = None) -> Dict[str, Any]:
        """Get shared knowledge, optionally filtered by topic"""
        if topic:
            return {k: v for k, v in self.shared_knowledge.items() if topic.lower() in k.lower()}
        return self.shared_knowledge
    
    def record_collaboration(self, participants: List[str], task: str, outcome: str):
        """Record collaborative interactions"""
        collaboration_record = {
            "participants": participants,
            "task": task,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        }
        self.collaboration_history.append(collaboration_record)


class MarketingCommunicationHub:
    """Communication hub for marketing crew"""
    
    def __init__(self):
        self.message_queue: List[Dict[str, Any]] = []
        self.role_channels: Dict[str, List[Dict[str, Any]]] = {
            "researcher": [],
            "analyst": [],
            "writer": [],
            "reviewer": [],
            "broadcast": []
        }
    
    async def send_message(self, sender: str, receiver: str, message: str, msg_type: str = "info") -> str:
        """Send message between crew members"""
        message_obj = {
            "id": f"msg_{len(self.message_queue)}",
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "read": False
        }
        
        self.message_queue.append(message_obj)
        
        # Route to appropriate channel
        if receiver in self.role_channels:
            self.role_channels[receiver].append(message_obj)
        elif receiver == "all":
            for channel in self.role_channels.values():
                channel.append(message_obj)
        
        return message_obj["id"]
    
    async def get_messages(self, role: str, unread_only: bool = True) -> List[Dict[str, Any]]:
        """Get messages for a specific role"""
        messages = self.role_channels.get(role, [])
        
        if unread_only:
            return [msg for msg in messages if not msg.get("read", False)]
        
        return messages
    
    async def broadcast_message(self, sender: str, message: str) -> List[str]:
        """Broadcast message to all crew members"""
        message_ids = []
        
        for role in ["researcher", "analyst", "writer", "reviewer"]:
            if role != sender:  # Don't send to sender
                msg_id = await self.send_message(sender, role, message, "broadcast")
                message_ids.append(msg_id)
        
        return message_ids


class MarketingTaskDelegator:
    """Intelligent task delegation for marketing crew"""
    
    def __init__(self):
        self.role_capabilities = {
            "researcher": [
                "market_research", "competitor_analysis", "trend_identification",
                "data_collection", "survey_design", "customer_insights"
            ],
            "analyst": [
                "data_analysis", "statistical_modeling", "pattern_recognition",
                "performance_metrics", "roi_calculation", "trend_analysis"
            ],
            "writer": [
                "content_creation", "copywriting", "storytelling",
                "brand_messaging", "campaign_copy", "social_media_content"
            ],
            "reviewer": [
                "quality_assurance", "content_review", "brand_compliance",
                "performance_evaluation", "feedback_integration", "approval_workflow"
            ]
        }
        
        self.task_type_mapping = {
            "research": "researcher",
            "analysis": "analyst", 
            "content": "writer",
            "review": "reviewer"
        }
    
    def delegate_task(self, task: MarketingTask) -> str:
        """Delegate task to most suitable role"""
        # Primary delegation based on task type
        if task.task_type in self.task_type_mapping:
            primary_role = self.task_type_mapping[task.task_type]
        else:
            # Analyze task description for capabilities
            primary_role = self._analyze_task_requirements(task.description)
        
        return primary_role
    
    def _analyze_task_requirements(self, description: str) -> str:
        """Analyze task description to determine best role"""
        desc_lower = description.lower()
        role_scores = {}
        
        for role, capabilities in self.role_capabilities.items():
            score = 0
            for capability in capabilities:
                capability_words = capability.split('_')
                for word in capability_words:
                    if word in desc_lower:
                        score += 1
            role_scores[role] = score
        
        # Return role with highest score
        best_role = max(role_scores, key=role_scores.get)
        return best_role if role_scores[best_role] > 0 else "researcher"  # default


class MarketingWorkflowCoordinator:
    """Coordinates marketing workflows and dependencies"""
    
    def __init__(self, memory_store: MarketingMemoryStore, comm_hub: MarketingCommunicationHub):
        self.memory_store = memory_store
        self.comm_hub = comm_hub
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
    
    async def create_campaign_workflow(self, campaign_name: str, tasks: List[MarketingTask]) -> Dict[str, Any]:
        """Create a marketing campaign workflow"""
        workflow = {
            "campaign_name": campaign_name,
            "tasks": {task.task_id: task for task in tasks},
            "status": "created",
            "progress": 0.0,
            "execution_order": [],
            "collaboration_points": [],
            "role_assignments": {}
        }
        
        # Analyze task dependencies
        await self._analyze_dependencies(workflow)
        
        # Create execution plan
        await self._create_execution_plan(workflow)
        
        # Assign roles to tasks
        await self._assign_roles(workflow)
        
        self.active_workflows[campaign_name] = workflow
        return workflow
    
    async def execute_campaign_workflow(self, campaign_name: str) -> Dict[str, Any]:
        """Execute marketing campaign workflow"""
        if campaign_name not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.active_workflows[campaign_name]
        workflow["status"] = "executing"
        
        try:
            execution_results = []
            
            # Execute tasks in dependency order
            for task_id in workflow["execution_order"]:
                task = workflow["tasks"][task_id]
                assigned_role = workflow["role_assignments"][task_id]
                
                # Notify crew about task assignment
                await self.comm_hub.send_message(
                    "coordinator", assigned_role,
                    f"Task assigned: {task.description}",
                    "task_assignment"
                )
                
                # Execute task with collaboration
                task_result = await self._execute_collaborative_task(task, assigned_role, workflow)
                execution_results.append(task_result)
                
                # Update workflow progress
                completed_tasks = sum(1 for t in workflow["tasks"].values() if t.status == "completed")
                workflow["progress"] = completed_tasks / len(workflow["tasks"])
            
            workflow["status"] = "completed"
            
            return {
                "success": True,
                "campaign": campaign_name,
                "tasks_completed": len(execution_results),
                "final_progress": workflow["progress"],
                "results": execution_results
            }
            
        except Exception as e:
            workflow["status"] = "failed"
            return {"success": False, "error": str(e)}
    
    async def _analyze_dependencies(self, workflow: Dict[str, Any]):
        """Analyze task dependencies"""
        tasks = workflow["tasks"]
        dependencies = {}
        
        for task_id, task in tasks.items():
            dependencies[task_id] = task.dependencies
        
        workflow["dependencies"] = dependencies
    
    async def _create_execution_plan(self, workflow: Dict[str, Any]):
        """Create execution plan respecting dependencies"""
        tasks = workflow["tasks"]
        dependencies = workflow["dependencies"]
        
        # Topological sort for execution order
        execution_order = []
        completed = set()
        
        while len(execution_order) < len(tasks):
            for task_id, task_deps in dependencies.items():
                if task_id not in completed and all(dep in completed for dep in task_deps):
                    execution_order.append(task_id)
                    completed.add(task_id)
                    break
        
        workflow["execution_order"] = execution_order
    
    async def _assign_roles(self, workflow: Dict[str, Any]):
        """Assign roles to tasks"""
        delegator = MarketingTaskDelegator()
        role_assignments = {}
        
        for task_id, task in workflow["tasks"].items():
            assigned_role = delegator.delegate_task(task)
            role_assignments[task_id] = assigned_role
            task.assigned_role = assigned_role
        
        workflow["role_assignments"] = role_assignments
    
    async def _execute_collaborative_task(self, task: MarketingTask, assigned_role: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with crew collaboration"""
        # Simulate task execution
        await asyncio.sleep(0.1)
        
        # Handle collaboration if required
        collaboration_result = None
        if task.collaboration_required:
            collaboration_result = await self._handle_task_collaboration(task, assigned_role)
        
        # Mock task execution result
        task_output = await self._generate_task_output(task, assigned_role)
        
        # Store results in shared memory
        self.memory_store.store_insight(
            assigned_role,
            f"task_completion_{task.task_type}",
            {
                "task_id": task.task_id,
                "output": task_output,
                "collaboration": collaboration_result
            }
        )
        
        # Update task status
        task.status = "completed"
        task.progress = 1.0
        
        return {
            "task_id": task.task_id,
            "assigned_role": assigned_role,
            "output": task_output,
            "collaboration": collaboration_result,
            "status": "completed"
        }
    
    async def _handle_task_collaboration(self, task: MarketingTask, assigned_role: str) -> Dict[str, Any]:
        """Handle collaborative aspects of task"""
        # Determine collaboration participants
        participants = [assigned_role]
        
        if task.task_type == "content":
            participants.extend(["researcher", "reviewer"])  # Content needs research input and review
        elif task.task_type == "analysis":
            participants.extend(["researcher"])  # Analysis needs research data
        elif task.task_type == "review":
            participants.extend(["writer", "analyst"])  # Review needs content to review
        
        # Simulate collaboration messages
        for participant in participants:
            if participant != assigned_role:
                await self.comm_hub.send_message(
                    assigned_role, participant,
                    f"Collaboration needed for: {task.description}",
                    "collaboration_request"
                )
        
        # Record collaboration in memory
        self.memory_store.record_collaboration(
            participants, task.description, "successful_collaboration"
        )
        
        return {
            "participants": participants,
            "collaboration_type": f"{task.task_type}_collaboration",
            "outcome": "successful",
            "insights_shared": len(participants) - 1
        }
    
    async def _generate_task_output(self, task: MarketingTask, role: str) -> str:
        """Generate mock task output based on role and task type"""
        outputs = {
            ("researcher", "research"): f"Market research completed for: {task.description}. Key findings include market size, competitor analysis, and customer insights.",
            ("analyst", "analysis"): f"Data analysis completed for: {task.description}. Performance metrics, ROI calculations, and trend analysis provided.",
            ("writer", "content"): f"Content created for: {task.description}. Includes compelling copy, brand messaging, and engagement strategies.",
            ("reviewer", "review"): f"Quality review completed for: {task.description}. Content approved with recommendations for optimization."
        }
        
        key = (role, task.task_type)
        return outputs.get(key, f"{role} completed {task.task_type} task: {task.description}")


class MarketingCrew:
    """
    Complete Marketing Crew demonstrating CrewAI multi-agent collaboration
    
    Features:
    - Role-based agent coordination (Researcher, Analyst, Writer, Reviewer)
    - Hierarchical task delegation
    - Shared memory systems
    - Inter-agent communication
    - Collaborative workflow execution
    - Multi-agent coordination patterns
    """
    
    def __init__(self, crew_name: str = "MarketingTeam"):
        self.crew_name = crew_name
        self.shared_memory = MarketingMemoryStore()
        self.communication_hub = MarketingCommunicationHub()
        self.workflow_coordinator = MarketingWorkflowCoordinator(
            self.shared_memory, self.communication_hub
        )
        
        # Crew configuration
        self.crew_config = {
            "crew_name": crew_name,
            "collaboration_pattern": "hierarchical",
            "delegation_strategy": "capability_based",
            "shared_memory": True,
            "roles": ["researcher", "analyst", "writer", "reviewer"],
            "communication_channels": ["direct", "broadcast", "role_specific"],
            "consensus_mechanism": "leader_approval"
        }
        
        print(f"🎯 Marketing Crew '{crew_name}' initialized")
        print(f"   Roles: {', '.join(self.crew_config['roles'])}")
        print(f"   Collaboration: {self.crew_config['collaboration_pattern']}")
        print(f"   Shared Memory: {self.crew_config['shared_memory']}")
    
    async def execute_marketing_campaign(self, campaign_name: str, campaign_brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete marketing campaign with crew collaboration
        
        Args:
            campaign_name: Name of the marketing campaign
            campaign_brief: Campaign requirements and objectives
        
        Returns:
            Campaign execution results
        """
        print(f"🚀 Starting marketing campaign: {campaign_name}")
        
        # Create campaign tasks
        tasks = self._create_campaign_tasks(campaign_brief)
        
        print(f"📋 Campaign tasks created: {len(tasks)}")
        for task in tasks:
            print(f"   - {task.description} ({task.task_type})")
        
        # Create collaborative workflow
        workflow = await self.workflow_coordinator.create_campaign_workflow(campaign_name, tasks)
        
        print(f"🔗 Workflow created with {len(workflow['execution_order'])} tasks")
        print(f"   Execution order: {' → '.join(workflow['execution_order'])}")
        
        # Execute campaign
        execution_result = await self.workflow_coordinator.execute_campaign_workflow(campaign_name)
        
        if execution_result["success"]:
            print(f"✅ Campaign '{campaign_name}' completed successfully")
            print(f"   Tasks completed: {execution_result['tasks_completed']}")
            print(f"   Final progress: {execution_result['final_progress']:.1%}")
        else:
            print(f"❌ Campaign '{campaign_name}' failed: {execution_result.get('error', 'Unknown error')}")
        
        # Generate campaign summary
        summary = self._generate_campaign_summary(campaign_name, execution_result)
        
        return {
            "crew_name": self.crew_name,
            "campaign": campaign_name,
            "execution_result": execution_result,
            "campaign_summary": summary,
            "crew_collaboration": True,
            "shared_memory_insights": len(self.shared_memory.shared_knowledge),
            "collaboration_events": len(self.shared_memory.collaboration_history)
        }
    
    def _create_campaign_tasks(self, campaign_brief: Dict[str, Any]) -> List[MarketingTask]:
        """Create tasks for marketing campaign"""
        campaign_type = campaign_brief.get("type", "standard")
        target_audience = campaign_brief.get("target_audience", "general")
        budget = campaign_brief.get("budget", "medium")
        
        # Standard campaign task sequence
        tasks = [
            MarketingTask(
                task_id="market_research",
                description=f"Conduct market research for {target_audience} audience",
                task_type="research",
                priority=1,
                assigned_role="researcher",
                dependencies=[],
                deliverables=["market_analysis_report", "competitor_insights", "audience_personas"],
                collaboration_required=False
            ),
            MarketingTask(
                task_id="data_analysis",
                description=f"Analyze market data and identify opportunities",
                task_type="analysis",
                priority=2,
                assigned_role="analyst",
                dependencies=["market_research"],
                deliverables=["data_insights", "opportunity_analysis", "recommendations"],
                collaboration_required=True
            ),
            MarketingTask(
                task_id="content_strategy",
                description=f"Develop content strategy for {campaign_type} campaign",
                task_type="content",
                priority=3,
                assigned_role="writer",
                dependencies=["market_research", "data_analysis"],
                deliverables=["content_strategy", "messaging_framework", "campaign_copy"],
                collaboration_required=True
            ),
            MarketingTask(
                task_id="campaign_review",
                description=f"Review and approve campaign materials",
                task_type="review",
                priority=4,
                assigned_role="reviewer",
                dependencies=["content_strategy"],
                deliverables=["approval_report", "optimization_recommendations", "final_approval"],
                collaboration_required=True
            )
        ]
        
        return tasks
    
    def _generate_campaign_summary(self, campaign_name: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive campaign summary"""
        summary = {
            "campaign_name": campaign_name,
            "execution_success": execution_result["success"],
            "tasks_completed": execution_result.get("tasks_completed", 0),
            "crew_performance": {
                "collaboration_events": len(self.shared_memory.collaboration_history),
                "shared_insights": len(self.shared_memory.shared_knowledge),
                "communication_efficiency": len(self.communication_hub.message_queue),
                "role_contributions": {
                    role: len(insights) for role, insights in self.shared_memory.role_insights.items()
                }
            },
            "key_deliverables": [],
            "collaboration_highlights": []
        }
        
        # Extract deliverables from task results
        if execution_result.get("results"):
            for result in execution_result["results"]:
                summary["key_deliverables"].append({
                    "task": result["task_id"],
                    "role": result["assigned_role"],
                    "output": result["output"][:100] + "..." if len(result["output"]) > 100 else result["output"]
                })
        
        # Extract collaboration highlights
        for collab in self.shared_memory.collaboration_history:
            summary["collaboration_highlights"].append({
                "participants": collab["participants"],
                "task": collab["task"][:50] + "..." if len(collab["task"]) > 50 else collab["task"],
                "outcome": collab["outcome"]
            })
        
        return summary
    
    async def get_crew_status(self) -> Dict[str, Any]:
        """Get comprehensive crew status"""
        # Get recent messages for each role
        role_messages = {}
        for role in self.crew_config["roles"]:
            messages = await self.communication_hub.get_messages(role, unread_only=False)
            role_messages[role] = len(messages)
        
        return {
            "crew_name": self.crew_name,
            "crew_config": self.crew_config,
            "active_workflows": len(self.workflow_coordinator.active_workflows),
            "shared_memory": {
                "total_insights": len(self.shared_memory.shared_knowledge),
                "role_insights": {role: len(insights) for role, insights in self.shared_memory.role_insights.items()},
                "collaboration_history": len(self.shared_memory.collaboration_history)
            },
            "communication": {
                "total_messages": len(self.communication_hub.message_queue),
                "role_messages": role_messages
            },
            "capabilities": [
                "role_based_collaboration",
                "hierarchical_delegation", 
                "shared_memory_system",
                "inter_agent_communication",
                "collaborative_workflows",
                "multi_agent_coordination"
            ]
        }


# Example usage and testing
async def test_marketing_crew():
    """Test the marketing crew system"""
    crew = MarketingCrew("EliteMarketingTeam")
    
    # Test campaign scenarios
    campaigns = [
        {
            "name": "Product Launch Campaign",
            "brief": {
                "type": "product_launch",
                "target_audience": "tech_professionals",
                "budget": "high",
                "objectives": ["awareness", "lead_generation", "conversions"]
            }
        },
        {
            "name": "Brand Awareness Campaign",
            "brief": {
                "type": "brand_awareness",
                "target_audience": "millennials",
                "budget": "medium",
                "objectives": ["brand_recognition", "engagement", "social_reach"]
            }
        }
    ]
    
    print("🧪 Testing Marketing Crew System")
    print("=" * 60)
    
    campaign_results = []
    
    for campaign_info in campaigns:
        print(f"\n📈 Campaign: {campaign_info['name']}")
        result = await crew.execute_marketing_campaign(
            campaign_info["name"], 
            campaign_info["brief"]
        )
        campaign_results.append(result)
        
        if result["execution_result"]["success"]:
            print(f"✅ Success: {result['shared_memory_insights']} insights generated")
            print(f"   Collaboration events: {result['collaboration_events']}")
        else:
            print(f"❌ Failed: {result['execution_result'].get('error', 'Unknown error')}")
    
    # Display crew status
    print(f"\n📊 Final Crew Status:")
    status = await crew.get_crew_status()
    print(f"   Workflows executed: {status['active_workflows']}")
    print(f"   Total insights: {status['shared_memory']['total_insights']}")
    print(f"   Messages exchanged: {status['communication']['total_messages']}")
    print(f"   Capabilities: {len(status['capabilities'])}")
    
    return {
        "crew_tested": True,
        "campaigns_executed": len(campaign_results),
        "successful_campaigns": sum(1 for r in campaign_results if r["execution_result"]["success"]),
        "final_status": status
    }


if __name__ == "__main__":
    asyncio.run(test_marketing_crew()) 