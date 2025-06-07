"""
Research Crew - Simple CrewAI Multi-Agent System

Demonstrates basic CrewAI capabilities:
- Peer-to-peer collaboration
- Role specialization
- Task coordination
- Shared objectives
- Democratic decision making
"""

# from crewai import Agent, Task, Crew  # Mock these for now
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass


# Mock CrewAI components for demonstration
class Agent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Crew:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass
class ResearchAgent:
    """Simple research agent definition"""
    role: str
    goal: str
    backstory: str
    tools: List[str]
    verbose: bool = True


class SimpleResearchCrew:
    """
    Simple research crew with peer-to-peer collaboration
    
    Demonstrates:
    - Role-based agent specialization
    - Collaborative research tasks
    - Peer-to-peer communication pattern
    - Democratic consensus mechanism
    - Shared research objectives
    """
    
    def __init__(self):
        self.crew_name = "AI Research Team"
        self.collaboration_pattern = "peer_to_peer"
        self.delegation_strategy = "expertise_matched"
        
        # Define research agents
        self.agents = {
            "literature_reviewer": ResearchAgent(
                role="Literature Reviewer",
                goal="Find and analyze relevant research papers and academic sources",
                backstory="Expert in academic research with deep knowledge of scientific literature databases and citation analysis",
                tools=["academic_search", "citation_analyzer", "paper_summarizer"]
            ),
            "data_analyst": ResearchAgent(
                role="Data Analyst", 
                goal="Analyze research data and identify statistical patterns",
                backstory="Specialist in statistical analysis and data interpretation with expertise in research methodologies",
                tools=["statistical_analyzer", "data_visualizer", "trend_detector"]
            ),
            "technical_writer": ResearchAgent(
                role="Technical Writer",
                goal="Synthesize research findings into comprehensive reports",
                backstory="Professional technical writer with experience in translating complex research into accessible documentation",
                tools=["report_generator", "content_organizer", "citation_formatter"]
            )
        }
        
        print(f"🔬 Research Crew '{self.crew_name}' initialized")
        print(f"   Collaboration: {self.collaboration_pattern}")
        print(f"   Agents: {', '.join(self.agents.keys())}")
    
    async def conduct_research_project(self, research_topic: str, research_scope: str = "comprehensive") -> Dict[str, Any]:
        """
        Conduct collaborative research project
        
        Args:
            research_topic: Topic to research
            research_scope: Scope of research (basic, detailed, comprehensive)
        
        Returns:
            Research project results
        """
        print(f"🔍 Starting research project: {research_topic}")
        print(f"   Scope: {research_scope}")
        
        # Create research tasks
        tasks = self._create_research_tasks(research_topic, research_scope)
        
        print(f"📋 Research tasks defined:")
        for task in tasks:
            print(f"   - {task['description']} (assigned to: {task['agent']})")
        
        # Execute collaborative research
        results = await self._execute_collaborative_research(tasks)
        
        # Generate research summary
        summary = self._generate_research_summary(research_topic, results)
        
        print(f"✅ Research project completed: {research_topic}")
        print(f"   Tasks completed: {len(results)}")
        print(f"   Collaboration events: {sum(r.get('collaborations', 0) for r in results)}")
        
        return {
            "crew_name": self.crew_name,
            "research_topic": research_topic,
            "research_scope": research_scope,
            "task_results": results,
            "research_summary": summary,
            "collaboration_pattern": self.collaboration_pattern,
            "agents_involved": list(self.agents.keys()),
            "peer_collaboration": True
        }
    
    def _create_research_tasks(self, topic: str, scope: str) -> List[Dict[str, Any]]:
        """Create research tasks based on topic and scope"""
        base_tasks = [
            {
                "id": "literature_review",
                "description": f"Conduct comprehensive literature review on {topic}",
                "agent": "literature_reviewer",
                "deliverables": ["literature_summary", "key_papers", "research_gaps"],
                "collaboration_required": False,
                "priority": 1
            },
            {
                "id": "data_analysis",
                "description": f"Analyze available data and research findings on {topic}",
                "agent": "data_analyst", 
                "deliverables": ["statistical_analysis", "trend_identification", "data_insights"],
                "collaboration_required": True,
                "priority": 2,
                "dependencies": ["literature_review"]
            },
            {
                "id": "report_synthesis",
                "description": f"Synthesize research findings into comprehensive report on {topic}",
                "agent": "technical_writer",
                "deliverables": ["research_report", "executive_summary", "recommendations"],
                "collaboration_required": True,
                "priority": 3,
                "dependencies": ["literature_review", "data_analysis"]
            }
        ]
        
        # Add scope-specific tasks
        if scope == "comprehensive":
            base_tasks.extend([
                {
                    "id": "peer_review",
                    "description": f"Peer review research findings and methodology",
                    "agent": "literature_reviewer",  # Cross-role review
                    "deliverables": ["review_feedback", "methodology_validation", "quality_assessment"],
                    "collaboration_required": True,
                    "priority": 4,
                    "dependencies": ["report_synthesis"]
                }
            ])
        
        return base_tasks
    
    async def _execute_collaborative_research(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute research tasks with collaboration"""
        results = []
        completed_tasks = set()
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.get("priority", 1))
        
        for task in sorted_tasks:
            # Check dependencies
            dependencies = task.get("dependencies", [])
            if not all(dep in completed_tasks for dep in dependencies):
                continue
            
            print(f"   🔬 Executing: {task['description']}")
            
            # Execute task
            task_result = await self._execute_research_task(task)
            
            # Handle collaboration if required
            if task.get("collaboration_required", False):
                collaboration_result = await self._handle_peer_collaboration(task, task_result)
                task_result["collaboration"] = collaboration_result
                task_result["collaborations"] = len(collaboration_result.get("participants", []))
            
            results.append(task_result)
            completed_tasks.add(task["id"])
        
        return results
    
    async def _execute_research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single research task"""
        # Simulate research work
        await asyncio.sleep(0.1)
        
        agent_name = task["agent"]
        agent = self.agents[agent_name]
        
        # Generate mock research output
        outputs = {
            "literature_review": f"Comprehensive literature review on {task['description']} completed. Found 25 relevant papers, identified 3 key research gaps, and summarized current state of knowledge.",
            "data_analysis": f"Statistical analysis completed for {task['description']}. Identified significant trends, performed correlation analysis, and generated data visualizations.",
            "report_synthesis": f"Research report synthesized for {task['description']}. Comprehensive 20-page report with executive summary, methodology, findings, and recommendations.",
            "peer_review": f"Peer review completed for {task['description']}. Methodology validated, quality assessed, and improvement recommendations provided."
        }
        
        task_output = outputs.get(task["id"], f"Task completed: {task['description']}")
        
        return {
            "task_id": task["id"],
            "agent": agent_name,
            "agent_role": agent.role,
            "description": task["description"],
            "output": task_output,
            "deliverables": task.get("deliverables", []),
            "tools_used": agent.tools,
            "status": "completed"
        }
    
    async def _handle_peer_collaboration(self, task: Dict[str, Any], task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle peer-to-peer collaboration"""
        task_agent = task["agent"]
        
        # Determine collaboration participants
        participants = [task_agent]
        
        # Add relevant collaborators based on task type
        if task["id"] == "data_analysis":
            participants.append("literature_reviewer")  # Needs literature context
        elif task["id"] == "report_synthesis":
            participants.extend(["literature_reviewer", "data_analyst"])  # Needs all inputs
        elif task["id"] == "peer_review":
            participants.extend(["data_analyst", "technical_writer"])  # Cross-role review
        
        # Remove duplicates and the primary agent
        collaborators = [p for p in set(participants) if p != task_agent]
        
        # Simulate peer collaboration
        collaboration_insights = []
        for collaborator in collaborators:
            collaborator_agent = self.agents[collaborator]
            insight = f"{collaborator_agent.role} provided domain expertise and feedback"
            collaboration_insights.append(insight)
        
        return {
            "collaboration_type": "peer_to_peer",
            "primary_agent": task_agent,
            "participants": participants,
            "collaborators": collaborators,
            "insights": collaboration_insights,
            "consensus_reached": True,
            "collaboration_quality": 0.9
        }
    
    def _generate_research_summary(self, topic: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive research summary"""
        total_deliverables = sum(len(r.get("deliverables", [])) for r in results)
        collaboration_events = sum(r.get("collaborations", 0) for r in results)
        
        # Extract key outputs
        key_findings = []
        for result in results:
            if result["task_id"] == "literature_review":
                key_findings.append("Literature review identified current research landscape")
            elif result["task_id"] == "data_analysis":
                key_findings.append("Data analysis revealed significant trends and patterns")
            elif result["task_id"] == "report_synthesis":
                key_findings.append("Comprehensive report synthesized all research findings")
            elif result["task_id"] == "peer_review":
                key_findings.append("Peer review validated methodology and quality")
        
        return {
            "research_topic": topic,
            "tasks_completed": len(results),
            "total_deliverables": total_deliverables,
            "collaboration_events": collaboration_events,
            "key_findings": key_findings,
            "research_quality": "high" if collaboration_events > 0 else "medium",
            "agents_contributed": len(set(r["agent"] for r in results)),
            "peer_collaboration_success": collaboration_events > 0,
            "recommendations": [
                "Research objectives successfully met",
                "High-quality collaborative process employed",
                "Peer review enhanced research validity",
                "Findings ready for publication/application"
            ]
        }
    
    def get_crew_configuration(self) -> Dict[str, Any]:
        """Get crew configuration details"""
        return {
            "crew_name": self.crew_name,
            "collaboration_pattern": self.collaboration_pattern,
            "delegation_strategy": self.delegation_strategy,
            "agent_count": len(self.agents),
            "agent_roles": [agent.role for agent in self.agents.values()],
            "capabilities": [
                "peer_to_peer_collaboration",
                "role_specialization", 
                "task_coordination",
                "democratic_consensus",
                "shared_objectives",
                "cross_functional_review"
            ],
            "tools_available": list(set(tool for agent in self.agents.values() for tool in agent.tools))
        }


# Example usage and testing
async def test_research_crew():
    """Test the research crew system"""
    crew = SimpleResearchCrew()
    
    # Test research projects
    research_projects = [
        {
            "topic": "Artificial Intelligence in Healthcare",
            "scope": "comprehensive"
        },
        {
            "topic": "Quantum Computing Applications",
            "scope": "detailed"
        },
        {
            "topic": "Sustainable Energy Technologies", 
            "scope": "basic"
        }
    ]
    
    print("🧪 Testing Research Crew System")
    print("=" * 60)
    
    project_results = []
    
    for project in research_projects:
        print(f"\n📚 Research Project: {project['topic']}")
        result = await crew.conduct_research_project(
            project["topic"],
            project["scope"]
        )
        project_results.append(result)
        
        summary = result["research_summary"]
        print(f"   Tasks: {summary['tasks_completed']}")
        print(f"   Deliverables: {summary['total_deliverables']}")
        print(f"   Collaboration: {summary['collaboration_events']} events")
        print(f"   Quality: {summary['research_quality']}")
    
    # Display crew configuration
    print(f"\n📊 Crew Configuration:")
    config = crew.get_crew_configuration()
    print(f"   Agents: {config['agent_count']}")
    print(f"   Roles: {', '.join(config['agent_roles'])}")
    print(f"   Tools: {len(config['tools_available'])}")
    print(f"   Capabilities: {len(config['capabilities'])}")
    
    return {
        "crew_tested": True,
        "projects_completed": len(project_results),
        "successful_projects": sum(1 for r in project_results if r["research_summary"]["research_quality"] == "high"),
        "total_collaborations": sum(r["research_summary"]["collaboration_events"] for r in project_results),
        "crew_config": config
    }


if __name__ == "__main__":
    asyncio.run(test_research_crew()) 