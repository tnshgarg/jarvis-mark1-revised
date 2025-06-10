#!/usr/bin/env python3
"""
UNIFIED WORKFLOW DEMO - Mark-1 Complete Agent Orchestration

This demo shows the complete Mark-1 workflow:
1. Agent discovery & repository scanning
2. Automatic adapter generation for new agents
3. Orchestration and task planning
4. Context management between agents
5. Task execution with clear visibility
"""

import asyncio
import sys
import os
import traceback
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import simplified database implementation
from database_simplified import (
    init_database, get_db_session, Agent, AgentType, AgentStatus, 
    AgentRepository, DatabaseError
)


class WorkflowLogger:
    """Simple logger with color support for CLI output"""
    
    COLORS = {
        "RESET": "\033[0m",
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "MAGENTA": "\033[95m",
        "CYAN": "\033[96m",
        "WHITE": "\033[97m"
    }
    
    @staticmethod
    def log(message, color=None, indent=0):
        """Log a message with optional color and indentation"""
        prefix = "  " * indent
        if color and color in WorkflowLogger.COLORS:
            print(f"{WorkflowLogger.COLORS[color]}{prefix}{message}{WorkflowLogger.COLORS['RESET']}")
        else:
            print(f"{prefix}{message}")
    
    @staticmethod
    def step(number, title):
        """Log a major step in the workflow"""
        print("\n" + "=" * 80)
        print(f"{WorkflowLogger.COLORS['CYAN']}STEP {number}: {title}{WorkflowLogger.COLORS['RESET']}")
        print("=" * 80)
    
    @staticmethod
    def substep(title):
        """Log a substep in the workflow"""
        print(f"\n{WorkflowLogger.COLORS['BLUE']}➤ {title}{WorkflowLogger.COLORS['RESET']}")
    
    @staticmethod
    def success(message, indent=0):
        """Log a success message"""
        prefix = "  " * indent
        print(f"{prefix}✅ {WorkflowLogger.COLORS['GREEN']}{message}{WorkflowLogger.COLORS['RESET']}")
    
    @staticmethod
    def error(message, indent=0):
        """Log an error message"""
        prefix = "  " * indent
        print(f"{prefix}❌ {WorkflowLogger.COLORS['RED']}{message}{WorkflowLogger.COLORS['RESET']}")
    
    @staticmethod
    def warning(message, indent=0):
        """Log a warning message"""
        prefix = "  " * indent
        print(f"{prefix}⚠️ {WorkflowLogger.COLORS['YELLOW']}{message}{WorkflowLogger.COLORS['RESET']}")
    
    @staticmethod
    def info(message, indent=0):
        """Log an info message"""
        prefix = "  " * indent
        print(f"{prefix}ℹ️ {WorkflowLogger.COLORS['WHITE']}{message}{WorkflowLogger.COLORS['RESET']}")
    
    @staticmethod
    def agent(name, message, indent=0):
        """Log an agent message"""
        prefix = "  " * indent
        print(f"{prefix}🤖 {WorkflowLogger.COLORS['MAGENTA']}[{name}]{WorkflowLogger.COLORS['RESET']} {message}")


class AgentAdapterGenerator:
    """Generates adapter code for new agent repositories"""
    
    @staticmethod
    async def generate_adapter(agent_data):
        """Generate adapter code for a new agent"""
        WorkflowLogger.substep(f"Generating adapter for agent: {agent_data['name']}")
        
        # Simulate adapter generation with some delay
        await asyncio.sleep(0.5)
        
        adapter_code = f"""
# Auto-generated adapter for {agent_data['name']}
# Framework: {agent_data['type']}
# Generated: {datetime.now().isoformat()}

from mark1.core.adapter import BaseAgentAdapter

class {agent_data['name'].replace(' ', '')}Adapter(BaseAgentAdapter):
    \"\"\"
    Adapter for {agent_data['name']} agent
    Type: {agent_data['type']}
    Capabilities: {', '.join(agent_data['capabilities'])}
    \"\"\"
    
    async def initialize(self):
        \"\"\"Initialize the agent\"\"\"
        self.logger.info(f"Initializing {{self.agent.name}} adapter")
        return True
    
    async def execute(self, task_input, context=None):
        \"\"\"Execute a task with this agent\"\"\"
        self.logger.info(f"Executing task with {{self.agent.name}}")
        
        # Load the agent module
        agent_module = self._load_agent_module()
        
        # Process task using agent capabilities
        result = await self._process_with_agent(agent_module, task_input, context)
        
        return result
"""
        
        # In a real implementation, we would save this to a file
        adapter_path = f"generated_output/{agent_data['name'].replace(' ', '_').lower()}_adapter.py"
        os.makedirs("generated_output", exist_ok=True)
        
        with open(adapter_path, "w") as f:
            f.write(adapter_code)
        
        WorkflowLogger.success(f"Generated adapter at: {adapter_path}", indent=1)
        
        return {
            "name": agent_data['name'],
            "path": adapter_path,
            "code": adapter_code
        }


class AgentDiscovery:
    """Discovers AI agents from repositories"""
    
    @staticmethod
    async def discover_agents(directory_path):
        """Discover agents in the given directory"""
        WorkflowLogger.substep(f"Scanning directory: {directory_path}")
        agents = []
        
        # Define agent patterns to look for
        agent_patterns = {
            "langchain": ["langchain", "chain", "llm"],
            "autogpt": ["autogpt", "autonomous", "agent"],
            "crewai": ["crewai", "crew", "agent"],
            "custom": ["agent", "task", "ai"]
        }
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                WorkflowLogger.warning(f"Directory {directory_path} doesn't exist", indent=1)
                return []
                
            file_count = 0
            agent_count = 0
            
            for file_path in directory.glob("**/*.py"):
                # Skip __init__.py and similar files
                if file_path.name.startswith("_"):
                    continue
                
                file_count += 1
                if file_count % 20 == 0:
                    WorkflowLogger.info(f"Scanned {file_count} files, found {agent_count} agents...", indent=1)
                
                try:
                    # Read file content
                    content = file_path.read_text()
                    
                    # Check for agent patterns
                    agent_type = AgentType.UNKNOWN.value
                    for pattern_type, keywords in agent_patterns.items():
                        if any(keyword in content.lower() for keyword in keywords):
                            agent_type = pattern_type
                            break
                    
                    # If any agent pattern was found
                    if agent_type != AgentType.UNKNOWN.value:
                        agent_count += 1
                        # Extract name from filename
                        name = file_path.stem.replace("_", " ").title()
                        
                        # Determine capabilities based on content
                        capabilities = []
                        if "text" in content.lower():
                            capabilities.append("text_processing")
                        if "analyze" in content.lower():
                            capabilities.append("analysis")
                        if "generate" in content.lower():
                            capabilities.append("generation")
                        
                        # Create agent data
                        agent_data = {
                            "name": name,
                            "type": agent_type,
                            "file_path": str(file_path),
                            "capabilities": capabilities,
                            "metadata": {
                                "discovered_at": datetime.now().isoformat(),
                                "size_bytes": file_path.stat().st_size
                            }
                        }
                        
                        agents.append(agent_data)
                        WorkflowLogger.log(f"Found agent: {name} ({agent_type})", color="GREEN", indent=1)
                except Exception as e:
                    WorkflowLogger.error(f"Error processing file {file_path}: {e}", indent=1)
            
            WorkflowLogger.success(f"Finished scanning. Found {len(agents)} agents in {file_count} files.", indent=1)
            return agents
        except Exception as e:
            WorkflowLogger.error(f"Error discovering agents: {e}", indent=1)
            return []


class TaskContext:
    """Manages context between agents in a workflow"""
    
    def __init__(self, task_id: str, initial_data: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.context_store = initial_data or {}
        self.history = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add(self, key: str, value: Any, source: str = "system"):
        """Add data to the context"""
        self.context_store[key] = value
        self.history.append({
            "action": "add",
            "key": key,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now().isoformat()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data from the context"""
        value = self.context_store.get(key, default)
        self.history.append({
            "action": "get",
            "key": key,
            "timestamp": datetime.now().isoformat()
        })
        return value
    
    def update(self, key: str, value: Any, source: str = "system"):
        """Update data in the context"""
        if key in self.context_store:
            self.context_store[key] = value
            self.history.append({
                "action": "update",
                "key": key,
                "source": source,
                "timestamp": datetime.now().isoformat()
            })
            self.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the context to a dictionary"""
        return {
            "task_id": self.task_id,
            "data": self.context_store,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def save(self, path: str):
        """Save the context to a file"""
        try:
            with open(path, "w") as f:
                json.dump(self.serialize(), f, indent=2)
            return True
        except Exception:
            return False


class Orchestrator:
    """Orchestrates agents to execute tasks"""
    
    def __init__(self):
        self.agents = []
        self.adapters = {}
        self.logger = WorkflowLogger
    
    async def initialize(self):
        """Initialize the orchestrator"""
        self.logger.substep("Initializing orchestrator")
        # In a real implementation, this would load configuration, connect to services, etc.
        await asyncio.sleep(0.5)
        self.logger.success("Orchestrator initialized", indent=1)
    
    async def register_agents(self, agents):
        """Register agents with the orchestrator"""
        self.logger.substep("Registering agents with orchestrator")
        self.agents = agents
        
        for agent in agents:
            # Register agent
            self.logger.info(f"Registering agent: {agent.name} ({agent.agent_type})", indent=1)
            
            # Simulate adapter loading
            await asyncio.sleep(0.2)
            
            # Add to adapters list
            adapter_key = f"{agent.agent_type}_{agent.name}"
            self.adapters[adapter_key] = {
                "agent_id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "capabilities": agent.capabilities,
                "status": "loaded"
            }
        
        self.logger.success(f"Registered {len(agents)} agents", indent=1)
    
    async def plan_workflow(self, task_description):
        """Plan a workflow for the given task"""
        self.logger.substep("Planning workflow for task")
        self.logger.info(f"Task description: {task_description}", indent=1)
        
        # Find suitable agents for this task
        suitable_agents = []
        
        # This is a simplified logic - in a real system, this would be more sophisticated
        for agent in self.agents:
            capabilities = agent.capabilities or []
            
            # Check if agent has relevant capabilities
            relevance_score = 0
            if "text_processing" in capabilities and "text" in task_description.lower():
                relevance_score += 1
            if "analysis" in capabilities and any(term in task_description.lower() for term in ["analyze", "analysis", "examine"]):
                relevance_score += 2
            if "generation" in capabilities and any(term in task_description.lower() for term in ["generate", "create", "produce"]):
                relevance_score += 2
            
            if relevance_score > 0:
                suitable_agents.append({
                    "agent": agent,
                    "relevance": relevance_score
                })
        
        # Sort agents by relevance
        suitable_agents.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Create a workflow plan (simplified)
        workflow = []
        
        # Take top 3 agents if available
        for i, agent_info in enumerate(suitable_agents[:3]):
            agent = agent_info["agent"]
            
            # Assign role based on position
            if i == 0:
                role = "primary"
            elif i == 1:
                role = "processor"
            else:
                role = "finalizer"
            
            workflow.append({
                "step": i + 1,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "agent_type": agent.agent_type,
                "role": role,
                "input": "previous_output" if i > 0 else "task_description",
                "estimated_time": 5  # seconds
            })
        
        if workflow:
            self.logger.success(f"Created workflow with {len(workflow)} steps", indent=1)
            for step in workflow:
                self.logger.info(f"Step {step['step']}: {step['agent_name']} ({step['role']})", indent=2)
        else:
            self.logger.warning("Could not create workflow - no suitable agents found", indent=1)
        
        return workflow
    
    async def execute_workflow(self, workflow, task_description):
        """Execute a workflow to complete a task"""
        self.logger.substep("Executing workflow")
        
        if not workflow:
            self.logger.error("No workflow to execute", indent=1)
            return None
        
        # Create a context for this task
        task_id = f"task_{int(time.time())}"
        context = TaskContext(task_id, {
            "task_description": task_description,
            "created_at": datetime.now().isoformat()
        })
        
        # Save initial context
        os.makedirs("data/contexts", exist_ok=True)
        context_path = f"data/contexts/{task_id}.json"
        context.save(context_path)
        self.logger.info(f"Created task context: {task_id}", indent=1)
        
        result = None
        for step in workflow:
            agent_name = step["agent_name"]
            agent_role = step["role"]
            
            self.logger.agent(agent_name, f"Starting execution (role: {agent_role})", indent=1)
            
            # Get input for this step
            if step["input"] == "task_description":
                step_input = task_description
            else:
                step_input = result
            
            # Simulate agent execution
            await asyncio.sleep(step["estimated_time"] * 0.2)  # Speed up for demo
            
            # Generate a simulated result based on agent type and role
            if "analysis" in self.agents[step["step"]-1].capabilities:
                if agent_role == "primary":
                    result = f"Initial analysis of the task: {task_description[:50]}... shows key elements that need further processing."
                elif agent_role == "processor":
                    result = f"Processed the initial analysis and identified 3 main components to address."
                else:
                    result = f"Finalized analysis with recommendations based on the processed data."
            elif "generation" in self.agents[step["step"]-1].capabilities:
                if agent_role == "primary":
                    result = f"Generated initial content based on: {task_description[:50]}..."
                elif agent_role == "processor":
                    result = f"Refined the generated content for better clarity and structure."
                else:
                    result = f"Finalized the content with formatting and additional details."
            else:
                result = f"Processed the task using standard methods for {agent_name}."
            
            # Update context
            context.add(f"step_{step['step']}_result", result, source=agent_name)
            context.save(context_path)
            
            self.logger.agent(agent_name, f"Completed execution with result: {result}", indent=1)
            self.logger.info(f"Updated context with step {step['step']} result", indent=2)
        
        # Create final result
        final_result = {
            "task_id": task_id,
            "workflow_steps": len(workflow),
            "agents_used": [step["agent_name"] for step in workflow],
            "execution_time": sum(step["estimated_time"] for step in workflow),
            "result": result,
            "context_path": context_path
        }
        
        # Save result
        os.makedirs("data/results", exist_ok=True)
        result_path = f"data/results/{task_id}.json"
        with open(result_path, "w") as f:
            json.dump(final_result, f, indent=2)
        
        self.logger.success(f"Workflow execution completed. Results saved to {result_path}", indent=1)
        return final_result


async def create_sample_agents():
    """Create sample agent files if no agents are found"""
    WorkflowLogger.substep("Creating sample agents for testing")
    
    # Create directories
    os.makedirs("test_agents/sample", exist_ok=True)
    
    # Sample agents with different capabilities
    sample_agents = [
        {
            "name": "text_processor_agent.py",
            "content": """
'''
Sample Text Processor Agent

This is a langchain-compatible agent that can process and analyze text.
'''

def process_text(text):
    '''Analyze and process text input'''
    return "Processed: " + text
            
def extract_entities(text):
    '''Extract named entities from text'''
    return ["Entity1", "Entity2", "Entity3"]
"""
        },
        {
            "name": "content_generator_agent.py",
            "content": """
'''
Sample Content Generator Agent

An AutoGPT-compatible agent that can generate various types of content.
'''

def generate_text(prompt, length=100):
    '''Generate text based on a prompt'''
    return f"Generated content based on: {prompt}"
    
def create_summary(text):
    '''Create a summary of longer text'''
    return f"Summary of {len(text)} characters of text"
"""
        },
        {
            "name": "data_analysis_agent.py",
            "content": """
'''
Sample Data Analysis Agent

A specialized agent for data analysis and insights.
'''

def analyze_data(data):
    '''Analyze a dataset and return insights'''
    return {
        "data_points": len(data),
        "insights": ["Insight 1", "Insight 2"],
        "recommendations": ["Recommendation 1", "Recommendation 2"]
    }
"""
        }
    ]
    
    # Write sample agents
    for agent in sample_agents:
        path = f"test_agents/sample/{agent['name']}"
        with open(path, "w") as f:
            f.write(agent["content"])
        WorkflowLogger.success(f"Created sample agent: {path}", indent=1)
    
    WorkflowLogger.success(f"Created {len(sample_agents)} sample agents", indent=1)


async def unified_workflow_demo():
    """
    Unified workflow demo showing the complete Mark-1 agent orchestration system
    """
    WorkflowLogger.log("\n" + "=" * 80, color="CYAN")
    WorkflowLogger.log("  MARK-1 UNIFIED WORKFLOW DEMO", color="CYAN")
    WorkflowLogger.log("  Complete Agent Orchestration System", color="CYAN")
    WorkflowLogger.log("=" * 80 + "\n", color="CYAN")
    
    WorkflowLogger.log("This demo shows the complete workflow:", color="WHITE")
    WorkflowLogger.log("1. Agent discovery & repository scanning", color="WHITE", indent=1)
    WorkflowLogger.log("2. Automatic adapter generation for new agents", color="WHITE", indent=1)
    WorkflowLogger.log("3. Orchestration and task planning", color="WHITE", indent=1)
    WorkflowLogger.log("4. Context management between agents", color="WHITE", indent=1)
    WorkflowLogger.log("5. Task execution with clear visibility", color="WHITE", indent=1)
    
    # Step 1: Initialize database
    WorkflowLogger.step(1, "Initialize Database")
    
    try:
        # Ensure aiosqlite is installed
        try:
            import aiosqlite
            WorkflowLogger.success("aiosqlite is installed", indent=1)
        except ImportError:
            WorkflowLogger.warning("aiosqlite is not installed. Installing now...", indent=1)
            os.system("pip install aiosqlite")
            import aiosqlite
            WorkflowLogger.success("aiosqlite installed successfully", indent=1)
        
        # Define settings override for a clean test database
        db_path = "data/unified_workflow_demo.sqlite"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        settings_override = {
            "database_url": f"sqlite:///{db_path}",
            "database_echo": False
        }
        
        # Use force_recreate=True to ensure tables use the updated schema
        await init_database(settings_override=settings_override, force_recreate=True)
        WorkflowLogger.success("Database initialized successfully", indent=1)
    except DatabaseError as e:
        WorkflowLogger.error(f"Database initialization failed: {e}", indent=1)
        traceback.print_exc()
        return
    
    # Step 2: Discover agents
    WorkflowLogger.step(2, "Discover AI Agents")
    
    # Look in test_agents directory
    agents_data = await AgentDiscovery.discover_agents("test_agents")
    
    if not agents_data:
        # Try agents directory if test_agents is empty
        agents_data = await AgentDiscovery.discover_agents("agents")
    
    if not agents_data:
        WorkflowLogger.warning("No agent directories found. Creating sample agents...", indent=1)
        await create_sample_agents()
        
        # Try again with the new sample
        agents_data = await AgentDiscovery.discover_agents("test_agents")
    
    WorkflowLogger.success(f"Found {len(agents_data)} potential AI agents", indent=1)
    
    # Step 3: Generate adapters for new agents
    WorkflowLogger.step(3, "Generate Agent Adapters")
    
    # Filter to just use 5 agents for the demo if there are many
    if len(agents_data) > 5:
        WorkflowLogger.info(f"Using only 5 of {len(agents_data)} agents for this demo", indent=1)
        # Filter to include diverse agent types
        agent_types = set()
        filtered_agents = []
        
        # First pass: get diverse agent types
        for agent in agents_data:
            if agent["type"] not in agent_types and len(agent_types) < 3:
                agent_types.add(agent["type"])
                filtered_agents.append(agent)
        
        # Second pass: fill remaining slots
        for agent in agents_data:
            if len(filtered_agents) < 5 and agent not in filtered_agents:
                filtered_agents.append(agent)
                if len(filtered_agents) >= 5:
                    break
        
        agents_data = filtered_agents
    
    # Generate adapters for all agents
    adapters = []
    for agent_data in agents_data:
        adapter = await AgentAdapterGenerator.generate_adapter(agent_data)
        adapters.append(adapter)
    
    WorkflowLogger.success(f"Generated {len(adapters)} agent adapters", indent=1)
    
    # Step 4: Register agents in database
    WorkflowLogger.step(4, "Register Agents in Database")
    
    registered_agents = []
    async with get_db_session() as session:
        agent_repo = AgentRepository()
        
        for agent_data in agents_data:
            try:
                agent = await agent_repo.create_agent(
                    session=session,
                    name=agent_data["name"],
                    agent_type=agent_data["type"],
                    framework="python",
                    file_path=agent_data["file_path"],
                    capabilities=agent_data["capabilities"],
                    metadata=agent_data["metadata"]
                )
                registered_agents.append(agent)
                WorkflowLogger.success(f"Registered: {agent.name} ({agent.agent_type})", indent=1)
            except Exception as e:
                WorkflowLogger.error(f"Failed to register {agent_data['name']}: {e}", indent=1)
    
    # Step 5: Initialize orchestrator
    WorkflowLogger.step(5, "Initialize Orchestrator")
    
    orchestrator = Orchestrator()
    await orchestrator.initialize()
    await orchestrator.register_agents(registered_agents)
    
    # Step 6: Plan task workflow
    WorkflowLogger.step(6, "Plan Task Workflow")
    
    task_description = """
    Analyze the current trends in renewable energy technologies and their potential 
    impact on reducing carbon emissions. Provide a summary of the most promising 
    technologies and their advantages/disadvantages.
    """
    
    WorkflowLogger.info(f"Task: {task_description.strip()}", indent=1)
    workflow = await orchestrator.plan_workflow(task_description)
    
    # Step 7: Execute task workflow
    WorkflowLogger.step(7, "Execute Task Workflow")
    
    if workflow:
        result = await orchestrator.execute_workflow(workflow, task_description)
        
        if result:
            WorkflowLogger.substep("Task Results")
            WorkflowLogger.info(f"Task ID: {result['task_id']}", indent=1)
            WorkflowLogger.info(f"Workflow Steps: {result['workflow_steps']}", indent=1)
            WorkflowLogger.info(f"Agents Used: {', '.join(result['agents_used'])}", indent=1)
            WorkflowLogger.info(f"Execution Time: {result['execution_time']} seconds", indent=1)
            WorkflowLogger.info(f"Result: {result['result']}", indent=1)
            WorkflowLogger.info(f"Context Path: {result['context_path']}", indent=1)
    else:
        WorkflowLogger.error("No workflow to execute", indent=1)
    
    # Step 8: Cleanup and summary
    WorkflowLogger.step(8, "Workflow Summary")
    
    WorkflowLogger.success("Unified workflow demonstration completed successfully", indent=1)
    WorkflowLogger.info(f"Discovered {len(agents_data)} agents", indent=1)
    WorkflowLogger.info(f"Generated {len(adapters)} adapters", indent=1)
    WorkflowLogger.info(f"Registered {len(registered_agents)} agents in database", indent=1)
    
    if workflow:
        WorkflowLogger.info(f"Created a {len(workflow)}-step workflow", indent=1)
        WorkflowLogger.info(f"Executed workflow with result: {result['result']}", indent=1)
    
    WorkflowLogger.log("\n" + "=" * 80, color="CYAN")
    WorkflowLogger.log("  MARK-1 WORKFLOW DEMO COMPLETED", color="CYAN")
    WorkflowLogger.log("=" * 80 + "\n", color="CYAN")


async def main():
    await unified_workflow_demo()


if __name__ == "__main__":
    asyncio.run(main()) 