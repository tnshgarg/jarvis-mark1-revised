"""
Multi-Agent LangChain System for Mark-1 Session 14 Testing

Demonstrates advanced multi-agent coordination features:
- Hierarchical agent communication
- Coordinator-subordinate patterns
- Shared memory and context
- Agent specialization and role definition
- Conflict resolution strategies
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage


@dataclass
class AgentRole:
    """Definition of an agent's role and capabilities"""
    name: str
    description: str
    specialization: str
    tools: List[str]
    priority: int  # Higher priority agents make final decisions


class SharedMemory:
    """Shared memory system for multi-agent coordination"""
    
    def __init__(self):
        self.conversations: List[Dict[str, Any]] = []
        self.shared_context: Dict[str, Any] = {}
        self.agent_outputs: Dict[str, Any] = {}
        self.coordination_log: List[Dict[str, Any]] = []
    
    def add_conversation(self, agent_name: str, message: str, role: str = "user"):
        """Add a conversation entry"""
        self.conversations.append({
            "agent": agent_name,
            "message": message,
            "role": role,
            "timestamp": __import__('time').time()
        })
    
    def update_context(self, key: str, value: Any, agent_name: str):
        """Update shared context"""
        self.shared_context[key] = {
            "value": value,
            "updated_by": agent_name,
            "timestamp": __import__('time').time()
        }
    
    def get_context(self, key: str) -> Any:
        """Get shared context value"""
        return self.shared_context.get(key, {}).get("value")
    
    def store_agent_output(self, agent_name: str, output: Any):
        """Store agent output for coordination"""
        self.agent_outputs[agent_name] = {
            "output": output,
            "timestamp": __import__('time').time()
        }
    
    def log_coordination_event(self, event_type: str, details: Dict[str, Any]):
        """Log coordination events"""
        self.coordination_log.append({
            "event_type": event_type,
            "details": details,
            "timestamp": __import__('time').time()
        })


@tool
def research_tool(query: str) -> str:
    """Research information about a topic"""
    # Mock research implementation
    research_data = {
        "artificial intelligence": "AI is a rapidly evolving field focused on creating intelligent machines...",
        "machine learning": "ML is a subset of AI that enables computers to learn without explicit programming...",
        "natural language processing": "NLP enables computers to understand and generate human language...",
        "robotics": "Robotics combines AI, engineering, and computer science to create autonomous machines...",
        "data science": "Data science extracts insights from structured and unstructured data...",
        "technology": "Technology encompasses tools, systems, and methods used to solve problems...",
        "business": "Business involves commercial activities aimed at generating profit...",
        "market analysis": "Market analysis evaluates the attractiveness and dynamics of a business market..."
    }
    
    query_lower = query.lower()
    for topic, info in research_data.items():
        if topic in query_lower:
            return f"Research on '{query}': {info}"
    
    return f"Research on '{query}': Limited information available. Recommend consulting specialized databases."


@tool
def analysis_tool(data: str) -> str:
    """Analyze data and provide insights"""
    # Mock analysis implementation
    word_count = len(data.split())
    sentiment_indicators = {
        "positive": ["good", "excellent", "great", "amazing", "outstanding", "beneficial"],
        "negative": ["bad", "poor", "terrible", "awful", "problematic", "concerning"],
        "technical": ["algorithm", "system", "method", "technology", "implementation", "framework"]
    }
    
    data_lower = data.lower()
    analysis_results = {}
    
    for category, indicators in sentiment_indicators.items():
        count = sum(1 for indicator in indicators if indicator in data_lower)
        analysis_results[category] = count
    
    dominant_aspect = max(analysis_results, key=analysis_results.get)
    
    return f"Analysis of data ({word_count} words): Dominant aspect is {dominant_aspect}. " \
           f"Characteristics: {analysis_results}"


@tool
def synthesis_tool(information: str) -> str:
    """Synthesize information into coherent recommendations"""
    # Mock synthesis implementation
    info_points = information.split('. ')
    key_points = [point.strip() for point in info_points if len(point.strip()) > 10]
    
    if len(key_points) >= 3:
        synthesis = f"Based on {len(key_points)} key insights, the synthesis reveals: " \
                   f"Primary theme emerges from the data patterns. " \
                   f"Recommendation: Integrate findings across {len(key_points)} dimensions."
    else:
        synthesis = f"Limited data points ({len(key_points)}) for comprehensive synthesis. " \
                   f"Recommend gathering additional information."
    
    return synthesis


class SpecializedAgent:
    """Base class for specialized agents in the multi-agent system"""
    
    def __init__(self, role: AgentRole, shared_memory: SharedMemory):
        self.role = role
        self.shared_memory = shared_memory
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create agent-specific tools
        self.tools = self._initialize_tools()
        
        # Create agent executor
        self.agent_executor = self._create_agent_executor()
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize tools based on agent specialization"""
        available_tools = {
            "research_tool": research_tool,
            "analysis_tool": analysis_tool,
            "synthesis_tool": synthesis_tool
        }
        
        agent_tools = []
        for tool_name in self.role.tools:
            if tool_name in available_tools:
                agent_tools.append(available_tools[tool_name])
        
        return agent_tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create agent executor with role-specific configuration"""
        prompt = PromptTemplate.from_template(f"""
        You are {self.role.name}, specialized in {self.role.specialization}.
        {self.role.description}
        
        Use the following tools to help with your tasks:
        {{tools}}
        
        Tool names: {{tool_names}}
        
        Always follow this format:
        Question: the input question you must answer
        Thought: think about what to do
        Action: the action to take, should be one of [{{tool_names}}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation as needed)
        Thought: I now know the final answer
        Final Answer: the final answer
        
        Question: {{input}}
        Thought: {{agent_scratchpad}}
        """)
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task with shared context"""
        try:
            # Update shared memory with task start
            self.shared_memory.add_conversation(self.role.name, task, "user")
            
            # Add context to input if provided
            enhanced_input = task
            if context:
                context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])
                enhanced_input = f"{task} [Context: {context_str}]"
            
            # Execute the task
            result = await self.agent_executor.ainvoke({"input": enhanced_input})
            
            # Store result in shared memory
            output = result.get("output", "No output generated")
            self.shared_memory.add_conversation(self.role.name, output, "assistant")
            self.shared_memory.store_agent_output(self.role.name, result)
            
            return {
                "agent": self.role.name,
                "task": task,
                "output": output,
                "success": True,
                "priority": self.role.priority
            }
            
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            self.shared_memory.add_conversation(self.role.name, error_msg, "error")
            
            return {
                "agent": self.role.name,
                "task": task,
                "output": error_msg,
                "success": False,
                "priority": self.role.priority,
                "error": str(e)
            }


class CoordinatorAgent(SpecializedAgent):
    """Coordinator agent that manages other agents"""
    
    def __init__(self, shared_memory: SharedMemory):
        coordinator_role = AgentRole(
            name="CoordinatorAgent",
            description="Manages and coordinates multiple specialized agents to achieve complex objectives",
            specialization="task_coordination",
            tools=["synthesis_tool", "analysis_tool"],
            priority=10  # Highest priority
        )
        super().__init__(coordinator_role, shared_memory)
        self.subordinate_agents: List[SpecializedAgent] = []
    
    def add_subordinate(self, agent: SpecializedAgent):
        """Add a subordinate agent"""
        self.subordinate_agents.append(agent)
        self.shared_memory.log_coordination_event("agent_added", {
            "coordinator": self.role.name,
            "subordinate": agent.role.name,
            "specialization": agent.role.specialization
        })
    
    async def coordinate_task(self, main_task: str) -> Dict[str, Any]:
        """Coordinate a complex task across multiple agents"""
        try:
            coordination_result = {
                "main_task": main_task,
                "coordinator": self.role.name,
                "subordinate_results": [],
                "final_synthesis": "",
                "success": True
            }
            
            # Log coordination start
            self.shared_memory.log_coordination_event("coordination_start", {
                "task": main_task,
                "subordinates": [agent.role.name for agent in self.subordinate_agents]
            })
            
            # Break down task for subordinates
            subtasks = await self._decompose_task(main_task)
            
            # Execute subtasks with subordinates
            subordinate_results = []
            for i, subtask in enumerate(subtasks):
                if i < len(self.subordinate_agents):
                    agent = self.subordinate_agents[i]
                    
                    # Add shared context
                    context = {
                        "main_task": main_task,
                        "subtask_id": i + 1,
                        "total_subtasks": len(subtasks)
                    }
                    
                    result = await agent.execute_task(subtask, context)
                    subordinate_results.append(result)
                    
                    # Update shared context with intermediate results
                    self.shared_memory.update_context(
                        f"subtask_{i+1}_result", 
                        result["output"], 
                        agent.role.name
                    )
            
            coordination_result["subordinate_results"] = subordinate_results
            
            # Synthesize results
            synthesis_input = self._prepare_synthesis_input(subordinate_results)
            synthesis_result = await self.execute_task(
                f"Synthesize the following results into a comprehensive answer for: {main_task}\n\nResults: {synthesis_input}"
            )
            
            coordination_result["final_synthesis"] = synthesis_result["output"]
            
            # Log coordination completion
            self.shared_memory.log_coordination_event("coordination_complete", {
                "task": main_task,
                "subordinate_count": len(subordinate_results),
                "success_rate": sum(1 for r in subordinate_results if r["success"]) / len(subordinate_results)
            })
            
            return coordination_result
            
        except Exception as e:
            return {
                "main_task": main_task,
                "coordinator": self.role.name,
                "subordinate_results": [],
                "final_synthesis": f"Coordination failed: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    async def _decompose_task(self, main_task: str) -> List[str]:
        """Decompose main task into subtasks for subordinates"""
        # Simple task decomposition based on agent specializations
        specializations = [agent.role.specialization for agent in self.subordinate_agents]
        
        decomposition_prompt = f"""
        Break down this task into {len(specializations)} subtasks for agents specialized in: {', '.join(specializations)}
        
        Main task: {main_task}
        
        Provide {len(specializations)} specific subtasks, one for each specialization.
        """
        
        result = await self.execute_task(decomposition_prompt)
        
        # Parse the result into subtasks (simplified)
        subtasks = []
        lines = result["output"].split('\n')
        for line in lines:
            if line.strip() and ('1.' in line or '2.' in line or '3.' in line):
                subtask = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                if subtask:
                    subtasks.append(subtask)
        
        # Fallback if parsing fails
        if not subtasks:
            subtasks = [
                f"Research aspects of: {main_task}",
                f"Analyze implications of: {main_task}",
                f"Provide synthesis for: {main_task}"
            ][:len(self.subordinate_agents)]
        
        return subtasks
    
    def _prepare_synthesis_input(self, subordinate_results: List[Dict[str, Any]]) -> str:
        """Prepare input for final synthesis"""
        synthesis_parts = []
        for result in subordinate_results:
            if result["success"]:
                synthesis_parts.append(f"{result['agent']}: {result['output']}")
            else:
                synthesis_parts.append(f"{result['agent']}: Failed - {result.get('error', 'Unknown error')}")
        
        return " | ".join(synthesis_parts)


class MultiAgentSystem:
    """Complete multi-agent system with coordination capabilities"""
    
    def __init__(self):
        self.shared_memory = SharedMemory()
        self.coordinator = CoordinatorAgent(self.shared_memory)
        self.specialized_agents: List[SpecializedAgent] = []
        
        # Initialize specialized agents
        self._initialize_specialized_agents()
        
        # Multi-agent configuration
        self.system_config = {
            "communication_protocol": "hierarchical",
            "coordinator_agent": "CoordinatorAgent",
            "shared_memory": True,
            "conflict_resolution": "priority_based",
            "agent_coordination": "enabled"
        }
    
    def _initialize_specialized_agents(self):
        """Initialize specialized agents"""
        
        # Research specialist
        research_role = AgentRole(
            name="ResearchAgent",
            description="Specializes in gathering and organizing information from various sources",
            specialization="information_research",
            tools=["research_tool"],
            priority=5
        )
        research_agent = SpecializedAgent(research_role, self.shared_memory)
        
        # Analysis specialist
        analysis_role = AgentRole(
            name="AnalysisAgent", 
            description="Specializes in analyzing data and extracting insights",
            specialization="data_analysis",
            tools=["analysis_tool"],
            priority=6
        )
        analysis_agent = SpecializedAgent(analysis_role, self.shared_memory)
        
        # Synthesis specialist
        synthesis_role = AgentRole(
            name="SynthesisAgent",
            description="Specializes in combining information into coherent recommendations",
            specialization="information_synthesis",
            tools=["synthesis_tool"],
            priority=7
        )
        synthesis_agent = SpecializedAgent(synthesis_role, self.shared_memory)
        
        # Add agents to system
        self.specialized_agents = [research_agent, analysis_agent, synthesis_agent]
        
        # Register with coordinator
        for agent in self.specialized_agents:
            self.coordinator.add_subordinate(agent)
    
    async def process_complex_query(self, query: str) -> Dict[str, Any]:
        """Process a complex query using the multi-agent system"""
        try:
            print(f"🤖 Multi-Agent System Processing: {query}")
            
            # Log query start
            self.shared_memory.log_coordination_event("query_start", {
                "query": query,
                "agent_count": len(self.specialized_agents) + 1
            })
            
            # Coordinate the task
            result = await self.coordinator.coordinate_task(query)
            
            # Add system metadata
            result["system_config"] = self.system_config
            result["coordination_log"] = self.shared_memory.coordination_log
            result["shared_context"] = self.shared_memory.shared_context
            result["conversation_history"] = self.shared_memory.conversations
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "system_config": self.system_config
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "coordinator": self.coordinator.role.name,
            "specialized_agents": [agent.role.name for agent in self.specialized_agents],
            "shared_memory_entries": len(self.shared_memory.conversations),
            "coordination_events": len(self.shared_memory.coordination_log),
            "system_config": self.system_config,
            "agent_specializations": {
                agent.role.name: agent.role.specialization 
                for agent in self.specialized_agents
            }
        }


# Example usage and testing
async def test_multi_agent_system():
    """Test the multi-agent system"""
    system = MultiAgentSystem()
    
    test_queries = [
        "What are the key trends in artificial intelligence and their business implications?",
        "Analyze the current state of machine learning research and provide recommendations for implementation",
        "How can natural language processing technologies be applied to improve customer service?",
        "What are the ethical considerations and practical challenges in deploying AI systems?"
    ]
    
    print("🚀 Testing Multi-Agent LangChain System")
    print("=" * 70)
    
    # Display system status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Coordinator: {status['coordinator']}")
    print(f"   Specialized Agents: {len(status['specialized_agents'])}")
    for agent, spec in status['agent_specializations'].items():
        print(f"     - {agent}: {spec}")
    print(f"   Communication: {status['system_config']['communication_protocol']}")
    
    # Process test queries
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"🔍 Query {i}: {query}")
        print("="*70)
        
        result = await system.process_complex_query(query)
        
        if result["success"]:
            print(f"✅ Coordination successful!")
            print(f"📝 Final Synthesis:")
            print(f"   {result['final_synthesis'][:200]}...")
            
            print(f"\n🤝 Agent Coordination:")
            for sub_result in result["subordinate_results"]:
                status_icon = "✅" if sub_result["success"] else "❌"
                print(f"   {status_icon} {sub_result['agent']}: {sub_result['output'][:100]}...")
            
            print(f"\n📈 Coordination Metrics:")
            print(f"   Subordinate Agents: {len(result['subordinate_results'])}")
            success_rate = sum(1 for r in result['subordinate_results'] if r['success']) / len(result['subordinate_results'])
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Coordination Events: {len(result['coordination_log'])}")
        else:
            print(f"❌ Coordination failed: {result.get('error', 'Unknown error')}")
        
        print("-" * 70)
    
    print(f"\n🎯 Multi-Agent System Test Complete!")
    print(f"Total coordination events: {len(system.shared_memory.coordination_log)}")
    print(f"Total conversations: {len(system.shared_memory.conversations)}")


if __name__ == "__main__":
    asyncio.run(test_multi_agent_system()) 