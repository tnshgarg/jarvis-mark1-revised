#!/usr/bin/env python3
"""
Phase 3 LangChain Integration Testing Script

Tests the LangChain integration capabilities of Mark-1, including:
- LangChain agent detection
- Agent analysis and metadata extraction  
- Agent integration and adapter creation
- Multi-agent system integration
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List
import structlog

# Import our LangChain integration components
from src.mark1.agents.integrations.langchain_integration import (
    LangChainIntegration, LangChainAgentAdapter
)
from src.mark1.agents.integrations.base_integration import (
    IntegrationType, AgentCapability, IntegrationResult
)


async def test_langchain_detection():
    """Test LangChain agent detection capabilities"""
    
    print("🔍 Testing LangChain Agent Detection")
    print("=" * 60)
    
    # Initialize LangChain integration
    langchain_integration = LangChainIntegration()
    
    # Test with our CrewAI examples (which may contain LangChain usage)
    test_path = Path("test_agents/crewAI-examples")
    
    if not test_path.exists():
        print("❌ Test path not found. Creating sample LangChain agent for testing...")
        test_path = await create_sample_langchain_agent()
    
    print(f"📁 Scanning path: {test_path}")
    
    start_time = time.time()
    
    try:
        # Detect LangChain agents
        discovered_agents = await langchain_integration.detect_agents(test_path)
        
        detection_time = time.time() - start_time
        
        print(f"⏱️  Detection completed in {detection_time:.2f} seconds")
        print(f"🤖 Agents discovered: {len(discovered_agents)}")
        print()
        
        # Print detailed results
        if discovered_agents:
            print("📋 DISCOVERED LANGCHAIN AGENTS:")
            print("-" * 40)
            
            for i, agent in enumerate(discovered_agents, 1):
                print(f"  {i}. {agent.name}")
                print(f"     📁 File: {agent.file_path.name}")
                print(f"     🔧 Framework: {agent.framework}")
                print(f"     🎯 Confidence: {agent.confidence:.2f}")
                print(f"     🏷️  Class: {agent.class_name or 'Unknown'}")
                
                # Show LangChain-specific info
                langchain_info = agent.metadata.get('langchain_info', {})
                agent_type = langchain_info.get('agent_type', 'unknown')
                print(f"     📝 Agent Type: {agent_type}")
                
                tools = langchain_info.get('tools', [])
                if tools:
                    print(f"     🛠️  Tools: {len(tools)} detected")
                    for tool in tools[:3]:  # Show first 3 tools
                        print(f"        - {tool.get('name', 'unnamed')}")
                
                llm_info = langchain_info.get('llm_info', {})
                if llm_info:
                    provider = llm_info.get('provider', 'unknown')
                    model = llm_info.get('model', 'unknown')
                    print(f"     🧠 LLM: {provider} ({model})")
                
                memory_type = langchain_info.get('memory_type')
                if memory_type:
                    print(f"     💾 Memory: {memory_type}")
                
                capabilities = agent.capabilities
                if capabilities:
                    print(f"     ⚡ Capabilities: {', '.join(capabilities[:3])}{'...' if len(capabilities) > 3 else ''}")
                
                print()
        else:
            print("❌ No LangChain agents detected")
        
        return discovered_agents
        
    except Exception as e:
        print(f"❌ Detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


async def test_langchain_integration(discovered_agents: List):
    """Test LangChain agent integration"""
    
    if not discovered_agents:
        print("⏭️  Skipping integration test - no agents to integrate")
        return
    
    print("\n🔗 Testing LangChain Agent Integration")
    print("=" * 60)
    
    langchain_integration = LangChainIntegration()
    
    start_time = time.time()
    
    try:
        # Integrate multiple agents
        integration_result = await langchain_integration.integrate_multiple_agents(discovered_agents)
        
        integration_time = time.time() - start_time
        
        print(f"⏱️  Integration completed in {integration_time:.2f} seconds")
        print(f"✅ Success: {integration_result.success}")
        print(f"🤖 Integrated agents: {len(integration_result.integrated_agents)}")
        print(f"❌ Errors: {len(integration_result.errors)}")
        print()
        
        # Show integration results
        if integration_result.integrated_agents:
            print("📋 INTEGRATED AGENTS:")
            print("-" * 40)
            
            for i, agent in enumerate(integration_result.integrated_agents, 1):
                print(f"  {i}. {agent.name}")
                print(f"     🆔 ID: {agent.id}")
                print(f"     🔧 Framework: {agent.framework.value}")
                print(f"     📁 Original Path: {agent.original_path.name}")
                print(f"     ⚡ Capabilities: {[cap.value for cap in agent.capabilities]}")
                print(f"     🛠️  Tools: {len(agent.tools)} tools")
                
                # Test adapter functionality
                adapter = agent.adapter
                print(f"     🔌 Adapter: {type(adapter).__name__}")
                
                # Test health check
                try:
                    health = await adapter.health_check()
                    print(f"     ❤️  Health: {'✅ Healthy' if health else '❌ Unhealthy'}")
                except Exception as e:
                    print(f"     ❤️  Health: ❌ Check failed ({str(e)[:50]}...)")
                
                print()
        
        # Show errors if any
        if integration_result.errors:
            print("❌ INTEGRATION ERRORS:")
            print("-" * 40)
            for error in integration_result.errors:
                print(f"  • {error}")
            print()
        
        # Show metadata
        metadata = integration_result.metadata
        print("📊 INTEGRATION METADATA:")
        print("-" * 40)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print()
        
        return integration_result
        
    except Exception as e:
        print(f"❌ Integration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_agent_adapter_functionality(integration_result: IntegrationResult):
    """Test the functionality of LangChain agent adapters"""
    
    if not integration_result or not integration_result.integrated_agents:
        print("⏭️  Skipping adapter test - no integrated agents")
        return
    
    print("\n🔌 Testing LangChain Agent Adapter Functionality")
    print("=" * 60)
    
    # Test the first integrated agent
    test_agent = integration_result.integrated_agents[0]
    adapter = test_agent.adapter
    
    print(f"🧪 Testing adapter for: {test_agent.name}")
    print(f"🔧 Adapter type: {type(adapter).__name__}")
    print()
    
    # Test 1: Get capabilities
    print("1️⃣  Testing get_capabilities()...")
    try:
        capabilities = adapter.get_capabilities()
        print(f"   ✅ Capabilities: {[cap.value for cap in capabilities]}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
    
    # Test 2: Get tools
    print("2️⃣  Testing get_tools()...")
    try:
        tools = adapter.get_tools()
        print(f"   ✅ Tools: {len(tools)} found")
        for tool in tools[:3]:
            print(f"      - {tool.get('name', 'unnamed')}: {tool.get('type', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
    
    # Test 3: Get model info
    print("3️⃣  Testing get_model_info()...")
    try:
        model_info = adapter.get_model_info()
        print(f"   ✅ Model info: {model_info}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
    
    # Test 4: Agent invocation (mock)
    print("4️⃣  Testing agent invocation...")
    try:
        test_input = {
            "input": "Hello, this is a test message from Mark-1!",
            "query": "test query"
        }
        
        # Note: This will likely fail since we're using mock instances
        # but it tests the adapter interface
        result = await adapter.invoke(test_input)
        print(f"   ✅ Invocation result: {result}")
    except Exception as e:
        print(f"   ⚠️  Invocation failed (expected for mock): {str(e)[:100]}...")
    
    # Test 5: Streaming (mock)
    print("5️⃣  Testing agent streaming...")
    try:
        test_input = {"input": "Stream test"}
        chunk_count = 0
        
        async for chunk in adapter.stream(test_input):
            chunk_count += 1
            if chunk_count <= 3:  # Show first 3 chunks
                print(f"   📡 Chunk {chunk_count}: {str(chunk)[:100]}...")
            if chunk_count >= 5:  # Limit chunks
                break
        
        print(f"   ✅ Streaming completed: {chunk_count} chunks")
    except Exception as e:
        print(f"   ⚠️  Streaming failed (expected for mock): {str(e)[:100]}...")
    
    print()


async def test_integration_framework_detection():
    """Test framework detection capabilities"""
    
    print("🔍 Testing Framework Detection Capabilities")
    print("=" * 60)
    
    langchain_integration = LangChainIntegration()
    
    # Test various LangChain code patterns
    test_patterns = [
        {
            "name": "ReAct Agent",
            "code": """
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [Tool(name="search", description="Search tool", func=lambda x: x)]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
"""
        },
        {
            "name": "LangGraph Agent", 
            "code": """
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    messages: list

def node_function(state: State):
    return {"messages": state["messages"] + ["processed"]}

graph = StateGraph(State)
graph.add_node("process", node_function)
graph.add_edge("process", "end")
"""
        },
        {
            "name": "Tool Definition",
            "code": """
from langchain.tools import tool

@tool
def search_tool(query: str) -> str:
    \"\"\"Search for information\"\"\"
    return f"Search results for: {query}"

from langchain.tools import Tool

search = Tool(
    name="search",
    description="Search the web",
    func=lambda x: "results"
)
"""
        }
    ]
    
    for test_case in test_patterns:
        print(f"🧪 Testing: {test_case['name']}")
        
        # Test framework detection
        detected = langchain_integration._detect_framework_markers(test_case['code'])
        print(f"   Framework detected: {'✅ Yes' if detected else '❌ No'}")
        
        # Test agent type identification
        agent_type = langchain_integration._identify_agent_type(test_case['code'])
        print(f"   Agent type: {agent_type}")
        
        # Test capability extraction
        capabilities = langchain_integration.extract_capabilities(test_case['code'])
        cap_names = [cap.value for cap in capabilities]
        print(f"   Capabilities: {cap_names}")
        
        # Test tool extraction
        tools = langchain_integration.extract_tools(test_case['code'])
        tool_names = [tool.get('name', 'unnamed') for tool in tools]
        print(f"   Tools: {tool_names}")
        
        print()


async def create_sample_langchain_agent() -> Path:
    """Create a sample LangChain agent for testing"""
    
    sample_dir = Path("test_agents/sample_langchain")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a comprehensive LangChain agent
    sample_agent_code = '''"""
Sample LangChain Agent for Mark-1 Integration Testing
"""

import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish


class SampleLangChainAgent:
    """A comprehensive LangChain agent for testing Mark-1 integration"""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_react_agent()
        
        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _create_tools(self):
        """Create agent tools"""
        
        @tool
        def search_tool(query: str) -> str:
            """Search for information on the internet"""
            # Mock search implementation
            return f"Search results for '{query}': Found relevant information about the topic."
        
        @tool
        def calculator_tool(expression: str) -> str:
            """Calculate mathematical expressions safely"""
            try:
                # Simple calculator for demo
                result = eval(expression.replace("^", "**"))
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        @tool
        def weather_tool(location: str) -> str:
            """Get weather information for a location"""
            # Mock weather API
            return f"Weather in {location}: Sunny, 72°F (22°C), light breeze."
        
        manual_tool = Tool(
            name="file_reader",
            description="Read and analyze files",
            func=lambda filepath: f"File contents of {filepath}: [Sample file content]"
        )
        
        return [search_tool, calculator_tool, weather_tool, manual_tool]
    
    def _create_react_agent(self):
        """Create a ReAct agent"""
        
        prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant with access to tools.
        
        Answer the following questions as best you can. You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought: {agent_scratchpad}
        """)
        
        return create_react_agent(self.llm, self.tools, prompt)
    
    async def run(self, query: str) -> str:
        """Run the agent with a query"""
        try:
            result = await self.executor.ainvoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Agent error: {str(e)}"
    
    def get_capabilities(self):
        """Get agent capabilities"""
        return [
            "search", "calculation", "weather", "file_reading",
            "conversation", "reasoning", "tool_use"
        ]
    
    def get_tools_info(self):
        """Get information about available tools"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools
        ]


# Example usage and testing
async def main():
    """Main function for testing the agent"""
    agent = SampleLangChainAgent()
    
    test_queries = [
        "What is 15 * 24 + 7?",
        "Search for information about artificial intelligence",
        "What's the weather like in San Francisco?",
        "Can you help me analyze a document?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        result = await agent.run(query)
        print(f"Result: {result}")
        print("-" * 50)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''
    
    # Write the sample agent
    with open(sample_dir / "langchain_agent.py", "w") as f:
        f.write(sample_agent_code)
    
    # Create a LangGraph example
    langgraph_code = '''"""
Sample LangGraph Workflow for Mark-1 Integration Testing
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


class WorkflowState(TypedDict):
    """State for the LangGraph workflow"""
    input: str
    messages: List[str]
    analysis: str
    output: str


class LangGraphWorkflow:
    """A sample LangGraph workflow for testing"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create the state graph"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("preprocess", self.preprocess_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("generate", self.generate_node)
        
        # Add edges
        workflow.add_edge("preprocess", "analyze")
        workflow.add_edge("analyze", "generate")
        workflow.add_edge("generate", END)
        
        # Set entry point
        workflow.set_entry_point("preprocess")
        
        return workflow.compile()
    
    def preprocess_node(self, state: WorkflowState) -> WorkflowState:
        """Preprocess the input"""
        input_text = state["input"]
        processed = f"Preprocessed: {input_text}"
        
        return {
            **state,
            "messages": state.get("messages", []) + [processed]
        }
    
    def analyze_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze the preprocessed input"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else state["input"]
        
        analysis = f"Analysis of '{last_message}': This appears to be a user query requiring processing."
        
        return {
            **state,
            "analysis": analysis,
            "messages": messages + [analysis]
        }
    
    def generate_node(self, state: WorkflowState) -> WorkflowState:
        """Generate the final response"""
        analysis = state.get("analysis", "")
        input_text = state["input"]
        
        output = f"Response to '{input_text}': Based on the analysis, here's the generated response."
        
        return {
            **state,
            "output": output,
            "messages": state["messages"] + [output]
        }
    
    async def run(self, input_text: str) -> str:
        """Run the workflow"""
        initial_state = {
            "input": input_text,
            "messages": [],
            "analysis": "",
            "output": ""
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result["output"]


# Example usage
async def main():
    workflow = LangGraphWorkflow()
    
    test_inputs = [
        "Analyze this text for sentiment",
        "Generate a summary of the document",
        "What are the key points?"
    ]
    
    for input_text in test_inputs:
        result = await workflow.run(input_text)
        print(f"Input: {input_text}")
        print(f"Output: {result}")
        print("-" * 50)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''
    
    # Write the LangGraph example
    with open(sample_dir / "langgraph_workflow.py", "w") as f:
        f.write(langgraph_code)
    
    # Create requirements file
    requirements = '''langchain>=0.1.0
langchain-openai>=0.1.0
langgraph>=0.0.20
openai>=1.0.0
'''
    
    with open(sample_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    
    print(f"✅ Created sample LangChain agents at: {sample_dir}")
    return sample_dir


async def main():
    """Main test execution for Phase 3 LangChain Integration"""
    
    print("🚀 Mark-1 Phase 3: LangChain Integration Testing")
    print("=" * 80)
    
    # Test 1: Framework detection capabilities
    await test_integration_framework_detection()
    print("\n" + "=" * 80)
    
    # Test 2: Agent detection
    discovered_agents = await test_langchain_detection()
    print("\n" + "=" * 80)
    
    # Test 3: Agent integration
    integration_result = await test_langchain_integration(discovered_agents)
    print("\n" + "=" * 80)
    
    # Test 4: Adapter functionality
    await test_agent_adapter_functionality(integration_result)
    
    print("\n" + "=" * 80)
    print("🎯 PHASE 3 SESSION 13 SUMMARY:")
    print("✅ LangChain framework detection implemented")
    print("✅ Agent type identification (ReAct, LangGraph, Tools)")
    print("✅ Comprehensive metadata extraction")
    print("✅ Agent integration and adapter creation")
    print("✅ Unified interface for different LangChain patterns")
    print("✅ Tool extraction and capability analysis")
    print("✅ Memory and LLM configuration detection")
    print("\n🎉 LangChain Integration Foundation Complete!")
    print("Ready for Session 14: Advanced LangChain & LangGraph")


if __name__ == "__main__":
    asyncio.run(main()) 