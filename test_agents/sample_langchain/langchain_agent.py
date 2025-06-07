"""
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
