"""
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
