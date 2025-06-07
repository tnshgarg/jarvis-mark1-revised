"""
Advanced LangGraph Workflow Example for Mark-1 Session 14 Testing

This example demonstrates complex LangGraph features:
- Advanced state management
- Conditional routing
- Parallel execution
- Multi-agent coordination
- Tool ecosystem integration
"""

import asyncio
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate


class AdvancedWorkflowState(TypedDict):
    """Complex state schema for advanced workflow"""
    input: str
    messages: List[str]
    analysis_results: Dict[str, Any]
    processing_stage: str
    confidence_score: float
    route_decision: str
    parallel_results: List[Dict[str, Any]]
    final_output: str
    error_log: List[str]
    execution_trace: List[Dict[str, Any]]


@tool
def sentiment_analysis_tool(text: str) -> Dict[str, Any]:
    """Analyze sentiment of the given text"""
    # Mock sentiment analysis
    words = text.lower().split()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.9, 0.5 + (positive_count * 0.1))
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.9, 0.5 + (negative_count * 0.1))
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "details": {
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_words": len(words)
        }
    }


@tool
def entity_extraction_tool(text: str) -> Dict[str, Any]:
    """Extract entities from the given text"""
    # Mock entity extraction
    import re
    
    # Simple patterns for demonstration
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    entities = {
        "emails": re.findall(email_pattern, text),
        "phones": re.findall(phone_pattern, text),
        "urls": re.findall(url_pattern, text),
        "mentions": re.findall(r'@\w+', text),
        "hashtags": re.findall(r'#\w+', text)
    }
    
    return {
        "entities": entities,
        "entity_count": sum(len(v) for v in entities.values()),
        "text_length": len(text)
    }


@tool
def content_classification_tool(text: str) -> Dict[str, Any]:
    """Classify the content type and topic"""
    # Mock classification
    keywords = {
        "technology": ["ai", "machine learning", "software", "computer", "programming", "tech"],
        "business": ["market", "sales", "revenue", "profit", "company", "business"],
        "science": ["research", "study", "experiment", "data", "analysis", "scientific"],
        "entertainment": ["movie", "music", "game", "fun", "entertainment", "show"],
        "news": ["breaking", "reported", "according", "news", "update", "latest"]
    }
    
    text_lower = text.lower()
    scores = {}
    
    for category, words in keywords.items():
        score = sum(1 for word in words if word in text_lower)
        scores[category] = score / len(words)  # Normalize
    
    top_category = max(scores, key=scores.get) if scores else "other"
    confidence = scores.get(top_category, 0)
    
    return {
        "category": top_category,
        "confidence": confidence,
        "all_scores": scores,
        "classification_method": "keyword_based"
    }


class AdvancedLangGraphWorkflow:
    """
    Complex LangGraph workflow demonstrating advanced features
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Create specialized agent tools
        self.tools = [
            sentiment_analysis_tool,
            entity_extraction_tool,
            content_classification_tool
        ]
        
        # Create the workflow graph
        self.graph = self._create_advanced_graph()
        
        # Multi-agent configuration
        self.multi_agent_config = {
            "coordinator_agent": "main_coordinator",
            "communication_protocol": "hierarchical",
            "shared_memory": True,
            "agents": {
                "sentiment_agent": "sentiment_analysis",
                "entity_agent": "entity_extraction", 
                "classification_agent": "content_classification"
            }
        }
    
    def _create_advanced_graph(self):
        """Create complex workflow with conditional routing and parallel execution"""
        workflow = StateGraph(AdvancedWorkflowState)
        
        # Add nodes for different processing stages
        workflow.add_node("initial_processing", self.initial_processing_node)
        workflow.add_node("content_analysis", self.content_analysis_node)
        workflow.add_node("route_decision", self.route_decision_node)
        workflow.add_node("simple_processing", self.simple_processing_node)
        workflow.add_node("complex_processing", self.complex_processing_node)
        workflow.add_node("parallel_analysis", self.parallel_analysis_node)
        workflow.add_node("result_aggregation", self.result_aggregation_node)
        workflow.add_node("quality_check", self.quality_check_node)
        workflow.add_node("final_output", self.final_output_node)
        
        # Add sequential edges
        workflow.add_edge("initial_processing", "content_analysis")
        workflow.add_edge("content_analysis", "route_decision")
        
        # Add conditional edges for routing
        workflow.add_conditional_edges(
            "route_decision",
            self.should_use_complex_processing,
            {
                "simple": "simple_processing",
                "complex": "complex_processing",
                "parallel": "parallel_analysis"
            }
        )
        
        # Connect processing paths
        workflow.add_edge("simple_processing", "quality_check")
        workflow.add_edge("complex_processing", "result_aggregation")
        workflow.add_edge("parallel_analysis", "result_aggregation")
        workflow.add_edge("result_aggregation", "quality_check")
        workflow.add_edge("quality_check", "final_output")
        workflow.add_edge("final_output", END)
        
        # Set entry point
        workflow.set_entry_point("initial_processing")
        
        return workflow.compile()
    
    def initial_processing_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Initial processing and state setup"""
        input_text = state["input"]
        
        # Initialize state
        processed_state = state.copy()
        processed_state.update({
            "messages": [f"Starting processing for input: {input_text[:50]}..."],
            "processing_stage": "initial",
            "confidence_score": 0.0,
            "analysis_results": {},
            "parallel_results": [],
            "error_log": [],
            "execution_trace": [{
                "stage": "initial_processing",
                "timestamp": __import__('time').time(),
                "action": "setup_state"
            }]
        })
        
        return processed_state
    
    def content_analysis_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Perform basic content analysis"""
        input_text = state["input"]
        
        # Basic analysis
        word_count = len(input_text.split())
        char_count = len(input_text)
        complexity_score = min(1.0, (word_count / 100) + (char_count / 1000))
        
        updated_state = state.copy()
        updated_state["analysis_results"] = {
            "word_count": word_count,
            "char_count": char_count,
            "complexity_score": complexity_score
        }
        updated_state["processing_stage"] = "analysis"
        updated_state["messages"].append(f"Analysis complete: {word_count} words, complexity: {complexity_score:.2f}")
        updated_state["execution_trace"].append({
            "stage": "content_analysis",
            "timestamp": __import__('time').time(),
            "action": "basic_analysis",
            "results": updated_state["analysis_results"]
        })
        
        return updated_state
    
    def route_decision_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Decide routing based on content complexity"""
        complexity = state["analysis_results"].get("complexity_score", 0)
        word_count = state["analysis_results"].get("word_count", 0)
        
        # Decision logic
        if complexity > 0.7 or word_count > 200:
            route = "complex"
        elif word_count > 50:
            route = "parallel"
        else:
            route = "simple"
        
        updated_state = state.copy()
        updated_state["route_decision"] = route
        updated_state["processing_stage"] = "routing"
        updated_state["messages"].append(f"Routing decision: {route} processing")
        updated_state["execution_trace"].append({
            "stage": "route_decision",
            "timestamp": __import__('time').time(),
            "action": "route_selection",
            "decision": route,
            "factors": {
                "complexity": complexity,
                "word_count": word_count
            }
        })
        
        return updated_state
    
    def should_use_complex_processing(self, state: AdvancedWorkflowState) -> Literal["simple", "complex", "parallel"]:
        """Conditional function for routing"""
        return state["route_decision"]
    
    def simple_processing_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Simple processing path"""
        input_text = state["input"]
        
        # Simple processing - just basic sentiment
        sentiment_result = sentiment_analysis_tool.invoke(input_text)
        
        updated_state = state.copy()
        updated_state["analysis_results"]["sentiment"] = sentiment_result
        updated_state["confidence_score"] = sentiment_result["confidence"]
        updated_state["processing_stage"] = "simple_complete"
        updated_state["messages"].append("Simple processing completed")
        updated_state["execution_trace"].append({
            "stage": "simple_processing",
            "timestamp": __import__('time').time(),
            "action": "sentiment_analysis",
            "results": sentiment_result
        })
        
        return updated_state
    
    def complex_processing_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Complex processing path with multiple analyses"""
        input_text = state["input"]
        
        # Perform multiple analyses
        sentiment_result = sentiment_analysis_tool.invoke(input_text)
        entity_result = entity_extraction_tool.invoke(input_text)
        classification_result = content_classification_tool.invoke(input_text)
        
        # Aggregate confidence
        avg_confidence = (
            sentiment_result["confidence"] + 
            classification_result["confidence"]
        ) / 2
        
        updated_state = state.copy()
        updated_state["analysis_results"].update({
            "sentiment": sentiment_result,
            "entities": entity_result,
            "classification": classification_result
        })
        updated_state["confidence_score"] = avg_confidence
        updated_state["processing_stage"] = "complex_complete"
        updated_state["messages"].append("Complex processing completed")
        updated_state["execution_trace"].append({
            "stage": "complex_processing",
            "timestamp": __import__('time').time(),
            "action": "multi_analysis",
            "results": {
                "sentiment": sentiment_result,
                "entities": entity_result,
                "classification": classification_result
            }
        })
        
        return updated_state
    
    def parallel_analysis_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Parallel analysis simulation"""
        input_text = state["input"]
        
        # Simulate parallel processing (in real implementation, this would be truly parallel)
        analyses = []
        
        # Sentiment analysis
        sentiment_result = sentiment_analysis_tool.invoke(input_text)
        analyses.append({
            "type": "sentiment",
            "result": sentiment_result,
            "processing_time": 0.1
        })
        
        # Entity extraction
        entity_result = entity_extraction_tool.invoke(input_text)
        analyses.append({
            "type": "entities",
            "result": entity_result,
            "processing_time": 0.15
        })
        
        # Classification
        classification_result = content_classification_tool.invoke(input_text)
        analyses.append({
            "type": "classification",
            "result": classification_result,
            "processing_time": 0.12
        })
        
        updated_state = state.copy()
        updated_state["parallel_results"] = analyses
        updated_state["processing_stage"] = "parallel_complete"
        updated_state["messages"].append(f"Parallel analysis completed: {len(analyses)} analyses")
        updated_state["execution_trace"].append({
            "stage": "parallel_analysis",
            "timestamp": __import__('time').time(),
            "action": "parallel_execution",
            "analysis_count": len(analyses)
        })
        
        return updated_state
    
    def result_aggregation_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Aggregate results from complex or parallel processing"""
        
        if state["processing_stage"] == "parallel_complete":
            # Aggregate parallel results
            aggregated = {}
            total_confidence = 0
            
            for analysis in state["parallel_results"]:
                analysis_type = analysis["type"]
                result = analysis["result"]
                
                aggregated[analysis_type] = result
                
                # Extract confidence if available
                if "confidence" in result:
                    total_confidence += result["confidence"]
            
            avg_confidence = total_confidence / len(state["parallel_results"])
            
            updated_state = state.copy()
            updated_state["analysis_results"].update(aggregated)
            updated_state["confidence_score"] = avg_confidence
        
        else:
            # Complex processing already has aggregated results
            updated_state = state.copy()
        
        updated_state["processing_stage"] = "aggregated"
        updated_state["messages"].append("Results aggregated successfully")
        updated_state["execution_trace"].append({
            "stage": "result_aggregation",
            "timestamp": __import__('time').time(),
            "action": "aggregate_results",
            "confidence": updated_state["confidence_score"]
        })
        
        return updated_state
    
    def quality_check_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Quality check and validation"""
        confidence = state["confidence_score"]
        analysis_results = state["analysis_results"]
        
        # Quality metrics
        completeness = len(analysis_results) / 4  # Expect up to 4 types of analysis
        quality_score = (confidence + completeness) / 2
        
        # Check for potential issues
        issues = []
        if confidence < 0.5:
            issues.append("Low confidence in analysis")
        if not analysis_results:
            issues.append("No analysis results found")
        
        updated_state = state.copy()
        updated_state["analysis_results"]["quality_metrics"] = {
            "confidence": confidence,
            "completeness": completeness,
            "quality_score": quality_score,
            "issues": issues
        }
        updated_state["processing_stage"] = "quality_checked"
        updated_state["messages"].append(f"Quality check: score {quality_score:.2f}")
        
        if issues:
            updated_state["error_log"].extend(issues)
        
        updated_state["execution_trace"].append({
            "stage": "quality_check",
            "timestamp": __import__('time').time(),
            "action": "quality_validation",
            "quality_score": quality_score,
            "issues": issues
        })
        
        return updated_state
    
    def final_output_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Generate final output"""
        analysis_results = state["analysis_results"]
        confidence = state["confidence_score"]
        
        # Create comprehensive output
        output_sections = []
        
        if "sentiment" in analysis_results:
            sentiment = analysis_results["sentiment"]
            output_sections.append(f"Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        
        if "classification" in analysis_results:
            classification = analysis_results["classification"]
            output_sections.append(f"Category: {classification['category']} (confidence: {classification['confidence']:.2f})")
        
        if "entities" in analysis_results:
            entities = analysis_results["entities"]
            entity_count = entities.get("entity_count", 0)
            output_sections.append(f"Entities found: {entity_count}")
        
        quality_metrics = analysis_results.get("quality_metrics", {})
        quality_score = quality_metrics.get("quality_score", 0)
        output_sections.append(f"Analysis quality: {quality_score:.2f}")
        
        final_output = " | ".join(output_sections)
        
        updated_state = state.copy()
        updated_state["final_output"] = final_output
        updated_state["processing_stage"] = "complete"
        updated_state["messages"].append("Final output generated")
        updated_state["execution_trace"].append({
            "stage": "final_output",
            "timestamp": __import__('time').time(),
            "action": "output_generation",
            "output_length": len(final_output)
        })
        
        return updated_state
    
    async def run(self, input_text: str) -> Dict[str, Any]:
        """Run the advanced workflow"""
        initial_state = {
            "input": input_text,
            "messages": [],
            "analysis_results": {},
            "processing_stage": "started",
            "confidence_score": 0.0,
            "route_decision": "",
            "parallel_results": [],
            "final_output": "",
            "error_log": [],
            "execution_trace": []
        }
        
        try:
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "output": result["final_output"],
                "confidence": result["confidence_score"],
                "processing_route": result["route_decision"],
                "stage": result["processing_stage"],
                "analysis_results": result["analysis_results"],
                "execution_trace": result["execution_trace"],
                "messages": result["messages"],
                "errors": result["error_log"],
                "workflow_complexity": "advanced"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": None,
                "workflow_complexity": "advanced"
            }
    
    def get_workflow_metadata(self) -> Dict[str, Any]:
        """Get metadata about this workflow"""
        return {
            "name": "AdvancedLangGraphWorkflow",
            "complexity": "advanced",
            "features": [
                "conditional_routing",
                "parallel_processing", 
                "state_management",
                "quality_checking",
                "error_handling",
                "execution_tracing"
            ],
            "nodes": [
                "initial_processing",
                "content_analysis", 
                "route_decision",
                "simple_processing",
                "complex_processing",
                "parallel_analysis",
                "result_aggregation",
                "quality_check",
                "final_output"
            ],
            "tools": [tool.name for tool in self.tools],
            "multi_agent_config": self.multi_agent_config,
            "state_schema": {
                "input": "str",
                "messages": "List[str]",
                "analysis_results": "Dict[str, Any]",
                "processing_stage": "str",
                "confidence_score": "float",
                "route_decision": "str",
                "parallel_results": "List[Dict[str, Any]]",
                "final_output": "str",
                "error_log": "List[str]",
                "execution_trace": "List[Dict[str, Any]]"
            }
        }


# Example usage and testing
async def main():
    """Test the advanced workflow"""
    workflow = AdvancedLangGraphWorkflow()
    
    test_inputs = [
        "This is a simple test message.",
        "I am absolutely thrilled with the amazing performance of this new AI system! It's fantastic and works perfectly for our business needs. Contact us at support@company.com for more information.",
        "The latest research study published in the journal of artificial intelligence demonstrates significant improvements in machine learning algorithms. The experiment involved complex data analysis and scientific methodology to validate the hypothesis.",
        "Breaking news: The technology company reported record revenue this quarter, with profits exceeding market expectations. According to industry analysts, this positive trend reflects strong business fundamentals."
    ]
    
    print("🧪 Testing Advanced LangGraph Workflow")
    print("=" * 60)
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n🔬 Test {i}: {input_text[:50]}...")
        
        result = await workflow.run(input_text)
        
        if result["success"]:
            print(f"✅ Success!")
            print(f"   Route: {result['processing_route']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Output: {result['output']}")
            print(f"   Stages: {len(result['execution_trace'])} execution steps")
        else:
            print(f"❌ Failed: {result['error']}")
        
        print("-" * 40)
    
    # Display workflow metadata
    metadata = workflow.get_workflow_metadata()
    print(f"\n📊 Workflow Metadata:")
    print(f"   Name: {metadata['name']}")
    print(f"   Complexity: {metadata['complexity']}")
    print(f"   Features: {', '.join(metadata['features'])}")
    print(f"   Nodes: {len(metadata['nodes'])}")
    print(f"   Tools: {len(metadata['tools'])}")


if __name__ == "__main__":
    asyncio.run(main()) 