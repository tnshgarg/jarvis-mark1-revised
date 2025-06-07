"""
Example Custom Agent for Mark-1 Orchestrator

This is a simple example of how to create a custom agent
that can be discovered and orchestrated by Mark-1.
"""

import asyncio
from typing import Dict, Any, List


class ExampleAgent:
    """
    Example agent that demonstrates basic Mark-1 integration
    """
    
    def __init__(self):
        self.name = "Example Agent"
        self.description = "A simple example agent for demonstration"
        self.capabilities = ["text_processing", "data_analysis"]
        self.version = "1.0.0"
    
    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task with the given context
        
        Args:
            task: Task description
            context: Additional context for the task
            
        Returns:
            Dict containing the task result
        """
        # Simulate some processing
        await asyncio.sleep(1)
        
        return {
            "status": "completed",
            "result": f"Processed task: {task}",
            "agent": self.name,
            "processing_time": 1.0
        }
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return self.capabilities
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "version": self.version,
            "type": "custom"
        }


# Mark-1 will automatically discover this agent
agent = ExampleAgent()
