"""
Base Integration Framework for Mark-1 Agent Orchestrator

Provides abstract base classes and common functionality for integrating
different AI agent frameworks into the Mark-1 ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.utils.exceptions import IntegrationError
from mark1.agents.discovery import DiscoveredAgent


class IntegrationType(Enum):
    """Types of agent integrations supported"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    CREWAI = "crewai"
    CUSTOM = "custom"
    OPENAI_ASSISTANT = "openai_assistant"
    ANTHROPIC_CLAUDE = "anthropic_claude"


class AgentCapability(Enum):
    """Standard agent capabilities that can be extracted"""
    CHAT = "chat"
    COMPLETION = "completion"
    SEARCH = "search"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    PLANNING = "planning"
    EXECUTION = "execution"
    MEMORY = "memory"
    TOOL_USE = "tool_use"
    MULTIMODAL = "multimodal"


@dataclass
class IntegratedAgent:
    """Represents an agent that has been integrated into Mark-1"""
    id: str
    name: str
    framework: IntegrationType
    original_path: Path
    adapter: 'BaseAgentAdapter'
    capabilities: List[AgentCapability]
    metadata: Dict[str, Any]
    tools: List[str] = None
    model_info: Dict[str, Any] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.model_info is None:
            self.model_info = {}
        if self.config is None:
            self.config = {}


@dataclass
class IntegrationResult:
    """Result of an integration operation"""
    success: bool
    integrated_agents: List[IntegratedAgent]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.integrated_agents is None:
            self.integrated_agents = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseAgentAdapter(ABC):
    """
    Abstract base class for agent adapters
    
    Agent adapters provide a unified interface for different agent frameworks,
    allowing Mark-1 to interact with agents regardless of their original framework.
    """
    
    def __init__(self, agent_instance: Any, metadata: Dict[str, Any]):
        self.agent_instance = agent_instance
        self.metadata = metadata
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with input data
        
        Args:
            input_data: Input parameters for the agent
            
        Returns:
            Agent response in standardized format
        """
        pass
    
    @abstractmethod
    async def stream(self, input_data: Dict[str, Any]):
        """
        Stream responses from the agent
        
        Args:
            input_data: Input parameters for the agent
            
        Yields:
            Streaming response chunks
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Get the capabilities of this agent
        
        Returns:
            List of agent capabilities
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get the tools available to this agent
        
        Returns:
            List of tool descriptions
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model used by this agent
        
        Returns:
            Model information dictionary
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the agent is healthy and responsive
        
        Returns:
            True if agent is healthy, False otherwise
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get adapter metadata"""
        return self.metadata.copy()
    
    def update_metadata(self, new_metadata: Dict[str, Any]):
        """Update adapter metadata"""
        self.metadata.update(new_metadata)


class BaseIntegration(ABC):
    """
    Abstract base class for framework integrations
    
    Integrations are responsible for discovering, analyzing, and adapting
    agents from specific frameworks into the Mark-1 ecosystem.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        self.integration_type = self.get_integration_type()
        
    @abstractmethod
    def get_integration_type(self) -> IntegrationType:
        """
        Get the type of integration this class handles
        
        Returns:
            Integration type enum value
        """
        pass
    
    @abstractmethod
    async def detect_agents(self, scan_path: Path) -> List[DiscoveredAgent]:
        """
        Detect agents of this framework type in the given path
        
        Args:
            scan_path: Path to scan for agents
            
        Returns:
            List of discovered agents
        """
        pass
    
    @abstractmethod
    async def integrate_agent(self, discovered_agent: DiscoveredAgent) -> IntegratedAgent:
        """
        Integrate a discovered agent into the Mark-1 system
        
        Args:
            discovered_agent: Agent discovered during scanning
            
        Returns:
            Integrated agent with adapter
        """
        pass
    
    @abstractmethod
    def create_adapter(self, agent_instance: Any, metadata: Dict[str, Any]) -> BaseAgentAdapter:
        """
        Create an adapter for the given agent instance
        
        Args:
            agent_instance: The framework-specific agent instance
            metadata: Agent metadata
            
        Returns:
            Agent adapter instance
        """
        pass
    
    @abstractmethod
    def extract_capabilities(self, agent_code: str, agent_instance: Any = None) -> List[AgentCapability]:
        """
        Extract capabilities from agent code or instance
        
        Args:
            agent_code: Source code of the agent
            agent_instance: Optional agent instance
            
        Returns:
            List of detected capabilities
        """
        pass
    
    @abstractmethod
    def extract_tools(self, agent_code: str, agent_instance: Any = None) -> List[Dict[str, Any]]:
        """
        Extract tools used by the agent
        
        Args:
            agent_code: Source code of the agent
            agent_instance: Optional agent instance
            
        Returns:
            List of tool descriptions
        """
        pass
    
    async def integrate_multiple_agents(self, discovered_agents: List[DiscoveredAgent]) -> IntegrationResult:
        """
        Integrate multiple discovered agents
        
        Args:
            discovered_agents: List of discovered agents to integrate
            
        Returns:
            Integration result with success/failure information
        """
        result = IntegrationResult(
            success=True,
            integrated_agents=[],
            errors=[],
            warnings=[],
            metadata={"integration_type": self.integration_type.value}
        )
        
        for discovered_agent in discovered_agents:
            try:
                integrated_agent = await self.integrate_agent(discovered_agent)
                result.integrated_agents.append(integrated_agent)
                
                self.logger.info("Agent integrated successfully",
                               agent_name=integrated_agent.name,
                               framework=integrated_agent.framework.value)
                
            except Exception as e:
                error_msg = f"Failed to integrate agent {discovered_agent.name}: {str(e)}"
                result.errors.append(error_msg)
                result.success = False
                
                self.logger.error("Agent integration failed",
                                agent_name=discovered_agent.name,
                                error=str(e))
        
        result.metadata.update({
            "total_agents": len(discovered_agents),
            "successful_integrations": len(result.integrated_agents),
            "failed_integrations": len(result.errors)
        })
        
        return result
    
    def validate_agent_structure(self, agent_code: str) -> List[str]:
        """
        Validate that the agent code has the expected structure for this framework
        
        Args:
            agent_code: Source code to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation - can be overridden by subclasses
        if not agent_code.strip():
            errors.append("Agent code is empty")
        
        return errors
    
    def supports_agent_type(self, agent_path: Path) -> bool:
        """
        Check if this integration supports the given agent type
        
        Args:
            agent_path: Path to the agent
            
        Returns:
            True if this integration can handle the agent
        """
        try:
            with open(agent_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._detect_framework_markers(content)
        except Exception:
            return False
    
    @abstractmethod
    def _detect_framework_markers(self, content: str) -> bool:
        """
        Detect framework-specific markers in the code
        
        Args:
            content: Source code content
            
        Returns:
            True if framework markers are found
        """
        pass
    
    def get_integration_info(self) -> Dict[str, Any]:
        """
        Get information about this integration
        
        Returns:
            Integration information dictionary
        """
        return {
            "type": self.integration_type.value,
            "class_name": self.__class__.__name__,
            "supported_capabilities": [cap.value for cap in AgentCapability],
            "version": "1.0.0"
        } 