"""
Mark-1 Agent Integration Framework

Provides seamless integration with popular AI agent frameworks including
LangChain, AutoGPT, CrewAI, and custom agent systems.

Session 15: AutoGPT & Autonomous Agent Integration
Enhanced with autonomous behavior support and goal-oriented task management.

Session 16: CrewAI & Multi-Agent Systems Integration
"""

from .langchain_integration import LangChainIntegration, LangChainAgentAdapter
from .advanced_langchain import (
    AdvancedLangChainIntegration, 
    AdvancedLangChainAgentAdapter,
    LangGraphWorkflow,
    LangGraphNode,
    LangGraphEdge,
    LangGraphStateSchema,
    LangGraphStateManager,
    MultiAgentConfiguration,
    WorkflowComplexity,
    LangGraphNodeType
)
from .autogpt_integration import (
    AutoGPTIntegration,
    AutoGPTAgentAdapter,
    AutonomyLevel,
    GoalType,
    AutonomousGoal,
    MemorySystem,
    AutoGPTAgentInfo,
    AutonomousGoalManager,
    MemoryManager,
    GoalDetector,
    MemorySystemAnalyzer
)
from .crewai_integration import (
    CrewAIIntegration,
    CrewAIAgentAdapter,
    CrewRole,
    CollaborationPattern,
    TaskDelegationStrategy,
    CrewMember,
    CrewTask,
    CrewConfiguration,
    CrewAIAgentDetector,
    RoleBasedTaskDelegator,
    InterAgentCommunicator,
    CollaborativeWorkflowEngine
)
from .base_integration import BaseIntegration, IntegrationError

# Custom Agent Integration Framework (Session 17)
from .custom_integration import (
    CustomAgentIntegration,
    GenericAgentAdapter,
    CustomAgentType,
    IntegrationProtocol,
    AdaptationStrategy,
    CustomIntegrationConfig,
    CustomAgentMetadata,
    IntegrationTemplate,
    CustomAgentSDK,
    GenericAgentDetector,
    DirectCallHandler,
    HTTPRestHandler,
    CLISubprocessHandler,
    WebSocketHandler
)

__all__ = [
    # Base classes
    "BaseIntegration",
    "BaseAgentAdapter", 
    "IntegrationType",
    "AgentCapability",
    "IntegratedAgent",
    "IntegrationError",
    
    # LangChain Integration (Session 13)
    "LangChainIntegration",
    "LangChainAgentAdapter",
    
    # Advanced LangChain Integration (Session 14)
    "AdvancedLangChainIntegration",
    "LangGraphAgentAdapter",
    "ChainType",
    "LangGraphNode",
    "LangGraphWorkflow",
    "LangChainToolRegistry",
    "ChainOrchestrator",
    "MultiAgentLangChainSystem",
    
    # AutoGPT Integration (Session 15)
    "AutoGPTIntegration",
    "AutoGPTAgentAdapter",
    "AutonomyLevel",
    "GoalType",
    "AutonomousGoal",
    "MemorySystem",
    "AutoGPTAgentInfo",
    "AutonomousGoalManager",
    "MemoryManager",
    "GoalDetector",
    "MemorySystemAnalyzer",
    
    # CrewAI Integration (Session 16)
    "CrewAIIntegration",
    "CrewAIAgentAdapter",
    "CrewRole",
    "CollaborationPattern",
    "TaskDelegationStrategy",
    "CrewMember",
    "CrewTask",
    "CrewConfiguration",
    "CrewAIAgentDetector",
    "RoleBasedTaskDelegator",
    "InterAgentCommunicator",
    "CollaborativeWorkflowEngine",
    
    # Custom Agent Integration Framework (Session 17)
    "CustomAgentIntegration",
    "GenericAgentAdapter",
    "CustomAgentType",
    "IntegrationProtocol",
    "AdaptationStrategy",
    "CustomIntegrationConfig",
    "CustomAgentMetadata",
    "IntegrationTemplate",
    "CustomAgentSDK",
    "GenericAgentDetector",
    "DirectCallHandler",
    "HTTPRestHandler",
    "CLISubprocessHandler",
    "WebSocketHandler"
]
