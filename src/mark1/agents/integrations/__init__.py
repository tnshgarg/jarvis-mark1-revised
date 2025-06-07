"""
Mark-1 Agent Integration Framework

Provides seamless integration with popular AI agent frameworks including
LangChain, AutoGPT, CrewAI, and custom agent systems.
"""

from .langchain_integration import LangChainIntegration, LangChainAgentAdapter
from .base_integration import BaseIntegration, IntegrationError

__all__ = [
    'LangChainIntegration',
    'LangChainAgentAdapter', 
    'BaseIntegration',
    'IntegrationError'
]
