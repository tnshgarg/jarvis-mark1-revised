#!/usr/bin/env python3
"""
Mark-1 Advanced AI Orchestration Module

This module provides advanced orchestration capabilities including:
- Multi-agent coordination and communication
- AI model integration and management
- Advanced workflow orchestration
- Performance optimization and monitoring
- Automation framework and scripting
"""

from .coordinator import AgentCoordinator, MultiAgentOrchestrator
from .model_manager import AIModelManager, ModelRouter
from .workflow_engine import AdvancedWorkflowEngine, WorkflowOptimizer
from .performance_monitor import PerformanceMonitor, SystemOptimizer
from .automation import AutomationFramework, ScriptEngine

__version__ = "1.0.0"
__all__ = [
    "AgentCoordinator",
    "MultiAgentOrchestrator", 
    "AIModelManager",
    "ModelRouter",
    "AdvancedWorkflowEngine",
    "WorkflowOptimizer",
    "PerformanceMonitor",
    "SystemOptimizer",
    "AutomationFramework",
    "ScriptEngine"
] 