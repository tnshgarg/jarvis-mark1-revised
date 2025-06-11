#!/usr/bin/env python3
"""
Mark-1 Universal Plugin System

This module provides the core plugin orchestration capabilities including:
- Universal plugin interface for any GitHub repository
- Plugin discovery, analysis, and registration
- Plugin lifecycle management and execution
- Plugin adapter system for different execution modes
- Plugin configuration and environment management
"""

from .base_plugin import (
    BasePlugin,
    PluginMetadata,
    PluginCapability,
    PluginStatus,
    PluginType,
    PluginResult,
    PluginException
)

from .plugin_manager import (
    PluginManager,
    PluginInstallationResult,
    PluginValidationResult
)

from .repository_analyzer import (
    RepositoryAnalyzer,
    RepositoryProfile,
    DependencyInfo,
    ServiceRequirement,
    EnvironmentVariable
)

from .plugin_adapter import (
    UniversalPluginAdapter,
    PluginExecutionContext,
    ExecutionMode,
    AdapterException
)

__version__ = "1.0.0"
__all__ = [
    # Base Plugin System
    "BasePlugin",
    "PluginMetadata", 
    "PluginCapability",
    "PluginStatus",
    "PluginType",
    "PluginResult",
    "PluginException",
    
    # Plugin Management
    "PluginManager",
    "PluginInstallationResult",
    "PluginValidationResult",
    
    # Repository Analysis
    "RepositoryAnalyzer",
    "RepositoryProfile",
    "DependencyInfo",
    "ServiceRequirement",
    "EnvironmentVariable",
    
    # Plugin Execution
    "UniversalPluginAdapter",
    "PluginExecutionContext",
    "ExecutionMode",
    "AdapterException"
]
