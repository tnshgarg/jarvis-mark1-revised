#!/usr/bin/env python3
"""
Base Plugin Interface for Mark-1 Universal Plugin System

Defines the core interfaces and data structures for plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid


class PluginType(Enum):
    """Types of plugins supported by Mark-1"""
    CLI_TOOL = "cli_tool"
    PYTHON_LIBRARY = "python_library"
    WEB_SERVICE = "web_service"
    CONTAINER = "container"
    SCRIPT = "script"
    AI_AGENT = "ai_agent"  # Legacy support for existing agents
    UNKNOWN = "unknown"


class PluginStatus(Enum):
    """Plugin lifecycle status"""
    DISCOVERED = "discovered"
    ANALYZING = "analyzing"
    CONFIGURING = "configuring"
    INSTALLING = "installing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class ExecutionMode(Enum):
    """Plugin execution modes"""
    SUBPROCESS = "subprocess"
    PYTHON_FUNCTION = "python_function"
    HTTP_API = "http_api"
    CONTAINER = "container"
    REMOTE = "remote"


@dataclass
class PluginCapability:
    """Represents a capability that a plugin provides"""
    name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PluginMetadata:
    """Complete metadata for a plugin"""
    plugin_id: str
    name: str
    description: str
    version: str
    author: str
    repository_url: str
    plugin_type: PluginType
    capabilities: List[PluginCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.SUBPROCESS
    entry_points: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PluginStatus = PluginStatus.DISCOVERED


@dataclass
class PluginResult:
    """Result of plugin execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    output_files: List[Path] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "output_files": [str(f) for f in self.output_files]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginResult':
        """Create from dictionary"""
        return cls(
            success=data.get("success", False),
            data=data.get("data"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            execution_time=data.get("execution_time", 0.0),
            output_files=[Path(f) for f in data.get("output_files", [])]
        )


class PluginException(Exception):
    """Base exception for plugin-related errors"""
    
    def __init__(self, message: str, plugin_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.plugin_id = plugin_id
        self.details = details or {}


class BasePlugin(ABC):
    """
    Abstract base class for all plugins in the Mark-1 system
    
    This defines the standard interface that all plugins must implement,
    regardless of their underlying technology or execution mode.
    """
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.plugin_id = metadata.plugin_id
        self._initialized = False
        self._execution_context: Optional[Dict[str, Any]] = None
    
    @property
    def name(self) -> str:
        """Get plugin name"""
        return self.metadata.name
    
    @property
    def version(self) -> str:
        """Get plugin version"""
        return self.metadata.version
    
    @property
    def capabilities(self) -> List[PluginCapability]:
        """Get plugin capabilities"""
        return self.metadata.capabilities
    
    @property
    def status(self) -> PluginStatus:
        """Get current plugin status"""
        return self.metadata.status
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> PluginResult:
        """
        Execute a plugin capability
        
        Args:
            capability: Name of the capability to execute
            inputs: Input data for the capability
            parameters: Optional execution parameters
            
        Returns:
            PluginResult with execution outcome
        """
        pass
    
    @abstractmethod
    async def validate_inputs(
        self,
        capability: str,
        inputs: Dict[str, Any]
    ) -> bool:
        """
        Validate inputs for a capability
        
        Args:
            capability: Name of the capability
            inputs: Input data to validate
            
        Returns:
            True if inputs are valid
        """
        pass
    
    @abstractmethod
    async def get_progress(self) -> Dict[str, Any]:
        """
        Get current execution progress
        
        Returns:
            Progress information
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup plugin resources
        """
        pass
    
    def get_capability(self, name: str) -> Optional[PluginCapability]:
        """Get a specific capability by name"""
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None
    
    def has_capability(self, name: str) -> bool:
        """Check if plugin has a specific capability"""
        return self.get_capability(name) is not None
    
    def set_execution_context(self, context: Dict[str, Any]) -> None:
        """Set execution context for the plugin"""
        self._execution_context = context
    
    def get_execution_context(self) -> Optional[Dict[str, Any]]:
        """Get current execution context"""
        return self._execution_context
