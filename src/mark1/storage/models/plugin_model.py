#!/usr/bin/env python3
"""
Plugin Storage Models for Mark-1 Universal Plugin System

SQLAlchemy models for storing plugin metadata, capabilities, and execution history.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..database import Base


class Plugin(Base):
    """Plugin metadata storage"""
    __tablename__ = "plugins"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    plugin_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    version = Column(String, default="1.0.0")
    author = Column(String)
    repository_url = Column(String, nullable=False)
    plugin_type = Column(String, nullable=False)  # PluginType enum value
    execution_mode = Column(String, nullable=False)  # ExecutionMode enum value
    status = Column(String, nullable=False, default="discovered")  # PluginStatus enum value
    
    # JSON fields for complex data
    dependencies = Column(JSON, default=list)
    environment_variables = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    entry_points = Column(JSON, default=dict)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    installed_at = Column(DateTime)
    last_used_at = Column(DateTime)
    
    # Statistics
    usage_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    average_execution_time = Column(Float, default=0.0)
    
    # Relationships
    capabilities = relationship("PluginCapability", back_populates="plugin", cascade="all, delete-orphan")
    executions = relationship("PluginExecution", back_populates="plugin", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "plugin_id": self.plugin_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "repository_url": self.repository_url,
            "plugin_type": self.plugin_type,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "dependencies": self.dependencies,
            "environment_variables": self.environment_variables,
            "configuration": self.configuration,
            "entry_points": self.entry_points,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "average_execution_time": self.average_execution_time,
            "capabilities": [cap.to_dict() for cap in self.capabilities] if self.capabilities else []
        }


class PluginCapability(Base):
    """Plugin capability storage"""
    __tablename__ = "plugin_capabilities"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    plugin_id = Column(String, ForeignKey("plugins.plugin_id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # JSON fields for complex data
    input_types = Column(JSON, default=list)
    output_types = Column(JSON, default=list)
    parameters = Column(JSON, default=dict)
    examples = Column(JSON, default=list)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Statistics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    average_execution_time = Column(Float, default=0.0)
    
    # Relationships
    plugin = relationship("Plugin", back_populates="capabilities")
    executions = relationship("PluginExecution", back_populates="capability")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "plugin_id": self.plugin_id,
            "name": self.name,
            "description": self.description,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "parameters": self.parameters,
            "examples": self.examples,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time
        }


class PluginExecution(Base):
    """Plugin execution history"""
    __tablename__ = "plugin_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    plugin_id = Column(String, ForeignKey("plugins.plugin_id"), nullable=False)
    capability_id = Column(String, ForeignKey("plugin_capabilities.id"), nullable=True)
    capability_name = Column(String, nullable=False)
    
    # Execution details
    execution_id = Column(String, unique=True, nullable=False)
    status = Column(String, nullable=False)  # success, error, timeout, cancelled
    started_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    execution_time = Column(Float, default=0.0)
    
    # Input/Output data
    inputs = Column(JSON, default=dict)
    parameters = Column(JSON, default=dict)
    outputs = Column(JSON, default=dict)
    error_message = Column(Text)
    
    # Execution context
    execution_mode = Column(String)
    environment_variables = Column(JSON, default=dict)
    working_directory = Column(String)
    
    # Resource usage
    memory_usage_mb = Column(Float, default=0.0)
    cpu_usage_percent = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    plugin = relationship("Plugin", back_populates="executions")
    capability = relationship("PluginCapability", back_populates="executions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "plugin_id": self.plugin_id,
            "capability_id": self.capability_id,
            "capability_name": self.capability_name,
            "execution_id": self.execution_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "error_message": self.error_message,
            "execution_mode": self.execution_mode,
            "environment_variables": self.environment_variables,
            "working_directory": self.working_directory,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PluginWorkflow(Base):
    """Plugin workflow storage"""
    __tablename__ = "plugin_workflows"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Workflow definition
    workflow_steps = Column(JSON, default=list)  # List of workflow steps
    plugin_chain = Column(JSON, default=list)    # Ordered list of plugin IDs
    
    # Execution details
    status = Column(String, default="created")  # created, running, completed, failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    execution_time = Column(Float, default=0.0)
    
    # Results
    inputs = Column(JSON, default=dict)
    outputs = Column(JSON, default=dict)
    error_message = Column(Text)
    
    # Statistics
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "workflow_steps": self.workflow_steps,
            "plugin_chain": self.plugin_chain,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error_message": self.error_message,
            "execution_count": self.execution_count,
            "success_count": self.success_count
        }
