"""
Task data models for Mark1 agent system.

This module defines the database models for task management, including
task definitions, executions, dependencies, and results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, Boolean,
    ForeignKey, JSON, Enum as SQLEnum, UniqueConstraint, Index, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from ..database import Base
from ...utils.exceptions import ValidationError

# Task-specific enums
class TaskStatus(str, Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class TaskType(str, Enum):
    """Task type classification."""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    PROCESSING = "processing"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    USER_DEFINED = "user_defined"

class DependencyType(str, Enum):
    """Task dependency relationship types."""
    REQUIRES = "requires"  # Must complete before this task starts
    BLOCKS = "blocks"      # This task blocks the dependent task
    RELATED = "related"    # Informational relationship
    SEQUENCE = "sequence"  # Sequential execution order


class Task(Base):
    """
    Core task model representing a unit of work in the system.
    
    Tasks can be standalone or part of complex workflows with dependencies.
    They track execution state, results, and metadata throughout their lifecycle.
    """
    __tablename__ = "tasks"
    
    # Primary identification - use String type for UUID for SQLite compatibility
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    
    # Task classification
    task_type = Column(SQLEnum(TaskType), nullable=False, default=TaskType.USER_DEFINED)
    priority = Column(SQLEnum(TaskPriority), nullable=False, default=TaskPriority.NORMAL)
    tags = Column(JSON, default=list)  # List of string tags for categorization
    
    # Execution configuration - use String type for UUID for SQLite compatibility
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    context_id = Column(String(36), ForeignKey("contexts.id"))
    
    # Task parameters and configuration
    parameters = Column(JSON, default=dict)  # Task-specific parameters
    constraints = Column(JSON, default=dict)  # Execution constraints
    expected_duration = Column(Integer)  # Expected duration in seconds
    max_retries = Column(Integer, default=3)
    timeout_seconds = Column(Integer, default=3600)  # 1 hour default
    
    # Scheduling
    scheduled_at = Column(DateTime(timezone=True))
    deadline = Column(DateTime(timezone=True))
    
    # Execution tracking
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    retry_count = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Results and output
    result_data = Column(JSON)  # Structured task results
    output_files = Column(JSON, default=list)  # List of output file references
    error_message = Column(Text)
    error_details = Column(JSON)  # Structured error information
    
    # Performance metrics
    execution_time_seconds = Column(Float)
    memory_usage_mb = Column(Float)
    cpu_usage_percentage = Column(Float)
    
    # Relationships
    agent = relationship("Agent")
    context = relationship("ContextModel", foreign_keys=[context_id])
    executions = relationship("TaskExecution", back_populates="task", cascade="all, delete-orphan")
    
    # Self-referential relationships for dependencies
    dependencies = relationship(
        "TaskDependency",
        foreign_keys="TaskDependency.dependent_task_id",
        back_populates="dependent_task",
        cascade="all, delete-orphan"
    )
    dependents = relationship(
        "TaskDependency",
        foreign_keys="TaskDependency.prerequisite_task_id",
        back_populates="prerequisite_task"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_task_status_priority', 'status', 'priority'),
        Index('idx_task_agent_status', 'agent_id', 'status'),
        Index('idx_task_created_at', 'created_at'),
        Index('idx_task_scheduled_at', 'scheduled_at'),
        Index('idx_task_deadline', 'deadline'),
    )
    
    @validates('priority')
    def validate_priority(self, key, priority):
        """Validate task priority value."""
        if priority not in TaskPriority:
            raise ValidationError(f"Invalid priority: {priority}")
        return priority
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate task status transitions."""
        if status not in TaskStatus:
            raise ValidationError(f"Invalid status: {status}")
        return status
    
    @validates('progress_percentage')
    def validate_progress(self, key, progress):
        """Validate progress percentage is between 0 and 100."""
        if progress is not None and (progress < 0 or progress > 100):
            raise ValidationError("Progress percentage must be between 0 and 100")
        return progress
    
    @validates('max_retries')
    def validate_max_retries(self, key, max_retries):
        """Validate max retries is non-negative."""
        if max_retries is not None and max_retries < 0:
            raise ValidationError("Max retries must be non-negative")
        return max_retries
    
    def is_ready_to_execute(self) -> bool:
        """Check if task is ready for execution based on dependencies."""
        if self.status != TaskStatus.PENDING:
            return False
        
        # Check if all prerequisite tasks are completed
        for dependency in self.dependencies:
            if dependency.dependency_type in [DependencyType.REQUIRES, DependencyType.SEQUENCE]:
                if dependency.prerequisite_task.status != TaskStatus.COMPLETED:
                    return False
        
        # Check scheduling constraints
        if self.scheduled_at and self.scheduled_at > datetime.now(timezone.utc):
            return False
        
        return True
    
    def can_retry(self) -> bool:
        """Check if task can be retried based on current state."""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def update_progress(self, percentage: float, message: str = None):
        """Update task progress with optional message."""
        self.progress_percentage = max(0, min(100, percentage))
        if message:
            # Store progress messages in result_data
            if not self.result_data:
                self.result_data = {}
            if 'progress_messages' not in self.result_data:
                self.result_data['progress_messages'] = []
            self.result_data['progress_messages'].append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'percentage': percentage,
                'message': message
            })
    
    def mark_started(self):
        """Mark task as started and update timestamps."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.progress_percentage = 0.0
    
    def mark_completed(self, result_data: Dict[str, Any] = None):
        """Mark task as completed with optional result data."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.progress_percentage = 100.0
        
        if result_data:
            self.result_data = result_data
        
        # Calculate execution time
        if self.started_at:
            duration = self.completed_at - self.started_at
            self.execution_time_seconds = duration.total_seconds()
    
    def mark_failed(self, error_message: str, error_details: Dict[str, Any] = None):
        """Mark task as failed with error information."""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.error_details = error_details or {}
        
        # Calculate execution time if started
        if self.started_at:
            failed_at = datetime.now(timezone.utc)
            duration = failed_at - self.started_at
            self.execution_time_seconds = duration.total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage,
            'agent_id': str(self.agent_id),
            'context_id': str(self.context_id) if self.context_id else None,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'execution_time_seconds': self.execution_time_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'result_data': self.result_data,
            'error_message': self.error_message
        }


class TaskDependency(Base):
    """
    Model for task dependencies and relationships.
    
    Defines how tasks relate to and depend on each other for execution ordering.
    """
    __tablename__ = "task_dependencies"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # The task that depends on another
    dependent_task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    
    # The task that must be completed first
    prerequisite_task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    
    # Type of dependency relationship
    dependency_type = Column(SQLEnum(DependencyType), nullable=False, default=DependencyType.REQUIRES)
    
    # Optional metadata about the dependency
    description = Column(Text)
    parameters = Column(JSON, default=dict)  # Dependency-specific parameters
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    dependent_task = relationship("Task", foreign_keys=[dependent_task_id], back_populates="dependencies")
    prerequisite_task = relationship("Task", foreign_keys=[prerequisite_task_id], back_populates="dependents")
    
    # Ensure no duplicate dependencies and no self-dependencies
    __table_args__ = (
        UniqueConstraint('dependent_task_id', 'prerequisite_task_id', 'dependency_type'),
        Index('idx_dependency_dependent', 'dependent_task_id'),
        Index('idx_dependency_prerequisite', 'prerequisite_task_id'),
    )
    
    @validates('dependent_task_id', 'prerequisite_task_id')
    def validate_no_self_dependency(self, key, task_id):
        """Prevent tasks from depending on themselves."""
        if key == 'dependent_task_id' and hasattr(self, 'prerequisite_task_id'):
            if task_id == self.prerequisite_task_id:
                raise ValidationError("Task cannot depend on itself")
        elif key == 'prerequisite_task_id' and hasattr(self, 'dependent_task_id'):
            if task_id == self.dependent_task_id:
                raise ValidationError("Task cannot depend on itself")
        return task_id


class TaskExecution(Base):
    """
    Model for tracking individual task execution attempts.
    
    Maintains detailed logs and metrics for each execution attempt,
    supporting debugging and performance analysis.
    """
    __tablename__ = "task_executions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    
    # Execution tracking
    attempt_number = Column(Integer, nullable=False)
    status = Column(SQLEnum(TaskStatus), nullable=False)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Results and metrics
    result_data = Column(JSON)
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # Performance metrics
    execution_time_seconds = Column(Float)
    memory_usage_mb = Column(Float)
    cpu_usage_percentage = Column(Float)
    
    # Execution environment info
    worker_id = Column(String(255))  # ID of worker/executor that ran the task
    environment_info = Column(JSON)  # System/environment details
    
    # Execution logs and output
    stdout_log = Column(Text)
    stderr_log = Column(Text)
    execution_log = Column(JSON)  # Structured execution logs
    
    # Relationships
    task = relationship("Task", back_populates="executions")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('task_id', 'attempt_number'),
        Index('idx_execution_task_attempt', 'task_id', 'attempt_number'),
        Index('idx_execution_status', 'status'),
        Index('idx_execution_started_at', 'started_at'),
    )
    
    def mark_completed(self, result_data: Dict[str, Any] = None):
        """Mark execution as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if result_data:
            self.result_data = result_data
        
        # Calculate execution time
        if self.started_at:
            duration = self.completed_at - self.started_at
            self.execution_time_seconds = duration.total_seconds()
    
    def mark_failed(self, error_message: str, error_details: Dict[str, Any] = None):
        """Mark execution as failed."""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.error_details = error_details or {}
        self.completed_at = datetime.now(timezone.utc)
        
        # Calculate execution time
        if self.started_at:
            duration = self.completed_at - self.started_at
            self.execution_time_seconds = duration.total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary representation."""
        return {
            'id': str(self.id),
            'task_id': str(self.task_id),
            'attempt_number': self.attempt_number,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time_seconds': self.execution_time_seconds,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percentage': self.cpu_usage_percentage,
            'worker_id': self.worker_id
        }


class TaskTemplate(Base):
    """
    Model for reusable task templates.
    
    Templates define common task configurations that can be instantiated
    multiple times with different parameters.
    """
    __tablename__ = "task_templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    
    # Template configuration
    task_type = Column(SQLEnum(TaskType), nullable=False)
    default_priority = Column(SQLEnum(TaskPriority), default=TaskPriority.NORMAL)
    
    # Default parameters and constraints
    default_parameters = Column(JSON, default=dict)
    parameter_schema = Column(JSON)  # JSON schema for parameter validation
    default_constraints = Column(JSON, default=dict)
    
    # Execution defaults
    default_timeout_seconds = Column(Integer, default=3600)
    default_max_retries = Column(Integer, default=3)
    expected_duration = Column(Integer)  # Expected duration in seconds
    
    # Template metadata
    tags = Column(JSON, default=list)
    category = Column(String(100))
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Relationships - tasks created from this template
    tasks = relationship("Task", secondary="task_template_usage", back_populates="templates")
    
    # Indexes
    __table_args__ = (
        Index('idx_template_category', 'category'),
        Index('idx_template_type', 'task_type'),
        Index('idx_template_active', 'is_active'),
    )
    
    def create_task(self, name: str, agent_id: str, parameters: Dict[str, Any] = None, **kwargs) -> 'Task':
        """Create a new task instance from this template."""
        # Merge template defaults with provided parameters
        final_parameters = self.default_parameters.copy()
        if parameters:
            final_parameters.update(parameters)
        
        # Create task with template defaults
        task_data = {
            'name': name,
            'description': kwargs.get('description', self.description),
            'task_type': self.task_type,
            'priority': kwargs.get('priority', self.default_priority),
            'agent_id': agent_id,
            'parameters': final_parameters,
            'constraints': kwargs.get('constraints', self.default_constraints.copy()),
            'timeout_seconds': kwargs.get('timeout_seconds', self.default_timeout_seconds),
            'max_retries': kwargs.get('max_retries', self.default_max_retries),
            'expected_duration': kwargs.get('expected_duration', self.expected_duration),
            'tags': kwargs.get('tags', self.tags.copy()),
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in task_data and hasattr(Task, key):
                task_data[key] = value
        
        return Task(**task_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'task_type': self.task_type.value,
            'default_priority': self.default_priority.value,
            'default_parameters': self.default_parameters,
            'parameter_schema': self.parameter_schema,
            'default_constraints': self.default_constraints,
            'default_timeout_seconds': self.default_timeout_seconds,
            'default_max_retries': self.default_max_retries,
            'expected_duration': self.expected_duration,
            'tags': self.tags,
            'category': self.category,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# Association table for task-template relationship
task_template_usage = Table(
    'task_template_usage',
    Base.metadata,
    Column('task_id', String(36), ForeignKey('tasks.id'), primary_key=True),
    Column('template_id', String(36), ForeignKey('task_templates.id'), primary_key=True),
    Column('created_at', DateTime(timezone=True), default=func.now()),
    extend_existing=True
)

# Add the back_populates relationship to Task
Task.templates = relationship("TaskTemplate", secondary=task_template_usage, back_populates="tasks")