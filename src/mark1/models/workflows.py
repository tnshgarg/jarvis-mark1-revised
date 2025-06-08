#!/usr/bin/env python3
"""
Workflow models for Mark-1 AI Orchestrator

This module defines workflow-related data structures and models for
workflow execution, tracking, and management within the Mark-1 system.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import uuid


class WorkflowStatus(Enum):
    """Workflow execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    """Workflow step status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    agent_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Workflow:
    """Workflow execution definition and state"""
    id: str
    name: str
    description: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    steps: List[Dict[str, Any]] = field(default_factory=list)  # For compatibility with WebSocket API
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    total_steps: int = 0
    completed_steps: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.steps, list) and self.steps:
            self.total_steps = len(self.steps)
    
    def calculate_progress(self) -> float:
        """Calculate workflow progress percentage"""
        if self.total_steps == 0:
            return 0.0
        
        completed = sum(1 for step in self.steps if step.get('status') == 'completed')
        self.completed_steps = completed
        self.progress = (completed / self.total_steps) * 100
        return self.progress
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get currently running step"""
        for step in self.steps:
            if step.get('status') == 'running':
                return step
        return None
    
    def is_completed(self) -> bool:
        """Check if workflow is completed"""
        return self.status == WorkflowStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if workflow has failed"""
        return self.status == WorkflowStatus.FAILED
    
    def is_running(self) -> bool:
        """Check if workflow is currently running"""
        return self.status == WorkflowStatus.RUNNING


@dataclass
class WorkflowTemplate:
    """Workflow template for creating new workflows"""
    id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    step_templates: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    
    def create_workflow(self, **kwargs) -> Workflow:
        """Create a workflow instance from this template"""
        workflow_id = kwargs.get('id', str(uuid.uuid4()))
        workflow_name = kwargs.get('name', self.name)
        
        # Convert step templates to workflow steps
        steps = []
        for template_step in self.step_templates:
            step = {
                'id': template_step.get('id', str(uuid.uuid4())),
                'name': template_step.get('name', ''),
                'type': template_step.get('type', 'generic'),
                'status': 'pending',
                'parameters': template_step.get('parameters', {}),
                'depends_on': template_step.get('depends_on', [])
            }
            steps.append(step)
        
        return Workflow(
            id=workflow_id,
            name=workflow_name,
            description=kwargs.get('description', self.description),
            steps=steps,
            created_by=kwargs.get('created_by'),
            tags=kwargs.get('tags', self.tags.copy()),
            metadata=kwargs.get('metadata', {})
        )


# Utility functions for workflow management
def create_simple_workflow(name: str, step_names: List[str], **kwargs) -> Workflow:
    """Create a simple workflow with sequential steps"""
    steps = []
    for i, step_name in enumerate(step_names):
        step = {
            'id': f"step_{i+1}",
            'name': step_name,
            'type': 'generic',
            'status': 'pending',
            'depends_on': [f"step_{i}"] if i > 0 else []
        }
        steps.append(step)
    
    return Workflow(
        id=kwargs.get('id', str(uuid.uuid4())),
        name=name,
        description=kwargs.get('description'),
        steps=steps,
        **kwargs
    )


def workflow_status_from_string(status_str: str) -> WorkflowStatus:
    """Convert string to WorkflowStatus enum"""
    try:
        return WorkflowStatus(status_str.lower())
    except ValueError:
        return WorkflowStatus.PENDING


def step_status_from_string(status_str: str) -> StepStatus:
    """Convert string to StepStatus enum"""
    try:
        return StepStatus(status_str.lower())
    except ValueError:
        return StepStatus.PENDING 