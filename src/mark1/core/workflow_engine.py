"""
Workflow Engine for Mark-1 Orchestrator

Provides workflow execution and management capabilities.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.utils.exceptions import WorkflowException


class WorkflowStatus(Enum):
    """Workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_id: str
    name: str
    description: str
    agent_id: Optional[str] = None
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None
    timeout: int = 300
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}


@dataclass
class Workflow:
    """Represents a complete workflow"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class WorkflowEngine:
    """
    Workflow execution engine
    
    Manages workflow execution, step coordination, and state management.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._initialized = False
        self._active_workflows: Dict[str, Workflow] = {}
        self._workflow_tasks: Dict[str, asyncio.Task] = {}
    
    async def initialize(self) -> None:
        """Initialize the workflow engine"""
        try:
            self.logger.info("Initializing workflow engine...")
            self._initialized = True
            self.logger.info("Workflow engine initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize workflow engine", error=str(e))
            raise WorkflowException(f"Workflow engine initialization failed: {e}")
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> Workflow:
        """
        Create a new workflow
        
        Args:
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps
            
        Returns:
            Created Workflow object
        """
        try:
            workflow_id = f"workflow_{datetime.now().timestamp()}"
            
            workflow_steps = []
            for i, step_data in enumerate(steps):
                step = WorkflowStep(
                    step_id=f"step_{i}",
                    name=step_data.get("name", f"Step {i+1}"),
                    description=step_data.get("description", ""),
                    agent_id=step_data.get("agent_id"),
                    dependencies=step_data.get("dependencies", []),
                    parameters=step_data.get("parameters", {}),
                    timeout=step_data.get("timeout", 300)
                )
                workflow_steps.append(step)
            
            workflow = Workflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                steps=workflow_steps
            )
            
            self._active_workflows[workflow_id] = workflow
            self.logger.info("Workflow created", workflow_id=workflow_id, name=name)
            
            return workflow
            
        except Exception as e:
            self.logger.error("Failed to create workflow", name=name, error=str(e))
            raise WorkflowException(f"Workflow creation failed: {e}")
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """
        Execute a workflow
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            True if execution started successfully
        """
        try:
            if workflow_id not in self._active_workflows:
                raise WorkflowException(f"Workflow not found: {workflow_id}")
            
            workflow = self._active_workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.CREATED:
                raise WorkflowException(f"Workflow {workflow_id} is not in CREATED state")
            
            # Start workflow execution task
            task = asyncio.create_task(self._execute_workflow_steps(workflow))
            self._workflow_tasks[workflow_id] = task
            
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now(timezone.utc)
            
            self.logger.info("Workflow execution started", workflow_id=workflow_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to execute workflow", workflow_id=workflow_id, error=str(e))
            raise WorkflowException(f"Workflow execution failed: {str(e)}")
    
    async def _execute_workflow_steps(self, workflow: Workflow) -> None:
        """
        Execute workflow steps in order
        
        Args:
            workflow: Workflow to execute
        """
        try:
            for step in workflow.steps:
                self.logger.info("Executing workflow step", 
                               workflow_id=workflow.workflow_id, 
                               step_id=step.step_id)
                
                # Execute step (placeholder - actual execution handled by orchestrator)
                step.status = "completed"
                step.completed_at = datetime.now(timezone.utc)
                
                self.logger.info("Workflow step completed", 
                               workflow_id=workflow.workflow_id, 
                               step_id=step.step_id)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now(timezone.utc)
            
            self.logger.info("Workflow completed", workflow_id=workflow.workflow_id)
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            self.logger.error("Workflow execution failed", 
                            workflow_id=workflow.workflow_id, 
                            error=str(e))
            raise
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """
        Pause a running workflow
        
        Args:
            workflow_id: ID of the workflow to pause
            
        Returns:
            True if paused successfully
        """
        try:
            if workflow_id not in self._active_workflows:
                return False
            
            workflow = self._active_workflows[workflow_id]
            
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.PAUSED
                
                # Cancel the execution task
                if workflow_id in self._workflow_tasks:
                    self._workflow_tasks[workflow_id].cancel()
                
                self.logger.info("Workflow paused", workflow_id=workflow_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("Failed to pause workflow", workflow_id=workflow_id, error=str(e))
            return False
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """
        Get workflow status
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Workflow status or None if not found
        """
        if workflow_id in self._active_workflows:
            return self._active_workflows[workflow_id].status
        return None
    
    async def list_workflows(self) -> List[Workflow]:
        """
        List all workflows
        
        Returns:
            List of all workflows
        """
        return list(self._active_workflows.values())
    
    @property
    def is_initialized(self) -> bool:
        """Check if the workflow engine is initialized"""
        return self._initialized
    
    async def shutdown(self) -> None:
        """Shutdown the workflow engine"""
        try:
            self.logger.info("Shutting down workflow engine...")
            
            # Cancel all active workflow tasks
            for workflow_id, task in self._workflow_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self.logger.info("Cancelled workflow", workflow_id=workflow_id)
            
            # Clear state
            self._active_workflows.clear()
            self._workflow_tasks.clear()
            self._initialized = False
            
            self.logger.info("Workflow engine shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during workflow engine shutdown", error=str(e))
            raise WorkflowException(f"Workflow engine shutdown failed: {e}")
