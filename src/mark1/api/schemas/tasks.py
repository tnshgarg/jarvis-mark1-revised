"""
Task API Schemas

Session 20: API Layer & REST Endpoints
Pydantic models for task-related API operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, validator
from .common import TimestampMixin, StatusEnum, PriorityEnum, PaginatedResponse


class TaskRequirementSchema(BaseModel):
    """Task requirement schema"""
    capability: str = Field(..., description="Required capability")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Capability parameters")
    priority: PriorityEnum = Field(PriorityEnum.MEDIUM, description="Requirement priority")
    
    class Config:
        json_schema_extra = {
            "example": {
                "capability": "data_processing",
                "parameters": {"format": "csv", "size_limit": "10MB"},
                "priority": "high"
            }
        }


class TaskCreateRequest(BaseModel):
    """Request schema for creating a new task"""
    description: str = Field(..., description="Task description", min_length=1)
    requirements: List[TaskRequirementSchema] = Field(
        default_factory=list,
        description="Task requirements"
    )
    priority: PriorityEnum = Field(PriorityEnum.MEDIUM, description="Task priority")
    context_id: Optional[str] = Field(None, description="Associated context ID")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Task input data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Task metadata")
    timeout_seconds: Optional[int] = Field(None, description="Task timeout", gt=0)
    auto_execute: bool = Field(False, description="Whether to execute task immediately")
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Task description cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Process customer data file",
                "requirements": [
                    {
                        "capability": "data_processing",
                        "parameters": {"format": "csv"},
                        "priority": "high"
                    }
                ],
                "priority": "high",
                "input_data": {"file_path": "/data/customers.csv"},
                "timeout_seconds": 300,
                "auto_execute": True
            }
        }


class TaskUpdateRequest(BaseModel):
    """Request schema for updating a task"""
    description: Optional[str] = Field(None, description="Updated description")
    priority: Optional[PriorityEnum] = Field(None, description="Updated priority")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    timeout_seconds: Optional[int] = Field(None, description="Updated timeout", gt=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Updated task description",
                "priority": "critical",
                "timeout_seconds": 600
            }
        }


class TaskExecutionRequest(BaseModel):
    """Request schema for task execution"""
    agent_id: Optional[str] = Field(None, description="Specific agent to use")
    async_execution: bool = Field(True, description="Whether to execute asynchronously")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Execution parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "agent_12345",
                "async_execution": True,
                "parameters": {"verbose": True}
            }
        }


class TaskResponse(TimestampMixin):
    """Response schema for task information"""
    id: str = Field(..., description="Task unique identifier")
    description: str = Field(..., description="Task description")
    status: StatusEnum = Field(..., description="Task status")
    priority: PriorityEnum = Field(..., description="Task priority")
    
    # Execution details
    assigned_agent_id: Optional[str] = Field(None, description="Assigned agent ID")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    
    # Requirements and data
    requirements: List[TaskRequirementSchema] = Field(
        default_factory=list,
        description="Task requirements"
    )
    input_data: Optional[Dict[str, Any]] = Field(None, description="Task input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Task output data")
    
    # Context and metadata
    context_id: Optional[str] = Field(None, description="Associated context ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Task metadata")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retries", ge=0)
    max_retries: int = Field(3, description="Maximum retry attempts", ge=0)
    
    # Progress tracking
    progress_percentage: float = Field(0.0, description="Task progress", ge=0, le=100)
    
    @classmethod
    def from_model(cls, task_model):
        """Create response from task model"""
        # Placeholder implementation
        return cls(
            id=str(task_model.id) if hasattr(task_model, 'id') else "unknown",
            description=getattr(task_model, 'description', 'Unknown Task'),
            status=StatusEnum.PENDING,
            priority=PriorityEnum.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            requirements=[],
            progress_percentage=0.0
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "task_12345",
                "description": "Process customer data file",
                "status": "completed",
                "priority": "high",
                "assigned_agent_id": "agent_67890",
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:35:00Z",
                "duration_seconds": 300.5,
                "requirements": [
                    {
                        "capability": "data_processing",
                        "priority": "high"
                    }
                ],
                "output_data": {"processed_records": 1000},
                "progress_percentage": 100.0,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }


class TaskListResponse(PaginatedResponse[TaskResponse]):
    """Response schema for listing tasks"""
    tasks: List[TaskResponse] = Field(..., alias="items", description="List of tasks")
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "tasks": [
                    {
                        "id": "task_1",
                        "description": "Data processing task",
                        "status": "completed",
                        "priority": "high",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:35:00Z"
                    }
                ],
                "total": 1,
                "limit": 10,
                "offset": 0,
                "has_next": False,
                "has_previous": False
            }
        }


class TaskLogResponse(BaseModel):
    """Response schema for task logs"""
    task_id: str = Field(..., description="Task ID")
    logs: List[Dict[str, Any]] = Field(..., description="Task execution logs")
    total_logs: int = Field(..., description="Total number of log entries")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_12345",
                "logs": [
                    {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "level": "INFO",
                        "message": "Task started",
                        "details": {}
                    },
                    {
                        "timestamp": "2024-01-15T10:35:00Z",
                        "level": "INFO", 
                        "message": "Task completed successfully",
                        "details": {"processed_records": 1000}
                    }
                ],
                "total_logs": 2
            }
        } 