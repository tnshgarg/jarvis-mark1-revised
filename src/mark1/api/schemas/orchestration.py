"""
Orchestration API Schemas

Session 20: API Layer & REST Endpoints
Pydantic models for orchestration and workflow operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from .common import TimestampMixin, StatusEnum, PriorityEnum


class OrchestrationRequest(BaseModel):
    """Request schema for orchestration workflow"""
    description: str = Field(..., description="Workflow description", min_length=1)
    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow requirements"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Workflow context")
    priority: PriorityEnum = Field(PriorityEnum.MEDIUM, description="Workflow priority")
    async_execution: bool = Field(True, description="Whether to execute asynchronously")
    timeout_seconds: Optional[int] = Field(None, description="Workflow timeout", gt=0)
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Workflow description cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Process customer data and generate report",
                "requirements": {
                    "data_processing": {"format": "csv", "size": "large"}
                },
                "context": {
                    "customer_segment": "enterprise"
                },
                "priority": "high",
                "async_execution": True
            }
        }


class OrchestrationResponse(TimestampMixin):
    """Response schema for orchestration status"""
    orchestration_id: str = Field(..., description="Orchestration unique identifier")
    status: str = Field(..., description="Orchestration status")
    task_ids: List[str] = Field(default_factory=list, description="Associated task IDs")
    agent_assignments: Dict[str, str] = Field(
        default_factory=dict,
        description="Task to agent assignments"
    )
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "orchestration_id": "orch_12345",
                "status": "running",
                "task_ids": ["task_1", "task_2"],
                "agent_assignments": {
                    "task_1": "agent_data_processor"
                },
                "estimated_completion": "2024-01-15T11:00:00Z",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }


class WorkflowRequest(BaseModel):
    """Request schema for workflow creation"""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Customer Data Processing Workflow",
                "description": "Complete workflow for processing customer data"
            }
        }


class WorkflowResponse(TimestampMixin):
    """Response schema for workflow information"""
    workflow_id: str = Field(..., description="Workflow unique identifier")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    status: str = Field(..., description="Workflow status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "workflow_12345",
                "name": "Customer Data Processing Workflow",
                "description": "Complete workflow for processing customer data",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        } 