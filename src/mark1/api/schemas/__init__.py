"""
Mark-1 API Schemas

Session 20: API Layer & REST Endpoints
Pydantic models for request/response validation and documentation
"""

from .agents import (
    AgentResponse, AgentCreateRequest, AgentUpdateRequest, AgentListResponse,
    AgentTestRequest, AgentStatusResponse
)
from .tasks import (
    TaskResponse, TaskCreateRequest, TaskUpdateRequest, TaskListResponse,
    TaskExecutionRequest, TaskLogResponse
)
from .contexts import (
    ContextResponse, ContextCreateRequest, ContextUpdateRequest, ContextListResponse,
    ContextShareRequest, ContextMergeRequest
)
from .orchestration import (
    OrchestrationRequest, OrchestrationResponse, WorkflowRequest, WorkflowResponse
)
from .system import (
    SystemStatusResponse, HealthResponse, MetricsResponse, ErrorResponse,
    DocumentationResponse
)
from .common import (
    BaseResponse, PaginatedResponse, ValidationErrorResponse,
    SuccessResponse, TimestampMixin
)

__all__ = [
    # Agent schemas
    'AgentResponse', 'AgentCreateRequest', 'AgentUpdateRequest', 'AgentListResponse',
    'AgentTestRequest', 'AgentStatusResponse',
    
    # Task schemas
    'TaskResponse', 'TaskCreateRequest', 'TaskUpdateRequest', 'TaskListResponse',
    'TaskExecutionRequest', 'TaskLogResponse',
    
    # Context schemas
    'ContextResponse', 'ContextCreateRequest', 'ContextUpdateRequest', 'ContextListResponse',
    'ContextShareRequest', 'ContextMergeRequest',
    
    # Orchestration schemas
    'OrchestrationRequest', 'OrchestrationResponse', 'WorkflowRequest', 'WorkflowResponse',
    
    # System schemas
    'SystemStatusResponse', 'HealthResponse', 'MetricsResponse', 'ErrorResponse',
    'DocumentationResponse',
    
    # Common schemas
    'BaseResponse', 'PaginatedResponse', 'ValidationErrorResponse',
    'SuccessResponse', 'TimestampMixin'
]
