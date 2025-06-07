"""
Common API Schemas

Session 20: API Layer & REST Endpoints
Base Pydantic models and common schemas used across the API
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum


T = TypeVar('T')


class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields"""
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    @validator('created_at', 'updated_at', pre=True)
    def validate_timestamps(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(True, description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SuccessResponse(BaseResponse):
    """Standard success response"""
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {"result": "example_data"}
            }
        }


class ErrorResponse(BaseResponse):
    """Standard error response"""
    success: bool = Field(False, description="Always false for error responses")
    error_code: str = Field(..., description="Machine-readable error code")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Validation failed",
                "timestamp": "2024-01-15T10:30:00Z",
                "error_code": "VALIDATION_ERROR",
                "error_type": "ValidationError",
                "details": {"field": "name", "issue": "Field is required"}
            }
        }


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific details"""
    validation_errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Request validation failed",
                "timestamp": "2024-01-15T10:30:00Z",
                "error_code": "VALIDATION_ERROR",
                "error_type": "ValidationError",
                "validation_errors": [
                    {
                        "field": "name",
                        "message": "Field is required",
                        "type": "missing"
                    }
                ]
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items", ge=0)
    limit: int = Field(..., description="Number of items per page", ge=1)
    offset: int = Field(..., description="Number of items skipped", ge=0)
    has_next: bool = Field(..., description="Whether there are more items")
    has_previous: bool = Field(..., description="Whether there are previous items")
    
    @validator('has_next', always=True)
    def calculate_has_next(cls, v, values):
        if 'total' in values and 'limit' in values and 'offset' in values:
            return values['offset'] + values['limit'] < values['total']
        return False
    
    @validator('has_previous', always=True)
    def calculate_has_previous(cls, v, values):
        return values.get('offset', 0) > 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": ["item1", "item2", "item3"],
                "total": 50,
                "limit": 10,
                "offset": 0,
                "has_next": True,
                "has_previous": False
            }
        }


class StatusEnum(str, Enum):
    """Common status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class PriorityEnum(str, Enum):
    """Priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetadataModel(BaseModel):
    """Flexible metadata model"""
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags")
    labels: Optional[Dict[str, str]] = Field(default_factory=dict, description="Key-value labels")
    annotations: Optional[Dict[str, str]] = Field(default_factory=dict, description="Annotations")
    custom: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tags": ["production", "critical"],
                "labels": {"environment": "prod", "version": "1.0"},
                "annotations": {"description": "Primary agent for data processing"},
                "custom": {"custom_field": "custom_value"}
            }
        }


class ResourceUsage(BaseModel):
    """Resource usage metrics"""
    cpu_percent: Optional[float] = Field(None, description="CPU usage percentage", ge=0, le=100)
    memory_mb: Optional[float] = Field(None, description="Memory usage in MB", ge=0)
    disk_mb: Optional[float] = Field(None, description="Disk usage in MB", ge=0)
    network_kb_in: Optional[float] = Field(None, description="Network input in KB", ge=0)
    network_kb_out: Optional[float] = Field(None, description="Network output in KB", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "cpu_percent": 25.5,
                "memory_mb": 512.0,
                "disk_mb": 1024.0,
                "network_kb_in": 156.8,
                "network_kb_out": 89.2
            }
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    total_requests: int = Field(0, description="Total number of requests", ge=0)
    successful_requests: int = Field(0, description="Number of successful requests", ge=0)
    failed_requests: int = Field(0, description="Number of failed requests", ge=0)
    average_response_time_ms: Optional[float] = Field(None, description="Average response time in milliseconds", ge=0)
    min_response_time_ms: Optional[float] = Field(None, description="Minimum response time in milliseconds", ge=0)
    max_response_time_ms: Optional[float] = Field(None, description="Maximum response time in milliseconds", ge=0)
    success_rate: Optional[float] = Field(None, description="Success rate as decimal", ge=0, le=1)
    
    @validator('success_rate', always=True)
    def calculate_success_rate(cls, v, values):
        total = values.get('total_requests', 0)
        successful = values.get('successful_requests', 0)
        if total > 0:
            return successful / total
        return None
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 1000,
                "successful_requests": 950,
                "failed_requests": 50,
                "average_response_time_ms": 250.5,
                "min_response_time_ms": 45.2,
                "max_response_time_ms": 2500.0,
                "success_rate": 0.95
            }
        }


class IDResponse(BaseModel):
    """Simple ID response"""
    id: str = Field(..., description="Resource identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "agent_12345"
            }
        }


class BulkOperation(BaseModel):
    """Bulk operation request"""
    operation: str = Field(..., description="Operation to perform")
    ids: List[str] = Field(..., description="List of resource IDs")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Operation parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operation": "delete",
                "ids": ["agent_1", "agent_2", "agent_3"],
                "parameters": {"force": True}
            }
        }


class BulkOperationResponse(BaseModel):
    """Bulk operation response"""
    total_requested: int = Field(..., description="Total number of items requested", ge=0)
    successful: int = Field(..., description="Number of successful operations", ge=0)
    failed: int = Field(..., description="Number of failed operations", ge=0)
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual operation results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_requested": 3,
                "successful": 2,
                "failed": 1,
                "results": [
                    {"id": "agent_1", "success": True},
                    {"id": "agent_2", "success": True},
                    {"id": "agent_3", "success": False, "error": "Not found"}
                ]
            }
        }


class FilterRequest(BaseModel):
    """Generic filter request"""
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", description="Sort order (asc/desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError("sort_order must be 'asc' or 'desc'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "filters": {"status": "active", "framework": "langchain"},
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        } 