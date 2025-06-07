"""
System API Schemas

Session 20: API Layer & REST Endpoints
Pydantic models for system-related API operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .common import BaseResponse, HealthStatus, ResourceUsage, PerformanceMetrics


class SystemStatusResponse(BaseResponse):
    """System status response"""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(..., description="Component status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "service": "Mark-1 AI Orchestrator",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "orchestrator": "healthy",
                    "agent_selector": "healthy",
                    "context_manager": "healthy",
                    "api": "healthy"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    checks: Dict[str, str] = Field(..., description="Individual component checks")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="Service version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "checks": {
                    "orchestrator": "healthy",
                    "database": "healthy",
                    "cache": "healthy"
                },
                "uptime_seconds": 3600,
                "version": "1.0.0"
            }
        }


class MetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    agent_metrics: Dict[str, Any] = Field(..., description="Agent performance metrics")
    task_metrics: Dict[str, Any] = Field(..., description="Task execution metrics")
    context_metrics: Dict[str, Any] = Field(..., description="Context management metrics")
    system_metrics: Dict[str, Any] = Field(..., description="System resource metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "agent_metrics": {
                    "total_agents": 5,
                    "active_agents": 4,
                    "average_response_time": 250
                },
                "task_metrics": {
                    "total_tasks": 150,
                    "completed_tasks": 142,
                    "success_rate": 0.947
                },
                "context_metrics": {
                    "total_contexts": 79,
                    "cache_hit_rate": 1.0,
                    "compression_ratio": 0.054
                },
                "system_metrics": {
                    "uptime_seconds": 3600,
                    "memory_usage_mb": 512,
                    "cpu_usage_percent": 25
                }
            }
        }


class ErrorResponse(BaseResponse):
    """Error response schema"""
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(..., description="Machine-readable error code")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    
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


class DocumentationResponse(BaseModel):
    """API documentation response"""
    title: str = Field(..., description="API title")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: List[Dict[str, Any]] = Field(..., description="Available endpoints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Mark-1 AI Orchestrator API",
                "version": "1.0.0",
                "description": "Comprehensive AI Agent Orchestration Platform",
                "endpoints": [
                    {
                        "path": "/agents",
                        "method": "GET",
                        "description": "List all agents"
                    }
                ]
            }
        } 