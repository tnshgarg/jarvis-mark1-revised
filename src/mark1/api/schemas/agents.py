"""
Agent API Schemas

Session 20: API Layer & REST Endpoints
Pydantic models for agent-related API operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID
from pydantic import BaseModel, Field, validator
from .common import TimestampMixin, MetadataModel, PerformanceMetrics, StatusEnum, PaginatedResponse


class AgentCapabilitySchema(BaseModel):
    """Agent capability schema"""
    name: str = Field(..., description="Capability name")
    category: str = Field(..., description="Capability category")
    description: Optional[str] = Field(None, description="Capability description")
    confidence: float = Field(1.0, description="Confidence score", ge=0, le=1)
    parameters: Optional[Dict[str, Any]] = Field(None, description="Capability parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "data_processing",
                "category": "data",
                "description": "Process and transform data files",
                "confidence": 0.95,
                "parameters": {"supported_formats": ["csv", "json", "xml"]}
            }
        }


class AgentEndpointSchema(BaseModel):
    """Agent endpoint configuration"""
    url: Optional[str] = Field(None, description="Agent endpoint URL")
    protocol: str = Field("http", description="Communication protocol")
    authentication: Optional[Dict[str, str]] = Field(None, description="Authentication details")
    timeout: int = Field(30, description="Request timeout in seconds", gt=0)
    retries: int = Field(3, description="Number of retries", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "http://localhost:8001/agent",
                "protocol": "http",
                "authentication": {"type": "bearer", "token": "xxx"},
                "timeout": 30,
                "retries": 3
            }
        }


class AgentCreateRequest(BaseModel):
    """Request schema for creating a new agent"""
    name: str = Field(..., description="Agent name", min_length=1, max_length=255)
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    description: Optional[str] = Field(None, description="Agent description")
    framework: str = Field(..., description="Agent framework (e.g., langchain, autogpt)")
    version: Optional[str] = Field(None, description="Agent version")
    
    # Capabilities
    capabilities: List[AgentCapabilitySchema] = Field(
        default_factory=list,
        description="List of agent capabilities"
    )
    
    # Endpoint configuration
    endpoint: Optional[AgentEndpointSchema] = Field(None, description="Agent endpoint configuration")
    
    # Configuration
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent-specific configuration")
    
    # Metadata
    metadata: Optional[MetadataModel] = Field(None, description="Agent metadata")
    
    # Operational settings
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent tasks", gt=0)
    health_check_interval: int = Field(300, description="Health check interval in seconds", gt=0)
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Agent name cannot be empty')
        return v.strip()
    
    @validator('framework')
    def validate_framework(cls, v):
        valid_frameworks = ['langchain', 'autogpt', 'crewai', 'custom', 'openai', 'anthropic']
        if v.lower() not in valid_frameworks:
            raise ValueError(f'Framework must be one of: {valid_frameworks}')
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "data_processor_agent",
                "display_name": "Data Processing Agent",
                "description": "Specialized agent for data processing and analysis",
                "framework": "langchain",
                "version": "1.0.0",
                "capabilities": [
                    {
                        "name": "csv_processing",
                        "category": "data",
                        "description": "Process CSV files",
                        "confidence": 0.9
                    }
                ],
                "endpoint": {
                    "url": "http://localhost:8001/agent",
                    "protocol": "http",
                    "timeout": 30
                },
                "metadata": {
                    "tags": ["production", "data"],
                    "labels": {"environment": "prod"}
                },
                "max_concurrent_tasks": 5
            }
        }


class AgentUpdateRequest(BaseModel):
    """Request schema for updating an existing agent"""
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    description: Optional[str] = Field(None, description="Agent description")
    version: Optional[str] = Field(None, description="Agent version")
    
    # Capabilities (can be updated)
    capabilities: Optional[List[AgentCapabilitySchema]] = Field(None, description="Updated capabilities")
    
    # Endpoint configuration
    endpoint: Optional[AgentEndpointSchema] = Field(None, description="Updated endpoint configuration")
    
    # Configuration
    configuration: Optional[Dict[str, Any]] = Field(None, description="Updated configuration")
    
    # Metadata
    metadata: Optional[MetadataModel] = Field(None, description="Updated metadata")
    
    # Operational settings
    max_concurrent_tasks: Optional[int] = Field(None, description="Maximum concurrent tasks", gt=0)
    health_check_interval: Optional[int] = Field(None, description="Health check interval in seconds", gt=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Updated agent description",
                "version": "1.1.0",
                "max_concurrent_tasks": 10
            }
        }


class AgentResponse(TimestampMixin):
    """Response schema for agent information"""
    id: str = Field(..., description="Agent unique identifier")
    name: str = Field(..., description="Agent name")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    description: Optional[str] = Field(None, description="Agent description")
    framework: str = Field(..., description="Agent framework")
    version: Optional[str] = Field(None, description="Agent version")
    
    # Status and health
    status: StatusEnum = Field(..., description="Agent status")
    health_status: str = Field("unknown", description="Agent health status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    
    # Capabilities
    capabilities: List[AgentCapabilitySchema] = Field(
        default_factory=list,
        description="Agent capabilities"
    )
    
    # Endpoint configuration
    endpoint: Optional[AgentEndpointSchema] = Field(None, description="Agent endpoint")
    
    # Performance metrics
    performance: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    
    # Configuration
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    
    # Metadata
    metadata: Optional[MetadataModel] = Field(None, description="Agent metadata")
    
    # Operational info
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent tasks")
    current_task_count: int = Field(0, description="Current number of tasks", ge=0)
    total_tasks_completed: int = Field(0, description="Total completed tasks", ge=0)
    
    @classmethod
    def from_model(cls, agent_model):
        """Create response from agent model"""
        # This is a placeholder implementation
        # In reality, this would convert from the actual agent model
        return cls(
            id=str(agent_model.id) if hasattr(agent_model, 'id') else "unknown",
            name=getattr(agent_model, 'name', 'Unknown Agent'),
            framework=getattr(agent_model, 'framework', 'unknown'),
            status=StatusEnum.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            capabilities=[],
            max_concurrent_tasks=5,
            current_task_count=0,
            total_tasks_completed=0
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "agent_12345",
                "name": "data_processor_agent",
                "display_name": "Data Processing Agent",
                "description": "Specialized agent for data processing",
                "framework": "langchain",
                "version": "1.0.0",
                "status": "active",
                "health_status": "healthy",
                "capabilities": [
                    {
                        "name": "csv_processing",
                        "category": "data",
                        "confidence": 0.9
                    }
                ],
                "performance": {
                    "total_requests": 100,
                    "successful_requests": 95,
                    "failed_requests": 5,
                    "average_response_time_ms": 250.5,
                    "success_rate": 0.95
                },
                "max_concurrent_tasks": 5,
                "current_task_count": 2,
                "total_tasks_completed": 150,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class AgentListResponse(PaginatedResponse[AgentResponse]):
    """Response schema for listing agents"""
    agents: List[AgentResponse] = Field(..., alias="items", description="List of agents")
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "agents": [
                    {
                        "id": "agent_1",
                        "name": "data_agent",
                        "framework": "langchain",
                        "status": "active",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:30:00Z"
                    }
                ],
                "total": 1,
                "limit": 10,
                "offset": 0,
                "has_next": False,
                "has_previous": False
            }
        }


class AgentStatusResponse(BaseModel):
    """Response schema for agent status"""
    id: str = Field(..., description="Agent ID")
    status: StatusEnum = Field(..., description="Agent status")
    health_status: str = Field(..., description="Health status")
    last_seen: Optional[datetime] = Field(None, description="Last seen timestamp")
    current_tasks: int = Field(0, description="Current task count", ge=0)
    uptime_seconds: Optional[int] = Field(None, description="Uptime in seconds", ge=0)
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "agent_12345",
                "status": "active",
                "health_status": "healthy",
                "last_seen": "2024-01-15T10:30:00Z",
                "current_tasks": 2,
                "uptime_seconds": 3600,
                "error_message": None
            }
        }


class AgentTestRequest(BaseModel):
    """Request schema for testing agent functionality"""
    test_type: str = Field("connectivity", description="Type of test to perform")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Test data")
    timeout: int = Field(30, description="Test timeout in seconds", gt=0)
    
    @validator('test_type')
    def validate_test_type(cls, v):
        valid_types = ['connectivity', 'capability', 'performance', 'full']
        if v not in valid_types:
            raise ValueError(f'Test type must be one of: {valid_types}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_type": "connectivity",
                "test_data": {"sample_input": "test data"},
                "timeout": 30
            }
        }


class AgentTestResponse(BaseModel):
    """Response schema for agent testing"""
    agent_id: str = Field(..., description="Agent ID")
    test_type: str = Field(..., description="Test type performed")
    success: bool = Field(..., description="Whether test passed")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    result: Optional[Dict[str, Any]] = Field(None, description="Test result data")
    error_message: Optional[str] = Field(None, description="Error message if test failed")
    timestamp: datetime = Field(..., description="Test timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "agent_12345",
                "test_type": "connectivity",
                "success": True,
                "response_time_ms": 150.5,
                "result": {"status": "ok", "version": "1.0.0"},
                "error_message": None,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        } 