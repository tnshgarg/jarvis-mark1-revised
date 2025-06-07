"""
Context API Schemas

Session 20: API Layer & REST Endpoints
Pydantic models for context-related API operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID
from pydantic import BaseModel, Field, validator
from .common import TimestampMixin, PriorityEnum, PaginatedResponse


class ContextCreateRequest(BaseModel):
    """Request schema for creating a new context"""
    key: str = Field(..., description="Context key/identifier", min_length=1)
    content: Dict[str, Any] = Field(..., description="Context content")
    context_type: str = Field("memory", description="Context type")
    scope: str = Field("agent", description="Context scope")
    priority: str = Field("medium", description="Context priority")
    tags: Optional[List[str]] = Field(None, description="Context tags")
    expires_in_hours: Optional[int] = Field(None, description="Expiration time in hours", gt=0)
    parent_context_id: Optional[str] = Field(None, description="Parent context ID for hierarchy")
    
    @validator('key')
    def validate_key(cls, v):
        if not v or not v.strip():
            raise ValueError('Context key cannot be empty')
        return v.strip()
    
    @validator('context_type')
    def validate_context_type(cls, v):
        valid_types = ['memory', 'session', 'conversation', 'task', 'agent']
        if v not in valid_types:
            raise ValueError(f'Context type must be one of: {valid_types}')
        return v
    
    @validator('scope')
    def validate_scope(cls, v):
        valid_scopes = ['global', 'agent', 'session', 'conversation', 'task']
        if v not in valid_scopes:
            raise ValueError(f'Context scope must be one of: {valid_scopes}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['low', 'medium', 'high', 'critical']
        if v not in valid_priorities:
            raise ValueError(f'Context priority must be one of: {valid_priorities}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "key": "user_session_data",
                "content": {
                    "user_id": "12345",
                    "preferences": {"theme": "dark"},
                    "session_data": {"login_time": "2024-01-15T10:30:00Z"}
                },
                "context_type": "session",
                "scope": "agent",
                "priority": "medium",
                "tags": ["session", "user_data"],
                "expires_in_hours": 24
            }
        }


class ContextUpdateRequest(BaseModel):
    """Request schema for updating a context"""
    content: Dict[str, Any] = Field(..., description="Updated context content")
    merge: bool = Field(True, description="Whether to merge with existing content")
    create_version: bool = Field(False, description="Whether to create a version backup")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": {
                    "user_id": "12345",
                    "preferences": {"theme": "light", "language": "en"},
                    "updated_at": "2024-01-15T11:00:00Z"
                },
                "merge": True,
                "create_version": True
            }
        }


class ContextShareRequest(BaseModel):
    """Request schema for sharing context between agents"""
    agent_ids: List[str] = Field(..., description="List of agent IDs to share with")
    permissions: List[str] = Field(
        default_factory=lambda: ["read"],
        description="Permissions to grant"
    )
    copy_on_share: bool = Field(True, description="Whether to create copies for each agent")
    
    @validator('permissions')
    def validate_permissions(cls, v):
        valid_permissions = ['read', 'write', 'delete', 'share']
        for perm in v:
            if perm not in valid_permissions:
                raise ValueError(f'Invalid permission: {perm}. Must be one of: {valid_permissions}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_ids": ["agent_1", "agent_2", "agent_3"],
                "permissions": ["read", "write"],
                "copy_on_share": True
            }
        }


class ContextMergeRequest(BaseModel):
    """Request schema for merging contexts"""
    context_ids: List[str] = Field(..., description="List of context IDs to merge", min_items=2)
    merge_strategy: str = Field("deep", description="Merge strategy")
    conflict_resolution: str = Field("latest", description="Conflict resolution strategy")
    create_backup: bool = Field(True, description="Whether to create backups before merging")
    
    @validator('merge_strategy')
    def validate_merge_strategy(cls, v):
        valid_strategies = ['shallow', 'deep', 'replace']
        if v not in valid_strategies:
            raise ValueError(f'Merge strategy must be one of: {valid_strategies}')
        return v
    
    @validator('conflict_resolution')
    def validate_conflict_resolution(cls, v):
        valid_resolutions = ['latest', 'oldest', 'priority', 'manual']
        if v not in valid_resolutions:
            raise ValueError(f'Conflict resolution must be one of: {valid_resolutions}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "context_ids": ["context_1", "context_2", "context_3"],
                "merge_strategy": "deep",
                "conflict_resolution": "latest",
                "create_backup": True
            }
        }


class ContextResponse(TimestampMixin):
    """Response schema for context information"""
    id: str = Field(..., description="Context unique identifier")
    key: str = Field(..., description="Context key")
    content: Dict[str, Any] = Field(..., description="Context content")
    context_type: str = Field(..., description="Context type")
    scope: str = Field(..., description="Context scope")
    priority: str = Field("medium", description="Context priority")
    
    # Hierarchy
    parent_context_id: Optional[str] = Field(None, description="Parent context ID")
    child_context_ids: List[str] = Field(default_factory=list, description="Child context IDs")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Context tags")
    size_bytes: Optional[int] = Field(None, description="Context size in bytes")
    compressed: bool = Field(False, description="Whether content is compressed")
    
    # Lifecycle
    version: int = Field(1, description="Context version")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    access_count: int = Field(0, description="Number of times accessed")
    
    # Sharing
    shared_with: List[str] = Field(default_factory=list, description="Agent IDs context is shared with")
    permissions: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Permissions per agent"
    )
    
    @classmethod
    def from_model(cls, context_model):
        """Create response from context model"""
        # Placeholder implementation
        return cls(
            id=str(context_model.id) if hasattr(context_model, 'id') else "unknown",
            key=getattr(context_model, 'key', 'unknown'),
            content=getattr(context_model, 'content', {}),
            context_type=getattr(context_model, 'context_type', 'memory'),
            scope=getattr(context_model, 'scope', 'agent'),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "context_12345",
                "key": "user_session_data",
                "content": {
                    "user_id": "12345",
                    "preferences": {"theme": "dark"},
                    "session_data": {"login_time": "2024-01-15T10:30:00Z"}
                },
                "context_type": "session",
                "scope": "agent",
                "priority": "medium",
                "tags": ["session", "user_data"],
                "size_bytes": 512,
                "compressed": False,
                "version": 1,
                "access_count": 5,
                "shared_with": ["agent_1", "agent_2"],
                "permissions": {
                    "agent_1": ["read", "write"],
                    "agent_2": ["read"]
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class ContextListResponse(PaginatedResponse[ContextResponse]):
    """Response schema for listing contexts"""
    contexts: List[ContextResponse] = Field(..., alias="items", description="List of contexts")
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "contexts": [
                    {
                        "id": "context_1",
                        "key": "session_data",
                        "context_type": "session",
                        "scope": "agent",
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