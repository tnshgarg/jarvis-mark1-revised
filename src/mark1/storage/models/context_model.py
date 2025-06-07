"""
Context Model for Mark1 Agent Framework

This module provides comprehensive context management for agents, including
conversation history, session tracking, and contextual state management.
Supports both short-term and long-term context storage with efficient
retrieval and management capabilities.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import uuid

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, Boolean, 
    ForeignKey, Index, JSON, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from ..database import Base
from ...utils.exceptions import ValidationError, ContextError
from ...utils.constants import (
    MAX_CONTEXT_SIZE, DEFAULT_CONTEXT_WINDOW, 
    CONTEXT_RETENTION_DAYS, MAX_CONVERSATION_LENGTH
)


class ContextType(Enum):
    """Enumeration of different context types."""
    CONVERSATION = "conversation"
    SESSION = "session"
    TASK = "task"
    SYSTEM = "system"
    USER_PREFERENCE = "user_preference"
    MEMORY = "memory"
    TOOL_USAGE = "tool_usage"


class ContextScope(Enum):
    """Enumeration of context scope levels."""
    GLOBAL = "global"
    AGENT = "agent"
    SESSION = "session"
    CONVERSATION = "conversation"
    TASK = "task"


class ContextPriority(Enum):
    """Enumeration of context priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ARCHIVE = "archive"


class ContextModel(Base):
    """
    Core context model for storing contextual information.
    
    This model handles various types of context data including conversation
    history, session state, task context, and system-level information.
    """
    
    __tablename__ = "contexts"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    context_key = Column(String(255), nullable=False, index=True)
    context_type = Column(String(50), nullable=False, index=True)
    context_scope = Column(String(50), nullable=False, index=True)
    priority = Column(String(20), nullable=False, default=ContextPriority.MEDIUM.value)
    
    # Hierarchical relationships
    parent_context_id = Column(UUID(as_uuid=True), ForeignKey("contexts.id"), nullable=True)
    agent_id = Column(UUID(as_uuid=True), nullable=True)  # Removed FK constraint temporarily
    session_id = Column(String(255), nullable=True, index=True)
    conversation_id = Column(String(255), nullable=True, index=True)
    task_id = Column(UUID(as_uuid=True), nullable=True)  # Removed FK constraint temporarily
    
    # Content storage
    title = Column(String(500), nullable=True)
    content = Column(JSON, nullable=True)  # Structured content
    raw_content = Column(Text, nullable=True)  # Raw text content
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata
    
    # Context management
    token_count = Column(Integer, nullable=True, default=0)
    size_bytes = Column(Integer, nullable=True, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    is_archived = Column(Boolean, nullable=False, default=False)
    
    # Temporal tracking
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    accessed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Versioning and tracking
    version = Column(Integer, nullable=False, default=1)
    checksum = Column(String(64), nullable=True)  # Content integrity
    
    # Relationships - using string references to avoid circular imports
    parent_context = relationship("ContextModel", remote_side=[id], backref="child_contexts")
    # Note: Agent and Task relationships will be added when those models are available
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_context_lookup", "context_key", "context_type", "context_scope"),
        Index("idx_context_hierarchy", "parent_context_id", "agent_id"),
        Index("idx_context_temporal", "created_at", "updated_at", "expires_at"),
        Index("idx_context_session", "session_id", "conversation_id"),
        Index("idx_context_active", "is_active", "is_archived"),
    )
    
    def __init__(self, **kwargs):
        """Initialize context model with validation."""
        super().__init__(**kwargs)
        self._validate_context_data()
    
    @validates('context_type')
    def validate_context_type(self, key, value):
        """Validate context type."""
        if value not in [ct.value for ct in ContextType]:
            raise ValidationError(f"Invalid context type: {value}")
        return value
    
    @validates('context_scope')
    def validate_context_scope(self, key, value):
        """Validate context scope."""
        if value not in [cs.value for cs in ContextScope]:
            raise ValidationError(f"Invalid context scope: {value}")
        return value
    
    @validates('priority')
    def validate_priority(self, key, value):
        """Validate context priority."""
        if value not in [cp.value for cp in ContextPriority]:
            raise ValidationError(f"Invalid context priority: {value}")
        return value
    
    def _validate_context_data(self):
        """Validate context data consistency."""
        if self.content and self.raw_content:
            # Ensure content consistency
            if isinstance(self.content, dict) and 'text' in self.content:
                if self.content['text'] != self.raw_content:
                    raise ValidationError("Content and raw_content mismatch")
        
        # Validate size constraints
        if self.size_bytes and self.size_bytes > MAX_CONTEXT_SIZE:
            raise ValidationError(f"Context size exceeds maximum: {self.size_bytes}")
    
    def update_content(self, content: Union[str, Dict[str, Any]], metadata: Optional[Dict] = None):
        """
        Update context content with automatic metadata management.
        
        Args:
            content: New content (string or structured data)
            metadata: Optional metadata to merge
        """
        if isinstance(content, str):
            self.raw_content = content
            self.content = {"text": content, "type": "text"}
        elif isinstance(content, dict):
            self.content = content
            if "text" in content:
                self.raw_content = content["text"]
        
        # Update size tracking
        self.size_bytes = len(json.dumps(self.content or {})) + len(self.raw_content or "")
        
        # Merge metadata
        if metadata:
            if self.extra_metadata:
                self.extra_metadata.update(metadata)
            else:
                self.extra_metadata = metadata
        
        # Update timestamps
        self.updated_at = datetime.now(timezone.utc)
        self.accessed_at = datetime.now(timezone.utc)
        self.version += 1
    
    def add_conversation_turn(self, role: str, message: str, metadata: Optional[Dict] = None):
        """
        Add a conversation turn to the context.
        
        Args:
            role: Speaker role (user, assistant, system)
            message: Message content
            metadata: Optional turn metadata
        """
        if self.context_type != ContextType.CONVERSATION.value:
            raise ContextError("Cannot add conversation turn to non-conversation context")
        
        if not self.content:
            self.content = {"turns": []}
        
        turn = {
            "role": role,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_id": str(uuid.uuid4())
        }
        
        if metadata:
            turn["metadata"] = metadata
        
        self.content["turns"].append(turn)
        self.update_content(self.content)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history with optional limit.
        
        Args:
            limit: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        if self.context_type != ContextType.CONVERSATION.value or not self.content:
            return []
        
        turns = self.content.get("turns", [])
        
        if limit:
            return turns[-limit:]
        return turns
    
    def get_token_count(self) -> int:
        """Get approximate token count for the context."""
        if self.token_count:
            return self.token_count
        
        # Simple approximation: 4 characters per token
        if self.raw_content:
            return len(self.raw_content) // 4
        elif self.content:
            return len(json.dumps(self.content)) // 4
        
        return 0
    
    def is_expired(self) -> bool:
        """Check if context has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def archive(self, reason: Optional[str] = None):
        """Archive the context."""
        self.is_archived = True
        self.is_active = False
        self.priority = ContextPriority.ARCHIVE.value
        
        if reason:
            if not self.extra_metadata:
                self.extra_metadata = {}
            self.extra_metadata["archive_reason"] = reason
            self.extra_metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """
        Convert context to dictionary representation.
        
        Args:
            include_content: Whether to include content data
            
        Returns:
            Dictionary representation of context
        """
        result = {
            "id": str(self.id),
            "context_key": self.context_key,
            "context_type": self.context_type,
            "context_scope": self.context_scope,
            "priority": self.priority,
            "title": self.title,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "token_count": self.get_token_count(),
            "size_bytes": self.size_bytes
        }
        
        if include_content:
            result.update({
                "content": self.content,
                "raw_content": self.raw_content,
                "metadata": self.extra_metadata
            })
        
        return result
    
    def __repr__(self):
        return (f"<ContextModel(id={self.id}, key={self.context_key}, "
                f"type={self.context_type}, scope={self.context_scope})>")


class ConversationContext(ContextModel):
    """
    Specialized context model for conversation management.
    Inherits from ContextModel with conversation-specific methods.
    """
    
    __mapper_args__ = {
        'polymorphic_identity': ContextType.CONVERSATION.value
    }
    
    def __init__(self, **kwargs):
        kwargs['context_type'] = ContextType.CONVERSATION.value
        super().__init__(**kwargs)
    
    def get_summary(self, max_length: int = 200) -> str:
        """Generate a summary of the conversation."""
        turns = self.get_conversation_history()
        if not turns:
            return "Empty conversation"
        
        # Simple summary generation
        total_text = " ".join([turn.get("message", "") for turn in turns])
        if len(total_text) <= max_length:
            return total_text
        
        return total_text[:max_length] + "..."
    
    def get_participants(self) -> List[str]:
        """Get list of conversation participants."""
        turns = self.get_conversation_history()
        participants = set()
        
        for turn in turns:
            if "role" in turn:
                participants.add(turn["role"])
        
        return list(participants)


class SessionContext(ContextModel):
    """
    Specialized context model for session management.
    Tracks session state and user interactions.
    """
    
    __mapper_args__ = {
        'polymorphic_identity': ContextType.SESSION.value
    }
    
    def __init__(self, **kwargs):
        kwargs['context_type'] = ContextType.SESSION.value
        super().__init__(**kwargs)
    
    def update_session_state(self, state_key: str, state_value: Any):
        """Update session state."""
        if not self.content:
            self.content = {"state": {}}
        
        if "state" not in self.content:
            self.content["state"] = {}
        
        self.content["state"][state_key] = state_value
        self.update_content(self.content)
    
    def get_session_state(self, state_key: str, default: Any = None) -> Any:
        """Get session state value."""
        if not self.content or "state" not in self.content:
            return default
        
        return self.content["state"].get(state_key, default)


class TaskContext(ContextModel):
    """
    Specialized context model for task-related context.
    Links to TaskModel and manages task-specific information.
    """
    
    __mapper_args__ = {
        'polymorphic_identity': ContextType.TASK.value
    }
    
    def __init__(self, **kwargs):
        kwargs['context_type'] = ContextType.TASK.value
        super().__init__(**kwargs)
    
    def add_task_event(self, event_type: str, event_data: Dict[str, Any]):
        """Add a task event to the context."""
        if not self.content:
            self.content = {"events": []}
        
        if "events" not in self.content:
            self.content["events"] = []
        
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4())
        }
        
        self.content["events"].append(event)
        self.update_content(self.content)
    
    def get_task_events(self, event_type: Optional[str] = None) -> List[Dict]:
        """Get task events, optionally filtered by type."""
        if not self.content or "events" not in self.content:
            return []
        
        events = self.content["events"]
        
        if event_type:
            return [event for event in events if event.get("type") == event_type]
        
        return events