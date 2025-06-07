"""
Context Manager for Mark-1 Orchestrator

Manages context sharing, memory, and state across agents and tasks.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.storage.models.context_model import ContextModel, ContextType, ContextScope
from mark1.storage.repositories.context_repository import ContextRepository
from mark1.storage.database import get_db_session
from mark1.utils.exceptions import ContextError
from mark1.utils.constants import MAX_CONTEXT_SIZE, CONTEXT_RETENTION_DAYS


class ContextOperationType(Enum):
    """Types of context operations"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"


@dataclass
class ContextEntry:
    """Represents a context entry"""
    id: str
    key: str
    content: Dict[str, Any]
    context_type: ContextType
    scope: ContextScope
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class ContextOperationResult:
    """Result of a context operation"""
    success: bool
    context_id: Optional[str] = None
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class ContextManager:
    """
    Context management system for Mark-1
    
    Handles context storage, retrieval, sharing, and lifecycle management
    across agents, tasks, and sessions.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._initialized = False
        self._context_cache: Dict[str, ContextEntry] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the context manager"""
        try:
            self.logger.info("Initializing context manager...")
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_contexts())
            
            self._initialized = True
            self.logger.info("Context manager initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize context manager", error=str(e))
            raise ContextError(f"Context manager initialization failed: {e}")
    
    async def create_context(
        self,
        key: str,
        content: Dict[str, Any],
        context_type: ContextType,
        scope: ContextScope,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        expires_in_hours: Optional[int] = None
    ) -> ContextOperationResult:
        """
        Create a new context entry
        
        Args:
            key: Context key identifier
            content: Context content
            context_type: Type of context
            scope: Context scope
            agent_id: Optional agent ID
            task_id: Optional task ID
            expires_in_hours: Optional expiration time in hours
            
        Returns:
            ContextOperationResult with creation details
        """
        try:
            self.logger.info("Creating context", key=key, type=context_type.value)
            
            # Validate content size
            if self._calculate_content_size(content) > MAX_CONTEXT_SIZE:
                return ContextOperationResult(
                    success=False,
                    message=f"Content size exceeds maximum allowed size: {MAX_CONTEXT_SIZE}"
                )
            
            # Calculate expiration
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
            elif context_type == ContextType.SESSION:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)  # Default session expiry
            
            # Create context record
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                context = await context_repo.create_context(
                    session=session,
                    context_key=key,
                    context_type=context_type.value,
                    context_scope=scope.value,
                    agent_id=agent_id,
                    task_id=task_id,
                    content=content,
                    expires_at=expires_at
                )
                
                await session.commit()
                context_id = str(context.id)
            
            # Add to cache
            context_entry = ContextEntry(
                id=context_id,
                key=key,
                content=content,
                context_type=context_type,
                scope=scope,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                expires_at=expires_at
            )
            
            self._context_cache[context_id] = context_entry
            
            self.logger.info("Context created successfully", context_id=context_id, key=key)
            
            return ContextOperationResult(
                success=True,
                context_id=context_id,
                message="Context created successfully"
            )
            
        except Exception as e:
            self.logger.error("Failed to create context", key=key, error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Context creation failed: {e}"
            )
    
    async def get_context(
        self,
        context_id: Optional[str] = None,
        key: Optional[str] = None,
        context_type: Optional[ContextType] = None
    ) -> ContextOperationResult:
        """
        Retrieve context by ID or key
        
        Args:
            context_id: Context ID
            key: Context key
            context_type: Context type filter
            
        Returns:
            ContextOperationResult with context data
        """
        try:
            # Check cache first
            if context_id and context_id in self._context_cache:
                context_entry = self._context_cache[context_id]
                
                # Check expiration
                if context_entry.expires_at and datetime.now(timezone.utc) > context_entry.expires_at:
                    await self._remove_expired_context(context_id)
                    return ContextOperationResult(
                        success=False,
                        message="Context has expired"
                    )
                
                return ContextOperationResult(
                    success=True,
                    context_id=context_id,
                    data=context_entry.content,
                    message="Context retrieved from cache"
                )
            
            # Query database
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                if context_id:
                    context = await context_repo.get_context_by_id(session, context_id)
                elif key:
                    context_type_str = context_type.value if context_type else None
                    context = await context_repo.get_context_by_key(session, key, context_type_str)
                else:
                    return ContextOperationResult(
                        success=False,
                        message="Either context_id or key must be provided"
                    )
                
                if not context:
                    return ContextOperationResult(
                        success=False,
                        message="Context not found"
                    )
                
                # Check expiration
                if context.expires_at and datetime.now(timezone.utc) > context.expires_at:
                    return ContextOperationResult(
                        success=False,
                        message="Context has expired"
                    )
                
                # Add to cache
                context_entry = ContextEntry(
                    id=str(context.id),
                    key=context.context_key,
                    content=context.content or {},
                    context_type=ContextType(context.context_type),
                    scope=ContextScope(context.context_scope),
                    created_at=context.created_at,
                    updated_at=context.updated_at,
                    expires_at=context.expires_at
                )
                
                self._context_cache[str(context.id)] = context_entry
                
                return ContextOperationResult(
                    success=True,
                    context_id=str(context.id),
                    data=context.content or {},
                    message="Context retrieved successfully"
                )
                
        except Exception as e:
            self.logger.error("Failed to get context", context_id=context_id, key=key, error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Context retrieval failed: {e}"
            )
    
    async def update_context(
        self,
        context_id: str,
        content: Dict[str, Any],
        merge: bool = True
    ) -> ContextOperationResult:
        """
        Update context content
        
        Args:
            context_id: Context ID to update
            content: New content
            merge: Whether to merge with existing content
            
        Returns:
            ContextOperationResult with update status
        """
        try:
            self.logger.info("Updating context", context_id=context_id)
            
            # Validate content size
            if self._calculate_content_size(content) > MAX_CONTEXT_SIZE:
                return ContextOperationResult(
                    success=False,
                    message=f"Content size exceeds maximum allowed size: {MAX_CONTEXT_SIZE}"
                )
            
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                # Get existing context
                existing_context = await context_repo.get_context_by_id(session, context_id)
                if not existing_context:
                    return ContextOperationResult(
                        success=False,
                        message="Context not found"
                    )
                
                # Prepare new content
                if merge and existing_context.content:
                    new_content = existing_context.content.copy()
                    new_content.update(content)
                else:
                    new_content = content
                
                # Update context
                success = await context_repo.update_context_content(
                    session=session,
                    context_id=context_id,
                    content=new_content
                )
                
                if success:
                    await session.commit()
                    
                    # Update cache
                    if context_id in self._context_cache:
                        self._context_cache[context_id].content = new_content
                        self._context_cache[context_id].updated_at = datetime.now(timezone.utc)
                    
                    self.logger.info("Context updated successfully", context_id=context_id)
                    
                    return ContextOperationResult(
                        success=True,
                        context_id=context_id,
                        message="Context updated successfully"
                    )
                else:
                    return ContextOperationResult(
                        success=False,
                        message="Context update failed"
                    )
                    
        except Exception as e:
            self.logger.error("Failed to update context", context_id=context_id, error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Context update failed: {e}"
            )
    
    async def share_context(
        self,
        context_id: str,
        target_agent_id: str,
        permissions: List[str] = None
    ) -> ContextOperationResult:
        """
        Share context with another agent
        
        Args:
            context_id: Context to share
            target_agent_id: Agent to share with
            permissions: List of permissions (read, write, etc.)
            
        Returns:
            ContextOperationResult with sharing status
        """
        try:
            self.logger.info("Sharing context", context_id=context_id, target_agent=target_agent_id)
            
            # Get the context
            result = await self.get_context(context_id=context_id)
            if not result.success:
                return result
            
            # Create a shared context entry
            # For now, we'll create a copy with shared scope
            # In a more advanced implementation, we'd have proper permission management
            
            shared_result = await self.create_context(
                key=f"shared_{context_id}_{target_agent_id}",
                content=result.data,
                context_type=ContextType.MEMORY,  # Shared contexts are memory type
                scope=ContextScope.AGENT,
                agent_id=target_agent_id
            )
            
            if shared_result.success:
                self.logger.info("Context shared successfully", 
                               original_context=context_id, 
                               shared_context=shared_result.context_id,
                               target_agent=target_agent_id)
            
            return shared_result
            
        except Exception as e:
            self.logger.error("Failed to share context", 
                            context_id=context_id, 
                            target_agent=target_agent_id, 
                            error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Context sharing failed: {e}"
            )
    
    async def list_contexts(
        self,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        context_type: Optional[ContextType] = None,
        scope: Optional[ContextScope] = None,
        limit: int = 100
    ) -> List[ContextEntry]:
        """
        List contexts with optional filtering
        
        Args:
            agent_id: Filter by agent ID
            task_id: Filter by task ID
            context_type: Filter by context type
            scope: Filter by scope
            limit: Maximum results to return
            
        Returns:
            List of context entries
        """
        try:
            contexts = []
            
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                db_contexts = await context_repo.list_contexts(
                    session=session,
                    limit=limit,
                    context_type=context_type.value if context_type else None,
                    context_scope=scope.value if scope else None
                )
                
                for context in db_contexts:
                    # Skip expired contexts
                    if context.expires_at and datetime.now(timezone.utc) > context.expires_at:
                        continue
                    
                    # Apply additional filters
                    if agent_id and context.agent_id != agent_id:
                        continue
                    
                    if task_id and context.task_id != task_id:
                        continue
                    
                    context_entry = ContextEntry(
                        id=str(context.id),
                        key=context.context_key,
                        content=context.content or {},
                        context_type=ContextType(context.context_type),
                        scope=ContextScope(context.context_scope),
                        created_at=context.created_at,
                        updated_at=context.updated_at,
                        expires_at=context.expires_at
                    )
                    
                    contexts.append(context_entry)
            
            return contexts
            
        except Exception as e:
            self.logger.error("Failed to list contexts", error=str(e))
            return []
    
    async def _cleanup_expired_contexts(self) -> None:
        """Background task to clean up expired contexts"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                current_time = datetime.now(timezone.utc)
                expired_context_ids = []
                
                # Check cache for expired contexts
                for context_id, context_entry in self._context_cache.items():
                    if context_entry.expires_at and current_time > context_entry.expires_at:
                        expired_context_ids.append(context_id)
                
                # Remove expired contexts
                for context_id in expired_context_ids:
                    await self._remove_expired_context(context_id)
                
                if expired_context_ids:
                    self.logger.info("Cleaned up expired contexts", count=len(expired_context_ids))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Context cleanup failed", error=str(e))
                await asyncio.sleep(3600)  # Wait before retrying
    
    async def _remove_expired_context(self, context_id: str) -> None:
        """Remove an expired context"""
        try:
            # Remove from cache
            if context_id in self._context_cache:
                del self._context_cache[context_id]
            
            # Archive in database (don't delete, just mark as archived)
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                context = await context_repo.get_context_by_id(session, context_id)
                
                if context:
                    context.is_archived = True
                    context.is_active = False
                    await session.commit()
                    
        except Exception as e:
            self.logger.error("Failed to remove expired context", context_id=context_id, error=str(e))
    
    def _calculate_content_size(self, content: Dict[str, Any]) -> int:
        """Calculate the size of content in bytes"""
        import json
        try:
            return len(json.dumps(content).encode('utf-8'))
        except Exception:
            return 0
    
    async def shutdown(self) -> None:
        """Shutdown the context manager"""
        try:
            self.logger.info("Shutting down context manager...")
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            self._context_cache.clear()
            self._initialized = False
            
            self.logger.info("Context manager shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during context manager shutdown", error=str(e))
    
    @property
    def is_initialized(self) -> bool:
        """Check if the context manager is initialized"""
        return self._initialized
    
    @property
    def cache_size(self) -> int:
        """Get current cache size"""
        return len(self._context_cache)
