"""
Context Repository for Mark-1 Orchestrator

Provides data access layer for context-related database operations.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
import structlog

from mark1.storage.database import DatabaseRepository
from mark1.storage.models.context_model import ContextModel
from mark1.utils.exceptions import DatabaseError, ContextError


class ContextRepository(DatabaseRepository):
    """
    Repository for context data access operations
    
    Provides methods for creating, reading, updating, and deleting contexts.
    """
    
    def __init__(self, db_manager=None):
        super().__init__(db_manager)
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def create_context(
        self,
        session: AsyncSession,
        context_key: str,
        context_type: str,
        context_scope: str,
        **kwargs
    ) -> ContextModel:
        """Create a new context record"""
        try:
            context = ContextModel(
                context_key=context_key,
                context_type=context_type,
                context_scope=context_scope,
                created_at=datetime.now(timezone.utc),
                **kwargs
            )
            
            created_context = await self.create(session, context)
            self.logger.info("Context created", context_id=created_context.id, context_key=context_key)
            
            return created_context
            
        except Exception as e:
            self.logger.error("Failed to create context", context_key=context_key, error=str(e))
            raise DatabaseError(f"Context creation failed: {e}")
    
    async def get_context_by_id(self, session: AsyncSession, context_id: str) -> Optional[ContextModel]:
        """Get context by ID"""
        try:
            context = await self.get_by_id(session, ContextModel, context_id)
            return context
            
        except Exception as e:
            self.logger.error("Failed to get context by ID", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context retrieval failed: {e}")
    
    async def get_context_by_key(
        self,
        session: AsyncSession,
        context_key: str,
        context_type: Optional[str] = None
    ) -> Optional[ContextModel]:
        """Get context by key and optional type"""
        try:
            query = select(ContextModel).where(ContextModel.context_key == context_key)
            
            if context_type:
                query = query.where(ContextModel.context_type == context_type)
            
            result = await session.execute(query)
            context = result.scalar_one_or_none()
            
            return context
            
        except Exception as e:
            self.logger.error("Failed to get context by key", context_key=context_key, error=str(e))
            raise DatabaseError(f"Context retrieval failed: {e}")
    
    async def list_contexts(
        self,
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100,
        context_type: Optional[str] = None,
        context_scope: Optional[str] = None
    ) -> List[ContextModel]:
        """List contexts with optional filtering and pagination"""
        try:
            query = select(ContextModel)
            
            if context_type:
                query = query.where(ContextModel.context_type == context_type)
            
            if context_scope:
                query = query.where(ContextModel.context_scope == context_scope)
            
            query = query.offset(offset).limit(limit).order_by(ContextModel.created_at.desc())
            
            result = await session.execute(query)
            contexts = result.scalars().all()
            
            return list(contexts)
            
        except Exception as e:
            self.logger.error("Failed to list contexts", error=str(e))
            raise DatabaseError(f"Context listing failed: {e}")
    
    async def update_context_content(
        self,
        session: AsyncSession,
        context_id: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update context content"""
        try:
            context = await self.get_by_id(session, ContextModel, context_id)
            if not context:
                return False
            
            context.update_content(content, metadata)
            await session.commit()
            
            self.logger.info("Context content updated", context_id=context_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to update context content", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context content update failed: {e}")
