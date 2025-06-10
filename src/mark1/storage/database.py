"""
Database initialization and connection management
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime, timezone

import structlog
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mark1.config.settings import get_settings
from mark1.utils.exceptions import DatabaseError

# Initialize logger
logger = structlog.get_logger(__name__)

# Create declarative base for all models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

# Get the async session
_session_factory = None
_engine: Optional[AsyncEngine] = None
_sync_engine = None


class DatabaseRepository:
    """
    Base repository class with common database operations
    All specific repositories should inherit from this class
    """
    
    def __init__(self, db_manager=None):
        # db_manager parameter is for compatibility, not used in this implementation
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def create(self, session: AsyncSession, model_instance) -> Any:
        """Create a new record"""
        try:
            session.add(model_instance)
            await session.flush()
            await session.refresh(model_instance)
            self.logger.debug("Created record", model=model_instance.__class__.__name__, id=getattr(model_instance, 'id', None))
            return model_instance
        except Exception as e:
            self.logger.error("Failed to create record", error=str(e))
            raise DatabaseError(f"Create operation failed: {e}")
    
    async def get_by_id(self, session: AsyncSession, model_class, record_id: Any) -> Optional[Any]:
        """Get record by ID"""
        try:
            result = await session.get(model_class, record_id)
            self.logger.debug("Retrieved record by ID", model=model_class.__name__, id=record_id, found=result is not None)
            return result
        except Exception as e:
            self.logger.error("Failed to get record by ID", error=str(e), model=model_class.__name__, id=record_id)
            raise DatabaseError(f"Get by ID operation failed: {e}")
    
    async def update(self, session: AsyncSession, model_instance, **kwargs) -> Any:
        """Update a record"""
        try:
            for key, value in kwargs.items():
                if hasattr(model_instance, key):
                    setattr(model_instance, key, value)
            
            # Update timestamp if model has updated_at field
            if hasattr(model_instance, 'updated_at'):
                model_instance.updated_at = datetime.now(timezone.utc)
            
            await session.flush()
            await session.refresh(model_instance)
            self.logger.debug("Updated record", model=model_instance.__class__.__name__, id=getattr(model_instance, 'id', None))
            return model_instance
        except Exception as e:
            self.logger.error("Failed to update record", error=str(e))
            raise DatabaseError(f"Update operation failed: {e}")
    
    async def delete(self, session: AsyncSession, model_instance) -> bool:
        """Delete a record"""
        try:
            await session.delete(model_instance)
            await session.flush()
            self.logger.debug("Deleted record", model=model_instance.__class__.__name__, id=getattr(model_instance, 'id', None))
            return True
        except Exception as e:
            self.logger.error("Failed to delete record", error=str(e))
            raise DatabaseError(f"Delete operation failed: {e}")
    
    async def list_with_pagination(self, session: AsyncSession, model_class, offset: int = 0, limit: int = 100, **filters) -> List[Any]:
        """List records with pagination and filtering"""
        try:
            query = select(model_class)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(model_class, key):
                    query = query.where(getattr(model_class, key) == value)
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            records = result.scalars().all()
            
            self.logger.debug("Listed records with pagination", 
                            model=model_class.__name__, 
                            count=len(records), 
                            offset=offset, 
                            limit=limit)
            return records
        except Exception as e:
            self.logger.error("Failed to list records", error=str(e))
            raise DatabaseError(f"List operation failed: {e}")


async def init_database(
    settings_override: Optional[Dict[str, Any]] = None,
    force_recreate: bool = False
) -> AsyncEngine:
    """
    Initialize the database, creating tables if they don't exist.
    
    Args:
        settings_override: Optional override for database settings
        force_recreate: Whether to drop and recreate all tables
        
    Returns:
        AsyncEngine instance
    """
    global _engine, _session_factory, _sync_engine
    
    try:
        settings = get_settings()
        
        if settings_override:
            for key, value in settings_override.items():
                setattr(settings, key, value)
        
        db_url = settings.database_url
        
        # Use aiosqlite for async SQLite
        if db_url.startswith("sqlite:"):
            # Convert to async URL
            if not db_url.startswith("sqlite+aiosqlite"):
                db_url = db_url.replace("sqlite:", "sqlite+aiosqlite:")
        
        logger.info(f"Initializing database with URL: {db_url}")
        
        connect_args = {}
        if "sqlite" in db_url:
            connect_args["check_same_thread"] = False
        
        # Create the async engine
        _engine = create_async_engine(
            db_url,
            echo=settings.database_echo,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True
        )
        
        # Create session factory
        _session_factory = sessionmaker(
            _engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        
        # Create the tables
        async with _engine.begin() as conn:
            if force_recreate:
                logger.warning("Dropping all tables!")
                await conn.run_sync(Base.metadata.drop_all)
            
            from mark1.storage.models.agent_model import Agent, Capability, agent_capabilities
            from mark1.storage.models.task_model import Task
            from mark1.storage.models.context_model import ContextModel
            
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        return _engine
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseError(f"Database initialization failed: {e}")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session"""
    global _session_factory
    
    if not _session_factory:
        await init_database()
    
    if not _session_factory:
        raise DatabaseError("Database not initialized")
    
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise
    finally:
        await session.close()


async def close_database() -> None:
    """Close database connections"""
    global _engine, _sync_engine
    
    if _engine:
        await _engine.dispose()
        _engine = None
        
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None 