"""
Database abstraction layer for Mark-1 Orchestrator
Provides unified interface for SQLAlchemy operations with async support
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData, event
from sqlalchemy.pool import StaticPool
import structlog
from datetime import datetime, timezone

from mark1.config.settings import get_settings
from mark1.utils.exceptions import DatabaseError, ConfigurationError

logger = structlog.get_logger(__name__)

# Create declarative base for all models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


class DatabaseManager:
    """
    Centralized database management for Mark-1 system
    Handles connection pooling, session management, and health monitoring
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize database connection and create session factory"""
        if self._is_initialized:
            logger.warning("Database already initialized")
            return
            
        try:
            # Get async URL
            database_url = self.settings.get_database_url(async_driver=True)
            
            # Get connection arguments and engine parameters
            connect_args = self._get_connect_args()
            engine_kwargs = {}
            
            # Handle SQLite-specific parameters
            if self.settings.database.url.startswith('sqlite'):
                engine_kwargs['poolclass'] = StaticPool
                # Remove poolclass from connect_args if it exists
                connect_args.pop('poolclass', None)
            
            # Create async engine with optimized settings
            engine_params = {
                'echo': self.settings.database.echo,
                'pool_pre_ping': True,  # Verify connections before use
                'connect_args': connect_args,
                **engine_kwargs
            }
            
            # Add pool parameters only for non-SQLite databases
            if not self.settings.database.url.startswith('sqlite'):
                engine_params.update({
                    'pool_size': self.settings.database.pool_size,
                    'max_overflow': self.settings.database.max_overflow,
                    'pool_timeout': self.settings.database.pool_timeout,
                    'pool_recycle': self.settings.database.pool_recycle,
                })
            
            self._engine = create_async_engine(database_url, **engine_params)
            
            # Create async session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True
            )
            
            # Set up event listeners
            self._setup_event_listeners()
            
            self._is_initialized = True
            
            # Create masked URL for logging (hide password)
            masked_url = self._mask_database_url(database_url)
            logger.info("Database initialized successfully", database_url=masked_url)
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _get_connect_args(self) -> Dict[str, Any]:
        """Get database-specific connection arguments"""
        connect_args = {}
        
        if self.settings.database.url.startswith('sqlite'):
            connect_args.update({
                'check_same_thread': False,
            })
        elif self.settings.database.url.startswith('postgresql'):
            connect_args.update({
                'server_settings': {
                    'application_name': 'mark1_orchestrator',
                    'jit': 'off'  # Disable JIT for better connection times
                }
            })
            
        return connect_args
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite-specific pragmas for better performance"""
            if self.settings.database.url.startswith('sqlite'):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.close()
        
        @event.listens_for(self._engine.sync_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries for monitoring"""
            conn.info.setdefault('query_start_time', []).append(datetime.now(timezone.utc))
            
        @event.listens_for(self._engine.sync_engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log query execution time"""
            total = datetime.now(timezone.utc) - conn.info['query_start_time'].pop(-1)
            slow_query_threshold = 1.0  # Default 1 second threshold
            if total.total_seconds() > slow_query_threshold:
                logger.warning("Slow query detected", 
                             duration=total.total_seconds(),
                             query=statement[:200])
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with automatic cleanup
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session for operations
        """
        if not self._is_initialized:
            await self.initialize()
            
        if not self._session_factory:
            raise DatabaseError("Session factory not initialized")
            
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()
    
    async def create_all_tables(self) -> None:
        """Create all database tables"""
        if not self._engine:
            await self.initialize()
            
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("All database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseError(f"Table creation failed: {e}")
    
    async def drop_all_tables(self) -> None:
        """Drop all database tables (use with caution)"""
        if not self._engine:
            await self.initialize()
            
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise DatabaseError(f"Table drop failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        if not self._is_initialized:
            return {"status": "not_initialized", "healthy": False}
            
        try:
            async with self.get_session() as session:
                # Simple query to test connection
                result = await session.execute("SELECT 1")
                result.fetchone()
                
                # Get connection pool status
                pool_status = {
                    "size": self._engine.pool.size(),
                    "checked_in": self._engine.pool.checkedin(),
                    "checked_out": self._engine.pool.checkedout(),
                }
                
                return {
                    "status": "healthy",
                    "healthy": True,
                    "pool_status": pool_status,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def close(self) -> None:
        """Close database connections and cleanup"""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
        self._is_initialized = False

    def _mask_database_url(self, url: str) -> str:
        """Mask sensitive information in database URL for logging"""
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            if parsed.password:
                # Replace password with asterisks
                netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
                masked_parsed = parsed._replace(netloc=netloc)
                return urlunparse(masked_parsed)
            return url
        except Exception:
            return url.split('@')[1] if '@' in url else url


class DatabaseRepository:
    """
    Base repository class with common database operations
    All specific repositories should inherit from this class
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
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
            from sqlalchemy import select
            
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


# Global database manager instance
db_manager = DatabaseManager()


# Utility functions for common database operations
async def init_database() -> None:
    """Initialize database and create tables"""
    await db_manager.initialize()
    await db_manager.create_all_tables()


async def close_database() -> None:
    """Close database connections"""
    await db_manager.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Convenience function to get database session"""
    async with db_manager.get_session() as session:
        yield session


# Database health check utility
async def check_database_health() -> Dict[str, Any]:
    """Check database health status"""
    return await db_manager.health_check()