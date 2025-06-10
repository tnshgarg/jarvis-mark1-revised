"""
Database initialization and connection management (Simplified Version)
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

import structlog
from sqlalchemy import Column, String, Integer, create_engine, MetaData, Text, Boolean, JSON, DateTime, ForeignKey, Table, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql import func

from mark1.config.settings import get_settings
from mark1.utils.exceptions import DatabaseError

# Create declarative base for all models
Base = declarative_base()

# Get the async session
_session_factory = None
_engine = None
_sync_engine = None

# Simple agent status enum
class AgentStatus(str, Enum):
    """Agent status enumeration"""
    DISCOVERED = "discovered"
    ANALYZING = "analyzing"
    READY = "ready"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"
    ARCHIVED = "archived"

# Simple agent type enum
class AgentType(str, Enum):
    """Agent type enumeration"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    CREWAI = "crewai"
    CUSTOM = "custom"
    UNKNOWN = "unknown"

# Simplified Agent model that works with SQLite
class Agent(Base):
    """Simplified Agent model for SQLite compatibility"""
    __tablename__ = 'agents'
    
    # Primary key as String instead of UUID for SQLite compatibility
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    agent_type = Column(String(50), nullable=False, default=AgentType.UNKNOWN.value)
    status = Column(String(50), nullable=False, default=AgentStatus.DISCOVERED.value)
    
    # Basic fields
    file_path = Column(String(1000), nullable=True)
    capabilities = Column(JSON, nullable=True)  # List of capability strings
    agent_metadata = Column(JSON, nullable=True)  # Flexible metadata
    framework = Column(String(100), nullable=True)  # Framework name
    confidence = Column(Integer, nullable=False, default=100)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    last_activity = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Agent(id={self.id}, name='{self.name}', type='{self.agent_type}', status='{self.status}')>"

class DatabaseRepository:
    """
    Base repository class with common database operations
    All specific repositories should inherit from this class
    """
    
    def __init__(self, db_manager=None):
        # db_manager is ignored in this simplified version
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def create(self, session: AsyncSession, model_instance) -> Any:
        """Create a new record"""
        try:
            session.add(model_instance)
            await session.flush()
            await session.refresh(model_instance)
            return model_instance
        except Exception as e:
            self.logger.error("Failed to create record", error=str(e))
            raise DatabaseError(f"Create operation failed: {e}")
    
    async def get_by_id(self, session: AsyncSession, model_class, record_id: Any) -> Optional[Any]:
        """Get record by ID"""
        try:
            result = await session.get(model_class, record_id)
            return result
        except Exception as e:
            self.logger.error("Failed to get record by ID", error=str(e))
            raise DatabaseError(f"Get by ID operation failed: {e}")


class AgentRepository(DatabaseRepository):
    """
    Repository for agent data access operations - Simplified version
    
    Provides methods for CRUD operations on agents
    """
    
    async def create_agent(
        self,
        session: AsyncSession,
        name: str,
        agent_type: str,
        framework: str,
        file_path: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: int = 100
    ) -> Agent:
        """Create a new agent record"""
        try:
            agent = Agent(
                name=name,
                agent_type=agent_type,
                status=AgentStatus.READY.value,
                framework=framework,
                file_path=file_path,
                capabilities=capabilities or [],
                agent_metadata=metadata or {},
                confidence=confidence,
                created_at=datetime.now(timezone.utc)
            )
            
            created_agent = await self.create(session, agent)
            self.logger.info("Agent created", agent_id=created_agent.id, name=name)
            
            return created_agent
            
        except Exception as e:
            self.logger.error("Failed to create agent", name=name, error=str(e))
            raise DatabaseError(f"Agent creation failed: {e}")
    
    async def get_agent_by_id(self, session: AsyncSession, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        try:
            agent = await self.get_by_id(session, Agent, agent_id)
            if agent:
                agent.last_activity = datetime.now(timezone.utc)
                await session.commit()
            
            return agent
            
        except Exception as e:
            self.logger.error("Failed to get agent by ID", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent retrieval failed: {e}")
    
    async def get_agent_by_name(self, session: AsyncSession, name: str) -> Optional[Agent]:
        """Get agent by name"""
        try:
            query = select(Agent).where(Agent.name == name)
            result = await session.execute(query)
            agent = result.scalar_one_or_none()
            
            if agent:
                agent.last_activity = datetime.now(timezone.utc)
                await session.commit()
            
            return agent
            
        except Exception as e:
            self.logger.error("Failed to get agent by name", name=name, error=str(e))
            raise DatabaseError(f"Agent retrieval failed: {e}")
    
    async def list_all(self, session: AsyncSession) -> List[Agent]:
        """Get all agents"""
        try:
            query = select(Agent).order_by(Agent.created_at.desc())
            result = await session.execute(query)
            agents = result.scalars().all()
            
            return list(agents)
            
        except Exception as e:
            self.logger.error("Failed to list all agents", error=str(e))
            raise DatabaseError(f"Agent listing failed: {e}")


class DatabaseError(Exception):
    """Database exception"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"[MARK1_DATABASEERROR] {message}")


async def init_database(
    settings_override: Optional[Dict[str, Any]] = None,
    force_recreate: bool = False
) -> None:
    """Initialize the database connection"""
    global _session_factory, _engine, _sync_engine

    logger = structlog.get_logger(__name__)

    try:
        # Use SQLite by default
        db_path = "data/mark1_db.sqlite"
        
        # Apply settings override if provided
        if settings_override and "database_url" in settings_override:
            db_url = settings_override["database_url"]
            # Extract path from sqlite URL
            if db_url.startswith("sqlite:///"):
                db_path = db_url.replace("sqlite:///", "")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Explicitly use aiosqlite for async operations
        db_url = f"sqlite+aiosqlite:///{db_path}"
        
        # Regular sqlite for sync operations
        sync_db_url = f"sqlite:///{db_path}"
        
        # Create engines with connect_args for SQLite
        _engine = create_async_engine(
            db_url,
            echo=settings_override.get("database_echo", False) if settings_override else False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        
        # Create sync engine for table creation
        _sync_engine = create_engine(
            sync_db_url,
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        
        # Create session factory
        _session_factory = sessionmaker(
            class_=AsyncSession,
            autocommit=False,
            autoflush=True,
            bind=_engine,
            expire_on_commit=False
        )
        
        # Create tables directly using the simplified models
        try:
            # Drop existing tables if force_recreate is True
            if force_recreate:
                logger.warning("Dropping existing tables due to force_recreate=True")
                Base.metadata.drop_all(_sync_engine)
            
            # Create tables with sync engine
            Base.metadata.create_all(_sync_engine)
            logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.warning(f"Issue with table creation: {e}")
            # Continue execution - tables might already exist
        
        logger.info("Database initialized successfully", database_url=db_url)
        
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise DatabaseError(f"Database operation failed: Table creation failed: {e}")


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