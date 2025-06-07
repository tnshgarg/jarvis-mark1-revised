"""
Task Repository for Mark-1 Orchestrator

Provides data access layer for task-related database operations.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
import structlog

from mark1.storage.database import DatabaseRepository
from mark1.storage.models.task_model import Task, TaskStatus
from mark1.utils.exceptions import DatabaseError


class TaskRepository(DatabaseRepository):
    """
    Repository for task data access operations
    
    Provides methods for creating, reading, updating, and deleting tasks.
    """
    
    def __init__(self, db_manager=None):
        super().__init__(db_manager)
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def create_task(
        self,
        session: AsyncSession,
        description: str,
        status: TaskStatus = TaskStatus.PENDING,
        **kwargs
    ) -> Task:
        """Create a new task record"""
        try:
            task = Task(
                description=description,
                status=status,
                created_at=datetime.now(timezone.utc),
                **kwargs
            )
            
            created_task = await self.create(session, task)
            self.logger.info("Task created", task_id=created_task.id, description=description)
            
            return created_task
            
        except Exception as e:
            self.logger.error("Failed to create task", description=description, error=str(e))
            raise DatabaseError(f"Task creation failed: {e}")
    
    async def get_task_by_id(self, session: AsyncSession, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        try:
            task = await self.get_by_id(session, Task, task_id)
            return task
            
        except Exception as e:
            self.logger.error("Failed to get task by ID", task_id=task_id, error=str(e))
            raise DatabaseError(f"Task retrieval failed: {e}")
    
    async def list_tasks(
        self,
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100,
        status_filter: Optional[TaskStatus] = None
    ) -> List[Task]:
        """List tasks with optional filtering and pagination"""
        try:
            query = select(Task)
            
            if status_filter:
                query = query.where(Task.status == status_filter)
            
            query = query.offset(offset).limit(limit).order_by(Task.created_at.desc())
            
            result = await session.execute(query)
            tasks = result.scalars().all()
            
            return list(tasks)
            
        except Exception as e:
            self.logger.error("Failed to list tasks", error=str(e))
            raise DatabaseError(f"Task listing failed: {e}")
    
    async def update_task_status(
        self,
        session: AsyncSession,
        task_id: str,
        status: TaskStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status"""
        try:
            query = (
                update(Task)
                .where(Task.id == task_id)
                .values(
                    status=status,
                    error_message=error_message,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            
            result = await session.execute(query)
            
            if result.rowcount == 0:
                return False
            
            self.logger.info("Task status updated", task_id=task_id, status=status)
            return True
            
        except Exception as e:
            self.logger.error("Failed to update task status", task_id=task_id, error=str(e))
            raise DatabaseError(f"Task status update failed: {e}")
