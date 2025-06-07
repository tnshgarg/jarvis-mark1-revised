"""
Agent Pool Management System for Mark-1 Orchestrator

Manages agent lifecycle, execution scheduling, resource allocation,
and provides load balancing across available agents.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass
from enum import Enum
import structlog
from concurrent.futures import ThreadPoolExecutor
import psutil
import signal

from mark1.config.settings import get_settings
from mark1.storage.database import get_db_session
from mark1.storage.models.agent_model import Agent, AgentStatus
from mark1.storage.repositories.agent_repository import AgentRepository
from mark1.utils.exceptions import (
    AgentException,
    AgentExecutionException,
    OrchestrationException,
    SecurityException
)


class AgentExecutionStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AgentExecution:
    """Information about agent execution"""
    execution_id: str
    agent_id: str
    task_id: str
    status: AgentExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    resource_usage: Optional[Dict[str, float]] = None
    

@dataclass
class AgentPoolStats:
    """Agent pool statistics"""
    total_agents: int
    available_agents: int
    busy_agents: int
    failed_agents: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    cpu_usage: float
    memory_usage: float


class AgentWorker:
    """Individual agent worker with resource monitoring"""
    
    def __init__(self, agent_id: str, max_memory_mb: int = 512, max_cpu_percent: int = 50):
        self.agent_id = agent_id
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.logger = structlog.get_logger(f"AgentWorker.{agent_id}")
        
        # Execution state
        self.current_execution: Optional[AgentExecution] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.process: Optional[Any] = None
        
        # Resource monitoring
        self.start_memory = 0
        self.start_cpu_time = 0
        
    async def initialize(self) -> None:
        """Initialize the agent worker"""
        try:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"agent-{self.agent_id}")
            self.logger.info("Agent worker initialized", agent_id=self.agent_id)
        except Exception as e:
            self.logger.error("Failed to initialize agent worker", agent_id=self.agent_id, error=str(e))
            raise AgentException(f"Worker initialization failed: {e}", self.agent_id)
    
    async def execute_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        timeout: float = 300.0
    ) -> AgentExecution:
        """Execute a task with resource monitoring"""
        execution_id = f"{self.agent_id}-{task_id}-{int(datetime.now().timestamp())}"
        
        execution = AgentExecution(
            execution_id=execution_id,
            agent_id=self.agent_id,
            task_id=task_id,
            status=AgentExecutionStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )
        
        self.current_execution = execution
        self.logger.info("Starting task execution", execution_id=execution_id, task_id=task_id)
        
        try:
            # Start resource monitoring
            self._start_resource_monitoring()
            
            # Execute the task with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._execute_task_sync, task_data),
                timeout=timeout
            )
            
            # Successful completion
            execution.status = AgentExecutionStatus.COMPLETED
            execution.result = result
            execution.end_time = datetime.now(timezone.utc)
            execution.resource_usage = self._get_resource_usage()
            
            self.logger.info("Task execution completed", execution_id=execution_id)
            
        except asyncio.TimeoutError:
            execution.status = AgentExecutionStatus.TIMEOUT
            execution.error = f"Task execution timeout after {timeout}s"
            execution.end_time = datetime.now(timezone.utc)
            
            self.logger.warning("Task execution timeout", execution_id=execution_id, timeout=timeout)
            
        except Exception as e:
            execution.status = AgentExecutionStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now(timezone.utc)
            
            self.logger.error("Task execution failed", execution_id=execution_id, error=str(e))
            
        finally:
            self.current_execution = None
            
        return execution
    
    def _execute_task_sync(self, task_data: Dict[str, Any]) -> Any:
        """Synchronous task execution (runs in thread pool)"""
        # This is where the actual agent execution would happen
        # For now, we'll simulate task execution
        import time
        import random
        
        # Simulate some work
        processing_time = random.uniform(1.0, 5.0)
        time.sleep(processing_time)
        
        # Simulate potential failures
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Simulated task failure")
        
        return {
            "status": "completed",
            "processing_time": processing_time,
            "result_data": f"Task completed by agent {self.agent_id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _start_resource_monitoring(self) -> None:
        """Start monitoring resource usage"""
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.start_cpu_time = process.cpu_times().user
        except Exception as e:
            self.logger.warning("Failed to start resource monitoring", error=str(e))
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            current_cpu_time = process.cpu_times().user
            
            return {
                "memory_mb": current_memory,
                "memory_delta_mb": current_memory - self.start_memory,
                "cpu_time_delta": current_cpu_time - self.start_cpu_time,
                "cpu_percent": process.cpu_percent()
            }
        except Exception as e:
            self.logger.warning("Failed to get resource usage", error=str(e))
            return {}
    
    async def cancel_execution(self) -> bool:
        """Cancel current execution"""
        if not self.current_execution:
            return False
        
        try:
            # Set status to cancelled
            self.current_execution.status = AgentExecutionStatus.CANCELLED
            self.current_execution.end_time = datetime.now(timezone.utc)
            
            # Try to interrupt the executor
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"agent-{self.agent_id}")
            
            self.logger.info("Execution cancelled", agent_id=self.agent_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to cancel execution", agent_id=self.agent_id, error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the agent worker"""
        try:
            if self.current_execution:
                await self.cancel_execution()
            
            if self.executor:
                self.executor.shutdown(wait=True)
                
            self.logger.info("Agent worker shutdown", agent_id=self.agent_id)
            
        except Exception as e:
            self.logger.error("Error during worker shutdown", agent_id=self.agent_id, error=str(e))


class AgentPool:
    """
    Agent pool management system
    
    Manages a pool of agent workers, handles load balancing,
    resource allocation, and execution scheduling.
    """
    
    def __init__(self, max_concurrent: int = 10, agent_timeout: int = 300):
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)
        
        # Pool configuration
        self.max_concurrent = max_concurrent
        self.agent_timeout = agent_timeout
        
        # Pool state
        self.workers: Dict[str, AgentWorker] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.active_executions: Dict[str, AgentExecution] = {}
        self.execution_history: List[AgentExecution] = []
        
        # Statistics
        self.stats = AgentPoolStats(
            total_agents=0,
            available_agents=0,
            busy_agents=0,
            failed_agents=0,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            average_execution_time=0.0,
            cpu_usage=0.0,
            memory_usage=0.0
        )
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the agent pool"""
        if self._initialized:
            self.logger.warning("Agent pool already initialized")
            return
        
        try:
            self.logger.info("Initializing agent pool", max_concurrent=self.max_concurrent)
            
            # Load available agents from database
            await self._load_available_agents()
            
            # Start background tasks
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._scheduler_task = asyncio.create_task(self._scheduling_loop())
            
            self._initialized = True
            self.logger.info("Agent pool initialized", agent_count=len(self.workers))
            
        except Exception as e:
            self.logger.error("Failed to initialize agent pool", error=str(e))
            raise OrchestrationException(f"Agent pool initialization failed: {e}")
    
    async def _load_available_agents(self) -> None:
        """Load available agents from database and create workers"""
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                agents = await agent_repo.list_by_status(session, AgentStatus.READY)
                
                for agent in agents:
                    await self._create_worker(agent.id)
                    
        except Exception as e:
            self.logger.error("Failed to load available agents", error=str(e))
            raise
    
    async def _create_worker(self, agent_id: str) -> AgentWorker:
        """Create a new agent worker"""
        try:
            worker = AgentWorker(
                agent_id=agent_id,
                max_memory_mb=self.settings.agents.sandbox_memory_limit_mb,
                max_cpu_percent=self.settings.agents.sandbox_cpu_limit_percent
            )
            
            await worker.initialize()
            self.workers[agent_id] = worker
            
            self.logger.debug("Agent worker created", agent_id=agent_id)
            return worker
            
        except Exception as e:
            self.logger.error("Failed to create agent worker", agent_id=agent_id, error=str(e))
            raise AgentException(f"Worker creation failed: {e}", agent_id)
    
    async def submit_task(
        self,
        agent_id: str,
        task_id: str,
        task_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """
        Submit a task for execution
        
        Args:
            agent_id: ID of the agent to execute the task
            task_id: Task identifier
            task_data: Task data and parameters
            priority: Task priority (higher = more priority)
            
        Returns:
            Execution ID
        """
        if not self._initialized:
            raise OrchestrationException("Agent pool not initialized")
        
        if agent_id not in self.workers:
            # Try to create worker for new agent
            try:
                await self._create_worker(agent_id)
            except Exception as e:
                raise AgentException(f"Agent not available: {e}", agent_id)
        
        # Create task submission
        execution_id = f"{agent_id}-{task_id}-{int(datetime.now().timestamp())}"
        
        task_submission = {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "task_data": task_data,
            "priority": priority,
            "submitted_at": datetime.now(timezone.utc)
        }
        
        # Add to queue
        await self.execution_queue.put(task_submission)
        
        self.logger.info("Task submitted to pool", 
                        execution_id=execution_id, 
                        agent_id=agent_id, 
                        task_id=task_id)
        
        return execution_id
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop for task execution"""
        while True:
            try:
                # Get next task from queue
                task_submission = await self.execution_queue.get()
                
                # Check if we're at capacity
                if len(self.active_executions) >= self.max_concurrent:
                    # Put task back and wait
                    await self.execution_queue.put(task_submission)
                    await asyncio.sleep(1.0)
                    continue
                
                # Execute the task
                asyncio.create_task(self._execute_task(task_submission))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in scheduling loop", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task_submission: Dict[str, Any]) -> None:
        """Execute a task using the appropriate worker"""
        execution_id = task_submission["execution_id"]
        agent_id = task_submission["agent_id"]
        task_id = task_submission["task_id"]
        task_data = task_submission["task_data"]
        
        try:
            worker = self.workers[agent_id]
            
            # Update agent status to busy
            await self._update_agent_status(agent_id, AgentStatus.BUSY)
            
            # Execute the task
            execution = await worker.execute_task(
                task_id=task_id,
                task_data=task_data,
                timeout=self.agent_timeout
            )
            
            # Store execution
            self.active_executions[execution_id] = execution
            
            # Update statistics
            await self._update_statistics(execution)
            
            # Move to history
            self.execution_history.append(execution)
            if len(self.execution_history) > 1000:  # Keep last 1000 executions
                self.execution_history = self.execution_history[-1000:]
            
        except Exception as e:
            self.logger.error("Task execution failed", 
                            execution_id=execution_id, 
                            agent_id=agent_id, 
                            error=str(e))
            
            # Create failed execution record
            execution = AgentExecution(
                execution_id=execution_id,
                agent_id=agent_id,
                task_id=task_id,
                status=AgentExecutionStatus.FAILED,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                error=str(e)
            )
            
            await self._update_statistics(execution)
            
        finally:
            # Update agent status back to available
            await self._update_agent_status(agent_id, AgentStatus.READY)
            
            # Remove from active executions
            self.active_executions.pop(execution_id, None)
    
    async def _update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update agent status in database"""
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                agent = await agent_repo.get_by_id(session, Agent, agent_id)
                
                if agent:
                    agent.status = status
                    agent.last_activity = datetime.now(timezone.utc)
                    await agent_repo.update(session, agent)
                    await session.commit()
                    
        except Exception as e:
            self.logger.error("Failed to update agent status", agent_id=agent_id, error=str(e))
    
    async def _update_statistics(self, execution: AgentExecution) -> None:
        """Update pool statistics"""
        self.stats.total_executions += 1
        
        if execution.status == AgentExecutionStatus.COMPLETED:
            self.stats.successful_executions += 1
        else:
            self.stats.failed_executions += 1
        
        # Update average execution time
        if execution.end_time and execution.start_time:
            execution_time = (execution.end_time - execution.start_time).total_seconds()
            
            if self.stats.total_executions > 1:
                self.stats.average_execution_time = (
                    (self.stats.average_execution_time * (self.stats.total_executions - 1) + execution_time) /
                    self.stats.total_executions
                )
            else:
                self.stats.average_execution_time = execution_time
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while True:
            try:
                await self._update_pool_stats()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _update_pool_stats(self) -> None:
        """Update pool statistics"""
        try:
            # Count agent states
            available_count = 0
            busy_count = 0
            failed_count = 0
            
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                agents = await agent_repo.list_all(session)
                
                for agent in agents:
                    if agent.status == AgentStatus.READY:
                        self.stats.available_agents += 1
                    elif agent.status == AgentStatus.BUSY:
                        busy_count += 1
                    elif agent.status == AgentStatus.ERROR:
                        failed_count += 1
            
            # Update stats
            self.stats.total_agents = len(self.workers)
            self.stats.busy_agents = busy_count
            self.stats.failed_agents = failed_count
            
            # Update system resource usage
            self.stats.cpu_usage = psutil.cpu_percent()
            self.stats.memory_usage = psutil.virtual_memory().percent
            
        except Exception as e:
            self.logger.error("Failed to update pool stats", error=str(e))
    
    async def get_execution_status(self, execution_id: str) -> Optional[AgentExecution]:
        """Get status of a specific execution"""
        # Check active executions
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        agent_id = execution.agent_id
        
        if agent_id in self.workers:
            return await self.workers[agent_id].cancel_execution()
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform pool health check"""
        try:
            healthy_workers = 0
            total_workers = len(self.workers)
            
            for worker in self.workers.values():
                if worker.current_execution is None:  # Worker is available
                    healthy_workers += 1
            
            return {
                "healthy": healthy_workers > 0,
                "total_workers": total_workers,
                "healthy_workers": healthy_workers,
                "active_executions": len(self.active_executions),
                "queue_size": self.execution_queue.qsize(),
                "stats": {
                    "total_executions": self.stats.total_executions,
                    "success_rate": (
                        self.stats.successful_executions / self.stats.total_executions 
                        if self.stats.total_executions > 0 else 0.0
                    ),
                    "average_execution_time": self.stats.average_execution_time,
                    "cpu_usage": self.stats.cpu_usage,
                    "memory_usage": self.stats.memory_usage
                }
            }
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {"healthy": False, "error": str(e)}
    
    async def get_pool_stats(self) -> AgentPoolStats:
        """Get current pool statistics"""
        return self.stats
    
    async def shutdown(self) -> None:
        """Shutdown the agent pool"""
        self.logger.info("Shutting down agent pool...")
        
        try:
            # Cancel background tasks
            if self._monitor_task:
                self._monitor_task.cancel()
            if self._scheduler_task:
                self._scheduler_task.cancel()
            
            # Cancel all active executions
            for execution_id in list(self.active_executions.keys()):
                await self.cancel_execution(execution_id)
            
            # Shutdown all workers
            for worker in self.workers.values():
                await worker.shutdown()
            
            self.logger.info("Agent pool shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during pool shutdown", error=str(e))
            raise OrchestrationException(f"Pool shutdown failed: {e}")
