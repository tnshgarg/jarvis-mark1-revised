"""
Mark-1 Core Orchestrator Engine

The central coordination system that manages agent discovery, task planning,
execution workflows, and inter-agent communication.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.config.settings import get_settings
from mark1.storage.database import get_db_session
from mark1.storage.models.agent_model import Agent, AgentStatus
from mark1.storage.models.task_model import Task, TaskStatus
from mark1.storage.models.context_model import ContextModel
from mark1.storage.repositories.agent_repository import AgentRepository
from mark1.storage.repositories.task_repository import TaskRepository
from mark1.storage.repositories.context_repository import ContextRepository
from mark1.agents.registry import AgentRegistry
from mark1.agents.discovery import AgentDiscovery
from mark1.agents.pool import AgentPool
from mark1.core.task_planner import TaskPlanner
from mark1.core.workflow_engine import WorkflowEngine
from mark1.core.context_manager import ContextManager
from mark1.scanning.codebase_scanner import CodebaseScanner
from mark1.llm.model_manager import ModelManager
from mark1.utils.exceptions import (
    OrchestrationException, 
    TaskExecutionException,
    AgentException,
    Mark1BaseException
)


class SystemStatus(Enum):
    """System status enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class SystemStatusInfo:
    """System status information"""
    overall_status: SystemStatus
    agent_count: int
    active_tasks: int
    database_status: str
    llm_status: str
    last_check: datetime
    components: Dict[str, str]


@dataclass
class ScanResult:
    """Codebase scan result"""
    path: Path
    agents: List[Dict[str, Any]]
    total_files_scanned: int
    scan_duration: float
    framework_distribution: Dict[str, int]
    
    async def save_to_file(self, output_path: Path):
        """Save scan results to JSON file"""
        import json
        
        data = {
            "scan_path": str(self.path),
            "scan_time": datetime.now(timezone.utc).isoformat(),
            "agents": self.agents,
            "statistics": {
                "total_files_scanned": self.total_files_scanned,
                "scan_duration_seconds": self.scan_duration,
                "framework_distribution": self.framework_distribution
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class OrchestrationResult:
    """Result of task orchestration"""
    task_id: str
    status: TaskStatus
    summary: str
    agents_used: List[str]
    execution_time: float
    result_data: Dict[str, Any]


class Mark1Orchestrator:
    """
    Main orchestrator engine for the Mark-1 system
    
    Coordinates agent discovery, task planning, execution workflows,
    and maintains system health and monitoring.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)
        
        # Core components
        self.agent_registry: Optional[AgentRegistry] = None
        self.agent_discovery: Optional[AgentDiscovery] = None
        self.agent_pool: Optional[AgentPool] = None
        self.task_planner: Optional[TaskPlanner] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.context_manager: Optional[ContextManager] = None
        self.codebase_scanner: Optional[CodebaseScanner] = None
        self.model_manager: Optional[ModelManager] = None
        
        # Repositories
        self.agent_repo: Optional[AgentRepository] = None
        self.task_repo: Optional[TaskRepository] = None
        self.context_repo: Optional[ContextRepository] = None
        
        # System state
        self._status = SystemStatus.INITIALIZING
        self._active_tasks: Set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the orchestrator and all its components"""
        try:
            self.logger.info("Initializing Mark-1 orchestrator...")
            
            # Initialize repositories
            await self._initialize_repositories()
            
            # Initialize core components
            await self._initialize_components()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._status = SystemStatus.HEALTHY
            self.logger.info("Mark-1 orchestrator initialized successfully")
            
        except Exception as e:
            self._status = SystemStatus.ERROR
            self.logger.error("Failed to initialize orchestrator", error=str(e))
            raise OrchestrationException(f"Orchestrator initialization failed: {e}")
    
    async def _initialize_repositories(self) -> None:
        """Initialize data repositories"""
        self.logger.debug("Initializing repositories...")
        
        async with get_db_session() as session:
            self.agent_repo = AgentRepository(session)
            self.task_repo = TaskRepository(session)
            self.context_repo = ContextRepository(session)
    
    async def _initialize_components(self) -> None:
        """Initialize core orchestrator components"""
        self.logger.debug("Initializing core components...")
        
        # Initialize agent management
        self.agent_registry = AgentRegistry()
        self.agent_discovery = AgentDiscovery(self.agent_registry)
        self.agent_pool = AgentPool(
            max_concurrent=self.settings.agents.max_concurrent_agents,
            agent_timeout=self.settings.agents.agent_timeout_seconds
        )
        
        # Initialize planning and execution
        self.task_planner = TaskPlanner()
        self.workflow_engine = WorkflowEngine()
        self.context_manager = ContextManager()
        
        # Initialize scanning and LLM
        self.codebase_scanner = CodebaseScanner()
        self.model_manager = ModelManager()
        
        # Initialize all components
        await self.agent_registry.initialize()
        await self.agent_discovery.initialize()
        await self.agent_pool.initialize()
        await self.task_planner.initialize()
        await self.workflow_engine.initialize()
        await self.context_manager.initialize()
        await self.codebase_scanner.initialize()
        await self.model_manager.initialize()
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks"""
        self.logger.debug("Starting background tasks...")
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start agent discovery if enabled
        if self.settings.agents.auto_discovery_enabled:
            asyncio.create_task(self._auto_discovery_loop())
    
    async def _health_check_loop(self) -> None:
        """Continuous health checking loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.settings.monitoring.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check failed", error=str(e))
                await asyncio.sleep(30)  # Wait before retry
    
    async def _auto_discovery_loop(self) -> None:
        """Automatic agent discovery loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._discover_new_agents()
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Auto-discovery failed", error=str(e))
                await asyncio.sleep(600)  # Wait longer before retry
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive system health check"""
        try:
            # Check component health
            components_health = await self._check_components_health()
            
            # Determine overall status
            if all(status == "healthy" for status in components_health.values()):
                if self._status == SystemStatus.ERROR:
                    self.logger.info("System recovered to healthy status")
                self._status = SystemStatus.HEALTHY
            else:
                unhealthy_components = [
                    comp for comp, status in components_health.items() 
                    if status != "healthy"
                ]
                self.logger.warning("System degraded", unhealthy_components=unhealthy_components)
                self._status = SystemStatus.DEGRADED
                
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            self._status = SystemStatus.ERROR
    
    async def _check_components_health(self) -> Dict[str, str]:
        """Check health of all components"""
        health_checks = {}
        
        try:
            # Check database
            async with get_db_session() as session:
                await session.execute("SELECT 1")
                health_checks["database"] = "healthy"
        except Exception:
            health_checks["database"] = "error"
        
        # Check agent pool
        if self.agent_pool:
            pool_health = await self.agent_pool.health_check()
            health_checks["agent_pool"] = "healthy" if pool_health["healthy"] else "error"
        
        # Check LLM provider
        if self.model_manager:
            llm_health = await self.model_manager.health_check()
            health_checks["llm"] = "healthy" if llm_health["healthy"] else "error"
        
        return health_checks
    
    async def _discover_new_agents(self) -> None:
        """Discover new agents in configured directories"""
        if not self.agent_discovery:
            return
            
        try:
            agents_dir = self.settings.agents_dir
            if agents_dir.exists():
                await self.agent_discovery.scan_directory(agents_dir)
                
        except Exception as e:
            self.logger.error("Agent auto-discovery failed", error=str(e))
    
    async def scan_codebase(
        self,
        path: Path,
        recursive: bool = True,
        framework_filter: Optional[List[str]] = None
    ) -> ScanResult:
        """
        Scan a codebase for AI agents
        
        Args:
            path: Path to scan
            recursive: Whether to scan recursively
            framework_filter: List of frameworks to filter for
            
        Returns:
            ScanResult with discovered agents and statistics
        """
        if not self.codebase_scanner:
            raise OrchestrationException("Codebase scanner not initialized")
        
        self.logger.info("Starting codebase scan", path=str(path))
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Perform the scan
            scan_results = await self.codebase_scanner.scan_directory(
                path=path,
                recursive=recursive,
                framework_filter=framework_filter
            )
            
            # Process results
            agents = []
            framework_distribution = {}
            
            for agent_info in scan_results.discovered_agents:
                # Convert to dictionary format
                agent_dict = {
                    "name": agent_info.name,
                    "framework": agent_info.framework,
                    "capabilities": agent_info.capabilities,
                    "file_path": str(agent_info.file_path),
                    "confidence": agent_info.confidence,
                    "metadata": agent_info.metadata
                }
                agents.append(agent_dict)
                
                # Update framework distribution
                framework = agent_info.framework
                framework_distribution[framework] = framework_distribution.get(framework, 0) + 1
            
            # Register discovered agents
            if self.agent_registry:
                for agent_dict in agents:
                    await self.agent_registry.register_from_scan(agent_dict)
            
            scan_duration = asyncio.get_event_loop().time() - start_time
            
            self.logger.info(
                "Codebase scan completed",
                agents_found=len(agents),
                duration=scan_duration,
                frameworks=list(framework_distribution.keys())
            )
            
            return ScanResult(
                path=path,
                agents=agents,
                total_files_scanned=scan_results.total_files_scanned,
                scan_duration=scan_duration,
                framework_distribution=framework_distribution
            )
            
        except Exception as e:
            self.logger.error("Codebase scan failed", error=str(e), path=str(path))
            raise OrchestrationException(f"Codebase scan failed: {e}")
    
    async def orchestrate_task(
        self,
        task_description: str,
        agent_filter: Optional[List[str]] = None,
        max_agents: int = 3,
        timeout: int = 300
    ) -> OrchestrationResult:
        """
        Orchestrate a task across multiple agents
        
        Args:
            task_description: Description of the task to perform
            agent_filter: Optional list of specific agent IDs to use
            max_agents: Maximum number of agents to involve
            timeout: Task timeout in seconds
            
        Returns:
            OrchestrationResult with execution details
        """
        if not all([self.task_planner, self.workflow_engine, self.agent_pool]):
            raise OrchestrationException("Core components not initialized")
        
        self.logger.info("Starting task orchestration", task=task_description)
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create task record
            async with get_db_session() as session:
                task = Task(
                    description=task_description,
                    status=TaskStatus.PENDING,
                    max_agents=max_agents,
                    timeout_seconds=timeout,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(task)
                await session.commit()
                task_id = task.id
            
            self._active_tasks.add(task_id)
            
            try:
                # Plan the task
                plan = await self.task_planner.create_plan(
                    task_description=task_description,
                    available_agents=await self._get_available_agents(agent_filter),
                    max_agents=max_agents
                )
                
                # Execute the workflow
                execution_result = await self.workflow_engine.execute_workflow(
                    plan=plan,
                    timeout=timeout
                )
                
                # Update task status
                async with get_db_session() as session:
                    task = await session.get(Task, task_id)
                    task.status = TaskStatus.COMPLETED
                    task.result_data = execution_result.result_data
                    task.completed_at = datetime.now(timezone.utc)
                    await session.commit()
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                result = OrchestrationResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    summary=execution_result.summary,
                    agents_used=execution_result.agents_used,
                    execution_time=execution_time,
                    result_data=execution_result.result_data
                )
                
                self.logger.info(
                    "Task orchestration completed",
                    task_id=task_id,
                    duration=execution_time,
                    agents_used=len(execution_result.agents_used)
                )
                
                return result
                
            except Exception as e:
                # Update task status to failed
                async with get_db_session() as session:
                    task = await session.get(Task, task_id)
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.now(timezone.utc)
                    await session.commit()
                
                raise TaskExecutionException(task_id, "execution", e)
                
            finally:
                self._active_tasks.discard(task_id)
                
        except Exception as e:
            self.logger.error("Task orchestration failed", error=str(e))
            raise OrchestrationException(f"Task orchestration failed: {e}")
    
    async def _get_available_agents(self, agent_filter: Optional[List[str]] = None) -> List[Agent]:
        """Get list of available agents, optionally filtered"""
        async with get_db_session() as session:
            agents = await self.agent_repo.list_by_status(session, AgentStatus.AVAILABLE)
            
            if agent_filter:
                agents = [agent for agent in agents if agent.id in agent_filter]
            
            return agents
    
    async def get_system_status(self) -> SystemStatusInfo:
        """Get comprehensive system status information"""
        try:
            # Count agents and tasks
            async with get_db_session() as session:
                agent_count = await self.agent_repo.count_by_status(session, AgentStatus.AVAILABLE)
                
            active_tasks = len(self._active_tasks)
            
            # Check component health
            components_health = await self._check_components_health()
            
            return SystemStatusInfo(
                overall_status=self._status,
                agent_count=agent_count,
                active_tasks=active_tasks,
                database_status=components_health.get("database", "unknown"),
                llm_status=components_health.get("llm", "unknown"),
                last_check=datetime.now(timezone.utc),
                components=components_health
            )
            
        except Exception as e:
            self.logger.error("Failed to get system status", error=str(e))
            raise OrchestrationException(f"Status check failed: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator"""
        self.logger.info("Shutting down Mark-1 orchestrator...")
        
        try:
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown components
            if self.agent_pool:
                await self.agent_pool.shutdown()
            
            if self.workflow_engine:
                await self.workflow_engine.shutdown()
            
            if self.model_manager:
                await self.model_manager.shutdown()
            
            self._status = SystemStatus.SHUTDOWN
            self.logger.info("Mark-1 orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))
            raise OrchestrationException(f"Shutdown failed: {e}")
    
    @property
    def status(self) -> SystemStatus:
        """Get current system status"""
        return self._status
    
    @property
    def active_tasks_count(self) -> int:
        """Get number of active tasks"""
        return len(self._active_tasks)
