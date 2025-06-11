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
from sqlalchemy import text

from mark1.config.settings import get_settings
from mark1.storage.database import get_db_session
from mark1.storage.models.agent_model import Agent, AgentStatus
from mark1.storage.models.task_model import Task, TaskStatus
from mark1.storage.models.context_model import ContextModel
from mark1.storage.repositories.agent_repository import AgentRepository
from mark1.storage.repositories.task_repository import TaskRepository
from mark1.storage.repositories.context_repository import ContextRepository
from mark1.agents.registry import AgentRegistry
from mark1.agents.discovery import AgentDiscoveryEngine
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
from mark1.plugins import (
    PluginManager,
    PluginMetadata,
    PluginInstallationResult,
    UniversalPluginAdapter
)
from mark1.storage.repositories.plugin_repository import PluginRepository


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
        self.agent_discovery: Optional[AgentDiscoveryEngine] = None
        self.agent_pool: Optional[AgentPool] = None
        self.task_planner: Optional[TaskPlanner] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.context_manager: Optional[ContextManager] = None
        self.codebase_scanner: Optional[CodebaseScanner] = None
        self.model_manager: Optional[ModelManager] = None

        # Plugin system components
        self.plugin_manager: Optional[PluginManager] = None

        # Repositories
        self.agent_repo: Optional[AgentRepository] = None
        self.task_repo: Optional[TaskRepository] = None
        self.context_repo: Optional[ContextRepository] = None
        self.plugin_repo: Optional[PluginRepository] = None
        
        # System state
        self._status = SystemStatus.INITIALIZING
        self._active_tasks: Set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._auto_discovery_task: Optional[asyncio.Task] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the orchestrator and all its components"""
        if self._initialized:
            self.logger.warning("Orchestrator already initialized")
            return
            
        try:
            self.logger.info("Initializing Mark-1 orchestrator...")
            
            # Initialize repositories
            await self._initialize_repositories()
            
            # Initialize core components
            await self._initialize_components()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._status = SystemStatus.HEALTHY
            self._initialized = True
            self.logger.info("Mark-1 orchestrator initialized successfully")
            
        except Exception as e:
            self._status = SystemStatus.ERROR
            self.logger.error("Failed to initialize orchestrator", error=str(e))
            raise OrchestrationException(f"Orchestrator initialization failed: {e}")
    
    async def _initialize_repositories(self) -> None:
        """Initialize data repositories"""
        self.logger.debug("Initializing repositories...")
        
        # Note: Repositories are created per-session basis in actual usage
        # This is just to ensure the structure is available
        self.logger.debug("Repository initialization completed")
    
    async def _initialize_components(self) -> None:
        """Initialize core orchestrator components"""
        self.logger.debug("Initializing core components...")
        
        try:
            # Initialize agent management
            self.agent_registry = AgentRegistry()
            await self.agent_registry.initialize()
            self.logger.debug("Agent registry initialized")
            
            self.agent_discovery = AgentDiscoveryEngine()
            await self.agent_discovery.initialize()
            self.logger.debug("Agent discovery engine initialized")
            
            self.agent_pool = AgentPool(
                max_concurrent=getattr(self.settings, 'max_concurrent_agents', 10),
                agent_timeout=getattr(self.settings, 'agent_timeout_seconds', 300)
            )
            await self.agent_pool.initialize()
            self.logger.debug("Agent pool initialized")
            
            # Initialize planning and execution
            self.task_planner = TaskPlanner()
            await self.task_planner.initialize()
            self.logger.debug("Task planner initialized")
            
            self.workflow_engine = WorkflowEngine()
            await self.workflow_engine.initialize()
            self.logger.debug("Workflow engine initialized")
            
            self.context_manager = ContextManager()
            await self.context_manager.initialize()
            self.logger.debug("Context manager initialized")
            
            # Initialize scanning and LLM
            self.codebase_scanner = CodebaseScanner()
            await self.codebase_scanner.initialize()
            self.logger.debug("Codebase scanner initialized")
            
            self.model_manager = ModelManager()
            await self.model_manager.initialize()
            self.logger.debug("Model manager initialized")

            # Initialize plugin system
            plugins_dir = Path(self.settings.data_dir) / "plugins"
            self.plugin_manager = PluginManager(plugins_directory=plugins_dir)
            self.logger.debug("Plugin manager initialized")

            self.logger.info("All core components initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize components", error=str(e))
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks"""
        self.logger.debug("Starting background tasks...")
        
        try:
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start auto-discovery task
            self._auto_discovery_task = asyncio.create_task(self._auto_discovery_loop())
            
            self.logger.debug("Background tasks started")
            
        except Exception as e:
            self.logger.error("Failed to start background tasks", error=str(e))
            raise
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check failed", error=str(e))
                await asyncio.sleep(30)
    
    async def _auto_discovery_loop(self) -> None:
        """Background auto-discovery loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._discover_new_agents()
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Auto-discovery failed", error=str(e))
                await asyncio.sleep(300)
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check"""
        try:
            # Check database connectivity
            async with get_db_session() as session:
                await session.execute(text("SELECT 1"))
            
            # Check component health
            component_health = await self._check_components_health()
            
            # Determine overall status
            if all(status == "healthy" for status in component_health.values()):
                self._status = SystemStatus.HEALTHY
            elif any(status == "error" for status in component_health.values()):
                self._status = SystemStatus.ERROR
            else:
                self._status = SystemStatus.DEGRADED
                
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            self._status = SystemStatus.ERROR
    
    async def _check_components_health(self) -> Dict[str, str]:
        """Check health of all components"""
        health = {}
        
        try:
            # Check agent pool health
            if self.agent_pool:
                pool_health = await self.agent_pool.health_check()
                health["agent_pool"] = "healthy" if pool_health.get("healthy", False) else "error"
            else:
                health["agent_pool"] = "not_initialized"
            
            # Check model manager health
            if self.model_manager:
                model_health = await self.model_manager.health_check()
                health["model_manager"] = "healthy" if model_health.healthy else "error"
            else:
                health["model_manager"] = "not_initialized"
            
            # Check other components
            components = {
                "agent_registry": self.agent_registry,
                "agent_discovery": self.agent_discovery,
                "task_planner": self.task_planner,
                "workflow_engine": self.workflow_engine,
                "context_manager": self.context_manager,
                "codebase_scanner": self.codebase_scanner
            }
            
            for component_name, component in components.items():
                if component:
                    health[component_name] = "healthy"
                else:
                    health[component_name] = "not_initialized"
                    
        except Exception as e:
            self.logger.error("Component health check failed", error=str(e))
            health["health_check"] = "error"
        
        return health
    
    async def _discover_new_agents(self) -> None:
        """Discover and register new agents"""
        try:
            if not self.agent_discovery:
                return
                
            # Discover agents in default paths
            discovered_agents = self.agent_discovery.discover_all()
            
            # Register new agents
            for agent_key, agent_metadata in discovered_agents.items():
                try:
                    if self.agent_registry:
                        existing_agent = await self.agent_registry.get_agent_by_name(agent_metadata.name)
                        if not existing_agent:
                            await self.agent_registry.register_from_scan({
                                "name": agent_metadata.name,
                                "framework": "detected",
                                "file_path": agent_metadata.file_path,
                                "capabilities": agent_metadata.capabilities,
                                "metadata": {
                                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                                    "module_path": agent_metadata.module_path,
                                    "class_name": agent_metadata.class_name
                                }
                            })
                            self.logger.info("Auto-discovered and registered new agent", 
                                           agent_name=agent_metadata.name)
                except Exception as e:
                    self.logger.warning("Failed to register discovered agent", 
                                       agent_name=agent_metadata.name, error=str(e))
                    
        except Exception as e:
            self.logger.error("Auto-discovery failed", error=str(e))
    
    async def scan_codebase(
        self,
        path: Path,
        recursive: bool = True,
        framework_filter: Optional[List[str]] = None
    ) -> ScanResult:
        """
        Scan codebase for agents and other AI components
        
        Args:
            path: Path to scan
            recursive: Whether to scan recursively
            framework_filter: Optional framework filter
            
        Returns:
            Scan results
        """
        if not self.codebase_scanner:
            raise OrchestrationException("Codebase scanner not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info("Starting codebase scan", path=str(path))
            
            # Perform the scan
            scan_results = await self.codebase_scanner.scan_directory(
                path,
                recursive=recursive,
                framework_filter=framework_filter
            )
            
            # Process results
            agents = []
            framework_distribution = {}
            
            # Convert discovered agents to expected format
            for agent_info in scan_results.discovered_agents:
                agents.append({
                    "name": agent_info.name,
                    "file_path": str(agent_info.file_path),
                    "framework": agent_info.framework,
                    "capabilities": agent_info.capabilities,
                    "confidence": agent_info.confidence,
                    "metadata": agent_info.metadata
                })
                
                framework = agent_info.framework
                framework_distribution[framework] = framework_distribution.get(framework, 0) + 1
            
            scan_duration = scan_results.scan_duration
            
            result = ScanResult(
                path=path,
                agents=agents,
                total_files_scanned=scan_results.total_files_scanned,
                scan_duration=scan_duration,
                framework_distribution=framework_distribution
            )
            
            self.logger.info("Codebase scan completed", 
                           agents_found=len(agents),
                           duration=scan_duration,
                           path=str(path))
            
            return result
            
        except Exception as e:
            self.logger.error("Codebase scan failed", path=str(path), error=str(e))
            raise OrchestrationException(f"Codebase scan failed: {e}")
    
    async def orchestrate_task(
        self,
        task_description: str,
        agent_filter: Optional[List[str]] = None,
        max_agents: int = 3,
        timeout: int = 300
    ) -> OrchestrationResult:
        """
        Orchestrate a task across available agents
        
        Args:
            task_description: Description of the task to execute
            agent_filter: Optional list of agent names to filter by
            max_agents: Maximum number of agents to use
            timeout: Task timeout in seconds
            
        Returns:
            Orchestration result
        """
        if not self._initialized:
            raise OrchestrationException("Orchestrator not initialized")
        
        start_time = datetime.now(timezone.utc)
        task_id = f"task_{int(start_time.timestamp())}"
        
        try:
            self.logger.info("Starting task orchestration", 
                           task_id=task_id, 
                           description=task_description)
            
            # Add to active tasks
            self._active_tasks.add(task_id)
            
            # Get available agents
            available_agents = await self._get_available_agents(agent_filter)
            
            if not available_agents:
                raise OrchestrationException("No available agents found")
            
            # Limit agents
            selected_agents = available_agents[:max_agents]
            
            # Plan the task
            if self.task_planner:
                task_plan = await self.task_planner.plan_task(
                    description=task_description,
                    available_agents=[agent.id for agent in selected_agents],
                    constraints={"timeout": timeout}
                )
            else:
                # Simple fallback plan
                task_plan = {
                    "steps": [{"agent_id": selected_agents[0].id, "action": task_description}],
                    "estimated_duration": 60
                }
            
            # Execute the task
            execution_results = []
            
            for step in task_plan.get("steps", []):
                agent_id = step.get("agent_id")
                action = step.get("action", task_description)
                
                if self.agent_pool:
                    execution_id = await self.agent_pool.submit_task(
                        agent_id=agent_id,
                        task_id=task_id,
                        task_data={"description": action, "parameters": step.get("parameters", {})},
                        priority=1
                    )
                    
                    # Wait for completion (simplified)
                    await asyncio.sleep(2)  # Give some time for execution
                    
                    execution = await self.agent_pool.get_execution_status(execution_id)
                    if execution:
                        execution_results.append({
                            "agent_id": agent_id,
                            "status": execution.status.value,
                            "result": execution.result,
                            "error": execution.error
                        })
            
            # Determine overall status
            if all(r.get("status") == "completed" for r in execution_results):
                overall_status = TaskStatus.COMPLETED
            elif any(r.get("status") == "failed" for r in execution_results):
                overall_status = TaskStatus.FAILED
            else:
                overall_status = TaskStatus.RUNNING
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = OrchestrationResult(
                task_id=task_id,
                status=overall_status,
                summary=f"Task executed across {len(selected_agents)} agents",
                agents_used=[agent.id for agent in selected_agents],
                execution_time=execution_time,
                result_data={
                    "execution_results": execution_results,
                    "task_plan": task_plan
                }
            )
            
            self.logger.info("Task orchestration completed", 
                           task_id=task_id,
                           status=overall_status.value,
                           execution_time=execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error("Task orchestration failed", 
                            task_id=task_id, 
                            error=str(e))
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return OrchestrationResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                summary=f"Task failed: {str(e)}",
                agents_used=[],
                execution_time=execution_time,
                result_data={"error": str(e)}
            )
            
        finally:
            # Remove from active tasks
            self._active_tasks.discard(task_id)
    
    async def _get_available_agents(self, agent_filter: Optional[List[str]] = None) -> List[Agent]:
        """Get list of available agents"""
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                agents = await agent_repo.list_by_status(session, AgentStatus.READY)
                
                if agent_filter:
                    agents = [agent for agent in agents if agent.name in agent_filter]
                
                return agents
                
        except Exception as e:
            self.logger.error("Failed to get available agents", error=str(e))
            return []
    
    async def get_system_status(self) -> SystemStatusInfo:
        """Get comprehensive system status"""
        try:
            # Get agent count
            agent_count = 0
            if self.agent_registry:
                agent_count = self.agent_registry.active_agent_count
            
            # Get component health
            component_health = await self._check_components_health()
            
            # Check database status
            db_status = "healthy"
            try:
                async with get_db_session() as session:
                    await session.execute(text("SELECT 1"))
            except Exception:
                db_status = "error"
            
            # Check LLM status
            llm_status = "healthy"
            try:
                if self.model_manager:
                    model_health = await self.model_manager.health_check()
                    llm_status = "healthy" if model_health.healthy else "error"
            except Exception:
                llm_status = "error"
            
            return SystemStatusInfo(
                overall_status=self._status,
                agent_count=agent_count,
                active_tasks=len(self._active_tasks),
                database_status=db_status,
                llm_status=llm_status,
                last_check=datetime.now(timezone.utc),
                components=component_health
            )
            
        except Exception as e:
            self.logger.error("Failed to get system status", error=str(e))
            return SystemStatusInfo(
                overall_status=SystemStatus.ERROR,
                agent_count=0,
                active_tasks=0,
                database_status="error",
                llm_status="error",
                last_check=datetime.now(timezone.utc),
                components={"error": str(e)}
            )
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all components"""
        try:
            self.logger.info("Shutting down Mark-1 orchestrator...")
            self._status = SystemStatus.SHUTDOWN
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._auto_discovery_task:
                self._auto_discovery_task.cancel()
                try:
                    await self._auto_discovery_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown components
            if self.agent_pool:
                await self.agent_pool.shutdown()
            
            if self.workflow_engine:
                await self.workflow_engine.shutdown()
            
            self.logger.info("Mark-1 orchestrator shutdown completed")
            
        except Exception as e:
            self.logger.error("Error during orchestrator shutdown", error=str(e))
            raise OrchestrationException(f"Shutdown failed: {e}")

    # Plugin Orchestration Methods

    async def install_plugin_from_repository(
        self,
        repository_url: str,
        branch: str = "main",
        force_reinstall: bool = False
    ) -> PluginInstallationResult:
        """
        Install a plugin from a GitHub repository

        Args:
            repository_url: GitHub repository URL
            branch: Branch to clone (default: main)
            force_reinstall: Force reinstallation if plugin exists

        Returns:
            PluginInstallationResult with installation outcome
        """
        if not self.plugin_manager:
            raise OrchestrationException("Plugin manager not initialized")

        try:
            self.logger.info("Installing plugin from repository",
                           repository_url=repository_url, branch=branch)

            result = await self.plugin_manager.install_plugin_from_repository(
                repository_url=repository_url,
                branch=branch,
                force_reinstall=force_reinstall
            )

            # Store plugin metadata in database if installation successful
            if result.success and result.plugin_metadata:
                async with get_db_session() as session:
                    plugin_repo = PluginRepository(session)
                    await plugin_repo.create_plugin(result.plugin_metadata)

            return result

        except Exception as e:
            self.logger.error("Plugin installation failed",
                            repository_url=repository_url, error=str(e))
            raise OrchestrationException(f"Plugin installation failed: {e}")

    async def list_installed_plugins(self) -> List[PluginMetadata]:
        """List all installed plugins"""
        if not self.plugin_manager:
            raise OrchestrationException("Plugin manager not initialized")

        return await self.plugin_manager.list_installed_plugins()

    async def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get metadata for a specific plugin"""
        if not self.plugin_manager:
            raise OrchestrationException("Plugin manager not initialized")

        return await self.plugin_manager.get_plugin_metadata(plugin_id)

    async def _get_available_plugins(self, plugin_filter: Optional[List[str]] = None) -> List[PluginMetadata]:
        """Get available plugins for orchestration"""
        if not self.plugin_manager:
            return []

        all_plugins = await self.plugin_manager.list_installed_plugins()

        if plugin_filter:
            return [p for p in all_plugins if p.plugin_id in plugin_filter]

        # Return only ready plugins
        return [p for p in all_plugins if p.status.value == "ready"]

    async def _plan_plugin_task(
        self,
        task_description: str,
        available_plugins: List[PluginMetadata],
        timeout: int
    ) -> Dict[str, Any]:
        """Plan task execution using available plugins"""
        # Simple planning logic - can be enhanced with LLM-based planning
        steps = []

        for plugin in available_plugins:
            if plugin.capabilities:
                # Use first available capability
                capability = plugin.capabilities[0]
                steps.append({
                    "plugin_id": plugin.plugin_id,
                    "capability": capability.name,
                    "inputs": {"description": task_description},
                    "parameters": {},
                    "estimated_duration": 60
                })

        return {
            "steps": steps,
            "estimated_duration": len(steps) * 60,
            "execution_mode": "sequential"
        }

    @property
    def status(self) -> SystemStatus:
        """Get current system status"""
        return self._status
    
    @property
    def active_tasks_count(self) -> int:
        """Get count of active tasks"""
        return len(self._active_tasks)
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self._status == SystemStatus.HEALTHY
    
    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized"""
        return self._initialized
