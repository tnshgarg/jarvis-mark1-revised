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
        
        # Repositories
        self.agent_repo: Optional[AgentRepository] = None
        self.task_repo: Optional[TaskRepository] = None
        self.context_repo: Optional[ContextRepository] = None
        
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
        timeout: int = 300,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """
        Orchestrate a task across available agents
        
        Args:
            task_description: Description of the task to execute
            agent_filter: Optional list of agent names to filter by
            max_agents: Maximum number of agents to use
            timeout: Task timeout in seconds
            context: Additional context for task execution
            
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
            
            # Decompose task into subtasks for different agents
            subtasks = self._decompose_complex_task(task_description, available_agents, context or {})
            
            # Limit agents based on subtasks
            selected_agents = available_agents[:min(max_agents, len(subtasks))]
            
            self.logger.info(f"Task decomposed into {len(subtasks)} subtasks for {len(selected_agents)} agents")
            
            # Execute subtasks in parallel
            execution_results = []
            execution_tasks = []
            
            for i, (subtask, agent) in enumerate(zip(subtasks, selected_agents)):
                if self.agent_pool:
                    # Prepare task data with context
                    task_data = {
                        "description": subtask["description"],
                        "parameters": {
                            **subtask.get("parameters", {}),
                            **(context or {}),
                            "subtask_index": i,
                            "total_subtasks": len(subtasks),
                            "agent_role": subtask.get("role", "worker")
                        }
                    }
                    
                    execution_task = asyncio.create_task(
                        self._execute_agent_subtask(agent.id, task_id, task_data, timeout)
                    )
                    execution_tasks.append(execution_task)
            
            # Wait for all subtasks to complete
            if execution_tasks:
                execution_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results and determine overall status
            successful_results = []
            failed_results = []
            
            for result in execution_results:
                if isinstance(result, Exception):
                    failed_results.append({"error": str(result)})
                else:
                    if result and result.get("status") == "completed":
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
            
            # Determine overall status
            if successful_results and not failed_results:
                overall_status = TaskStatus.COMPLETED
                summary = f"Task completed successfully with {len(successful_results)} agents"
            elif successful_results and failed_results:
                overall_status = TaskStatus.COMPLETED
                summary = f"Task partially completed: {len(successful_results)} succeeded, {len(failed_results)} failed"
            else:
                overall_status = TaskStatus.FAILED
                summary = f"Task failed: all {len(failed_results)} subtasks failed"
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Aggregate results for final output
            aggregated_result = self._aggregate_results(
                task_description, 
                successful_results, 
                failed_results, 
                context or {}
            )
            
            result = OrchestrationResult(
                task_id=task_id,
                status=overall_status,
                summary=summary,
                agents_used=[str(agent.id) for agent in selected_agents],
                execution_time=execution_time,
                result_data={
                    "subtasks": subtasks,
                    "execution_results": successful_results,
                    "failed_results": failed_results,
                    "aggregated_result": aggregated_result,
                    "output_files": aggregated_result.get("files_generated", [])
                }
            )
            
            self.logger.info("Task orchestration completed", 
                           task_id=task_id,
                           status=overall_status.value,
                           execution_time=execution_time,
                           successful_subtasks=len(successful_results),
                           failed_subtasks=len(failed_results))
            
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
    
    def _decompose_complex_task(
        self, 
        task_description: str, 
        available_agents: List[Agent], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose a complex task into subtasks for different agents"""
        
        task_lower = task_description.lower()
        
        if "website" in task_lower or "landing page" in task_lower:
            return self._decompose_website_task(task_description, context)
        elif "analysis" in task_lower or "analyze" in task_lower:
            return self._decompose_analysis_task(task_description, context)
        elif "code" in task_lower and "generate" in task_lower:
            return self._decompose_coding_task(task_description, context)
        else:
            return self._decompose_generic_task(task_description, context)
    
    def _decompose_website_task(self, task_description: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose website development task"""
        return [
            {
                "description": f"Plan and design the structure for: {task_description}",
                "role": "planner",
                "parameters": {"task_type": "planning", "output_format": "design_doc"}
            },
            {
                "description": f"Generate HTML structure and content for: {task_description}",
                "role": "developer",
                "parameters": {"task_type": "html_generation", "output_format": "html_file"}
            },
            {
                "description": f"Create CSS styling and responsive design for: {task_description}",
                "role": "designer", 
                "parameters": {"task_type": "css_generation", "output_format": "css_file"}
            }
        ]
    
    def _decompose_analysis_task(self, task_description: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose analysis task"""
        return [
            {
                "description": f"Gather and collect data for: {task_description}",
                "role": "researcher",
                "parameters": {"task_type": "data_collection", "output_format": "dataset"}
            },
            {
                "description": f"Analyze and process data for: {task_description}",
                "role": "analyst",
                "parameters": {"task_type": "data_analysis", "output_format": "analysis_report"}
            }
        ]
    
    def _decompose_coding_task(self, task_description: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose coding task"""
        return [
            {
                "description": f"Design architecture and plan implementation for: {task_description}",
                "role": "architect", 
                "parameters": {"task_type": "architecture_design", "output_format": "design_doc"}
            },
            {
                "description": f"Generate code implementation for: {task_description}",
                "role": "developer",
                "parameters": {"task_type": "code_generation", "output_format": "source_code"}
            }
        ]
    
    def _decompose_generic_task(self, task_description: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose generic task"""
        return [
            {
                "description": f"Analyze and understand requirements for: {task_description}",
                "role": "analyst",
                "parameters": {"task_type": "requirement_analysis", "output_format": "requirements_doc"}
            },
            {
                "description": f"Execute and implement solution for: {task_description}",
                "role": "executor",
                "parameters": {"task_type": "implementation", "output_format": "solution"}
            }
        ]
    
    async def _execute_agent_subtask(
        self, 
        agent_id: str, 
        task_id: str, 
        task_data: Dict[str, Any], 
        timeout: int
    ) -> Dict[str, Any]:
        """Execute a subtask with a specific agent"""
        try:
            if self.agent_pool:
                execution_id = await self.agent_pool.submit_task(
                    agent_id=agent_id,
                    task_id=task_id,
                    task_data=task_data,
                    priority=1
                )
                
                # Wait for completion with timeout
                max_wait_time = min(timeout, 120)  # Max 2 minutes per subtask
                wait_start = datetime.now(timezone.utc)
                
                while (datetime.now(timezone.utc) - wait_start).total_seconds() < max_wait_time:
                    execution = await self.agent_pool.get_execution_status(execution_id)
                    if execution and execution.status in [
                        execution.status.COMPLETED, 
                        execution.status.FAILED, 
                        execution.status.TIMEOUT,
                        execution.status.CANCELLED
                    ]:
                        return {
                            "agent_id": agent_id,
                            "execution_id": execution_id,
                            "status": "completed" if execution.status.value == "completed" else "failed",
                            "result": execution.result,
                            "error": execution.error,
                            "execution_time": (
                                execution.end_time - execution.start_time
                            ).total_seconds() if execution.end_time else 0
                        }
                    await asyncio.sleep(2)  # Check every 2 seconds
                
                # Timeout occurred
                return {
                    "agent_id": agent_id,
                    "execution_id": execution_id,
                    "status": "timeout",
                    "error": f"Subtask timeout after {max_wait_time}s"
                }
            else:
                raise Exception("Agent pool not available")
                
        except Exception as e:
            return {
                "agent_id": agent_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _aggregate_results(
        self, 
        original_task: str, 
        successful_results: List[Dict[str, Any]], 
        failed_results: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        
        aggregated = {
            "original_task": original_task,
            "total_agents": len(successful_results) + len(failed_results),
            "successful_agents": len(successful_results),
            "failed_agents": len(failed_results),
            "files_generated": [],
            "combined_output": "",
            "summary": ""
        }
        
        # Combine outputs from all successful agents
        combined_outputs = []
        files_generated = []
        
        for result in successful_results:
            if result.get("result") and isinstance(result["result"], dict):
                result_data = result["result"].get("result_data", "")
                combined_outputs.append(f"Agent {result.get('agent_id', 'unknown')}:\n{result_data}")
                
                # Check for file generation information
                if "generated successfully" in str(result_data):
                    files_generated.append(f"Files from Agent {result.get('agent_id', 'unknown')}")
        
        aggregated["combined_output"] = "\n\n" + "="*50 + "\n\n".join(combined_outputs)
        aggregated["files_generated"] = files_generated
        
        # Generate summary
        if successful_results:
            aggregated["summary"] = f"""
🎉 MULTI-AGENT TASK COMPLETION
=============================

✅ Original Task: {original_task}

📊 Execution Summary:
• Total Agents Used: {aggregated['total_agents']}
• Successful Executions: {aggregated['successful_agents']}
• Failed Executions: {aggregated['failed_agents']}
• Files Generated: {len(files_generated)}

🎯 Results:
{aggregated['combined_output']}

✨ Task completed through collaborative multi-agent orchestration!
"""
        else:
            aggregated["summary"] = f"❌ Task failed - no agents completed their subtasks successfully"
        
        return aggregated
    
    async def _get_available_agents(self, agent_filter: Optional[List[str]] = None) -> List[Agent]:
        """Get list of available agents"""
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository()
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

    async def _update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update agent status in database"""
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository()
                agent = await agent_repo.get_by_id(session, Agent, agent_id)
                
                if agent:
                    agent.status = status
                    agent.last_activity = datetime.now(timezone.utc)
                    await agent_repo.update(session, agent)
                    await session.commit()
                    
        except Exception as e:
            self.logger.error("Failed to update agent status", agent_id=agent_id, error=str(e))
