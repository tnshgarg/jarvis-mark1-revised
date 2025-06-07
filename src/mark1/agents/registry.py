"""
Agent Registry System for Mark-1 Orchestrator

Manages agent registration, lifecycle, metadata, and provides
a centralized interface for agent discovery and management.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.config.settings import get_settings
from mark1.storage.database import get_db_session
from mark1.storage.models.agent_model import Agent, AgentStatus, AgentType
from mark1.storage.repositories.agent_repository import AgentRepository
from mark1.utils.exceptions import (
    AgentException,
    AgentRegistrationException,
    AgentNotFoundException,
    ValidationException
)


class RegistrationStatus(Enum):
    """Agent registration status"""
    PENDING = "pending"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class AgentRegistrationInfo:
    """Information about agent registration"""
    agent_id: str
    name: str
    framework: str
    file_path: Path
    capabilities: List[str]
    metadata: Dict[str, Any]
    confidence: float
    registration_time: datetime
    status: RegistrationStatus


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    last_execution_time: Optional[datetime]
    success_rate: float


class AgentRegistry:
    """
    Central registry for agent management and discovery
    
    Manages agent registration, lifecycle, metadata, and provides
    a unified interface for agent discovery and selection.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)
        
        # Registry state
        self._registered_agents: Dict[str, AgentRegistrationInfo] = {}
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._discovery_callbacks: List[callable] = []
        self._initialized = False
        
        # Agent type mappings
        self._framework_to_type = {
            "langchain": AgentType.LANGCHAIN,
            "autogpt": AgentType.AUTOGPT,
            "crewai": AgentType.CREWAI,
            "custom": AgentType.CUSTOM
        }
        
    async def initialize(self) -> None:
        """Initialize the agent registry"""
        if self._initialized:
            self.logger.warning("Agent registry already initialized")
            return
            
        try:
            self.logger.info("Initializing agent registry...")
            
            # Load existing agents from database
            await self._load_existing_agents()
            
            # Initialize metrics tracking
            await self._initialize_metrics()
            
            self._initialized = True
            self.logger.info("Agent registry initialized successfully", 
                           registered_agents=len(self._registered_agents))
            
        except Exception as e:
            self.logger.error("Failed to initialize agent registry", error=str(e))
            raise AgentRegistrationException("registry", f"Initialization failed: {e}")
    
    async def _load_existing_agents(self) -> None:
        """Load existing agents from database"""
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                agents = await agent_repo.list_all(session)
                
                for agent in agents:
                    registration_info = AgentRegistrationInfo(
                        agent_id=agent.id,
                        name=agent.name,
                        framework=agent.framework,
                        file_path=Path(agent.file_path) if agent.file_path else None,
                        capabilities=agent.capabilities or [],
                        metadata=agent.extra_metadata or {},
                        confidence=agent.confidence or 0.0,
                        registration_time=agent.created_at,
                        status=self._agent_status_to_registration_status(agent.status)
                    )
                    
                    self._registered_agents[agent.id] = registration_info
                    
        except Exception as e:
            self.logger.error("Failed to load existing agents", error=str(e))
            raise
    
    async def _initialize_metrics(self) -> None:
        """Initialize agent metrics tracking"""
        try:
            # Load metrics from database for existing agents
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                
                for agent_id in self._registered_agents.keys():
                    metrics = await agent_repo.get_agent_metrics(session, agent_id)
                    if metrics:
                        self._agent_metrics[agent_id] = AgentMetrics(
                            total_executions=metrics.get("total_executions", 0),
                            successful_executions=metrics.get("successful_executions", 0),
                            failed_executions=metrics.get("failed_executions", 0),
                            average_execution_time=metrics.get("average_execution_time", 0.0),
                            last_execution_time=metrics.get("last_execution_time"),
                            success_rate=metrics.get("success_rate", 0.0)
                        )
                    else:
                        self._agent_metrics[agent_id] = AgentMetrics(
                            total_executions=0,
                            successful_executions=0,
                            failed_executions=0,
                            average_execution_time=0.0,
                            last_execution_time=None,
                            success_rate=0.0
                        )
                        
        except Exception as e:
            self.logger.error("Failed to initialize metrics", error=str(e))
            # Continue without metrics for now
    
    def _agent_status_to_registration_status(self, agent_status: AgentStatus) -> RegistrationStatus:
        """Convert agent status to registration status"""
        mapping = {
            AgentStatus.AVAILABLE: RegistrationStatus.ACTIVE,
            AgentStatus.BUSY: RegistrationStatus.ACTIVE,
            AgentStatus.DISABLED: RegistrationStatus.DISABLED,
            AgentStatus.ERROR: RegistrationStatus.ERROR
        }
        return mapping.get(agent_status, RegistrationStatus.PENDING)
    
    async def register_agent(
        self,
        name: str,
        framework: str,
        file_path: Optional[Path] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Register a new agent
        
        Args:
            name: Agent name
            framework: Framework name (langchain, autogpt, etc.)
            file_path: Path to agent file
            capabilities: List of agent capabilities
            metadata: Additional metadata
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            Agent ID
        """
        if not self._initialized:
            raise AgentRegistrationException(name, "Registry not initialized")
        
        try:
            self.logger.info("Registering new agent", name=name, framework=framework)
            
            # Validate input
            await self._validate_registration_data(name, framework, capabilities)
            
            # Create agent record
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                
                agent = Agent(
                    name=name,
                    type=self._framework_to_type.get(framework.lower(), AgentType.CUSTOM),
                    framework=framework,
                    file_path=str(file_path) if file_path else None,
                    capabilities=capabilities or [],
                    extra_metadata=metadata or {},
                    confidence=confidence,
                    status=AgentStatus.AVAILABLE,
                    created_at=datetime.now(timezone.utc)
                )
                
                agent = await agent_repo.create(session, agent)
                await session.commit()
                
                agent_id = agent.id
            
            # Add to registry
            registration_info = AgentRegistrationInfo(
                agent_id=agent_id,
                name=name,
                framework=framework,
                file_path=file_path,
                capabilities=capabilities or [],
                metadata=metadata or {},
                confidence=confidence,
                registration_time=datetime.now(timezone.utc),
                status=RegistrationStatus.ACTIVE
            )
            
            self._registered_agents[agent_id] = registration_info
            
            # Initialize metrics
            self._agent_metrics[agent_id] = AgentMetrics(
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_execution_time=0.0,
                last_execution_time=None,
                success_rate=0.0
            )
            
            # Notify discovery callbacks
            await self._notify_discovery_callbacks("registered", registration_info)
            
            self.logger.info("Agent registered successfully", 
                           agent_id=agent_id, name=name, framework=framework)
            
            return agent_id
            
        except Exception as e:
            self.logger.error("Agent registration failed", name=name, error=str(e))
            raise AgentRegistrationException(name, str(e))
    
    async def register_from_scan(self, scan_data: Dict[str, Any]) -> str:
        """
        Register an agent from scan results
        
        Args:
            scan_data: Agent data from codebase scan
            
        Returns:
            Agent ID
        """
        return await self.register_agent(
            name=scan_data["name"],
            framework=scan_data["framework"],
            file_path=Path(scan_data["file_path"]) if scan_data.get("file_path") else None,
            capabilities=scan_data.get("capabilities", []),
            metadata=scan_data.get("metadata", {}),
            confidence=scan_data.get("confidence", 0.0)
        )
    
    async def _validate_registration_data(
        self,
        name: str,
        framework: str,
        capabilities: Optional[List[str]]
    ) -> None:
        """Validate agent registration data"""
        errors = []
        
        if not name or len(name.strip()) == 0:
            errors.append("Agent name cannot be empty")
        
        if not framework or len(framework.strip()) == 0:
            errors.append("Framework cannot be empty")
        
        if capabilities:
            for capability in capabilities:
                if not isinstance(capability, str):
                    errors.append(f"Invalid capability type: {type(capability)}")
        
        # Check for duplicate names
        for existing_info in self._registered_agents.values():
            if existing_info.name == name and existing_info.status == RegistrationStatus.ACTIVE:
                errors.append(f"Agent with name '{name}' already registered")
                break
        
        if errors:
            raise ValidationException("Agent registration validation failed", 
                                    field="registration_data", 
                                    value={"name": name, "framework": framework})
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent
        
        Args:
            agent_id: Agent ID to unregister
            
        Returns:
            True if successful
        """
        if agent_id not in self._registered_agents:
            raise AgentNotFoundException(agent_id)
        
        try:
            self.logger.info("Unregistering agent", agent_id=agent_id)
            
            # Update database
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                agent = await agent_repo.get_by_id(session, Agent, agent_id)
                
                if agent:
                    agent.status = AgentStatus.DISABLED
                    await agent_repo.update(session, agent)
                    await session.commit()
            
            # Update registry
            registration_info = self._registered_agents[agent_id]
            registration_info.status = RegistrationStatus.DISABLED
            
            # Notify callbacks
            await self._notify_discovery_callbacks("unregistered", registration_info)
            
            self.logger.info("Agent unregistered successfully", agent_id=agent_id)
            return True
            
        except Exception as e:
            self.logger.error("Agent unregistration failed", agent_id=agent_id, error=str(e))
            raise AgentException(f"Unregistration failed: {e}", agent_id=agent_id)
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistrationInfo]:
        """Get agent registration information"""
        return self._registered_agents.get(agent_id)
    
    async def get_agent_by_name(self, name: str) -> Optional[AgentRegistrationInfo]:
        """Get agent by name"""
        for info in self._registered_agents.values():
            if info.name == name and info.status == RegistrationStatus.ACTIVE:
                return info
        return None
    
    async def list_agents(
        self,
        framework_filter: Optional[str] = None,
        capability_filter: Optional[List[str]] = None,
        status_filter: Optional[RegistrationStatus] = None
    ) -> List[AgentRegistrationInfo]:
        """
        List registered agents with optional filtering
        
        Args:
            framework_filter: Filter by framework
            capability_filter: Filter by capabilities
            status_filter: Filter by registration status
            
        Returns:
            List of matching agents
        """
        agents = list(self._registered_agents.values())
        
        # Apply filters
        if framework_filter:
            agents = [a for a in agents if a.framework.lower() == framework_filter.lower()]
        
        if capability_filter:
            agents = [
                a for a in agents 
                if any(cap in a.capabilities for cap in capability_filter)
            ]
        
        if status_filter:
            agents = [a for a in agents if a.status == status_filter]
        
        return agents
    
    async def search_agents(
        self,
        query: str,
        framework: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[AgentRegistrationInfo]:
        """
        Search agents by query string
        
        Args:
            query: Search query
            framework: Optional framework filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching agents
        """
        query_lower = query.lower()
        matching_agents = []
        
        for info in self._registered_agents.values():
            if info.status != RegistrationStatus.ACTIVE:
                continue
                
            if info.confidence < min_confidence:
                continue
                
            if framework and info.framework.lower() != framework.lower():
                continue
            
            # Search in name and capabilities
            if (query_lower in info.name.lower() or 
                any(query_lower in cap.lower() for cap in info.capabilities)):
                matching_agents.append(info)
        
        # Sort by confidence score
        matching_agents.sort(key=lambda x: x.confidence, reverse=True)
        
        return matching_agents
    
    async def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get agent performance metrics"""
        return self._agent_metrics.get(agent_id)
    
    async def update_agent_metrics(
        self,
        agent_id: str,
        execution_time: float,
        success: bool
    ) -> None:
        """Update agent performance metrics"""
        if agent_id not in self._agent_metrics:
            return
        
        metrics = self._agent_metrics[agent_id]
        
        # Update counters
        metrics.total_executions += 1
        if success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
        
        # Update average execution time
        if metrics.total_executions > 1:
            total_time = metrics.average_execution_time * (metrics.total_executions - 1) + execution_time
            metrics.average_execution_time = total_time / metrics.total_executions
        else:
            metrics.average_execution_time = execution_time
        
        # Update success rate
        metrics.success_rate = metrics.successful_executions / metrics.total_executions
        
        # Update last execution time
        metrics.last_execution_time = datetime.now(timezone.utc)
        
        # Persist to database
        try:
            async with get_db_session() as session:
                agent_repo = AgentRepository(session)
                await agent_repo.update_agent_metrics(session, agent_id, {
                    "total_executions": metrics.total_executions,
                    "successful_executions": metrics.successful_executions,
                    "failed_executions": metrics.failed_executions,
                    "average_execution_time": metrics.average_execution_time,
                    "success_rate": metrics.success_rate,
                    "last_execution_time": metrics.last_execution_time
                })
                await session.commit()
        except Exception as e:
            self.logger.error("Failed to update agent metrics", agent_id=agent_id, error=str(e))
    
    def add_discovery_callback(self, callback: callable) -> None:
        """Add callback for agent discovery events"""
        self._discovery_callbacks.append(callback)
    
    def remove_discovery_callback(self, callback: callable) -> None:
        """Remove discovery callback"""
        if callback in self._discovery_callbacks:
            self._discovery_callbacks.remove(callback)
    
    async def _notify_discovery_callbacks(
        self,
        event_type: str,
        agent_info: AgentRegistrationInfo
    ) -> None:
        """Notify all discovery callbacks"""
        for callback in self._discovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, agent_info)
                else:
                    callback(event_type, agent_info)
            except Exception as e:
                self.logger.error("Discovery callback failed", error=str(e))
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        active_agents = len([a for a in self._registered_agents.values() 
                           if a.status == RegistrationStatus.ACTIVE])
        
        framework_distribution = {}
        for info in self._registered_agents.values():
            if info.status == RegistrationStatus.ACTIVE:
                framework = info.framework
                framework_distribution[framework] = framework_distribution.get(framework, 0) + 1
        
        return {
            "total_registered": len(self._registered_agents),
            "active_agents": active_agents,
            "framework_distribution": framework_distribution,
            "average_confidence": sum(a.confidence for a in self._registered_agents.values()) / 
                                len(self._registered_agents) if self._registered_agents else 0.0
        }
    
    @property
    def registered_agent_count(self) -> int:
        """Get count of registered agents"""
        return len(self._registered_agents)
    
    @property
    def active_agent_count(self) -> int:
        """Get count of active agents"""
        return len([a for a in self._registered_agents.values() 
                   if a.status == RegistrationStatus.ACTIVE])