"""
Agent Repository for Mark-1 Orchestrator

Provides data access layer for agent-related database operations
including CRUD operations, status management, and metrics tracking.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload
import structlog

from mark1.storage.database import DatabaseRepository
from mark1.storage.models.agent_model import Agent, AgentStatus, AgentType
from mark1.utils.exceptions import DatabaseError, AgentNotFoundException


class AgentRepository(DatabaseRepository):
    """
    Repository for agent data access operations
    
    Provides methods for creating, reading, updating, and deleting agents,
    as well as specialized queries for agent management.
    """
    
    def __init__(self, db_manager=None):
        # Initialize with the parent class first, pass db_manager
        # to maintain compatibility with both simplified and original
        # This will correctly handle both the original and simplified implementations
        try:
            super().__init__(db_manager)
        except TypeError:
            # If the parent init doesn't accept db_manager, try without it
            super().__init__()
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def create_agent(
        self,
        session: AsyncSession,
        name: str,
        agent_type: AgentType,
        framework: str,
        file_path: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> Agent:
        """
        Create a new agent record
        
        Args:
            session: Database session
            name: Agent name
            agent_type: Type of agent
            framework: Framework name
            file_path: Path to agent file
            capabilities: List of capabilities
            metadata: Additional metadata
            confidence: Confidence score
            
        Returns:
            Created agent instance
        """
        try:
            agent = Agent(
                name=name,
                type=agent_type,
                framework=framework,
                file_path=file_path,
                capabilities=capabilities or [],
                metadata=metadata or {},
                confidence=confidence,
                status=AgentStatus.READY,
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
                # Update last accessed time
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
    
    async def list_agents(
        self,
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100,
        status_filter: Optional[AgentStatus] = None,
        type_filter: Optional[AgentType] = None,
        framework_filter: Optional[str] = None
    ) -> List[Agent]:
        """
        List agents with optional filtering and pagination
        
        Args:
            session: Database session
            offset: Number of records to skip
            limit: Maximum number of records to return
            status_filter: Filter by agent status
            type_filter: Filter by agent type
            framework_filter: Filter by framework
            
        Returns:
            List of agents
        """
        try:
            query = select(Agent)
            
            # Apply filters
            if status_filter:
                query = query.where(Agent.status == status_filter)
            
            if type_filter:
                query = query.where(Agent.type == type_filter)
            
            if framework_filter:
                query = query.where(Agent.framework.ilike(f"%{framework_filter}%"))
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Order by created_at descending
            query = query.order_by(Agent.created_at.desc())
            
            result = await session.execute(query)
            agents = result.scalars().all()
            
            self.logger.debug("Listed agents", count=len(agents), filters={
                "status": status_filter,
                "type": type_filter,
                "framework": framework_filter
            })
            
            return list(agents)
            
        except Exception as e:
            self.logger.error("Failed to list agents", error=str(e))
            raise DatabaseError(f"Agent listing failed: {e}")
    
    async def list_by_status(self, session: AsyncSession, status: AgentStatus) -> List[Agent]:
        """Get all agents with specific status"""
        try:
            query = select(Agent).where(Agent.status == status)
            result = await session.execute(query)
            agents = result.scalars().all()
            
            return list(agents)
            
        except Exception as e:
            self.logger.error("Failed to list agents by status", status=status, error=str(e))
            raise DatabaseError(f"Agent listing by status failed: {e}")
    
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
    
    async def update_agent_status(
        self,
        session: AsyncSession,
        agent_id: str,
        status: AgentStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update agent status
        
        Args:
            session: Database session
            agent_id: Agent ID
            status: New status
            error_message: Optional error message for failed status
            
        Returns:
            True if updated successfully
        """
        try:
            query = (
                update(Agent)
                .where(Agent.id == agent_id)
                .values(
                    status=status,
                    error_message=error_message,
                    last_activity=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
            )
            
            result = await session.execute(query)
            
            if result.rowcount == 0:
                raise AgentNotFoundException(agent_id)
            
            self.logger.info("Agent status updated", agent_id=agent_id, status=status)
            return True
            
        except AgentNotFoundException:
            raise
        except Exception as e:
            self.logger.error("Failed to update agent status", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent status update failed: {e}")
    
    async def update_agent_metadata(
        self,
        session: AsyncSession,
        agent_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update agent metadata"""
        try:
            agent = await self.get_by_id(session, Agent, agent_id)
            if not agent:
                raise AgentNotFoundException(agent_id)
            
            # Merge with existing metadata
            if agent.extra_metadata:
                agent.extra_metadata.update(metadata)
            else:
                agent.extra_metadata = metadata
            
            agent.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            self.logger.info("Agent metadata updated", agent_id=agent_id)
            return True
            
        except AgentNotFoundException:
            raise
        except Exception as e:
            self.logger.error("Failed to update agent metadata", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent metadata update failed: {e}")
    
    async def update_agent_capabilities(
        self,
        session: AsyncSession,
        agent_id: str,
        capabilities: List[str]
    ) -> bool:
        """Update agent capabilities"""
        try:
            query = (
                update(Agent)
                .where(Agent.id == agent_id)
                .values(
                    capabilities=capabilities,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            
            result = await session.execute(query)
            
            if result.rowcount == 0:
                raise AgentNotFoundException(agent_id)
            
            self.logger.info("Agent capabilities updated", agent_id=agent_id)
            return True
            
        except AgentNotFoundException:
            raise
        except Exception as e:
            self.logger.error("Failed to update agent capabilities", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent capabilities update failed: {e}")
    
    async def search_agents(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int = 50
    ) -> List[Agent]:
        """
        Search agents by name, capabilities, or metadata
        
        Args:
            session: Database session
            query_text: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching agents
        """
        try:
            # Create search conditions
            search_term = f"%{query_text.lower()}%"
            
            query = select(Agent).where(
                or_(
                    Agent.name.ilike(search_term),
                    Agent.framework.ilike(search_term),
                    Agent.capabilities.op('?|')(f'{{"{query_text.lower()}"}}'.split(','))
                )
            ).limit(limit)
            
            result = await session.execute(query)
            agents = result.scalars().all()
            
            self.logger.debug("Searched agents", query=query_text, results=len(agents))
            
            return list(agents)
            
        except Exception as e:
            self.logger.error("Failed to search agents", query=query_text, error=str(e))
            raise DatabaseError(f"Agent search failed: {e}")
    
    async def get_agents_by_capability(
        self,
        session: AsyncSession,
        capability_name: str
    ) -> List[Agent]:
        """Get agents that have a specific capability"""
        try:
            from mark1.storage.models.agent_model import Agent, agent_capabilities
            from mark1.storage.models.agent_model import Capability
            
            # Use the correct join syntax for the agent_capabilities association table
            query = (
                select(Agent)
                .join(agent_capabilities)
                .join(Capability)
                .where(
                    and_(
                        Capability.name == capability_name,
                        Agent.status == AgentStatus.READY
                    )
                )
            )
            
            result = await session.execute(query)
            agents = result.scalars().all()
            
            return list(agents)
            
        except Exception as e:
            self.logger.error("Failed to get agents by capability", capability=capability_name, error=str(e))
            raise DatabaseError(f"Agent capability query failed: {e}")
    
    async def count_by_status(self, session: AsyncSession, status: AgentStatus) -> int:
        """Count agents by status"""
        try:
            query = select(func.count(Agent.id)).where(Agent.status == status)
            result = await session.execute(query)
            count = result.scalar()
            
            return count or 0
            
        except Exception as e:
            self.logger.error("Failed to count agents by status", status=status, error=str(e))
            raise DatabaseError(f"Agent count failed: {e}")
    
    async def get_agent_statistics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        try:
            # Count by status
            status_counts = {}
            for status in AgentStatus:
                count = await self.count_by_status(session, status)
                status_counts[status.value] = count
            
            # Count by type
            type_query = select(Agent.type, func.count(Agent.id)).group_by(Agent.type)
            type_result = await session.execute(type_query)
            type_counts = {row[0].value: row[1] for row in type_result}
            
            # Count by framework
            framework_query = select(Agent.framework, func.count(Agent.id)).group_by(Agent.framework)
            framework_result = await session.execute(framework_query)
            framework_counts = {row[0]: row[1] for row in framework_result}
            
            # Get total count
            total_query = select(func.count(Agent.id))
            total_result = await session.execute(total_query)
            total_count = total_result.scalar()
            
            return {
                "total_agents": total_count,
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "framework_distribution": framework_counts
            }
            
        except Exception as e:
            self.logger.error("Failed to get agent statistics", error=str(e))
            raise DatabaseError(f"Agent statistics failed: {e}")
    
    async def delete_agent(self, session: AsyncSession, agent_id: str) -> bool:
        """
        Delete an agent (soft delete by setting status to disabled)
        
        Args:
            session: Database session
            agent_id: Agent ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Instead of hard delete, we'll update status to disabled
            result = await self.update_agent_status(
                session, 
                agent_id, 
                AgentStatus.DISABLED,
                "Agent deleted"
            )
            
            self.logger.info("Agent deleted (soft delete)", agent_id=agent_id)
            return result
            
        except Exception as e:
            self.logger.error("Failed to delete agent", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent deletion failed: {e}")
    
    async def get_agent_metrics(self, session: AsyncSession, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent performance metrics"""
        try:
            agent = await self.get_by_id(session, Agent, agent_id)
            if not agent:
                return None
            
            # Return metrics from agent metadata
            return agent.extra_metadata.get("metrics", {})
            
        except Exception as e:
            self.logger.error("Failed to get agent metrics", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent metrics retrieval failed: {e}")
    
    async def update_agent_metrics(
        self,
        session: AsyncSession,
        agent_id: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """Update agent performance metrics"""
        try:
            agent = await self.get_by_id(session, Agent, agent_id)
            if not agent:
                raise AgentNotFoundException(agent_id)
            
            # Update metrics in metadata
            if not agent.extra_metadata:
                agent.extra_metadata = {}
            
            agent.extra_metadata["metrics"] = metrics
            agent.extra_metadata["metrics_updated_at"] = datetime.now(timezone.utc).isoformat()
            agent.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            self.logger.debug("Agent metrics updated", agent_id=agent_id)
            return True
            
        except AgentNotFoundException:
            raise
        except Exception as e:
            self.logger.error("Failed to update agent metrics", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent metrics update failed: {e}")
    
    async def get_recently_active_agents(
        self,
        session: AsyncSession,
        hours: int = 24,
        limit: int = 50
    ) -> List[Agent]:
        """Get agents that were active within the specified hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            query = (
                select(Agent)
                .where(Agent.last_activity >= cutoff_time)
                .order_by(Agent.last_activity.desc())
                .limit(limit)
            )
            
            result = await session.execute(query)
            agents = result.scalars().all()
            
            return list(agents)
            
        except Exception as e:
            self.logger.error("Failed to get recently active agents", error=str(e))
            raise DatabaseError(f"Recent agents query failed: {e}")
    
    async def update_agent_heartbeat(self, session: AsyncSession, agent_id: str) -> bool:
        """Update agent heartbeat timestamp"""
        try:
            query = (
                update(Agent)
                .where(Agent.id == agent_id)
                .values(last_activity=datetime.now(timezone.utc))
            )
            
            result = await session.execute(query)
            
            if result.rowcount == 0:
                raise AgentNotFoundException(agent_id)
            
            return True
            
        except AgentNotFoundException:
            raise
        except Exception as e:
            self.logger.error("Failed to update agent heartbeat", agent_id=agent_id, error=str(e))
            raise DatabaseError(f"Agent heartbeat update failed: {e}")
