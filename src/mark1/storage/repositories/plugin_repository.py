#!/usr/bin/env python3
"""
Plugin Repository for Mark-1 Universal Plugin System

Database operations for plugin management.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, select
import structlog

from ..models.plugin_model import Plugin, PluginCapability, PluginExecution, PluginWorkflow
from ...plugins.base_plugin import PluginMetadata, PluginCapability as PluginCapabilityData


logger = structlog.get_logger(__name__)


class PluginRepository:
    """Repository for plugin database operations"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = structlog.get_logger(__name__)
    
    async def create_plugin(self, metadata: PluginMetadata) -> Plugin:
        """Create a new plugin record"""
        try:
            plugin = Plugin(
                plugin_id=metadata.plugin_id,
                name=metadata.name,
                description=metadata.description,
                version=metadata.version,
                author=metadata.author,
                repository_url=metadata.repository_url,
                plugin_type=metadata.plugin_type.value,
                execution_mode=metadata.execution_mode.value,
                status=metadata.status.value,
                dependencies=metadata.dependencies,
                environment_variables=metadata.environment_variables,
                configuration=metadata.configuration,
                entry_points=metadata.entry_points,
                installed_at=datetime.now(timezone.utc)
            )
            
            self.db.add(plugin)
            
            # Add capabilities
            for cap_data in metadata.capabilities:
                capability = PluginCapability(
                    plugin_id=metadata.plugin_id,
                    name=cap_data.name,
                    description=cap_data.description,
                    input_types=cap_data.input_types,
                    output_types=cap_data.output_types,
                    parameters=cap_data.parameters,
                    examples=cap_data.examples
                )
                self.db.add(capability)
            
            self.db.commit()
            self.db.refresh(plugin)
            
            self.logger.info("Plugin created in database", plugin_id=metadata.plugin_id)
            return plugin
        
        except Exception as e:
            self.db.rollback()
            self.logger.error("Failed to create plugin", plugin_id=metadata.plugin_id, error=str(e))
            raise
    
    async def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID"""
        try:
            from sqlalchemy import select
            stmt = select(Plugin).where(Plugin.plugin_id == plugin_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error("Failed to get plugin", plugin_id=plugin_id, error=str(e))
            return None
    
    async def get_all_plugins(self, status: Optional[str] = None) -> List[Plugin]:
        """Get all plugins, optionally filtered by status"""
        try:
            from sqlalchemy import select
            stmt = select(Plugin)
            if status:
                stmt = stmt.where(Plugin.status == status)
            stmt = stmt.order_by(Plugin.created_at.desc())
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error("Failed to get all plugins", error=str(e))
            return []
    
    async def update_plugin_status(self, plugin_id: str, status: str) -> bool:
        """Update plugin status"""
        try:
            plugin = await self.get_plugin(plugin_id)
            if plugin:
                plugin.status = status
                plugin.updated_at = datetime.now(timezone.utc)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            self.logger.error("Failed to update plugin status", plugin_id=plugin_id, error=str(e))
            return False
    
    async def update_plugin_usage_stats(
        self,
        plugin_id: str,
        execution_time: float,
        success: bool
    ) -> bool:
        """Update plugin usage statistics"""
        try:
            plugin = await self.get_plugin(plugin_id)
            if plugin:
                plugin.usage_count += 1
                if success:
                    plugin.success_count += 1
                else:
                    plugin.error_count += 1
                
                # Update average execution time
                total_time = plugin.average_execution_time * (plugin.usage_count - 1) + execution_time
                plugin.average_execution_time = total_time / plugin.usage_count
                
                plugin.last_used_at = datetime.now(timezone.utc)
                plugin.updated_at = datetime.now(timezone.utc)
                
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            self.logger.error("Failed to update plugin usage stats", plugin_id=plugin_id, error=str(e))
            return False
    
    async def delete_plugin(self, plugin_id: str) -> bool:
        """Delete plugin and all related data"""
        try:
            plugin = await self.get_plugin(plugin_id)
            if plugin:
                self.db.delete(plugin)
                self.db.commit()
                self.logger.info("Plugin deleted from database", plugin_id=plugin_id)
                return True
            return False
        except Exception as e:
            self.db.rollback()
            self.logger.error("Failed to delete plugin", plugin_id=plugin_id, error=str(e))
            return False
    
    async def search_plugins(
        self,
        query: str,
        plugin_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Plugin]:
        """Search plugins by name, description, or capabilities"""
        try:
            from sqlalchemy import select
            stmt = select(Plugin)

            # Text search
            if query:
                search_filter = or_(
                    Plugin.name.ilike(f"%{query}%"),
                    Plugin.description.ilike(f"%{query}%")
                )
                stmt = stmt.where(search_filter)

            # Type filter
            if plugin_type:
                stmt = stmt.where(Plugin.plugin_type == plugin_type)

            stmt = stmt.limit(limit)
            result = await self.db.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            self.logger.error("Failed to search plugins", query=query, error=str(e))
            return []
    
    async def get_plugin_capabilities(self, plugin_id: str) -> List[PluginCapability]:
        """Get all capabilities for a plugin"""
        try:
            from sqlalchemy import select
            stmt = select(PluginCapability).where(PluginCapability.plugin_id == plugin_id)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error("Failed to get plugin capabilities", plugin_id=plugin_id, error=str(e))
            return []
    
    async def record_plugin_execution(
        self,
        plugin_id: str,
        capability_name: str,
        execution_id: str,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
        status: str,
        execution_time: float,
        outputs: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[PluginExecution]:
        """Record plugin execution (non-blocking, returns None on failure)"""
        try:
            # Check if plugin exists first
            plugin = await self.get_plugin(plugin_id)
            if not plugin:
                self.logger.warning("Plugin not found for execution recording", plugin_id=plugin_id)
                return None

            # Get capability ID if exists
            from sqlalchemy import select
            stmt = select(PluginCapability).where(
                and_(
                    PluginCapability.plugin_id == plugin_id,
                    PluginCapability.name == capability_name
                )
            )
            result = await self.db.execute(stmt)
            capability = result.scalar_one_or_none()

            execution = PluginExecution(
                plugin_id=plugin_id,
                capability_id=capability.id if capability else None,
                capability_name=capability_name,
                execution_id=execution_id,
                status=status,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                execution_time=execution_time,
                inputs=inputs,
                parameters=parameters,
                outputs=outputs or {},
                error_message=error_message
            )

            self.db.add(execution)
            await self.db.commit()
            await self.db.refresh(execution)

            # Update plugin usage stats (non-blocking)
            try:
                await self.update_plugin_usage_stats(
                    plugin_id, execution_time, status == "success"
                )
            except Exception as stats_error:
                self.logger.warning("Failed to update plugin stats", error=str(stats_error))

            return execution

        except Exception as e:
            try:
                await self.db.rollback()
            except:
                pass  # Ignore rollback errors
            self.logger.warning("Failed to record plugin execution (non-critical)",
                              plugin_id=plugin_id, error=str(e))
            return None
    
    async def get_plugin_execution_history(
        self,
        plugin_id: str,
        limit: int = 100
    ) -> List[PluginExecution]:
        """Get execution history for a plugin"""
        try:
            stmt = select(PluginExecution).where(
                PluginExecution.plugin_id == plugin_id
            ).order_by(desc(PluginExecution.started_at)).limit(limit)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error("Failed to get plugin execution history", 
                            plugin_id=plugin_id, error=str(e))
            return []
    
    async def get_plugin_stats(self, plugin_id: str) -> Dict[str, Any]:
        """Get comprehensive plugin statistics"""
        try:
            plugin = await self.get_plugin(plugin_id)
            if not plugin:
                return {}
            
            # Get execution statistics
            total_stmt = select(func.count(PluginExecution.id)).where(
                PluginExecution.plugin_id == plugin_id
            )
            total_result = await self.db.execute(total_stmt)
            total_executions = total_result.scalar() or 0

            success_stmt = select(func.count(PluginExecution.id)).where(
                and_(
                    PluginExecution.plugin_id == plugin_id,
                    PluginExecution.status == "success"
                )
            )
            success_result = await self.db.execute(success_stmt)
            successful_executions = success_result.scalar() or 0

            avg_stmt = select(func.avg(PluginExecution.execution_time)).where(
                PluginExecution.plugin_id == plugin_id
            )
            avg_result = await self.db.execute(avg_stmt)
            avg_execution_time = avg_result.scalar() or 0.0

            # Recent activity
            recent_stmt = select(PluginExecution).where(
                PluginExecution.plugin_id == plugin_id
            ).order_by(desc(PluginExecution.started_at)).limit(10)
            recent_result = await self.db.execute(recent_stmt)
            recent_executions = recent_result.scalars().all()
            
            return {
                "plugin_id": plugin_id,
                "name": plugin.name,
                "status": plugin.status,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
                "average_execution_time": float(avg_execution_time),
                "last_used_at": plugin.last_used_at.isoformat() if plugin.last_used_at else None,
                "recent_executions": [exec.to_dict() for exec in recent_executions]
            }
        
        except Exception as e:
            self.logger.error("Failed to get plugin stats", plugin_id=plugin_id, error=str(e))
            return {}
