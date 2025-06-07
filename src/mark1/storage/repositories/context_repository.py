"""
Context Repository for Mark-1 Orchestrator

Session 19: Advanced Context Management
Provides enhanced data access layer for context-related database operations.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set, Union
from uuid import UUID
import uuid as uuid_module
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, Integer
from sqlalchemy.orm import selectinload
import structlog

from mark1.storage.database import DatabaseRepository, get_db_session
from mark1.storage.models.context_model import ContextModel, ContextType, ContextScope, ContextPriority
from mark1.utils.exceptions import DatabaseError, ContextError


class ContextRepository(DatabaseRepository):
    """
    Enhanced repository for context data access operations
    
    Session 19: Advanced Context Management
    Provides methods for creating, reading, updating, and deleting contexts
    with support for hierarchies, versioning, and advanced querying.
    """
    
    def __init__(self, db_manager=None):
        super().__init__(db_manager)
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def create_context(
        self,
        session: AsyncSession,
        context_key: str,
        context_type: str,
        context_scope: str,
        priority: str = ContextPriority.MEDIUM.value,
        parent_context_id: Optional[Union[str, UUID]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        content: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> ContextModel:
        """Create a new context record with enhanced features"""
        try:
            # Convert parent_context_id to UUID if it's a string
            if parent_context_id and isinstance(parent_context_id, str):
                try:
                    parent_context_id = uuid_module.UUID(parent_context_id)
                except ValueError:
                    # If it's not a valid UUID, set it to None
                    self.logger.warning("Invalid parent context ID format", parent_context_id=parent_context_id)
                    parent_context_id = None
            
            # Calculate content metrics
            content = content or {}
            raw_content = kwargs.get('raw_content')
            
            # Calculate size and token count
            import json
            content_str = json.dumps(content) if content else ""
            size_bytes = len(content_str.encode('utf-8'))
            token_count = len(content_str.split()) if content_str else 0
            
            # Generate checksum
            import hashlib
            checksum = hashlib.sha256(content_str.encode('utf-8')).hexdigest() if content_str else None
            
            context = ContextModel(
                context_key=context_key,
                context_type=context_type,
                context_scope=context_scope,
                priority=priority,
                parent_context_id=parent_context_id,
                agent_id=agent_id,
                task_id=task_id,
                session_id=session_id,
                title=title,
                content=content,
                raw_content=raw_content,
                expires_at=expires_at,
                size_bytes=size_bytes,
                token_count=token_count,
                checksum=checksum,
                created_at=datetime.now(timezone.utc),
                **kwargs
            )
            
            # Add tags as metadata
            if tags:
                if not context.extra_metadata:
                    context.extra_metadata = {}
                context.extra_metadata['tags'] = tags
            
            created_context = await self.create(session, context)
            self.logger.info("Enhanced context created", 
                           context_id=created_context.id, 
                           context_key=context_key,
                           size_bytes=size_bytes,
                           has_parent=parent_context_id is not None)
            
            return created_context
            
        except Exception as e:
            self.logger.error("Failed to create enhanced context", context_key=context_key, error=str(e))
            raise DatabaseError(f"Enhanced context creation failed: {e}")
    
    async def get_by_id(self, context_id: Union[str, UUID], session: Optional[AsyncSession] = None) -> Optional[ContextModel]:
        """Get context by ID"""
        should_close_session = session is None
        if session is None:
            session = self.session_factory()
        
        try:
            # Convert string ID to UUID if needed
            if isinstance(context_id, str):
                context_id = uuid_module.UUID(context_id)
            
            query = select(ContextModel).where(
                and_(
                    ContextModel.id == context_id,
                    ContextModel.is_active == True
                )
            )
            
            result = await session.execute(query)
            context = result.scalar_one_or_none()
            
            if context:
                self.logger.debug("Retrieved context by ID", context_id=context_id, context_key=context.context_key)
            else:
                self.logger.warning("Context not found", context_id=context_id)
            
            return context
            
        except Exception as e:
            self.logger.error("Failed to get context by ID", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context retrieval failed: {e}")
        finally:
            if should_close_session:
                await session.close()
    
    async def get_context_by_key(
        self,
        session: AsyncSession,
        context_key: str,
        context_type: Optional[str] = None,
        context_scope: Optional[str] = None
    ) -> Optional[ContextModel]:
        """Get context by key with optional type and scope filters"""
        try:
            query = select(ContextModel).where(
                ContextModel.context_key == context_key,
                ContextModel.is_active == True
            )
            
            if context_type:
                query = query.where(ContextModel.context_type == context_type)
            
            if context_scope:
                query = query.where(ContextModel.context_scope == context_scope)
            
            result = await session.execute(query)
            context = result.scalar_one_or_none()
            
            if context:
                # Update accessed timestamp
                context.accessed_at = datetime.now(timezone.utc)
                await session.commit()
            
            return context
            
        except Exception as e:
            self.logger.error("Failed to get context by key", context_key=context_key, error=str(e))
            raise DatabaseError(f"Context retrieval failed: {e}")

    async def list_contexts(
        self,
        session: AsyncSession,
        limit: int = 100,
        offset: int = 0,
        context_type: Optional[str] = None,
        context_scope: Optional[str] = None,
        priority: Optional[str] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_context_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_archived: bool = False,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[ContextModel]:
        """Enhanced context listing with advanced filtering"""
        try:
            query = select(ContextModel)
            
            # Base filters
            filters = []
            if not include_archived:
                filters.append(ContextModel.is_active == True)
                filters.append(ContextModel.is_archived == False)
            
            # Type filters
            if context_type:
                filters.append(ContextModel.context_type == context_type)
            if context_scope:
                filters.append(ContextModel.context_scope == context_scope)
            if priority:
                filters.append(ContextModel.priority == priority)
            
            # Relationship filters
            if agent_id:
                filters.append(ContextModel.agent_id == agent_id)
            if task_id:
                filters.append(ContextModel.task_id == task_id)
            if session_id:
                filters.append(ContextModel.session_id == session_id)
            if parent_context_id:
                filters.append(ContextModel.parent_context_id == parent_context_id)
            
            # Tag filters
            if tags:
                tag_filters = []
                for tag in tags:
                    tag_filters.append(
                        ContextModel.extra_metadata['tags'].astext.contains(tag)
                    )
                if tag_filters:
                    filters.append(or_(*tag_filters))
            
            if filters:
                query = query.where(and_(*filters))
            
            # Ordering
            if hasattr(ContextModel, order_by):
                order_column = getattr(ContextModel, order_by)
                if order_desc:
                    query = query.order_by(order_column.desc())
                else:
                    query = query.order_by(order_column.asc())
            
            # Pagination
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            contexts = result.scalars().all()
            
            return list(contexts)
            
        except Exception as e:
            self.logger.error("Failed to list contexts", error=str(e))
            raise DatabaseError(f"Context listing failed: {e}")

    async def get_context_hierarchy(
        self, 
        session: AsyncSession, 
        root_context_id: str,
        max_depth: int = 10
    ) -> List[ContextModel]:
        """Get complete context hierarchy starting from root"""
        try:
            hierarchy_contexts = []
            
            # Recursive CTE query to get hierarchy
            cte_query = select(
                ContextModel.id,
                ContextModel.parent_context_id,
                func.cast(0, Integer).label('depth')
            ).where(
                ContextModel.id == root_context_id,
                ContextModel.is_active == True
            ).cte(name='hierarchy', recursive=True)
            
            # Recursive part
            recursive_query = select(
                ContextModel.id,
                ContextModel.parent_context_id,
                (cte_query.c.depth + 1).label('depth')
            ).select_from(
                ContextModel.join(cte_query, ContextModel.parent_context_id == cte_query.c.id)
            ).where(
                ContextModel.is_active == True,
                cte_query.c.depth < max_depth
            )
            
            cte_query = cte_query.union_all(recursive_query)
            
            # Get all contexts in hierarchy
            final_query = select(ContextModel).select_from(
                ContextModel.join(cte_query, ContextModel.id == cte_query.c.id)
            ).order_by(cte_query.c.depth, ContextModel.created_at)
            
            result = await session.execute(final_query)
            hierarchy_contexts = result.scalars().all()
            
            return list(hierarchy_contexts)
            
        except Exception as e:
            self.logger.error("Failed to get context hierarchy", root_context=root_context_id, error=str(e))
            return []

    async def get_context_children(
        self, 
        session: AsyncSession, 
        parent_context_id: str,
        recursive: bool = False
    ) -> List[ContextModel]:
        """Get direct children or entire subtree of a context"""
        try:
            if recursive:
                return await self.get_context_hierarchy(session, parent_context_id)
            else:
                query = select(ContextModel).where(
                    ContextModel.parent_context_id == parent_context_id,
                    ContextModel.is_active == True
                ).order_by(ContextModel.created_at)
                
                result = await session.execute(query)
                children = result.scalars().all()
                
                return list(children)
                
        except Exception as e:
            self.logger.error("Failed to get context children", parent_context=parent_context_id, error=str(e))
            return []

    async def update_context_content(
        self,
        session: AsyncSession,
        context_id: str,
        content: Dict[str, Any],
        update_metadata: bool = True
    ) -> bool:
        """Update context content with metadata refresh"""
        try:
            # Calculate new metrics
            import json
            content_str = json.dumps(content)
            size_bytes = len(content_str.encode('utf-8'))
            token_count = len(content_str.split())
            
            # Generate new checksum
            import hashlib
            checksum = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
            
            update_data = {
                'content': content,
                'updated_at': datetime.now(timezone.utc)
            }
            
            if update_metadata:
                update_data.update({
                    'size_bytes': size_bytes,
                    'token_count': token_count,
                    'checksum': checksum
                })
            
            query = update(ContextModel).where(
                ContextModel.id == context_id
            ).values(**update_data)
            
            result = await session.execute(query)
            success = result.rowcount > 0
            
            if success:
                self.logger.debug("Context content updated", 
                                context_id=context_id, 
                                size_bytes=size_bytes)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to update context content", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context content update failed: {e}")

    async def update_context_metadata(
        self,
        session: AsyncSession,
        context_id: str,
        title: Optional[str] = None,
        priority: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update context metadata"""
        try:
            update_data = {'updated_at': datetime.now(timezone.utc)}
            
            if title is not None:
                update_data['title'] = title
            if priority is not None:
                update_data['priority'] = priority
            if expires_at is not None:
                update_data['expires_at'] = expires_at
            
            # Handle tags and extra metadata
            if tags is not None or extra_metadata is not None:
                # Get current context to merge metadata
                context = await self.get_by_id(context_id)
                if context:
                    current_metadata = context.extra_metadata or {}
                    
                    if tags is not None:
                        current_metadata['tags'] = tags
                    
                    if extra_metadata is not None:
                        current_metadata.update(extra_metadata)
                    
                    update_data['extra_metadata'] = current_metadata
            
            query = update(ContextModel).where(
                ContextModel.id == context_id
            ).values(**update_data)
            
            result = await session.execute(query)
            return result.rowcount > 0
            
        except Exception as e:
            self.logger.error("Failed to update context metadata", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context metadata update failed: {e}")

    async def archive_context(self, session: AsyncSession, context_id: str) -> bool:
        """Archive a context (soft delete)"""
        try:
            query = update(ContextModel).where(
                ContextModel.id == context_id
            ).values(
                is_archived=True,
                is_active=False,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await session.execute(query)
            success = result.rowcount > 0
            
            if success:
                self.logger.info("Context archived", context_id=context_id)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to archive context", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context archival failed: {e}")

    async def restore_context(self, session: AsyncSession, context_id: str) -> bool:
        """Restore an archived context"""
        try:
            query = update(ContextModel).where(
                ContextModel.id == context_id
            ).values(
                is_archived=False,
                is_active=True,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await session.execute(query)
            success = result.rowcount > 0
            
            if success:
                self.logger.info("Context restored", context_id=context_id)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to restore context", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context restoration failed: {e}")

    async def delete_context(self, session: AsyncSession, context_id: str) -> bool:
        """Permanently delete a context"""
        try:
            query = delete(ContextModel).where(ContextModel.id == context_id)
            result = await session.execute(query)
            success = result.rowcount > 0
            
            if success:
                self.logger.warning("Context permanently deleted", context_id=context_id)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to delete context", context_id=context_id, error=str(e))
            raise DatabaseError(f"Context deletion failed: {e}")

    async def bulk_archive_contexts(
        self, 
        session: AsyncSession, 
        context_ids: List[str]
    ) -> Dict[str, bool]:
        """Bulk archive multiple contexts"""
        try:
            results = {}
            
            if not context_ids:
                return results
            
            query = update(ContextModel).where(
                ContextModel.id.in_(context_ids)
            ).values(
                is_archived=True,
                is_active=False,
                updated_at=datetime.now(timezone.utc)
            )
            
            result = await session.execute(query)
            archived_count = result.rowcount
            
            # Mark all as successful (simplified)
            for context_id in context_ids:
                results[context_id] = True
            
            self.logger.info("Bulk archive completed", 
                           requested=len(context_ids), 
                           archived=archived_count)
            
            return results
            
        except Exception as e:
            self.logger.error("Failed to bulk archive contexts", error=str(e))
            raise DatabaseError(f"Bulk archive failed: {e}")

    async def get_context_statistics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive context statistics"""
        try:
            stats = {}
            
            # Total counts
            total_query = select(func.count(ContextModel.id)).where(ContextModel.is_active == True)
            total_result = await session.execute(total_query)
            stats['total_contexts'] = total_result.scalar()
            
            # Archived counts
            archived_query = select(func.count(ContextModel.id)).where(ContextModel.is_archived == True)
            archived_result = await session.execute(archived_query)
            stats['archived_contexts'] = archived_result.scalar()
            
            # By type
            type_query = select(
                ContextModel.context_type,
                func.count(ContextModel.id)
            ).where(
                ContextModel.is_active == True
            ).group_by(ContextModel.context_type)
            
            type_result = await session.execute(type_query)
            stats['by_type'] = dict(type_result.all())
            
            # By scope
            scope_query = select(
                ContextModel.context_scope,
                func.count(ContextModel.id)
            ).where(
                ContextModel.is_active == True
            ).group_by(ContextModel.context_scope)
            
            scope_result = await session.execute(scope_query)
            stats['by_scope'] = dict(scope_result.all())
            
            # By priority
            priority_query = select(
                ContextModel.priority,
                func.count(ContextModel.id)
            ).where(
                ContextModel.is_active == True
            ).group_by(ContextModel.priority)
            
            priority_result = await session.execute(priority_query)
            stats['by_priority'] = dict(priority_result.all())
            
            # Size statistics
            size_query = select(
                func.avg(ContextModel.size_bytes),
                func.max(ContextModel.size_bytes),
                func.sum(ContextModel.size_bytes)
            ).where(
                ContextModel.is_active == True,
                ContextModel.size_bytes.isnot(None)
            )
            
            size_result = await session.execute(size_query)
            avg_size, max_size, total_size = size_result.one()
            
            stats['size_stats'] = {
                'average_size_bytes': float(avg_size) if avg_size else 0,
                'max_size_bytes': int(max_size) if max_size else 0,
                'total_size_bytes': int(total_size) if total_size else 0
            }
            
            # Hierarchical statistics
            hierarchy_query = select(func.count(ContextModel.id)).where(
                ContextModel.parent_context_id.isnot(None),
                ContextModel.is_active == True
            )
            hierarchy_result = await session.execute(hierarchy_query)
            stats['hierarchical_contexts'] = hierarchy_result.scalar()
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get context statistics", error=str(e))
            return {}

    async def cleanup_expired_contexts(self, session: AsyncSession) -> int:
        """Clean up expired contexts"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Archive expired contexts
            query = update(ContextModel).where(
                ContextModel.expires_at < current_time,
                ContextModel.is_active == True
            ).values(
                is_archived=True,
                is_active=False,
                updated_at=current_time
            )
            
            result = await session.execute(query)
            cleaned_count = result.rowcount
            
            if cleaned_count > 0:
                self.logger.info("Cleaned up expired contexts", count=cleaned_count)
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error("Failed to cleanup expired contexts", error=str(e))
            return 0

    async def search_contexts(
        self,
        session: AsyncSession,
        search_query: str,
        context_type: Optional[str] = None,
        limit: int = 50
    ) -> List[ContextModel]:
        """Search contexts by content and metadata"""
        try:
            # Build search filters
            search_filters = [ContextModel.is_active == True]
            
            if context_type:
                search_filters.append(ContextModel.context_type == context_type)
            
            # Text search in title, content, and raw_content
            text_search = or_(
                ContextModel.title.ilike(f'%{search_query}%'),
                ContextModel.raw_content.ilike(f'%{search_query}%'),
                # JSON content search would need specific database support
            )
            search_filters.append(text_search)
            
            query = select(ContextModel).where(
                and_(*search_filters)
            ).order_by(
                ContextModel.updated_at.desc()
            ).limit(limit)
            
            result = await session.execute(query)
            contexts = result.scalars().all()
            
            return list(contexts)
            
        except Exception as e:
            self.logger.error("Failed to search contexts", search_query=search_query, error=str(e))
            return []

    async def update_enhanced_context(
        self,
        session: AsyncSession,
        context_id: Union[str, UUID],
        content: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update context with enhanced features"""
        try:
            # Convert string ID to UUID if needed
            if isinstance(context_id, str):
                context_id = uuid_module.UUID(context_id)
                
            if tags is not None or extra_metadata is not None:
                # Get current context to merge metadata
                context = await self.get_by_id(context_id, session)
                if context:
                    current_metadata = context.extra_metadata or {}
                    
                    # Merge tags into metadata
                    if tags is not None:
                        current_metadata['tags'] = list(tags)
                    
                    # Merge additional metadata
                    if extra_metadata:
                        current_metadata.update(extra_metadata)
                    
                    # Use the merged metadata
                    extra_metadata = current_metadata
            
            update_data = {
                'content': content,
                'updated_at': datetime.now(timezone.utc)
            }
            
            if extra_metadata:
                update_data.update({
                    'size_bytes': extra_metadata.get('size_bytes'),
                    'token_count': extra_metadata.get('token_count'),
                    'checksum': extra_metadata.get('checksum')
                })
            
            query = update(ContextModel).where(
                ContextModel.id == context_id
            ).values(**update_data)
            
            result = await session.execute(query)
            return result.rowcount > 0
            
        except Exception as e:
            self.logger.error("Failed to update enhanced context", context_id=context_id, error=str(e))
            raise DatabaseError(f"Enhanced context update failed: {e}")
