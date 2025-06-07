"""
Advanced Context Manager for Mark-1 Orchestrator

Session 19: Advanced Context Management
Provides intelligent context sharing, memory optimization, and lifecycle management
"""

import asyncio
import json
import gzip
import hashlib
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict, OrderedDict
import threading
import time

from mark1.storage.models.context_model import ContextModel, ContextType, ContextScope, ContextPriority
from mark1.storage.repositories.context_repository import ContextRepository
from mark1.storage.database import get_db_session
from mark1.utils.exceptions import ContextError
from mark1.utils.constants import MAX_CONTEXT_SIZE, CONTEXT_RETENTION_DAYS


class ContextOperationType(Enum):
    """Types of context operations"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"
    MERGE = "merge"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    RESTORE = "restore"


class ContextCacheStrategy(Enum):
    """Context caching strategies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    PRIORITY = "priority"  # Priority-based
    ADAPTIVE = "adaptive"  # Adaptive strategy


class ContextCompressionLevel(Enum):
    """Context compression levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


@dataclass
class ContextEntry:
    """Enhanced context entry with advanced features"""
    id: str
    key: str
    content: Dict[str, Any]
    context_type: ContextType
    scope: ContextScope
    priority: ContextPriority = ContextPriority.MEDIUM
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Advanced features
    version: int = 1
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    access_count: int = 0
    size_bytes: int = 0
    is_compressed: bool = False
    is_cached: bool = True
    is_dirty: bool = False
    checksum: Optional[str] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def update_access(self):
        """Update access tracking"""
        self.accessed_at = datetime.now(timezone.utc)
        self.access_count += 1
        
    def calculate_size(self) -> int:
        """Calculate content size"""
        try:
            if self.is_compressed:
                # Estimate compressed size
                return len(gzip.compress(json.dumps(self.content).encode('utf-8')))
            else:
                return len(json.dumps(self.content).encode('utf-8'))
        except Exception:
            return 0
    
    def calculate_checksum(self) -> str:
        """Calculate content checksum"""
        try:
            content_str = json.dumps(self.content, sort_keys=True)
            return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        except Exception:
            return ""


@dataclass
class ContextOperationResult:
    """Enhanced context operation result"""
    success: bool
    context_id: Optional[str] = None
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    operation_type: Optional[ContextOperationType] = None
    operation_time: float = 0.0
    cache_hit: bool = False
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextCacheStats:
    """Context cache statistics"""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    compression_ratio: float = 0.0
    average_access_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class AdvancedContextCache:
    """Advanced caching system for contexts"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 strategy: ContextCacheStrategy = ContextCacheStrategy.ADAPTIVE,
                 compression_threshold: int = 1024):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.compression_threshold = compression_threshold
        
        # Cache storage
        self._cache: OrderedDict[str, ContextEntry] = OrderedDict()
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_times: Dict[str, float] = {}
        self._size_tracking: Dict[str, int] = {}
        
        # Statistics
        self.stats = ContextCacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = structlog.get_logger(__name__)
    
    def get(self, context_id: str) -> Optional[ContextEntry]:
        """Get context from cache"""
        with self._lock:
            if context_id in self._cache:
                entry = self._cache[context_id]
                entry.update_access()
                
                # Update cache strategy data
                self._access_counts[context_id] += 1
                self._access_times[context_id] = time.time()
                
                # Move to end for LRU
                if self.strategy == ContextCacheStrategy.LRU:
                    self._cache.move_to_end(context_id)
                
                self.stats.hit_count += 1
                return entry
            
            self.stats.miss_count += 1
            return None
    
    def put(self, context_id: str, entry: ContextEntry) -> bool:
        """Put context in cache with intelligent eviction"""
        with self._lock:
            # Check if we need compression
            entry.size_bytes = entry.calculate_size()
            if entry.size_bytes > self.compression_threshold and not entry.is_compressed:
                self._compress_entry(entry)
            
            # Check cache limits
            if len(self._cache) >= self.max_size or self._get_total_size() + entry.size_bytes > self.max_memory_bytes:
                self._evict_entries()
            
            # Add to cache
            self._cache[context_id] = entry
            self._access_counts[context_id] = 1
            self._access_times[context_id] = time.time()
            self._size_tracking[context_id] = entry.size_bytes
            
            entry.is_cached = True
            self.stats.total_entries = len(self._cache)
            self.stats.total_size_bytes = self._get_total_size()
            
            return True
    
    def remove(self, context_id: str) -> bool:
        """Remove context from cache"""
        with self._lock:
            if context_id in self._cache:
                del self._cache[context_id]
                self._access_counts.pop(context_id, None)
                self._access_times.pop(context_id, None)
                self._size_tracking.pop(context_id, None)
                
                self.stats.total_entries = len(self._cache)
                self.stats.total_size_bytes = self._get_total_size()
                return True
            return False
    
    def _compress_entry(self, entry: ContextEntry):
        """Compress context entry content"""
        try:
            original_content = json.dumps(entry.content)
            compressed_content = gzip.compress(original_content.encode('utf-8'))
            
            # Store compressed content as base64
            import base64
            entry.content = {"_compressed": base64.b64encode(compressed_content).decode('utf-8')}
            entry.is_compressed = True
            entry.size_bytes = len(compressed_content)
            
            # Update compression stats
            original_size = len(original_content.encode('utf-8'))
            compression_ratio = len(compressed_content) / original_size
            self.stats.compression_ratio = (self.stats.compression_ratio + compression_ratio) / 2
            
        except Exception as e:
            self.logger.warning("Failed to compress context entry", error=str(e))
    
    def _decompress_entry(self, entry: ContextEntry) -> Dict[str, Any]:
        """Decompress context entry content"""
        try:
            if entry.is_compressed and "_compressed" in entry.content:
                import base64
                compressed_data = base64.b64decode(entry.content["_compressed"])
                decompressed_content = gzip.decompress(compressed_data).decode('utf-8')
                return json.loads(decompressed_content)
            return entry.content
        except Exception as e:
            self.logger.warning("Failed to decompress context entry", error=str(e))
            return entry.content
    
    def _evict_entries(self):
        """Evict entries based on strategy"""
        if not self._cache:
            return
        
        if self.strategy == ContextCacheStrategy.LRU:
            # Remove least recently used
            context_id, _ = self._cache.popitem(last=False)
        elif self.strategy == ContextCacheStrategy.LFU:
            # Remove least frequently used
            context_id = min(self._access_counts.items(), key=lambda x: x[1])[0]
            del self._cache[context_id]
        elif self.strategy == ContextCacheStrategy.TTL:
            # Remove expired entries
            current_time = time.time()
            expired_entries = [
                cid for cid, entry in self._cache.items()
                if entry.expires_at and datetime.now(timezone.utc) > entry.expires_at
            ]
            for cid in expired_entries:
                del self._cache[cid]
            
            # If no expired entries, fall back to LRU
            if not expired_entries and self._cache:
                context_id, _ = self._cache.popitem(last=False)
        elif self.strategy == ContextCacheStrategy.PRIORITY:
            # Remove lowest priority
            lowest_priority_entry = min(
                self._cache.items(),
                key=lambda x: (x[1].priority.value, x[1].accessed_at)
            )
            context_id = lowest_priority_entry[0]
            del self._cache[context_id]
        else:  # ADAPTIVE
            # Use adaptive strategy based on access patterns
            if len(self._cache) > self.max_size * 0.8:
                # Use LFU for high cache usage
                context_id = min(self._access_counts.items(), key=lambda x: x[1])[0]
            else:
                # Use LRU for normal usage
                context_id, _ = self._cache.popitem(last=False)
            del self._cache[context_id]
        
        # Clean up tracking data
        if context_id in self._access_counts:
            del self._access_counts[context_id]
        if context_id in self._access_times:
            del self._access_times[context_id]
        if context_id in self._size_tracking:
            del self._size_tracking[context_id]
        
        self.stats.eviction_count += 1
    
    def _get_total_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(self._size_tracking.values())
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._access_times.clear()
            self._size_tracking.clear()
            self.stats = ContextCacheStats()


class AdvancedContextManager:
    """
    Advanced Context Manager for Mark-1 Orchestrator
    
    Session 19: Advanced Context Management
    Provides intelligent context sharing, memory optimization, and lifecycle management
    """
    
    def __init__(self, 
                 cache_size: int = 1000,
                 cache_memory_mb: int = 100,
                 cache_strategy: ContextCacheStrategy = ContextCacheStrategy.ADAPTIVE,
                 auto_compression: bool = True,
                 compression_threshold: int = 1024):
        self.logger = structlog.get_logger(__name__)
        
        # Cache configuration
        self.cache = AdvancedContextCache(
            max_size=cache_size,
            max_memory_mb=cache_memory_mb,
            strategy=cache_strategy,
            compression_threshold=compression_threshold
        )
        
        # Configuration
        self.auto_compression = auto_compression
        self.compression_threshold = compression_threshold
        
        # State tracking
        self._initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        
        # Context relationships
        self._context_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self._context_dependencies: Dict[str, Set[str]] = defaultdict(set)  # context -> dependencies
        
        # Performance tracking
        self._operation_stats = defaultdict(int)
        self._operation_times = defaultdict(list)
        
        # Context sharing
        self._shared_contexts: Dict[str, Set[str]] = defaultdict(set)  # context -> agents
        self._agent_contexts: Dict[str, Set[str]] = defaultdict(set)  # agent -> contexts
    
    async def initialize(self) -> None:
        """Initialize the advanced context manager"""
        try:
            self.logger.info("Initializing advanced context manager...")
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            self._sync_task = asyncio.create_task(self._background_sync())
            
            self._initialized = True
            self.logger.info("Advanced context manager initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize advanced context manager", error=str(e))
            raise ContextError(f"Advanced context manager initialization failed: {e}")

    async def create_context(
        self,
        key: str,
        content: Dict[str, Any],
        context_type: ContextType,
        scope: ContextScope,
        priority: ContextPriority = ContextPriority.MEDIUM,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_context_id: Optional[str] = None,
        expires_in_hours: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        auto_compress: bool = None
    ) -> ContextOperationResult:
        """
        Create a new context entry with advanced features
        """
        start_time = time.time()
        
        try:
            self.logger.info("Creating advanced context", key=key, type=context_type.value, scope=scope.value)
            
            # Use instance setting if not specified
            if auto_compress is None:
                auto_compress = self.auto_compression
            
            # Validate content size
            content_size = len(json.dumps(content).encode('utf-8'))
            if content_size > MAX_CONTEXT_SIZE:
                return ContextOperationResult(
                    success=False,
                    message=f"Content size exceeds maximum allowed size: {MAX_CONTEXT_SIZE}",
                    operation_type=ContextOperationType.CREATE,
                    operation_time=time.time() - start_time
                )
            
            # Calculate expiration
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
            elif context_type == ContextType.SESSION:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
            elif context_type == ContextType.CONVERSATION:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=12)
            
            # Create context record in database
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                context = await context_repo.create_context(
                    session=session,
                    context_key=key,
                    context_type=context_type.value,
                    context_scope=scope.value,
                    priority=priority.value,
                    agent_id=agent_id,
                    task_id=task_id,
                    session_id=session_id,
                    parent_context_id=parent_context_id,
                    content=content,
                    expires_at=expires_at,
                    tags=list(tags) if tags else []
                )
                
                await session.commit()
                context_id = str(context.id)
            
            # Create enhanced context entry
            context_entry = ContextEntry(
                id=context_id,
                key=key,
                content=content.copy(),
                context_type=context_type,
                scope=scope,
                priority=priority,
                agent_id=agent_id,
                task_id=task_id,
                session_id=session_id,
                parent_id=parent_context_id,
                expires_at=expires_at,
                tags=tags or set(),
                size_bytes=content_size
            )
            
            # Calculate checksum
            context_entry.checksum = context_entry.calculate_checksum()
            
            # Auto-compress if needed
            if auto_compress and content_size > self.compression_threshold:
                self.cache._compress_entry(context_entry)
            
            # Add to cache
            self.cache.put(context_id, context_entry)
            
            # Update hierarchy
            if parent_context_id:
                self._context_hierarchy[parent_context_id].add(context_id)
            
            # Update agent associations
            if agent_id:
                self._agent_contexts[agent_id].add(context_id)
            
            operation_time = time.time() - start_time
            self._operation_stats[ContextOperationType.CREATE] += 1
            self._operation_times[ContextOperationType.CREATE].append(operation_time)
            
            self.logger.info("Advanced context created successfully", 
                           context_id=context_id, 
                           key=key,
                           size_bytes=content_size,
                           compressed=context_entry.is_compressed)
            
            return ContextOperationResult(
                success=True,
                context_id=context_id,
                message="Advanced context created successfully",
                operation_type=ContextOperationType.CREATE,
                operation_time=operation_time,
                compressed=context_entry.is_compressed,
                metadata={
                    "size_bytes": content_size,
                    "checksum": context_entry.checksum,
                    "priority": priority.value,
                    "auto_compressed": context_entry.is_compressed
                }
            )
            
        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error("Failed to create advanced context", key=key, error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Advanced context creation failed: {e}",
                operation_type=ContextOperationType.CREATE,
                operation_time=operation_time
            )

    async def get_context(
        self,
        context_id: Optional[str] = None,
        key: Optional[str] = None,
        context_type: Optional[ContextType] = None,
        decompress: bool = True
    ) -> ContextOperationResult:
        """
        Retrieve context with advanced caching and decompression
        """
        start_time = time.time()
        cache_hit = False
        
        try:
            # Try cache first
            if context_id:
                cached_entry = self.cache.get(context_id)
                if cached_entry:
                    cache_hit = True
                    
                    # Check expiration
                    if cached_entry.expires_at and datetime.now(timezone.utc) > cached_entry.expires_at:
                        await self._remove_expired_context(context_id)
                        return ContextOperationResult(
                            success=False,
                            message="Context has expired",
                            operation_type=ContextOperationType.READ,
                            operation_time=time.time() - start_time,
                            cache_hit=cache_hit
                        )
                    
                    # Decompress if needed
                    content = cached_entry.content
                    if decompress and cached_entry.is_compressed:
                        content = self.cache._decompress_entry(cached_entry)
                    
                    operation_time = time.time() - start_time
                    self._operation_stats[ContextOperationType.READ] += 1
                    self._operation_times[ContextOperationType.READ].append(operation_time)
                    
                    return ContextOperationResult(
                        success=True,
                        context_id=context_id,
                        data=content,
                        message="Context retrieved from cache",
                        operation_type=ContextOperationType.READ,
                        operation_time=operation_time,
                        cache_hit=cache_hit,
                        compressed=cached_entry.is_compressed,
                        metadata={
                            "access_count": cached_entry.access_count,
                            "version": cached_entry.version,
                            "checksum": cached_entry.checksum
                        }
                    )
            
            # Query database
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                if context_id:
                    context = await context_repo.get_context_by_id(session, context_id)
                elif key:
                    context_type_str = context_type.value if context_type else None
                    context = await context_repo.get_context_by_key(session, key, context_type_str)
                else:
                    return ContextOperationResult(
                        success=False,
                        message="Either context_id or key must be provided",
                        operation_type=ContextOperationType.READ,
                        operation_time=time.time() - start_time,
                        cache_hit=cache_hit
                    )
                
                if not context:
                    return ContextOperationResult(
                        success=False,
                        message="Context not found",
                        operation_type=ContextOperationType.READ,
                        operation_time=time.time() - start_time,
                        cache_hit=cache_hit
                    )
                
                # Check expiration
                if context.expires_at and datetime.now(timezone.utc) > context.expires_at:
                    return ContextOperationResult(
                        success=False,
                        message="Context has expired",
                        operation_type=ContextOperationType.READ,
                        operation_time=time.time() - start_time,
                        cache_hit=cache_hit
                    )
                
                # Create context entry and add to cache
                context_entry = ContextEntry(
                    id=str(context.id),
                    key=context.context_key,
                    content=context.content or {},
                    context_type=ContextType(context.context_type),
                    scope=ContextScope(context.context_scope),
                    priority=ContextPriority(context.priority),
                    created_at=context.created_at,
                    updated_at=context.updated_at,
                    expires_at=context.expires_at,
                    version=context.version,
                    parent_id=str(context.parent_context_id) if context.parent_context_id else None,
                    agent_id=str(context.agent_id) if context.agent_id else None,
                    task_id=str(context.task_id) if context.task_id else None,
                    session_id=context.session_id,
                    checksum=context.checksum
                )
                
                # Add to cache
                self.cache.put(str(context.id), context_entry)
                
                operation_time = time.time() - start_time
                self._operation_stats[ContextOperationType.READ] += 1
                self._operation_times[ContextOperationType.READ].append(operation_time)
                
                return ContextOperationResult(
                    success=True,
                    context_id=str(context.id),
                    data=context.content or {},
                    message="Context retrieved from database",
                    operation_type=ContextOperationType.READ,
                    operation_time=operation_time,
                    cache_hit=cache_hit,
                    metadata={
                        "version": context.version,
                        "checksum": context.checksum,
                        "size_bytes": context.size_bytes or 0
                    }
                )
                
        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error("Failed to get advanced context", 
                            context_id=context_id, 
                            key=key, 
                            error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Advanced context retrieval failed: {e}",
                operation_type=ContextOperationType.READ,
                operation_time=operation_time,
                cache_hit=cache_hit
            )

    async def update_context(
        self,
        context_id: str,
        content: Dict[str, Any],
        merge: bool = True,
        create_version: bool = False
    ) -> ContextOperationResult:
        """
        Update context content with advanced versioning and merging
        """
        start_time = time.time()
        
        try:
            self.logger.info("Updating advanced context", context_id=context_id, merge=merge, create_version=create_version)
            
            # Validate content size
            content_size = len(json.dumps(content).encode('utf-8'))
            if content_size > MAX_CONTEXT_SIZE:
                return ContextOperationResult(
                    success=False,
                    message=f"Content size exceeds maximum allowed size: {MAX_CONTEXT_SIZE}",
                    operation_type=ContextOperationType.UPDATE,
                    operation_time=time.time() - start_time
                )
            
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                
                # Get existing context
                context = await context_repo.get_by_id(context_id, session)
                if not context:
                    return ContextOperationResult(
                        success=False,
                        message=f"Context not found: {context_id}",
                        operation_type=ContextOperationType.UPDATE
                    )
                
                # Create version backup if requested
                if create_version:
                    version_key = f"{context.context_key}_v{context.version + 1}"
                    await context_repo.create_context(
                        session=session,
                        context_key=version_key,
                        context_type=context.context_type,
                        context_scope=context.context_scope,
                        priority=context.priority,
                        parent_context_id=str(context.id) if context.id else None,
                        content=context.content,
                        extra_metadata={"version_of": str(context.id), "version_number": context.version}
                    )
                
                # Merge or replace content
                if merge:
                    merged_content = self._merge_content(context.content or {}, content)
                else:
                    merged_content = content
                
                # Update context - directly update the database record
                context.content = merged_content
                context.updated_at = datetime.now(timezone.utc)
                context.version += 1
                
                await session.commit()
                
                # Update cache
                cached_entry = self.cache.get(context_id)
                if cached_entry:
                    cached_entry.content = merged_content
                    cached_entry.updated_at = datetime.now(timezone.utc)
                    cached_entry.version = context.version
                    cached_entry.is_dirty = False
                    cached_entry.checksum = cached_entry.calculate_checksum()
                    
                    # Recompress if needed
                    if cached_entry.is_compressed:
                        self.cache._compress_entry(cached_entry)
                
                operation_time = time.time() - start_time
                self._operation_stats[ContextOperationType.UPDATE] += 1
                self._operation_times[ContextOperationType.UPDATE].append(operation_time)
                
                self.logger.info("Advanced context updated successfully", 
                               context_id=context_id,
                               version=context.version,
                               size_bytes=content_size)
                
                return ContextOperationResult(
                    success=True,
                    context_id=context_id,
                    message="Advanced context updated successfully",
                    operation_type=ContextOperationType.UPDATE,
                    operation_time=operation_time,
                    metadata={
                        "version": context.version,
                        "size_bytes": content_size,
                        "merge_applied": merge,
                        "version_created": create_version
                    }
                )
                
        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error("Failed to update advanced context", context_id=context_id, error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Advanced context update failed: {e}",
                operation_type=ContextOperationType.UPDATE,
                operation_time=operation_time
            )

    async def share_context(
        self,
        context_id: str,
        target_agent_ids: List[str],
        permissions: List[str] = None,
        copy_on_share: bool = True,
        share_children: bool = False
    ) -> ContextOperationResult:
        """
        Advanced context sharing with permissions and hierarchy support
        """
        start_time = time.time()
        
        try:
            self.logger.info("Sharing advanced context", 
                           context_id=context_id, 
                           target_agents=target_agent_ids,
                           copy_on_share=copy_on_share,
                           share_children=share_children)
            
            # Get the context
            result = await self.get_context(context_id=context_id)
            if not result.success:
                return ContextOperationResult(
                    success=False,
                    message=result.message,
                    operation_type=ContextOperationType.SHARE,
                    operation_time=time.time() - start_time
                )
            
            shared_context_ids = []
            
            for target_agent_id in target_agent_ids:
                if copy_on_share:
                    # Create a copy for each agent
                    cached_entry = self.cache.get(context_id)
                    original_key = cached_entry.key if cached_entry else f"context_{context_id}"
                    
                    shared_result = await self.create_context(
                        key=f"shared_{original_key}_{target_agent_id}",
                        content=result.data,
                        context_type=ContextType.MEMORY,
                        scope=ContextScope.AGENT,
                        priority=ContextPriority.MEDIUM,
                        tags={f"shared_from_{context_id}", f"agent_{target_agent_id}"}
                    )
                    
                    if shared_result.success:
                        shared_context_ids.append(shared_result.context_id)
                        
                        # Update sharing tracking
                        self._shared_contexts[context_id].add(target_agent_id)
                        self._agent_contexts[target_agent_id].add(shared_result.context_id)
                else:
                    # Direct access sharing (reference sharing)
                    self._shared_contexts[context_id].add(target_agent_id)
                    self._agent_contexts[target_agent_id].add(context_id)
                    shared_context_ids.append(context_id)
            
            # Share children if requested
            if share_children and context_id in self._context_hierarchy:
                for child_id in self._context_hierarchy[context_id]:
                    child_share_result = await self.share_context(
                        child_id, 
                        target_agent_ids, 
                        permissions, 
                        copy_on_share, 
                        False  # Prevent infinite recursion
                    )
            
            operation_time = time.time() - start_time
            self._operation_stats[ContextOperationType.SHARE] += 1
            self._operation_times[ContextOperationType.SHARE].append(operation_time)
            
            self.logger.info("Advanced context sharing completed", 
                           original_context=context_id,
                           shared_contexts=len(shared_context_ids),
                           target_agents=len(target_agent_ids))
            
            return ContextOperationResult(
                success=True,
                context_id=context_id,
                message=f"Context shared with {len(target_agent_ids)} agents",
                operation_type=ContextOperationType.SHARE,
                operation_time=operation_time,
                metadata={
                    "shared_context_ids": shared_context_ids,
                    "target_agents": target_agent_ids,
                    "copy_on_share": copy_on_share,
                    "children_shared": share_children
                }
            )
            
        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error("Failed to share advanced context", 
                            context_id=context_id, 
                            target_agents=target_agent_ids, 
                            error=str(e))
            return ContextOperationResult(
                success=False,
                message=f"Advanced context sharing failed: {e}",
                operation_type=ContextOperationType.SHARE,
                operation_time=operation_time
            )

    async def merge_contexts(
        self,
        source_context_ids: List[str],
        target_key: str,
        merge_strategy: str = "deep",
        conflict_resolution: str = "latest",
        create_backup: bool = True
    ) -> ContextOperationResult:
        """
        Merge multiple contexts into a new context with intelligent conflict resolution
        """
        start_time = time.time()
        
        try:
            self.logger.info("Merging contexts", 
                           source_contexts=source_context_ids,
                           target_key=target_key,
                           strategy=merge_strategy)
            
            # Get all source contexts
            source_contexts = []
            for context_id in source_context_ids:
                result = await self.get_context(context_id=context_id)
                if result.success:
                    source_contexts.append({
                        "id": context_id,
                        "data": result.data,
                        "metadata": result.metadata
                    })
            
            if not source_contexts:
                return ContextOperationResult(
                    success=False,
                    message="No valid source contexts found",
                    operation_type=ContextOperationType.MERGE,
                    operation_time=time.time() - start_time
                )
            
            # Create backups if requested
            backup_ids = []
            if create_backup:
                for i, context in enumerate(source_contexts):
                    backup_result = await self.create_context(
                        key=f"backup_{target_key}_{i}_{int(time.time())}",
                        content=context["data"],
                        context_type=ContextType.MEMORY,
                        scope=ContextScope.GLOBAL,
                        priority=ContextPriority.LOW,
                        expires_in_hours=168,  # 7 days
                        tags={"backup", "merge_operation"}
                    )
                    if backup_result.success:
                        backup_ids.append(backup_result.context_id)
            
            # Perform intelligent merge
            merged_content = await self._intelligent_merge(
                source_contexts, 
                merge_strategy, 
                conflict_resolution
            )
            
            # Create merged context
            merge_result = await self.create_context(
                key=target_key,
                content=merged_content,
                context_type=ContextType.MEMORY,
                scope=ContextScope.AGENT,
                priority=ContextPriority.HIGH,
                tags={"merged_context", f"sources_{len(source_contexts)}"}
            )
            
            if merge_result.success:
                operation_time = time.time() - start_time
                self._operation_stats[ContextOperationType.MERGE] += 1
                self._operation_times[ContextOperationType.MERGE].append(operation_time)
                
                self.logger.info("Context merge completed successfully",
                               merged_context=merge_result.context_id,
                               source_count=len(source_contexts),
                               backup_count=len(backup_ids))
                
                return ContextOperationResult(
                    success=True,
                    context_id=merge_result.context_id,
                    message=f"Successfully merged {len(source_contexts)} contexts",
                    operation_type=ContextOperationType.MERGE,
                    operation_time=operation_time,
                    metadata={
                        "source_context_ids": source_context_ids,
                        "backup_ids": backup_ids,
                        "merge_strategy": merge_strategy,
                        "conflict_resolution": conflict_resolution
                    }
                )
            else:
                return merge_result
                
        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error("Failed to merge contexts", 
                            source_contexts=source_context_ids,
                            error=str(e),
                            error_type=type(e).__name__,
                            traceback=str(e.__traceback__))
            return ContextOperationResult(
                success=False,
                message=f"Context merge failed: {type(e).__name__}: {str(e)}",
                operation_type=ContextOperationType.MERGE,
                operation_time=operation_time
            )

    async def get_context_hierarchy(self, root_context_id: str) -> Dict[str, Any]:
        """Get complete context hierarchy starting from root"""
        try:
            hierarchy = {
                "root": root_context_id,
                "children": {},
                "metadata": {}
            }
            
            # Get root context info
            root_result = await self.get_context(context_id=root_context_id)
            if root_result.success:
                hierarchy["metadata"]["root"] = {
                    "key": root_result.metadata.get("key"),
                    "size": root_result.metadata.get("size_bytes", 0),
                    "version": root_result.metadata.get("version", 1)
                }
            
            # Recursively build hierarchy
            await self._build_hierarchy_tree(root_context_id, hierarchy["children"])
            
            return hierarchy
            
        except Exception as e:
            self.logger.error("Failed to get context hierarchy", root_context=root_context_id, error=str(e))
            return {}

    async def get_agent_contexts(self, agent_id: str, include_shared: bool = True) -> List[str]:
        """Get all contexts associated with an agent"""
        try:
            agent_contexts = list(self._agent_contexts.get(agent_id, set()))
            
            if include_shared:
                # Add contexts shared with this agent
                for context_id, shared_agents in self._shared_contexts.items():
                    if agent_id in shared_agents:
                        agent_contexts.append(context_id)
            
            return list(set(agent_contexts))  # Remove duplicates
            
        except Exception as e:
            self.logger.error("Failed to get agent contexts", agent_id=agent_id, error=str(e))
            return []

    async def optimize_context_storage(self) -> Dict[str, Any]:
        """Optimize context storage through compression and cleanup"""
        try:
            start_time = time.time()
            
            optimization_stats = {
                "contexts_processed": 0,
                "contexts_compressed": 0,
                "contexts_archived": 0,
                "space_saved_bytes": 0,
                "operation_time": 0.0
            }
            
            # Compress large uncompressed contexts
            for context_id, context_entry in self.cache._cache.items():
                optimization_stats["contexts_processed"] += 1
                
                if not context_entry.is_compressed and context_entry.size_bytes > self.compression_threshold:
                    original_size = context_entry.size_bytes
                    self.cache._compress_entry(context_entry)
                    optimization_stats["contexts_compressed"] += 1
                    optimization_stats["space_saved_bytes"] += original_size - context_entry.size_bytes
                
                # Archive old, rarely accessed contexts
                days_since_access = (datetime.now(timezone.utc) - context_entry.accessed_at).days
                if days_since_access > 7 and context_entry.access_count < 3:
                    await self._archive_context(context_id)
                    optimization_stats["contexts_archived"] += 1
            
            optimization_stats["operation_time"] = time.time() - start_time
            
            self.logger.info("Context storage optimization completed", **optimization_stats)
            return optimization_stats
            
        except Exception as e:
            self.logger.error("Context storage optimization failed", error=str(e))
            return {}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            metrics = {
                "cache_stats": {
                    "total_entries": self.cache.stats.total_entries,
                    "total_size_bytes": self.cache.stats.total_size_bytes,
                    "hit_rate": self.cache.stats.hit_rate,
                    "miss_rate": self.cache.stats.miss_rate,
                    "eviction_count": self.cache.stats.eviction_count,
                    "compression_ratio": self.cache.stats.compression_ratio
                },
                "operation_stats": dict(self._operation_stats),
                "operation_times": {},
                "context_distribution": {
                    "by_type": defaultdict(int),
                    "by_scope": defaultdict(int),
                    "by_priority": defaultdict(int)
                },
                "sharing_stats": {
                    "total_shared_contexts": len(self._shared_contexts),
                    "total_agent_associations": sum(len(agents) for agents in self._shared_contexts.values()),
                    "average_shares_per_context": 0
                },
                "hierarchy_stats": {
                    "total_hierarchies": len(self._context_hierarchy),
                    "average_children_per_parent": 0,
                    "max_hierarchy_depth": 0
                }
            }
            
            # Calculate average operation times
            for op_type, times in self._operation_times.items():
                if times:
                    metrics["operation_times"][op_type.value] = {
                        "average": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "count": len(times)
                    }
            
            # Analyze context distribution
            for context_entry in self.cache._cache.values():
                metrics["context_distribution"]["by_type"][context_entry.context_type.value] += 1
                metrics["context_distribution"]["by_scope"][context_entry.scope.value] += 1
                metrics["context_distribution"]["by_priority"][context_entry.priority.value] += 1
            
            # Calculate sharing statistics
            if self._shared_contexts:
                total_shares = sum(len(agents) for agents in self._shared_contexts.values())
                metrics["sharing_stats"]["average_shares_per_context"] = total_shares / len(self._shared_contexts)
            
            # Calculate hierarchy statistics
            if self._context_hierarchy:
                children_counts = [len(children) for children in self._context_hierarchy.values()]
                metrics["hierarchy_stats"]["average_children_per_parent"] = sum(children_counts) / len(children_counts)
                metrics["hierarchy_stats"]["max_hierarchy_depth"] = await self._calculate_max_hierarchy_depth()
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to get performance metrics", error=str(e))
            return {}

    # Helper methods
    async def _smart_merge_content(self, old_content: Dict[str, Any], new_content: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently merge old and new content"""
        try:
            merged = old_content.copy()
            
            for key, value in new_content.items():
                if key in merged:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        # Recursively merge dictionaries
                        merged[key] = await self._smart_merge_content(merged[key], value)
                    elif isinstance(merged[key], list) and isinstance(value, list):
                        # Merge lists (handle unhashable types safely)
                        try:
                            # Try to remove duplicates if items are hashable
                            merged[key] = list(set(merged[key] + value))
                        except TypeError:
                            # If items are unhashable, just concatenate
                            merged[key] = merged[key] + value
                    else:
                        # Override with new value
                        merged[key] = value
                else:
                    # Add new key
                    merged[key] = value
            
            return merged
            
        except Exception as e:
            self.logger.warning("Smart merge failed, using simple merge", error=str(e))
            return {**old_content, **new_content}

    async def _intelligent_merge(self, source_contexts: List[Dict], strategy: str, conflict_resolution: str) -> Dict[str, Any]:
        """Perform intelligent merge of multiple contexts"""
        try:
            if strategy == "shallow":
                # Simple key-value merge
                merged = {}
                for context in source_contexts:
                    if "data" in context and context["data"]:
                        merged.update(context["data"])
                return merged
            
            elif strategy == "deep":
                # Deep merge with conflict resolution
                merged = {}
                # Sort by version, but handle missing/invalid metadata gracefully
                sorted_contexts = []
                for context in source_contexts:
                    if "data" in context:
                        version = 0
                        try:
                            metadata = context.get("metadata", {})
                            if metadata and isinstance(metadata, dict):
                                version = metadata.get("version", 0)
                        except (AttributeError, TypeError):
                            version = 0
                        sorted_contexts.append((version, context))
                
                # Sort by version and merge
                sorted_contexts.sort(key=lambda x: x[0])
                for version, context in sorted_contexts:
                    merged = await self._smart_merge_content(merged, context["data"])
                return merged
            
            elif strategy == "weighted":
                # Weight-based merge (by context priority/access count)
                weighted_contexts = []
                for context in source_contexts:
                    if "data" in context:
                        weight = 1  # Default weight
                        try:
                            metadata = context.get("metadata", {})
                            if metadata and isinstance(metadata, dict):
                                weight = metadata.get("access_count", 1)
                        except (AttributeError, TypeError):
                            weight = 1
                        weighted_contexts.append((weight, context["data"]))
                
                # Sort by weight and merge
                weighted_contexts.sort(key=lambda x: x[0], reverse=True)
                merged = {}
                for weight, data in weighted_contexts:
                    merged = await self._smart_merge_content(merged, data)
                return merged
            
            else:
                # Default to simple merge
                merged = {}
                for context in source_contexts:
                    if "data" in context and context["data"]:
                        merged.update(context["data"])
                return merged
                
        except Exception as e:
            self.logger.error("Intelligent merge failed", 
                             error=str(e),
                             error_type=type(e).__name__,
                             strategy=strategy,
                             source_count=len(source_contexts))
            # Fallback to simple merge
            merged = {}
            for context in source_contexts:
                if "data" in context and context["data"]:
                    merged.update(context["data"])
            return merged

    async def _build_hierarchy_tree(self, parent_id: str, tree: Dict[str, Any]):
        """Recursively build context hierarchy tree"""
        try:
            if parent_id in self._context_hierarchy:
                for child_id in self._context_hierarchy[parent_id]:
                    tree[child_id] = {}
                    await self._build_hierarchy_tree(child_id, tree[child_id])
        except Exception as e:
            self.logger.warning("Failed to build hierarchy tree", parent_id=parent_id, error=str(e))

    async def _calculate_max_hierarchy_depth(self) -> int:
        """Calculate maximum hierarchy depth"""
        try:
            max_depth = 0
            
            for root_id in self._context_hierarchy.keys():
                depth = await self._get_hierarchy_depth(root_id, 0)
                max_depth = max(max_depth, depth)
            
            return max_depth
        except Exception:
            return 0

    async def _get_hierarchy_depth(self, context_id: str, current_depth: int) -> int:
        """Get hierarchy depth for a specific context"""
        try:
            if context_id not in self._context_hierarchy:
                return current_depth
            
            max_child_depth = current_depth
            for child_id in self._context_hierarchy[context_id]:
                child_depth = await self._get_hierarchy_depth(child_id, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        except Exception:
            return current_depth

    async def _archive_context(self, context_id: str):
        """Archive a context (move to cold storage)"""
        try:
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                context = await context_repo.get_by_id(context_id, session)
                
                if context:
                    context.is_archived = True
                    context.is_active = False
                    await session.commit()
                    
                    # Remove from cache
                    self.cache.remove(context_id)
                    
                    self.logger.debug("Context archived", context_id=context_id)
                    
        except Exception as e:
            self.logger.warning("Failed to archive context", context_id=context_id, error=str(e))

    async def _background_cleanup(self):
        """Enhanced background cleanup with optimization"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                current_time = datetime.now(timezone.utc)
                
                # Clean expired contexts
                expired_count = 0
                for context_id, context_entry in list(self.cache._cache.items()):
                    if context_entry.expires_at and current_time > context_entry.expires_at:
                        await self._remove_expired_context(context_id)
                        expired_count += 1
                
                # Optimize storage periodically
                if expired_count > 10:
                    await self.optimize_context_storage()
                
                if expired_count > 0:
                    self.logger.info("Background cleanup completed", expired_contexts=expired_count)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Background cleanup failed", error=str(e))
                await asyncio.sleep(1800)

    async def _background_sync(self):
        """Background synchronization of dirty contexts"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                dirty_contexts = []
                for context_id, context_entry in self.cache._cache.items():
                    if context_entry.is_dirty:
                        dirty_contexts.append((context_id, context_entry))
                
                # Sync dirty contexts to database
                for context_id, context_entry in dirty_contexts:
                    try:
                        async with get_db_session() as session:
                            context_repo = ContextRepository(session)
                            await context_repo.update_context_content(
                                session=session,
                                context_id=context_id,
                                content=context_entry.content
                            )
                            await session.commit()
                            context_entry.is_dirty = False
                    except Exception as e:
                        self.logger.warning("Failed to sync dirty context", context_id=context_id, error=str(e))
                
                if dirty_contexts:
                    self.logger.debug("Background sync completed", synced_contexts=len(dirty_contexts))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Background sync failed", error=str(e))
                await asyncio.sleep(300)

    async def _remove_expired_context(self, context_id: str) -> None:
        """Remove an expired context with cleanup"""
        try:
            # Remove from cache
            self.cache.remove(context_id)
            
            # Clean up relationships
            if context_id in self._context_hierarchy:
                del self._context_hierarchy[context_id]
            
            # Remove from parent's children
            for parent_id, children in self._context_hierarchy.items():
                children.discard(context_id)
            
            # Clean up sharing relationships
            if context_id in self._shared_contexts:
                for agent_id in self._shared_contexts[context_id]:
                    self._agent_contexts[agent_id].discard(context_id)
                del self._shared_contexts[context_id]
            
            # Archive in database
            await self._archive_context(context_id)
                    
        except Exception as e:
            self.logger.error("Failed to remove expired context", context_id=context_id, error=str(e))

    async def shutdown(self) -> None:
        """Enhanced shutdown with cleanup"""
        try:
            self.logger.info("Shutting down advanced context manager...")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            
            # Sync any remaining dirty contexts
            dirty_contexts = [
                (context_id, context_entry)
                for context_id, context_entry in self.cache._cache.items()
                if context_entry.is_dirty
            ]
            
            for context_id, context_entry in dirty_contexts:
                try:
                    async with get_db_session() as session:
                        context_repo = ContextRepository(session)
                        await context_repo.update_context_content(
                            session=session,
                            context_id=context_id,
                            content=context_entry.content
                        )
                        await session.commit()
                except Exception as e:
                    self.logger.warning("Failed to sync context during shutdown", context_id=context_id, error=str(e))
            
            # Clear all data structures
            self.cache.clear()
            self._context_hierarchy.clear()
            self._context_dependencies.clear()
            self._shared_contexts.clear()
            self._agent_contexts.clear()
            self._operation_stats.clear()
            self._operation_times.clear()
            
            self._initialized = False
            
            self.logger.info("Advanced context manager shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during advanced context manager shutdown", error=str(e))

    @property
    def is_initialized(self) -> bool:
        """Check if the context manager is initialized"""
        return self._initialized

    @property
    def cache_size(self) -> int:
        """Get current cache size"""
        return len(self.cache._cache)

    @property
    def total_contexts(self) -> int:
        """Get total number of contexts managed"""
        return len(self.cache._cache)

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        return self.cache.stats.hit_rate

    @property
    def compression_ratio(self) -> float:
        """Get average compression ratio"""
        return self.cache.stats.compression_ratio

    def _merge_content(self, old_content: Dict[str, Any], new_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smart merge of context content
        
        Args:
            old_content: Existing content dictionary
            new_content: New content to merge
            
        Returns:
            Merged content dictionary
        """
        if not old_content:
            return new_content
        if not new_content:
            return old_content
            
        # Create a copy of old content to avoid modifying original
        merged = old_content.copy()
        
        # Merge new content recursively
        for key, value in new_content.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_content(merged[key], value)
            else:
                # Overwrite with new value
                merged[key] = value
        
        return merged


# Backward compatibility alias
ContextManager = AdvancedContextManager
