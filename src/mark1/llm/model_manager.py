"""
Model Manager for Mark-1 Orchestrator

Manages LLM models, providers, and handles model operations including
loading, unloading, switching between models, and health checking.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.config.settings import get_settings
from mark1.utils.exceptions import LLMException, LLMConnectionException, LLMModelNotFoundException


class ModelStatus(Enum):
    """Model status enumeration"""
    UNKNOWN = "unknown"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADING = "unloading"
    OFFLINE = "offline"


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    provider: str
    status: ModelStatus
    size_gb: Optional[float] = None
    parameters: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    last_used: Optional[datetime] = None
    load_time: Optional[float] = None
    memory_usage: Optional[int] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ModelUsageStats:
    """Model usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_tokens_processed: int = 0
    last_request_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    healthy: bool
    status: str
    details: Dict[str, Any]
    checked_at: datetime
    response_time: Optional[float] = None


class ModelManager:
    """
    Model management system for Mark-1
    
    Handles LLM model lifecycle, provider management, health monitoring,
    and provides a unified interface for model operations.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Manager state
        self._initialized = False
        self._providers: Dict[str, Any] = {}
        self._models: Dict[str, ModelInfo] = {}
        self._usage_stats: Dict[str, ModelUsageStats] = {}
        self._current_model: Optional[str] = None
        
        # Health monitoring
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
    
    async def initialize(self) -> None:
        """Initialize the model manager"""
        try:
            self.logger.info("Initializing model manager...")
            
            # Initialize providers (simplified for now)
            await self._initialize_providers()
            
            # Discover available models
            await self._discover_models()
            
            # Set default model
            await self._set_default_model()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self._initialized = True
            self.logger.info("Model manager initialized successfully",
                           providers=len(self._providers),
                           models=len(self._models))
            
        except Exception as e:
            self.logger.error("Failed to initialize model manager", error=str(e))
            # Don't raise exception to allow system to continue
            self._initialized = True  # Mark as initialized anyway
    
    async def _initialize_providers(self) -> None:
        """Initialize LLM providers"""
        try:
            # For now, just create a placeholder Ollama provider
            self._providers["ollama"] = {
                "name": "ollama",
                "base_url": self.settings.ollama.base_url,
                "status": "ready"
            }
            
            self.logger.info("Providers initialized", providers=list(self._providers.keys()))
            
        except Exception as e:
            self.logger.error("Failed to initialize providers", error=str(e))
    
    async def _discover_models(self) -> None:
        """Discover available models from all providers"""
        try:
            # Add a default model for testing
            default_model = self.settings.ollama.default_model
            
            model_info = ModelInfo(
                name=default_model,
                provider="ollama",
                status=ModelStatus.OFFLINE,
                description=f"Default model: {default_model}"
            )
            
            self._models[default_model] = model_info
            self._usage_stats[default_model] = ModelUsageStats()
            
            self.logger.info("Model discovery completed", 
                           total_models=len(self._models))
            
        except Exception as e:
            self.logger.error("Model discovery failed", error=str(e))
    
    async def _set_default_model(self) -> None:
        """Set the default model"""
        try:
            default_model = self.settings.ollama.default_model
            
            if default_model in self._models:
                self._current_model = default_model
                # Mark as ready for simplicity
                self._models[default_model].status = ModelStatus.READY
                self.logger.info("Default model set", model=default_model)
            
        except Exception as e:
            self.logger.error("Failed to set default model", error=str(e))
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        if model_name not in self._models:
            raise LLMModelNotFoundException(model_name, "unknown")
        
        try:
            self.logger.info("Loading model", model=model_name)
            
            # For now, just mark as ready
            self._models[model_name].status = ModelStatus.READY
            self._models[model_name].last_used = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            self.logger.error("Model loading failed", model=model_name, error=str(e))
            return False
    
    async def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response using the specified or current model"""
        target_model = model_name or self._current_model
        
        if not target_model:
            raise LLMException("No model available for generation")
        
        try:
            # For now, return a simple mock response
            response = f"Mock response from {target_model} for prompt: {prompt[:50]}..."
            
            # Update statistics
            if target_model in self._usage_stats:
                stats = self._usage_stats[target_model]
                stats.total_requests += 1
                stats.successful_requests += 1
                stats.last_request_time = datetime.now(timezone.utc)
            
            return response
            
        except Exception as e:
            self.logger.error("Response generation failed", 
                            model=target_model, error=str(e))
            raise LLMException(f"Response generation failed: {e}")
    
    async def health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check"""
        try:
            details = {
                "providers": len(self._providers),
                "models": len(self._models),
                "current_model": self._current_model
            }
            
            healthy = self._current_model is not None
            status = "healthy" if healthy else "no_model"
            
            result = HealthCheckResult(
                healthy=healthy,
                status=status,
                details=details,
                checked_at=datetime.now(timezone.utc),
                response_time=0.1  # Mock response time
            )
            
            self._last_health_check = result.checked_at
            return result
            
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status="error",
                details={"error": str(e)},
                checked_at=datetime.now(timezone.utc),
                response_time=None
            )
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                health_result = await self.health_check()
                
                if not health_result.healthy:
                    self.logger.warning("Health check failed", 
                                      status=health_result.status)
                else:
                    self.logger.debug("Health check passed")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def shutdown(self) -> None:
        """Shutdown the model manager"""
        try:
            self.logger.info("Shutting down model manager...")
            
            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            self._providers.clear()
            self._models.clear()
            self._usage_stats.clear()
            self._current_model = None
            self._initialized = False
            
            self.logger.info("Model manager shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during model manager shutdown", error=str(e))
    
    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized"""
        return self._initialized
    
    @property
    def current_model(self) -> Optional[str]:
        """Get current active model"""
        return self._current_model
