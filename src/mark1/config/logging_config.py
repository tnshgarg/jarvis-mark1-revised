"""
Mark-1 Orchestrator Logging Configuration

This module provides comprehensive logging setup using structlog for structured
logging with support for JSON output, colored console output, and various
processors for enhanced log information.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler

from mark1.config.settings import Settings, get_settings


class ContextualFilter(logging.Filter):
    """
    A logging filter that adds contextual information to log records
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add contextual information to the log record"""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class AsyncLogProcessor:
    """
    Async-aware log processor for handling async context information
    """
    
    def __call__(self, logger, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log events with async context"""
        # Add async task information if available
        try:
            import asyncio
            current_task = asyncio.current_task()
            if current_task:
                event_dict["task_id"] = id(current_task)
                event_dict["task_name"] = current_task.get_name()
        except RuntimeError:
            # No async context available
            pass
        
        return event_dict


class AgentContextProcessor:
    """
    Processor for adding agent-specific context to logs
    """
    
    def __call__(self, logger, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add agent context information"""
        # This will be populated by the agent system when available
        agent_context = getattr(logger, "_agent_context", None)
        if agent_context:
            event_dict.update({
                "agent_id": agent_context.get("id"),
                "agent_type": agent_context.get("type"),
                "agent_name": agent_context.get("name"),
                "workflow_id": agent_context.get("workflow_id")
            })
        
        return event_dict


class SensitiveDataProcessor:
    """
    Processor for sanitizing sensitive data from logs
    """
    
    SENSITIVE_KEYS = {
        "password", "token", "secret", "key", "auth", "credential",
        "api_key", "access_token", "refresh_token", "jwt", "bearer"
    }
    
    def __call__(self, logger, name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive data"""
        return self._sanitize_dict(event_dict)
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        sanitized = {}
        
        for key, value in data.items():
            if self._is_sensitive_key(key):
                sanitized[key] = self._mask_value(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = self._sanitize_sequence(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key is considered sensitive"""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.SENSITIVE_KEYS)
    
    def _mask_value(self, value: Any) -> str:
        """Mask sensitive values"""
        if not value:
            return str(value)
        
        str_value = str(value)
        if len(str_value) <= 4:
            return "*" * len(str_value)
        
        return str_value[:2] + "*" * (len(str_value) - 4) + str_value[-2:]
    
    def _sanitize_sequence(self, seq: Union[List, tuple]) -> Union[List, tuple]:
        """Sanitize sequence data"""
        sanitized = []
        for item in seq:
            if isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item))
            else:
                sanitized.append(item)
        
        return type(seq)(sanitized)


def add_log_level(level_name: str, level_value: int) -> None:
    """
    Add a custom log level to the logging system
    
    Args:
        level_name: Name of the log level
        level_value: Numeric value of the log level
    """
    # Add to logging module
    logging.addLevelName(level_value, level_name)
    
    # Add method to Logger class
    def log_method(self, message, *args, **kwargs):
        if self.isEnabledFor(level_value):
            self._log(level_value, message, args, **kwargs)
    
    setattr(logging.Logger, level_name.lower(), log_method)
    
    # Add to structlog
    setattr(structlog.get_logger().__class__, level_name.lower(), log_method)


def setup_logging(settings: Optional[Settings] = None) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        settings: Application settings instance
    """
    if settings is None:
        settings = get_settings()
    
    # Ensure log directory exists
    settings.create_directories()
    
    # Add custom log levels
    add_log_level("TRACE", 5)
    add_log_level("SUCCESS", 25)
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.TimeStamper(fmt="ISO", utc=True),
        AsyncLogProcessor(),
        AgentContextProcessor(),
        SensitiveDataProcessor(),
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Add appropriate final processor based on environment
    if settings.is_development():
        # Development: colorized console output
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    else:
        # Production: JSON output
        processors.append(
            structlog.processors.JSONRenderer()
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.value)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging configuration
    logging_config = _get_logging_config(settings)
    logging.config.dictConfig(logging_config)
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, settings.log_level.value))
    
    # Configure third-party loggers
    _configure_third_party_loggers(settings)


def _get_logging_config(settings: Settings) -> Dict[str, Any]:
    """
    Generate logging configuration dictionary
    
    Args:
        settings: Application settings
        
    Returns:
        Dictionary configuration for logging
    """
    log_file_path = settings.log_dir / "mark1.log"
    error_log_path = settings.log_dir / "mark1-error.log"
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)s | "
                    "%(filename)s:%(lineno)d | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s | %(name)s | %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": (
                    "%(asctime)s %(name)s %(levelname)s %(filename)s "
                    "%(lineno)d %(message)s"
                )
            }
        },
        "filters": {
            "context_filter": {
                "()": ContextualFilter,
                "context": {
                    "service": "mark1-orchestrator",
                    "version": settings.version,
                    "environment": settings.environment.value
                }
            }
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler" if settings.is_development() else "logging.StreamHandler",
                "level": settings.log_level.value,
                "formatter": "simple" if settings.is_development() else "json",
                "filters": ["context_filter"],
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filters": ["context_filter"],
                "filename": str(log_file_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filters": ["context_filter"],
                "filename": str(error_log_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "mark1": {
                "level": "DEBUG",
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": settings.log_level.value,
            "handlers": ["console", "file"]
        }
    }
    
    return config


def _configure_third_party_loggers(settings: Settings) -> None:
    """
    Configure third-party library loggers
    
    Args:
        settings: Application settings
    """
    # Suppress noisy third-party loggers in production
    if settings.is_production():
        noisy_loggers = [
            "urllib3.connectionpool",
            "requests.packages.urllib3",
            "asyncio",
            "multipart.multipart"
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Configure specific library loggers
    library_configs = {
        "sqlalchemy.engine": logging.INFO if settings.database.echo else logging.WARNING,
        "sqlalchemy.pool": logging.WARNING,
        "redis": logging.INFO,
        "chromadb": logging.INFO,
        "langchain": logging.INFO,
        "ollama": logging.INFO
    }
    
    for logger_name, level in library_configs.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name, defaults to calling module name
        
    Returns:
        Configured structlog BoundLogger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'mark1')
    
    return structlog.get_logger(name)


def set_agent_context(logger: structlog.BoundLogger, agent_context: Dict[str, Any]) -> structlog.BoundLogger:
    """
    Add agent context to a logger
    
    Args:
        logger: The logger instance
        agent_context: Agent context information
        
    Returns:
        Logger with bound agent context
    """
    return logger.bind(
        agent_id=agent_context.get("id"),
        agent_type=agent_context.get("type"),
        agent_name=agent_context.get("name"),
        workflow_id=agent_context.get("workflow_id")
    )


def log_execution_time(logger: structlog.BoundLogger):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.info(
                    "Function executed successfully",
                    function=func.__name__,
                    execution_time=execution_time,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.info(
                    "Async function executed successfully",
                    function=func.__name__,
                    execution_time=execution_time,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    "Async function execution failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_api_request(logger: structlog.BoundLogger):
    """
    Decorator to log API requests and responses
    
    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def wrapper(request, *args, **kwargs):
            start_time = time.perf_counter()
            
            # Log request
            logger.info(
                "API request received",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if hasattr(request, 'client') else None,
                user_agent=request.headers.get("user-agent"),
                content_type=request.headers.get("content-type")
            )
            
            try:
                response = await func(request, *args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Log successful response
                logger.info(
                    "API request completed successfully",
                    method=request.method,
                    url=str(request.url),
                    status_code=getattr(response, 'status_code', 200),
                    execution_time=execution_time
                )
                
                return response
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                
                # Log error response
                logger.error(
                    "API request failed",
                    method=request.method,
                    url=str(request.url),
                    error=str(e),
                    error_type=type(e).__name__,
                    execution_time=execution_time
                )
                raise
        
        return wrapper
    return decorator


class LoggingMiddleware:
    """
    Middleware for automatic request/response logging
    """
    
    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        self.logger = logger or get_logger("middleware")
    
    async def __call__(self, request, call_next):
        """Process request and log details"""
        import time
        
        start_time = time.perf_counter()
        
        # Log incoming request
        self.logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if hasattr(request, 'client') else None
        )
        
        try:
            response = await call_next(request)
            execution_time = time.perf_counter() - start_time
            
            # Log successful response
            self.logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                execution_time=execution_time
            )
            
            return response
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            # Log error
            self.logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time
            )
            raise


# Convenience functions for common logging patterns
def log_agent_action(logger: structlog.BoundLogger, agent_id: str, action: str, **kwargs):
    """Log agent action with standardized format"""
    logger.info(
        "Agent action",
        agent_id=agent_id,
        action=action,
        **kwargs
    )


def log_workflow_step(logger: structlog.BoundLogger, workflow_id: str, step: str, status: str, **kwargs):
    """Log workflow step with standardized format"""
    logger.info(
        "Workflow step",
        workflow_id=workflow_id,
        step=step,
        status=status,
        **kwargs
    )


def log_performance_metric(logger: structlog.BoundLogger, metric_name: str, value: float, unit: str = "", **kwargs):
    """Log performance metric with standardized format"""
    logger.info(
        "Performance metric",
        metric_name=metric_name,
        value=value,
        unit=unit,
        **kwargs
    )