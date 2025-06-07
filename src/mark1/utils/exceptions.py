"""
Custom Exception Hierarchy for Mark-1 Orchestrator
Provides structured error handling across all system components
"""

from typing import Any, Dict, Optional, List
import traceback
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better error classification"""
    CONFIGURATION = "configuration"
    AGENT = "agent"
    LLM = "llm"
    SCANNING = "scanning"
    STORAGE = "storage"
    NETWORK = "network"
    SECURITY = "security"
    ORCHESTRATION = "orchestration"
    INTEGRATION = "integration"
    VALIDATION = "validation"
    TASK = "task"


class Mark1BaseException(Exception):
    """
    Base exception class for all Mark-1 system exceptions
    Provides structured error information and context
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.CONFIGURATION,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.cause = cause
        self.traceback_info = traceback.format_exc() if cause else None
        
        super().__init__(self.message)
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception class"""
        class_name = self.__class__.__name__
        return f"MARK1_{class_name.upper().replace('EXCEPTION', '_ERROR')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_info
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# Configuration-related exceptions
class ConfigurationException(Mark1BaseException):
    """Base class for configuration-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        super().__init__(message, **kwargs)


class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration values are invalid or missing"""
    
    def __init__(self, config_key: str, value: Any = None, expected_type: str = None):
        self.config_key = config_key
        self.value = value
        self.expected_type = expected_type
        
        message = f"Invalid configuration for '{config_key}'"
        if value is not None:
            message += f": got '{value}'"
        if expected_type:
            message += f", expected {expected_type}"
            
        context = {
            "config_key": config_key,
            "value": value,
            "expected_type": expected_type
        }
        
        super().__init__(message, context=context, severity=ErrorSeverity.HIGH)


class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing"""
    
    def __init__(self, config_keys: List[str]):
        self.config_keys = config_keys
        
        if len(config_keys) == 1:
            message = f"Missing required configuration: {config_keys[0]}"
        else:
            message = f"Missing required configurations: {', '.join(config_keys)}"
            
        context = {"missing_keys": config_keys}
        super().__init__(message, context=context, severity=ErrorSeverity.CRITICAL)


# Backward compatibility aliases
ConfigurationError = ConfigurationException


# Agent-related exceptions
class AgentException(Mark1BaseException):
    """Base class for agent-related errors"""
    
    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        self.agent_id = agent_id
        kwargs.setdefault('category', ErrorCategory.AGENT)
        
        if agent_id:
            message = f"Agent {agent_id}: {message}"
            
        context = kwargs.get('context', {})
        context['agent_id'] = agent_id
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class AgentNotFoundException(AgentException):
    """Raised when an agent is not found"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent not found: {agent_id}",
            agent_id=agent_id,
            severity=ErrorSeverity.MEDIUM
        )


class AgentRegistrationException(AgentException):
    """Raised when agent registration fails"""
    
    def __init__(self, agent_name: str, reason: str, error: Optional[Exception] = None):
        message = f"Agent registration failed for '{agent_name}': {reason}"
        context = {"agent_name": agent_name, "reason": reason}
        super().__init__(
            message,
            context=context,
            cause=error,
            severity=ErrorSeverity.HIGH
        )


class AgentExecutionException(AgentException):
    """Raised when agent execution fails"""
    
    def __init__(self, agent_id: str, task: str, error: Exception):
        message = f"Agent execution failed for task: {task}"
        context = {"task": task}
        super().__init__(
            message,
            agent_id=agent_id,
            context=context,
            cause=error,
            severity=ErrorSeverity.HIGH
        )


class AgentTimeoutException(AgentException):
    """Raised when agent execution times out"""
    
    def __init__(self, agent_id: str, timeout_seconds: int):
        message = f"Agent execution timed out after {timeout_seconds} seconds"
        context = {"timeout_seconds": timeout_seconds}
        super().__init__(
            message,
            agent_id=agent_id,
            context=context,
            severity=ErrorSeverity.HIGH
        )


class AgentLoadError(AgentException):
    """Raised when agent loading fails"""
    
    def __init__(self, agent_path: str, reason: str, error: Optional[Exception] = None):
        message = f"Failed to load agent from '{agent_path}': {reason}"
        context = {"agent_path": agent_path, "reason": reason}
        super().__init__(
            message,
            context=context,
            cause=error,
            severity=ErrorSeverity.HIGH
        )


class DiscoveryError(AgentException):
    """Raised when agent discovery fails"""
    
    def __init__(self, path: str, reason: str, error: Optional[Exception] = None):
        message = f"Agent discovery failed for path '{path}': {reason}"
        context = {"discovery_path": path, "reason": reason}
        super().__init__(
            message,
            context=context,
            cause=error,
            severity=ErrorSeverity.MEDIUM
        )


# Task-related exceptions
class TaskException(Mark1BaseException):
    """Base class for task-related errors"""
    
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        self.task_id = task_id
        kwargs.setdefault('category', ErrorCategory.TASK)
        
        if task_id:
            message = f"Task {task_id}: {message}"
            
        context = kwargs.get('context', {})
        context['task_id'] = task_id
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class TaskNotFoundException(TaskException):
    """Raised when a task is not found"""
    
    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            task_id=task_id,
            severity=ErrorSeverity.MEDIUM
        )


class TaskExecutionException(TaskException):
    """Raised when task execution fails"""
    
    def __init__(self, task_id: str, reason: str, error: Optional[Exception] = None):
        message = f"Task execution failed: {reason}"
        context = {"failure_reason": reason}
        super().__init__(
            message,
            task_id=task_id,
            context=context,
            cause=error,
            severity=ErrorSeverity.HIGH
        )


class TaskTimeoutException(TaskException):
    """Raised when task execution times out"""
    
    def __init__(self, task_id: str, timeout_seconds: int):
        message = f"Task execution timed out after {timeout_seconds} seconds"
        context = {"timeout_seconds": timeout_seconds}
        super().__init__(
            message,
            task_id=task_id,
            context=context,
            severity=ErrorSeverity.HIGH
        )


class TaskPlanningException(TaskException):
    """Raised when task planning fails"""
    
    def __init__(self, message: str, task_description: Optional[str] = None, error: Optional[Exception] = None):
        context = {"task_description": task_description}
        super().__init__(
            message,
            context=context,
            cause=error,
            severity=ErrorSeverity.MEDIUM
        )


# LLM-related exceptions
class LLMException(Mark1BaseException):
    """Base class for LLM-related errors"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        self.model_name = model_name
        kwargs.setdefault('category', ErrorCategory.LLM)
        kwargs.setdefault('context', {}).update({"model_name": model_name})
        super().__init__(message, **kwargs)


class LLMConnectionException(LLMException):
    """Raised when LLM connection fails"""
    
    def __init__(self, provider: str, endpoint: str, error: Optional[Exception] = None):
        self.provider = provider
        self.endpoint = endpoint
        
        message = f"Failed to connect to {provider} at {endpoint}"
        context = {"provider": provider, "endpoint": endpoint}
        super().__init__(
            message, 
            context=context, 
            cause=error, 
            severity=ErrorSeverity.CRITICAL
        )


class LLMModelNotFoundException(LLMException):
    """Raised when requested LLM model is not available"""
    
    def __init__(self, model_name: str, provider: str):
        message = f"Model '{model_name}' not available on {provider}"
        context = {"provider": provider}
        super().__init__(message, model_name=model_name, context=context, severity=ErrorSeverity.HIGH)


class LLMRateLimitException(LLMException):
    """Raised when LLM rate limits are exceeded"""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f", retry after {retry_after} seconds"
            
        context = {"provider": provider, "retry_after": retry_after}
        super().__init__(message, context=context, severity=ErrorSeverity.MEDIUM)


# Scanning-related exceptions
class ScanningException(Mark1BaseException):
    """Base class for codebase scanning errors"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        self.file_path = file_path
        kwargs.setdefault('category', ErrorCategory.SCANNING)
        kwargs.setdefault('context', {}).update({"file_path": file_path})
        super().__init__(message, **kwargs)


class InvalidCodebaseException(ScanningException):
    """Raised when codebase structure is invalid or unsupported"""
    
    def __init__(self, path: str, reason: str):
        message = f"Invalid codebase at '{path}': {reason}"
        super().__init__(message, file_path=path, severity=ErrorSeverity.HIGH)


class ParseException(ScanningException):
    """Raised when code parsing fails"""
    
    def __init__(self, file_path: str, language: str, error: Exception):
        message = f"Failed to parse {language} file: {file_path}"
        context = {"language": language, "parse_error": str(error)}
        super().__init__(
            message, 
            file_path=file_path, 
            context=context, 
            cause=error, 
            severity=ErrorSeverity.MEDIUM
        )


# Storage-related exceptions
class StorageException(Mark1BaseException):
    """Base class for storage-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.STORAGE)
        super().__init__(message, **kwargs)


class DatabaseError(StorageException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, error: Optional[Exception] = None):
        self.operation = operation
        
        if operation:
            message = f"Database {operation} failed: {message}"
        else:
            message = f"Database operation failed: {message}"
            
        context = {"operation": operation}
        super().__init__(
            message, 
            context=context, 
            cause=error, 
            severity=ErrorSeverity.HIGH
        )


class DatabaseConnectionException(StorageException):
    """Raised when database connection fails"""
    
    def __init__(self, database_url: str, error: Exception):
        message = f"Failed to connect to database: {database_url}"
        context = {"database_url": database_url}
        super().__init__(
            message, 
            context=context, 
            cause=error, 
            severity=ErrorSeverity.CRITICAL
        )


class VectorStoreException(StorageException):
    """Raised when vector store operations fail"""
    
    def __init__(self, operation: str, collection: Optional[str] = None, error: Optional[Exception] = None):
        message = f"Vector store operation failed: {operation}"
        if collection:
            message += f" (collection: {collection})"
            
        context = {"operation": operation, "collection": collection}
        super().__init__(message, context=context, cause=error, severity=ErrorSeverity.HIGH)


class ContextError(StorageException):
    """Raised when context operations fail"""
    
    def __init__(self, message: str, context_id: Optional[str] = None, error: Optional[Exception] = None):
        self.context_id = context_id
        
        if context_id:
            message = f"Context error for {context_id}: {message}"
        else:
            message = f"Context operation failed: {message}"
            
        context = {"context_id": context_id}
        super().__init__(message, context=context, cause=error, severity=ErrorSeverity.HIGH)


# Network-related exceptions
class NetworkException(Mark1BaseException):
    """Base class for network-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        super().__init__(message, **kwargs)


class APIConnectionException(NetworkException):
    """Raised when API connection fails"""
    
    def __init__(self, url: str, status_code: Optional[int] = None, error: Optional[Exception] = None):
        message = f"API connection failed: {url}"
        if status_code:
            message += f" (status: {status_code})"
            
        context = {"url": url, "status_code": status_code}
        super().__init__(message, context=context, cause=error, severity=ErrorSeverity.HIGH)


# Security-related exceptions
class SecurityException(Mark1BaseException):
    """Base class for security-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class AuthenticationException(SecurityException):
    """Raised when authentication fails"""
    
    def __init__(self, reason: str = "Invalid credentials"):
        message = f"Authentication failed: {reason}"
        super().__init__(message)


class AuthorizationException(SecurityException):
    """Raised when authorization fails"""
    
    def __init__(self, resource: str, action: str, user: Optional[str] = None):
        message = f"Access denied to {resource} for action {action}"
        if user:
            message += f" (user: {user})"
            
        context = {"resource": resource, "action": action, "user": user}
        super().__init__(message, context=context)


class SandboxViolationException(SecurityException):
    """Raised when sandbox security is violated"""
    
    def __init__(self, agent_id: str, violation_type: str, details: str):
        message = f"Sandbox violation by agent {agent_id}: {violation_type}"
        context = {
            "agent_id": agent_id,
            "violation_type": violation_type,
            "details": details
        }
        super().__init__(message, context=context)


# Orchestration-related exceptions
class OrchestrationException(Mark1BaseException):
    """Base class for orchestration-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.ORCHESTRATION)
        super().__init__(message, **kwargs)


class TaskExecutionException(OrchestrationException):
    """Raised when task execution fails"""
    
    def __init__(self, task_id: str, stage: str, error: Optional[Exception] = None):
        message = f"Task execution failed at stage '{stage}': {task_id}"
        context = {"task_id": task_id, "stage": stage}
        super().__init__(message, context=context, cause=error, severity=ErrorSeverity.HIGH)


class WorkflowException(OrchestrationException):
    """Raised when workflow execution fails"""
    
    def __init__(self, workflow_id: str, step: Optional[str] = None, error: Optional[Exception] = None):
        message = f"Workflow execution failed: {workflow_id}"
        if step:
            message += f" at step '{step}'"
            
        context = {"workflow_id": workflow_id, "step": step}
        super().__init__(message, context=context, cause=error, severity=ErrorSeverity.HIGH)


# Integration-related exceptions
class IntegrationException(Mark1BaseException):
    """Base class for integration-related errors"""
    
    def __init__(self, message: str, integration_type: Optional[str] = None, **kwargs):
        self.integration_type = integration_type
        kwargs.setdefault('category', ErrorCategory.INTEGRATION)
        kwargs.setdefault('context', {}).update({"integration_type": integration_type})
        super().__init__(message, **kwargs)


class LangChainIntegrationException(IntegrationException):
    """Raised when LangChain integration fails"""
    
    def __init__(self, component: str, error: Exception):
        message = f"LangChain integration failed for component: {component}"
        context = {"component": component}
        super().__init__(
            message, 
            integration_type="langchain", 
            context=context, 
            cause=error, 
            severity=ErrorSeverity.HIGH
        )


# Validation-related exceptions
class ValidationException(Mark1BaseException):
    """Base class for validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        self.field = field
        self.value = value
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('context', {}).update({"field": field, "value": value})
        super().__init__(message, **kwargs)


class SchemaValidationException(ValidationException):
    """Raised when schema validation fails"""
    
    def __init__(self, schema_name: str, errors: List[str]):
        self.schema_name = schema_name
        self.validation_errors = errors
        
        message = f"Schema validation failed for {schema_name}: {'; '.join(errors)}"
        context = {"schema_name": schema_name, "validation_errors": errors}
        super().__init__(message, context=context, severity=ErrorSeverity.MEDIUM)


# Utility functions for exception handling
def handle_exception(
    exc: Exception, 
    logger=None, 
    reraise: bool = True, 
    convert_to_mark1: bool = True
) -> Optional[Mark1BaseException]:
    """
    Utility function to handle exceptions consistently
    
    Args:
        exc: The exception to handle
        logger: Optional logger instance
        reraise: Whether to reraise the exception
        convert_to_mark1: Whether to convert to Mark1BaseException
    
    Returns:
        Mark1BaseException if convert_to_mark1 is True and reraise is False
    """
    if isinstance(exc, Mark1BaseException):
        mark1_exc = exc
    elif convert_to_mark1:
        mark1_exc = Mark1BaseException(
            message=str(exc),
            cause=exc,
            severity=ErrorSeverity.HIGH
        )
    else:
        mark1_exc = None
    
    if logger and mark1_exc:
        logger.error("Exception occurred", extra=mark1_exc.to_dict())
    
    if reraise:
        if mark1_exc and not isinstance(exc, Mark1BaseException):
            raise mark1_exc from exc
        else:
            raise exc
    
    return mark1_exc


def create_error_context(**kwargs) -> Dict[str, Any]:
    """Create standardized error context dictionary"""
    return {k: v for k, v in kwargs.items() if v is not None}


# Backwards compatibility aliases
ValidationError = ValidationException  # For backwards compatibility


class AnalysisException(Mark1BaseException):
    """Exception raised during code analysis operations"""
    def __init__(self, message: str, analysis_type: str = None, file_path: str = None):
        super().__init__(message)
        self.analysis_type = analysis_type
        self.file_path = file_path


class ParseException(Mark1BaseException):
    """Exception raised during code parsing operations"""
    def __init__(self, message: str, file_path: str = None, language: str = None):
        super().__init__(message)
        self.file_path = file_path
        self.language = language


class DetectionException(Mark1BaseException):
    """Exception raised during detection operations"""
    def __init__(self, message: str, detection_type: str = None):
        super().__init__(message)
        self.detection_type = detection_type


class ReplacementException(Mark1BaseException):
    """Exception raised during code replacement operations"""
    def __init__(self, message: str, replacement_type: str = None):
        super().__init__(message)
        self.replacement_type = replacement_type


class ScanException(Mark1BaseException):
    """Exception raised during codebase scanning operations"""
    def __init__(self, message: str, scan_path: str = None):
        super().__init__(message)
        self.scan_path = scan_path


class IntegrationError(Mark1BaseException):
    """Exception raised when agent integration fails"""
    pass