"""
Mark-1 Orchestrator System Constants

This module contains all system-wide constants, enums, and configuration values
used throughout the Mark-1 Orchestrator system.
"""

from enum import Enum, IntEnum
from typing import Dict, List, Set, Tuple
import re


# =============================================================================
# VERSION AND SYSTEM INFO
# =============================================================================

VERSION = "1.0.0"
SYSTEM_NAME = "Mark-1 Orchestrator"
SYSTEM_DESCRIPTION = "Advanced AI Agent Orchestration System"
API_VERSION = "v1"


# =============================================================================
# AGENT FRAMEWORK CONSTANTS
# =============================================================================

class AgentFramework(Enum):
    """Supported agent frameworks"""
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    AUTOGPT = "autogpt"
    CREWAI = "crewai"
    CUSTOM = "custom"
    OPENAI_ASSISTANTS = "openai_assistants"
    LLAMAINDEX = "llamaindex"


class AgentType(Enum):
    """Different types of agents"""
    REACT = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    CONVERSATIONAL = "conversational"
    TOOL_CALLING = "tool_calling"
    MULTI_AGENT = "multi_agent"
    WORKFLOW = "workflow"
    RETRIEVAL = "retrieval"
    CODE_GENERATION = "code_generation"


class AgentStatus(Enum):
    """Agent lifecycle status"""
    DISCOVERED = "discovered"
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


# =============================================================================
# TASK AND WORKFLOW CONSTANTS
# =============================================================================

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class TaskPriority(IntEnum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class WorkflowStatus(Enum):
    """Workflow execution status"""
    CREATED = "created"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# LLM PROVIDER CONSTANTS
# =============================================================================

class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


# LLM API call patterns for detection and replacement
LLM_CALL_PATTERNS = {
    "openai": [
        r"openai\.ChatCompletion\.create",
        r"client\.chat\.completions\.create",
        r"await.*openai.*chat",
        r"OpenAI\(\)\.chat\.completions",
        r"openai\.Completion\.create"
    ],
    "anthropic": [
        r"anthropic\.messages\.create",
        r"client\.messages\.create",
        r"Anthropic\(\)\.messages",
        r"anthropic\.completions\.create"
    ],
    "langchain": [
        r"ChatOpenAI\(",
        r"OpenAI\(",
        r"ChatAnthropic\(",
        r"llm\.invoke",
        r"llm\.ainvoke",
        r"llm\.generate",
        r"llm\.agenerate"
    ],
    "huggingface": [
        r"transformers\.pipeline",
        r"AutoTokenizer\.from_pretrained",
        r"AutoModelForCausalLM\.from_pretrained",
        r"pipeline\([\"']text-generation[\"']\)"
    ]
}

# Default model configurations
DEFAULT_MODELS = {
    LLMProvider.OLLAMA: {
        "chat": "llama3.1:8b",
        "embedding": "nomic-embed-text",
        "code": "codellama:7b",
        "reasoning": "llama3.1:70b"
    }
}


# =============================================================================
# COMMUNICATION AND MESSAGING
# =============================================================================

class MessageType(Enum):
    """Message types for inter-agent communication"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    COORDINATION = "coordination"
    DATA_SHARING = "data_sharing"


class CommunicationProtocol(Enum):
    """Communication protocols"""
    REDIS_PUBSUB = "redis_pubsub"
    WEBSOCKET = "websocket"
    HTTP_REST = "http_rest"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"


# =============================================================================
# CONTEXT AND MEMORY CONSTANTS
# =============================================================================

# Context size limits (in bytes)
MAX_CONTEXT_SIZE = 10 * 1024 * 1024  # 10MB maximum context size
DEFAULT_CONTEXT_WINDOW = 4096  # Default context window in tokens
MAX_CONVERSATION_LENGTH = 1000  # Maximum conversation turns

# Context retention settings
CONTEXT_RETENTION_DAYS = 30  # Default context retention period in days

# Token estimation constants
AVERAGE_CHARS_PER_TOKEN = 4  # Rough estimation for token counting
MAX_TOKENS_PER_CONTEXT = 32000  # Maximum tokens in a single context


# =============================================================================
# FILE EXTENSIONS AND LANGUAGE SUPPORT
# =============================================================================

SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".clj": "clojure"
}

CODE_FILE_EXTENSIONS = set(SUPPORTED_LANGUAGES.keys())

CONFIG_FILE_PATTERNS = [
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "composer.json",
    "Gemfile",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.toml",
    "*.ini",
    "*.cfg",
    "*.conf"
]

DOCUMENTATION_EXTENSIONS = {
    ".md", ".rst", ".txt", ".adoc", ".tex"
}


# =============================================================================
# AGENT PATTERN RECOGNITION
# =============================================================================

LANGCHAIN_PATTERNS = {
    "react_agent": {
        "signatures": [
            "from langchain.agents import create_react_agent",
            "AgentExecutor.from_agent_and_tools",
            "ReActSingleInputOutputParser"
        ],
        "indicators": ["thought", "action", "observation", "Final Answer"],
        "confidence_threshold": 0.8
    },
    "conversational_agent": {
        "signatures": [
            "from langchain.agents import create_conversational_agent",
            "ConversationalAgent",
            "ConversationalChatAgent"
        ],
        "indicators": ["chat_history", "human_input", "ai_response"],
        "confidence_threshold": 0.7
    },
    "langgraph_agent": {
        "signatures": [
            "from langgraph import StateGraph",
            "StateGraph()",
            "add_node",
            "add_edge",
            "compile()"
        ],
        "indicators": ["state", "nodes", "edges", "workflow"],
        "confidence_threshold": 0.9
    }
}

AUTOGPT_PATTERNS = {
    "classic_autogpt": {
        "signatures": [
            "class.*Agent.*:",
            "def.*execute.*task",
            "memory.*add"
        ],
        "indicators": ["goals", "memory", "resources", "constraints"],
        "confidence_threshold": 0.7
    }
}

CREWAI_PATTERNS = {
    "crew_agent": {
        "signatures": [
            "from crewai import Agent",
            "from crewai import Task",
            "from crewai import Crew"
        ],
        "indicators": ["role", "goal", "backstory", "tools", "verbose"],
        "confidence_threshold": 0.9
    }
}


# =============================================================================
# CAPABILITY CATEGORIES
# =============================================================================

class CapabilityCategory(Enum):
    """Categories of agent capabilities"""
    DATA_PROCESSING = "data_processing"
    WEB_SCRAPING = "web_scraping"
    FILE_OPERATIONS = "file_operations"
    API_INTEGRATION = "api_integration"
    DATABASE_OPERATIONS = "database_operations"
    CODE_GENERATION = "code_generation"
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    REASONING = "reasoning"
    PLANNING = "planning"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    SEARCH = "search"
    COMMUNICATION = "communication"


CAPABILITY_KEYWORDS = {
    CapabilityCategory.DATA_PROCESSING: [
        "pandas", "numpy", "csv", "json", "xml", "parse", "transform", "clean"
    ],
    CapabilityCategory.WEB_SCRAPING: [
        "requests", "beautifulsoup", "selenium", "scrapy", "crawl", "fetch", "extract"
    ],
    CapabilityCategory.FILE_OPERATIONS: [
        "file", "directory", "read", "write", "upload", "download", "compress"
    ],
    CapabilityCategory.API_INTEGRATION: [
        "api", "rest", "graphql", "webhook", "http", "client", "request"
    ],
    CapabilityCategory.DATABASE_OPERATIONS: [
        "database", "sql", "mongodb", "postgres", "mysql", "query", "insert", "update"
    ],
    CapabilityCategory.CODE_GENERATION: [
        "generate", "template", "scaffold", "create", "build", "compile"
    ],
    CapabilityCategory.TEXT_PROCESSING: [
        "nlp", "tokenize", "sentiment", "summarize", "translate", "extract"
    ],
    CapabilityCategory.REASONING: [
        "reason", "infer", "deduce", "conclude", "analyze", "evaluate"
    ],
    CapabilityCategory.PLANNING: [
        "plan", "schedule", "strategy", "workflow", "sequence", "optimize"
    ]
}


# =============================================================================
# SECURITY AND SANDBOXING
# =============================================================================

class SecurityLevel(Enum):
    """Security levels for agent execution"""
    UNRESTRICTED = "unrestricted"
    LIMITED = "limited"
    SANDBOXED = "sandboxed"
    ISOLATED = "isolated"


DANGEROUS_FUNCTIONS = {
    "file_system": [
        "os.system", "subprocess.run", "subprocess.call", "os.remove",
        "shutil.rmtree", "os.rmdir", "pathlib.Path.unlink"
    ],
    "network": [
        "socket.socket", "urllib.request.urlopen", "requests.get",
        "requests.post", "httpx.get", "httpx.post"
    ],
    "process": [
        "os.kill", "signal.kill", "multiprocessing.Process",
        "threading.Thread", "asyncio.create_subprocess"
    ],
    "imports": [
        "importlib.import_module", "__import__", "exec", "eval"
    ]
}

ALLOWED_MODULES = {
    "standard": [
        "json", "re", "datetime", "math", "random", "string", "collections",
        "itertools", "functools", "operator", "typing", "dataclasses"
    ],
    "data": [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly"
    ],
    "ml": [
        "sklearn", "torch", "tensorflow", "transformers", "langchain"
    ]
}


# =============================================================================
# PERFORMANCE AND MONITORING
# =============================================================================

class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


PERFORMANCE_THRESHOLDS = {
    "task_execution_timeout": 300,  # seconds
    "agent_response_timeout": 30,   # seconds
    "memory_usage_limit": 1024,     # MB
    "cpu_usage_limit": 80,          # percentage
    "concurrent_tasks_limit": 10,
    "queue_size_limit": 100
}

HEALTH_CHECK_INTERVALS = {
    "agent_heartbeat": 30,          # seconds
    "system_metrics": 60,           # seconds
    "database_connection": 120,     # seconds
    "external_services": 300        # seconds
}


# =============================================================================
# API AND HTTP CONSTANTS
# =============================================================================

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


HTTP_STATUS_CODES = {
    "SUCCESS": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "NO_CONTENT": 204,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503
}

API_RATE_LIMITS = {
    "default": 100,      # requests per minute
    "authenticated": 500,
    "premium": 1000
}


# =============================================================================
# STORAGE AND DATABASE
# =============================================================================

class StorageType(Enum):
    """Types of storage backends"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    REDIS = "redis"
    MONGODB = "mongodb"
    CHROMADB = "chromadb"


DATABASE_POOL_SETTINGS = {
    "min_connections": 5,
    "max_connections": 20,
    "connection_timeout": 30,
    "idle_timeout": 300,
    "max_lifetime": 3600
}

REDIS_SETTINGS = {
    "default_ttl": 3600,        # seconds
    "max_connections": 10,
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "retry_on_timeout": True
}


# =============================================================================
# DIRECTORY AND PATH CONSTANTS
# =============================================================================

DEFAULT_DIRECTORIES = {
    "agents": "agents",
    "data": "data",
    "logs": "logs",
    "cache": "cache",
    "models": "models",
    "temp": "temp",
    "config": "config",
    "scripts": "scripts"
}

LOG_FILE_PATTERNS = {
    "main": "mark1_{date}.log",
    "agent": "agent_{agent_id}_{date}.log",
    "task": "task_{task_id}_{date}.log",
    "error": "error_{date}.log",
    "audit": "audit_{date}.log"
}


# =============================================================================
# REGEX PATTERNS
# =============================================================================

REGEX_PATTERNS = {
    "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    "uuid": re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    "semver": re.compile(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+)?$'),
    "agent_id": re.compile(r'^[a-zA-Z0-9_-]+$'),
    "task_id": re.compile(r'^task_[0-9a-f]{8}$'),
    "python_import": re.compile(r'^(?:from\s+[\w.]+\s+)?import\s+[\w.,\s*]+$'),
    "function_call": re.compile(r'(\w+)\s*\([^)]*\)'),
    "class_definition": re.compile(r'class\s+(\w+)(?:\([^)]*\))?:'),
    "llm_api_key": re.compile(r'sk-[a-zA-Z0-9]{48}|sk-[a-zA-Z0-9]{32}')
}


# =============================================================================
# ERROR CODES AND MESSAGES
# =============================================================================

ERROR_CODES = {
    # Agent related errors
    "AGENT_NOT_FOUND": "AGT001",
    "AGENT_REGISTRATION_FAILED": "AGT002",
    "AGENT_EXECUTION_FAILED": "AGT003",
    "AGENT_TIMEOUT": "AGT004",
    "AGENT_RESOURCE_LIMIT": "AGT005",
    
    # Task related errors
    "TASK_CREATION_FAILED": "TSK001",
    "TASK_EXECUTION_FAILED": "TSK002",
    "TASK_TIMEOUT": "TSK003",
    "TASK_DEPENDENCY_FAILED": "TSK004",
    
    # Workflow related errors
    "WORKFLOW_INVALID": "WFL001",
    "WORKFLOW_EXECUTION_FAILED": "WFL002",
    "WORKFLOW_CYCLE_DETECTED": "WFL003",
    
    # System related errors
    "DATABASE_CONNECTION_FAILED": "SYS001",
    "REDIS_CONNECTION_FAILED": "SYS002",
    "LLM_SERVICE_UNAVAILABLE": "SYS003",
    "CONFIGURATION_ERROR": "SYS004",
    "RESOURCE_EXHAUSTED": "SYS005",
    
    # Security related errors
    "UNAUTHORIZED_ACCESS": "SEC001",
    "INSUFFICIENT_PERMISSIONS": "SEC002",
    "MALICIOUS_CODE_DETECTED": "SEC003",
    "SANDBOX_VIOLATION": "SEC004",
    
    # Integration related errors
    "LANGCHAIN_INTEGRATION_FAILED": "INT001",
    "EXTERNAL_API_ERROR": "INT002",
    "MODEL_LOADING_FAILED": "INT003"
}


# =============================================================================
# FEATURE FLAGS
# =============================================================================

class FeatureFlag(Enum):
    """System feature flags"""
    AGENT_SANDBOXING = "agent_sandboxing"
    DISTRIBUTED_EXECUTION = "distributed_execution"
    ADVANCED_MONITORING = "advanced_monitoring"
    AUTOMATIC_SCALING = "automatic_scaling"
    SECURITY_SCANNING = "security_scanning"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    EXPERIMENTAL_FEATURES = "experimental_features"


DEFAULT_FEATURE_FLAGS = {
    FeatureFlag.AGENT_SANDBOXING: True,
    FeatureFlag.DISTRIBUTED_EXECUTION: False,
    FeatureFlag.ADVANCED_MONITORING: True,
    FeatureFlag.AUTOMATIC_SCALING: False,
    FeatureFlag.SECURITY_SCANNING: True,
    FeatureFlag.PERFORMANCE_OPTIMIZATION: True,
    FeatureFlag.EXPERIMENTAL_FEATURES: False
}


# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_CONFIG = {
    "system": {
        "max_concurrent_agents": 10,
        "task_queue_size": 100,
        "heartbeat_interval": 30,
        "cleanup_interval": 300
    },
    "llm": {
        "provider": LLMProvider.OLLAMA.value,
        "timeout": 30,
        "max_retries": 3,
        "temperature": 0.7,
        "max_tokens": 2048
    },
    "storage": {
        "type": StorageType.SQLITE.value,
        "connection_pool_size": 10,
        "cache_ttl": 3600
    },
    "security": {
        "level": SecurityLevel.SANDBOXED.value,
        "enable_code_scanning": True,
        "max_execution_time": 300
    },
    "monitoring": {
        "enable_metrics": True,
        "metrics_interval": 60,
        "enable_tracing": False
    }
}


# =============================================================================
# AGENT DISCOVERY AND SCANNING
# =============================================================================

# Default paths to search for agents
AGENT_DISCOVERY_PATHS = [
    "agents/",
    "src/agents/",
    "mark1/agents/",
    "."
]

# File patterns for agent detection
AGENT_FILE_PATTERNS = [
    "**/*agent*.py",
    "**/*bot*.py", 
    "**/*assistant*.py",
    "**/agent.py",
    "**/main.py"
]

# Agent detection keywords and patterns
AGENT_KEYWORDS = [
    'agent', 'bot', 'assistant', 'ai', 'llm', 'gpt', 'chat',
    'executor', 'runner', 'processor', 'handler', 'orchestrator',
    'manager', 'controller', 'coordinator', 'planner', 'worker'
]

# Framework detection patterns
FRAMEWORK_DETECTION_PATTERNS = {
    "langchain": [
        r"from langchain",
        r"import langchain",
        r"LangChain",
        r"BaseAgent",
        r"AgentExecutor",
        r"Chain"
    ],
    "crewai": [
        r"from crewai",
        r"import crewai",
        r"CrewAI",
        r"Agent.*crew",
        r"Task.*crew"
    ],
    "autogpt": [
        r"from autogpt",
        r"import autogpt",
        r"AutoGPT",
        r"auto_gpt"
    ],
    "openai_assistants": [
        r"openai\.beta\.assistants",
        r"Assistant.*openai",
        r"client\.beta\.assistants"
    ]
}


# =============================================================================
# EXPORT ALL CONSTANTS
# =============================================================================

__all__ = [
    # Version and system
    "VERSION", "SYSTEM_NAME", "SYSTEM_DESCRIPTION", "API_VERSION",
    
    # Enums
    "AgentFramework", "AgentType", "AgentStatus",
    "TaskStatus", "TaskPriority", "WorkflowStatus",
    "LLMProvider", "MessageType", "CommunicationProtocol",
    "CapabilityCategory", "SecurityLevel", "MetricType",
    "HTTPMethod", "StorageType", "FeatureFlag",
    
    # Patterns and configurations
    "LLM_CALL_PATTERNS", "DEFAULT_MODELS",
    "SUPPORTED_LANGUAGES", "CODE_FILE_EXTENSIONS",
    "CONFIG_FILE_PATTERNS", "DOCUMENTATION_EXTENSIONS",
    "LANGCHAIN_PATTERNS", "AUTOGPT_PATTERNS", "CREWAI_PATTERNS",
    "CAPABILITY_KEYWORDS", "DANGEROUS_FUNCTIONS", "ALLOWED_MODULES",
    
    # Thresholds and limits
    "PERFORMANCE_THRESHOLDS", "HEALTH_CHECK_INTERVALS",
    "HTTP_STATUS_CODES", "API_RATE_LIMITS",
    "DATABASE_POOL_SETTINGS", "REDIS_SETTINGS",
    
    # Paths and patterns
    "DEFAULT_DIRECTORIES", "LOG_FILE_PATTERNS", "REGEX_PATTERNS",
    
    # Errors and features
    "ERROR_CODES", "DEFAULT_FEATURE_FLAGS", "DEFAULT_CONFIG"
]