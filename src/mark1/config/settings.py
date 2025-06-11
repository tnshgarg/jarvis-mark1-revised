"""
Mark-1 Orchestrator Configuration Management

This module provides comprehensive configuration management for the Mark-1 system
using Pydantic Settings for validation and environment variable support.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# Load .env file explicitly
from dotenv import load_dotenv
load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings

try:
    from pydantic import (
        BaseModel,
        Field,
        field_validator,
        model_validator,
        SecretStr
    )
    from pydantic.types import PositiveInt
    PYDANTIC_V2 = True
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import (
        BaseModel,
        Field,
        validator as field_validator,
        root_validator as model_validator,
        SecretStr
    )
    from pydantic import PositiveInt
    PYDANTIC_V2 = False


class LogLevel(str, Enum):
    """Enumeration of available log levels"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class Environment(str, Enum):
    """Enumeration of deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    
    url: str = Field(
        default="postgresql://mark1:password@localhost:5432/mark1_db",
        description="Database connection URL"
    )
    pool_size: PositiveInt = Field(default=10, description="Connection pool size")
    max_overflow: PositiveInt = Field(default=20, description="Max overflow connections")
    pool_timeout: PositiveInt = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: PositiveInt = Field(default=1800, description="Pool recycle time")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    
    @field_validator('url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format"""
        try:
            parsed = urlparse(v)
            # Handle SQLite URLs which can have different formats
            if parsed.scheme == 'sqlite':
                return v
            # For other databases, require netloc
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid database URL format")
            return v
        except Exception as e:
            raise ValueError(f"Invalid database URL: {e}")


class RedisConfig(BaseModel):
    """Redis configuration settings"""
    
    host: str = Field(default="localhost", description="Redis host")
    port: PositiveInt = Field(default=6379, description="Redis port")
    password: Optional[SecretStr] = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    ssl: bool = Field(default=False, description="Enable SSL connection")
    socket_timeout: float = Field(default=30.0, description="Socket timeout")
    connection_pool_max_connections: PositiveInt = Field(
        default=50, description="Max connections in pool"
    )
    
    @property
    def url(self) -> str:
        """Generate Redis URL from configuration"""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class ChromaDBConfig(BaseModel):
    """ChromaDB vector store configuration"""
    
    host: str = Field(default="localhost", description="ChromaDB host")
    port: PositiveInt = Field(default=8000, description="ChromaDB port")
    collection_name: str = Field(default="mark1_vectors", description="Collection name")
    ssl: bool = Field(default=False, description="Enable SSL")
    timeout: float = Field(default=30.0, description="Request timeout")
    
    @property
    def url(self) -> str:
        """Generate ChromaDB URL"""
        scheme = "https" if self.ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"


class OllamaConfig(BaseModel):
    """Ollama LLM provider configuration"""
    
    base_url: str = Field(default="https://f6da-103-167-213-208.ngrok-free.app", description="Ollama base URL")
    default_model: str = Field(default="llama2", description="Default model name")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: Optional[PositiveInt] = Field(
        default=2048, description="Maximum tokens to generate"
    )
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v):
        """Validate Ollama base URL"""
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid base URL format")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base URL: {e}")


class SecurityConfig(BaseModel):
    """Security configuration settings"""
    
    secret_key: SecretStr = Field(
        default="change-me-in-production",
        description="Secret key for encryption"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: PositiveInt = Field(
        default=30, description="Access token expiration"
    )
    refresh_token_expire_days: PositiveInt = Field(
        default=7, description="Refresh token expiration"
    )
    password_min_length: int = Field(default=8, description="Minimum password length")
    max_login_attempts: int = Field(default=5, description="Max failed login attempts")
    lockout_duration_minutes: int = Field(
        default=15, description="Account lockout duration"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration"""
    
    enable_prometheus: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    prometheus_port: PositiveInt = Field(
        default=8090, description="Prometheus metrics port"
    )
    enable_health_checks: bool = Field(
        default=True, description="Enable health check endpoints"
    )
    health_check_interval: PositiveInt = Field(
        default=30, description="Health check interval in seconds"
    )
    enable_performance_tracking: bool = Field(
        default=True, description="Enable performance tracking"
    )
    max_metrics_retention_days: PositiveInt = Field(
        default=30, description="Metrics retention period"
    )


class ScanningConfig(BaseModel):
    """Codebase scanning configuration"""
    
    max_file_size_mb: PositiveInt = Field(
        default=10, description="Maximum file size to scan (MB)"
    )
    excluded_extensions: List[str] = Field(
        default=[".pyc", ".pyo", ".so", ".dll", ".exe", ".bin"],
        description="File extensions to exclude from scanning"
    )
    excluded_directories: List[str] = Field(
        default=["__pycache__", ".git", ".svn", "node_modules", ".venv"],
        description="Directories to exclude from scanning"
    )
    max_scan_depth: PositiveInt = Field(
        default=10, description="Maximum directory depth to scan"
    )
    enable_ml_pattern_detection: bool = Field(
        default=True, description="Enable ML-based pattern detection"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Pattern detection confidence threshold"
    )


class AgentConfig(BaseModel):
    """Agent management configuration"""
    
    max_concurrent_agents: PositiveInt = Field(
        default=10, description="Maximum concurrent agents"
    )
    agent_timeout_seconds: PositiveInt = Field(
        default=300, description="Agent execution timeout"
    )
    enable_sandboxing: bool = Field(
        default=True, description="Enable agent sandboxing"
    )
    sandbox_memory_limit_mb: PositiveInt = Field(
        default=512, description="Sandbox memory limit (MB)"
    )
    sandbox_cpu_limit_percent: int = Field(
        default=50, ge=1, le=100, description="Sandbox CPU limit (%)"
    )
    enable_agent_registry: bool = Field(
        default=True, description="Enable agent registry"
    )
    auto_discovery_enabled: bool = Field(
        default=True, description="Enable automatic agent discovery"
    )


class APIConfig(BaseModel):
    """API server configuration"""
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: PositiveInt = Field(default=8000, description="API server port")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    workers: PositiveInt = Field(default=1, description="Number of worker processes")
    max_request_size: PositiveInt = Field(
        default=16 * 1024 * 1024, description="Max request size in bytes"
    )
    cors_origins: List[str] = Field(
        default=["*"], description="CORS allowed origins"
    )
    rate_limit_requests: PositiveInt = Field(
        default=100, description="Rate limit: requests per minute"
    )
    enable_docs: bool = Field(default=True, description="Enable API documentation")


class Settings(BaseSettings):
    """
    Main configuration class for Mark-1 Orchestrator
    
    This class combines all configuration sections and provides
    environment-aware settings management.
    """
    
    # Core Settings
    app_name: str = Field(default="Mark-1 Orchestrator", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # Paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent,
        description="Base directory path"
    )
    data_dir: Path = Field(default=None, description="Data directory path")
    log_dir: Path = Field(default=None, description="Log directory path")
    agents_dir: Path = Field(default=None, description="Agents directory path")
    
    # Configuration Sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    scanning: ScanningConfig = Field(default_factory=ScanningConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        env_prefix = "MARK1_"
        case_sensitive = False
        extra = "allow"
        
    @model_validator(mode='before')
    @classmethod
    def set_default_paths(cls, values):
        """Set default paths based on base directory"""
        # Handle both dict and model instance
        if hasattr(values, 'get'):
            data = values
        else:
            data = values.__dict__ if hasattr(values, '__dict__') else values
        
        base_dir = data.get("base_dir", Path.cwd())
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        
        if not data.get("data_dir"):
            data["data_dir"] = base_dir / "data"
            
        if not data.get("log_dir"):
            data["log_dir"] = base_dir / "data" / "logs"
            
        if not data.get("agents_dir"):
            data["agents_dir"] = base_dir / "agents"
            
        return data
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate and normalize environment value"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @field_validator('log_level', mode='before')
    @classmethod
    def validate_log_level(cls, v):
        """Validate and normalize log level value"""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            self.log_dir,
            self.agents_dir,
            self.data_dir / "models",
            self.data_dir / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == Environment.TESTING
    
    def get_database_url(self, async_driver: bool = False) -> str:
        """Get database URL with optional async driver"""
        url = self.database.url
        if async_driver:
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://")
            elif url.startswith("sqlite://"):
                url = url.replace("sqlite://", "sqlite+aiosqlite://")
        return url
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        return self.redis.url
    
    def get_chromadb_url(self) -> str:
        """Get ChromaDB connection URL"""
        return self.chromadb.url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.model_dump()
    
    def __repr__(self) -> str:
        return f"<Settings(environment={self.environment}, debug={self.debug})>"


# Global settings instance (lazy-loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance
    
    This function provides a way to access settings throughout the application
    and can be overridden for testing purposes.
    
    Returns:
        Settings: The global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings_from_file(file_path: Union[str, Path]) -> Settings:
    """
    Load settings from a specific file
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Settings: A new settings instance loaded from the file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    # Set environment variable to point to the specific file
    os.environ["MARK1_CONFIG_FILE"] = str(file_path)
    
    return Settings(_env_file=str(file_path))


def override_settings(**kwargs) -> Settings:
    """
    Create a new settings instance with overridden values
    
    This is useful for testing or temporary configuration changes.
    
    Args:
        **kwargs: Settings to override
        
    Returns:
        Settings: New settings instance with overridden values
    """
    current_settings = get_settings()
    settings_dict = current_settings.model_dump()
    settings_dict.update(kwargs)
    
    return Settings(**settings_dict)