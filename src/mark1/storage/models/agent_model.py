"""
Agent data models for Mark-1 Orchestrator
Defines the database schema for agent management and metadata
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey, Table, UniqueConstraint, Index, Float
# Use String instead of UUID for SQLite compatibility
# from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import structlog

from mark1.storage.database import Base
from mark1.utils.exceptions import ValidationError

logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    DISCOVERED = "discovered"
    ANALYZING = "analyzing"
    READY = "ready"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"
    ARCHIVED = "archived"
    INACTIVE = "inactive"


class AgentType(str, Enum):
    """Agent type enumeration"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    CREWAI = "crewai"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class IntegrationType(str, Enum):
    """Integration pattern enumeration"""
    DIRECT = "direct"
    WRAPPER = "wrapper"
    ADAPTER = "adapter"
    PROXY = "proxy"
    NATIVE = "native"


# Association table for agent capabilities (many-to-many)
agent_capabilities = Table(
    'agent_capabilities',
    Base.metadata,
    Column('agent_id', String(36), ForeignKey('agents.id'), primary_key=True),
    Column('capability_id', String(36), ForeignKey('capabilities.id'), primary_key=True),
    Column('added_at', DateTime(timezone=True), default=func.now()),
    Column('capability_level', String(20), default='standard'),  # basic, standard, advanced, expert
    Column('priority', Integer, default=100),  # Lower is higher priority
    extend_existing=True
)

# Association table for agent dependencies (many-to-many self-referencing)
agent_dependencies = Table(
    'agent_dependencies',
    Base.metadata,
    # Use String instead of UUID for SQLite compatibility
    Column('parent_id', String(36), ForeignKey('agents.id', ondelete='CASCADE'), primary_key=True),
    Column('dependency_id', String(36), ForeignKey('agents.id', ondelete='CASCADE'), primary_key=True),
    Column('dependency_type', String(50), nullable=False, default='requires'),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    UniqueConstraint('parent_id', 'dependency_id', name='unique_agent_dependency'),
    extend_existing=True
)


class Agent(Base):
    """
    Agent model representing AI agents in the Mark-1 system.
    
    This model stores agent metadata, capabilities, and operational status
    for both built-in and external AI agents.
    """
    __tablename__ = "agents"
    
    # Primary identification
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Classification
    agent_type = Column(String(50), nullable=False, default=AgentType.UNKNOWN.value, index=True)
    framework = Column(String(100))  # langchain, autogpt, llama_index, etc.
    category = Column(String(100))  # general, specialized, tool-using, etc.
    tags = Column(JSON, default=list)
    
    # Status tracking and health
    health_score = Column(Integer, nullable=False, default=100)
    error_count = Column(Integer, default=0)
    last_error = Column(Text)
    
    # Capabilities (many-to-many relationship)
    capabilities = relationship(
        "Capability",
        secondary=agent_capabilities,
        backref="agents",
        lazy="joined"
    )
    
    # Agent classification
    framework_version = Column(String(100), nullable=True)
    integration_type = Column(String(50), nullable=False, default=IntegrationType.DIRECT.value)
    vendor = Column(String(100), nullable=True)
    
    # Agent configuration
    initialization_params = Column(JSON, nullable=True)  # Structured init parameters
    config_schema = Column(JSON, nullable=True)  # JSON schema for configuration validation
    default_config = Column(JSON, nullable=True)  # Default configuration values
    
    # API Integration
    api_key_required = Column(Boolean, default=False)
    api_endpoint = Column(String(500), nullable=True)
    api_documentation_url = Column(String(500), nullable=True)
    
    # Operational data
    status = Column(String(50), nullable=False, default=AgentStatus.INACTIVE.value, index=True)
    is_available = Column(Boolean, default=True, index=True)
    is_remote = Column(Boolean, default=False)
    is_built_in = Column(Boolean, default=False)
    priority = Column(Integer, default=100)  # Lower value = higher priority
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)  # Average response time
    success_rate = Column(Float, nullable=True)  # Percentage of successful calls
    error_rate = Column(Float, nullable=True)  # Percentage of failed calls
    usage_count = Column(Integer, default=0)  # Number of times agent has been used
    
    # Resource usage
    cpu_usage = Column(Float, nullable=True)  # Average CPU usage percentage
    memory_usage_mb = Column(Float, nullable=True)  # Average memory usage in MB
    tokens_per_minute = Column(Integer, nullable=True)  # Token processing rate
    cost_per_1k_tokens = Column(Float, nullable=True)  # Cost in USD per 1000 tokens
    
    # Location and path info
    source_path = Column(String(500), nullable=True)  # File system path or URL
    entry_point = Column(String(255), nullable=True)  # Entry function or method
    module_path = Column(String(500), nullable=True)  # Python module path
    
    # Metadata and tags
    extra_metadata = Column(JSON, nullable=True)  # Flexible metadata storage
    notes = Column(Text, nullable=True)
    
    # Security and permissions
    required_permissions = Column(JSON, nullable=True)  # List of required permissions
    security_level = Column(String(50), default="standard")  # security clearance level
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_activity = Column(DateTime(timezone=True), nullable=True)
    
    # Self-referencing relationships for dependencies
    dependencies_as_parent = relationship(
        "Agent",
        secondary=agent_dependencies,
        primaryjoin=id == agent_dependencies.c.parent_id,
        secondaryjoin=id == agent_dependencies.c.dependency_id,
        backref="dependent_agents",
        lazy="select",
        viewonly=True  # Add viewonly=True to prevent SQLite issues
    )
    
    # Indexes for query optimization
    __table_args__ = (
        Index('idx_agent_framework', 'framework'),
        Index('idx_agent_created', 'created_at'),
        Index('idx_agent_updated', 'updated_at'),
        Index('idx_agent_type_status', 'agent_type', 'status'),
        Index('idx_agent_health_updated', 'health_score', 'updated_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name='{self.name}', type='{self.agent_type}', status='{self.status}')>"
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate agent status"""
        if status not in [s.value for s in AgentStatus]:
            raise ValidationError(f"Invalid agent status: {status}")
        return status
    
    @validates('agent_type')
    def validate_agent_type(self, key, agent_type):
        """Validate agent type"""
        if agent_type not in [t.value for t in AgentType]:
            raise ValidationError(f"Invalid agent type: {agent_type}")
        return agent_type
    
    @hybrid_property
    def is_healthy(self) -> bool:
        """Check if agent is in a healthy state"""
        return self.status not in [AgentStatus.ERROR, AgentStatus.DISABLED]
    
    @hybrid_property
    def failure_rate(self) -> float:
        """Calculate current failure rate"""
        if self.total_executions == 0:
            return 0.0
        return (self.failed_executions / self.total_executions) * 100
    
    def update_health_metrics(self, execution_success: bool, response_time: Optional[int] = None) -> None:
        """Update health metrics after task execution"""
        self.total_executions += 1
        
        if not execution_success:
            self.failed_executions += 1
            self.error_count += 1
            # Decrease health score based on recent failures
            self.health_score = max(0, self.health_score - min(10, self.error_count))
        else:
            # Gradually improve health score on success
            self.health_score = min(100, self.health_score + 1)
        
        # Update success rate
        self.success_rate = int(((self.total_executions - self.failed_executions) / self.total_executions) * 100)
        
        # Update average response time
        if response_time is not None:
            if self.average_response_time is None:
                self.average_response_time = response_time
            else:
                # Exponential moving average
                self.average_response_time = int(0.7 * self.average_response_time + 0.3 * response_time)
        
        self.last_health_check = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)
    
    def add_capability(self, capability: 'Capability') -> None:
        """Add a capability to this agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def remove_capability(self, capability: 'Capability') -> None:
        """Remove a capability from this agent"""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if self.default_config and isinstance(self.default_config, dict):
            return self.default_config.get(key, default)
        return default
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set configuration value"""
        if self.default_config is None:
            self.default_config = {}
        elif not isinstance(self.default_config, dict):
            self.default_config = {}
        
        self.default_config[key] = value
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the agent"""
        if self.tags is None:
            self.tags = []
        elif not isinstance(self.tags, list):
            self.tags = []
        
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the agent"""
        if self.tags and isinstance(self.tags, list) and tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if agent has a specific tag"""
        return self.tags is not None and isinstance(self.tags, list) and tag in self.tags
    
    def to_dict(self, include_relationships: bool = False) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        data = {
            'id': str(self.id),
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'agent_type': self.agent_type,
            'framework_version': self.framework_version,
            'integration_type': self.integration_type,
            'status': self.status,
            'health_score': self.health_score,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'is_integrated': self.is_integrated,
            'is_healthy': self.is_healthy,
            'is_available': self.is_available,
            'success_rate': self.success_rate,
            'total_executions': self.total_executions,
            'average_response_time': self.average_response_time,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }
        
        if include_relationships:
            data['capabilities'] = [cap.to_dict() for cap in self.capabilities] if self.capabilities else []
        
        return data


class Capability(Base):
    """
    Agent capabilities model
    Represents specific abilities or skills that agents can perform
    """
    __tablename__ = 'capabilities'
    
    # Primary identification
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    display_name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Capability classification
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100), nullable=True)
    skill_level = Column(String(20), nullable=False, default='intermediate')  # basic, intermediate, advanced, expert
    
    # Technical specifications
    input_types = Column(JSON, nullable=True)  # List of supported input types
    output_types = Column(JSON, nullable=True)  # List of supported output types
    required_tools = Column(JSON, nullable=True)  # List of required tools/dependencies
    optional_tools = Column(JSON, nullable=True)  # List of optional tools that enhance capability
    
    # Performance characteristics
    typical_response_time = Column(Integer, nullable=True)  # milliseconds
    resource_intensity = Column(String(20), nullable=False, default='medium')  # low, medium, high
    parallel_execution = Column(Boolean, nullable=False, default=True)
    
    # Quality metrics
    accuracy_score = Column(Integer, nullable=False, default=85)  # percentage
    reliability_score = Column(Integer, nullable=False, default=95)  # percentage
    complexity_score = Column(Integer, nullable=False, default=5)  # 1-10 scale
    
    # Usage statistics
    usage_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Configuration
    parameters_schema = Column(JSON, nullable=True)  # JSON Schema for capability parameters
    examples = Column(JSON, nullable=True)  # Example inputs/outputs
    
    # Metadata
    tags = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    agents = relationship(
        "Agent",
        secondary=agent_capabilities,
        back_populates="capabilities",
        lazy="select"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_capability_category_skill', 'category', 'skill_level'),
        Index('idx_capability_performance', 'accuracy_score', 'reliability_score'),
    )
    
    def __repr__(self) -> str:
        return f"<Capability(id={self.id}, name='{self.name}', category='{self.category}')>"
    
    @validates('skill_level')
    def validate_skill_level(self, key, skill_level):
        """Validate skill level"""
        valid_levels = ['basic', 'intermediate', 'advanced', 'expert']
        if skill_level not in valid_levels:
            raise ValidationError(f"Invalid skill level: {skill_level}. Must be one of: {valid_levels}")
        return skill_level
    
    @validates('resource_intensity')
    def validate_resource_intensity(self, key, resource_intensity):
        """Validate resource intensity"""
        valid_intensities = ['low', 'medium', 'high']
        if resource_intensity not in valid_intensities:
            raise ValidationError(f"Invalid resource intensity: {resource_intensity}. Must be one of: {valid_intensities}")
        return resource_intensity
    
    @validates('accuracy_score')
    def validate_accuracy_score(self, key, accuracy_score):
        """Validate accuracy score is between 0 and 100"""
        if not 0 <= accuracy_score <= 100:
            raise ValidationError(f"Accuracy score must be between 0 and 100, got: {accuracy_score}")
        return accuracy_score
    
    @validates('reliability_score')
    def validate_reliability_score(self, key, reliability_score):
        """Validate reliability score is between 0 and 100"""
        if not 0 <= reliability_score <= 100:
            raise ValidationError(f"Reliability score must be between 0 and 100, got: {reliability_score}")
        return reliability_score
    
    @validates('complexity_score')
    def validate_complexity_score(self, key, complexity_score):
        """Validate complexity score is between 1 and 10"""
        if not 1 <= complexity_score <= 10:
            raise ValidationError(f"Complexity score must be between 1 and 10, got: {complexity_score}")
        return complexity_score
    
    @hybrid_property
    def success_rate(self) -> float:
        """Calculate success rate for this capability"""
        if self.usage_count == 0:
            return 0.0
        return (self.success_count / self.usage_count) * 100
    
    @hybrid_property
    def is_reliable(self) -> bool:
        """Check if capability is reliable (>= 80% success rate and reliability score)"""
        return self.success_rate >= 80.0 and self.reliability_score >= 80
    
    def update_usage_metrics(self, success: bool) -> None:
        """Update usage metrics after capability execution"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.last_used = datetime.now(timezone.utc)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the capability"""
        if self.tags is None:
            self.tags = []
        elif not isinstance(self.tags, list):
            self.tags = []
        
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the capability"""
        if self.tags and isinstance(self.tags, list) and tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if capability has a specific tag"""
        return self.tags is not None and isinstance(self.tags, list) and tag in self.tags
    
    def to_dict(self, include_relationships: bool = False) -> Dict[str, Any]:
        """Convert capability to dictionary representation"""
        data = {
            'id': str(self.id),
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'category': self.category,
            'subcategory': self.subcategory,
            'skill_level': self.skill_level,
            'input_types': self.input_types,
            'output_types': self.output_types,
            'typical_response_time': self.typical_response_time,
            'resource_intensity': self.resource_intensity,
            'parallel_execution': self.parallel_execution,
            'accuracy_score': self.accuracy_score,
            'reliability_score': self.reliability_score,
            'complexity_score': self.complexity_score,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'is_reliable': self.is_reliable,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None
        }
        
        if include_relationships:
            data['agents'] = [{'id': str(agent.id), 'name': agent.name} for agent in self.agents] if self.agents else []
        
        return data


class AgentMetrics(Base):
    """
    Historical metrics and performance tracking for agents
    Stores time-series data for monitoring and analytics
    """
    __tablename__ = 'agent_metrics'
    
    # Primary identification
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String(36), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Metric data
    metric_type = Column(String(50), nullable=False, index=True)  # response_time, success_rate, health_score, etc.
    metric_value = Column(Integer, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # ms, percentage, score, etc.
    
    # Context information
    context = Column(JSON, nullable=True)  # Additional context about the metric
    tags = Column(JSON, nullable=True)  # Tags for filtering and grouping
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    agent = relationship("Agent", backref="metrics")
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_agent_metrics_type_time', 'agent_id', 'metric_type', 'recorded_at'),
    )
    
    def __repr__(self) -> str:
        return f"<AgentMetrics(id={self.id}, agent_id={self.agent_id}, type='{self.metric_type}', value={self.metric_value})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'metric_type': self.metric_type,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'context': self.context,
            'tags': self.tags,
            'recorded_at': self.recorded_at.isoformat(),
            'created_at': self.created_at.isoformat()
        }


class AgentConfiguration(Base):
    """
    Agent configuration templates and presets
    Allows for storing and managing different agent configurations
    """
    __tablename__ = 'agent_configurations'
    
    # Primary identification
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    agent_id = Column(String(36), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Configuration metadata
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default='1.0.0')
    is_default = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Configuration data
    config_data = Column(JSON, nullable=False)
    parameter_overrides = Column(JSON, nullable=True)
    environment_variables = Column(JSON, nullable=True)
    resource_limits = Column(JSON, nullable=True)
    
    # Validation and schema
    validation_schema = Column(JSON, nullable=True)
    last_validated = Column(DateTime(timezone=True), nullable=True)
    validation_errors = Column(JSON, nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, nullable=False, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    tags = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    agent = relationship("Agent", backref="configurations")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_config_name_version', 'agent_id', 'name', 'version'),
        UniqueConstraint('agent_id', 'name', 'version', name='unique_agent_config'),
    )
    
    def __repr__(self) -> str:
        return f"<AgentConfiguration(id={self.id}, name='{self.name}', version='{self.version}')>"
    
    @validates('config_data')
    def validate_config_data(self, key, config_data):
        """Validate configuration data is not empty"""
        if not config_data:
            raise ValidationError("Configuration data cannot be empty")
        return config_data
    
    def update_usage(self) -> None:
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
    
    def to_dict(self, include_config: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary representation"""
        data = {
            'id': str(self.id),
            'agent_id': str(self.agent_id),
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'is_default': self.is_default,
            'is_active': self.is_active,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if include_config:
            data.update({
                'config_data': self.config_data,
                'parameter_overrides': self.parameter_overrides,
                'environment_variables': self.environment_variables,
                'resource_limits': self.resource_limits
            })
        
        return data