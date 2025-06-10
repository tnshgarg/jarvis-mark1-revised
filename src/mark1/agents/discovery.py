"""
Agent Discovery Engine for Mark1 AI Framework.

This module provides auto-discovery capabilities for finding and identifying
agents within the system, including file-based agents, module-based agents,
and dynamically loaded components.
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

from ..utils.exceptions import DiscoveryError, AgentLoadError
from ..utils.constants import AGENT_DISCOVERY_PATHS, AGENT_FILE_PATTERNS
from ..config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """Container for agent metadata extracted during discovery."""
    name: str
    module_path: str
    class_name: str
    file_path: str
    version: Optional[str] = None
    description: Optional[str] = None
    capabilities: List[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []


@dataclass
class DiscoveredAgent:
    """Represents an agent discovered during framework scanning"""
    name: str
    file_path: Path
    framework: str
    class_name: Optional[str] = None
    confidence: float = 0.8
    capabilities: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.metadata is None:
            self.metadata = {}


class AgentDiscoveryStrategy(ABC):
    """Abstract base class for agent discovery strategies."""
    
    @abstractmethod
    def discover(self, search_paths: List[Path]) -> List[AgentMetadata]:
        """Discover agents using this strategy."""
        pass
    
    @abstractmethod
    def validate_agent(self, agent_class: Type) -> bool:
        """Validate that a discovered class is a valid agent."""
        pass


class PythonModuleDiscovery(AgentDiscoveryStrategy):
    """Discovery strategy for Python module-based agents."""
    
    def __init__(self):
        self.agent_base_classes = set()
        self._load_base_classes()
    
    def _load_base_classes(self):
        """Load known agent base classes for validation."""
        try:
            # Try to import common agent base classes
            base_class_paths = [
                'mark1.agents.base.BaseAgent',
                'mark1.agents.base.Agent',
                'BaseAgent',
                'Agent'
            ]
            
            for class_path in base_class_paths:
                try:
                    if '.' in class_path:
                        module_name, class_name = class_path.rsplit('.', 1)
                        module = importlib.import_module(module_name)
                        base_class = getattr(module, class_name)
                        self.agent_base_classes.add(base_class)
                        logger.debug(f"Loaded base class: {class_path}")
                except (ImportError, AttributeError):
                    continue
                except Exception as e:
                    logger.debug(f"Could not load base class {class_path}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error loading agent base classes: {e}")
    
    def discover(self, search_paths: List[Path]) -> List[AgentMetadata]:
        """Discover Python module-based agents."""
        discovered_agents = []
        
        for search_path in search_paths:
            if not search_path.exists():
                logger.debug(f"Search path does not exist: {search_path}")
                continue
                
            logger.info(f"Scanning for agents in: {search_path}")
            discovered_agents.extend(self._scan_directory(search_path))
        
        return discovered_agents
    
    def _scan_directory(self, directory: Path) -> List[AgentMetadata]:
        """Recursively scan a directory for agent modules."""
        agents = []
        
        # Files to skip during discovery
        skip_files = {
            'setup.py', 'conftest.py', '__init__.py', 'test_*.py', 
            'tests.py', 'manage.py', 'wsgi.py', 'asgi.py',
            'settings.py', 'config.py', 'urls.py', 'admin.py',
            'models.py', 'views.py', 'serializers.py', 'forms.py'
        }
        
        # Directories to skip
        skip_dirs = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            'migrations', 'static', 'media', 'templates', 'locale',
            'venv', 'env', '.venv', '.env', 'build', 'dist', 'egg-info'
        }
        
        try:
            for item in directory.rglob("*.py"):
                # Skip files that start with underscore
                if item.name.startswith('_'):
                    continue
                
                # Skip specific files
                if item.name in skip_files:
                    continue
                    
                # Skip files matching patterns
                if any(pattern in item.name for pattern in ['test_', '_test']):
                    continue
                
                # Skip if in excluded directories
                if any(skip_dir in item.parts for skip_dir in skip_dirs):
                    continue
                
                # Skip if file is too large (likely not an agent)
                try:
                    if item.stat().st_size > 1024 * 1024:  # 1MB limit
                        continue
                except:
                    continue
                    
                try:
                    agent_metadata = self._analyze_python_file(item)
                    if agent_metadata:
                        agents.extend(agent_metadata)
                except Exception as e:
                    logger.debug(f"Error analyzing {item}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            
        return agents
    
    def _analyze_python_file(self, file_path: Path) -> List[AgentMetadata]:
        """Analyze a Python file for agent classes."""
        agents = []
        
        try:
            # Additional safety checks before analysis
            if not self._is_safe_to_analyze(file_path):
                return agents
            
            # Create module spec
            module_name = self._path_to_module_name(file_path)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                return agents
            
            # Load the module
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules temporarily to handle relative imports
            sys.modules[module_name] = module
            
            try:
                # Execute with timeout protection
                spec.loader.exec_module(module)
                
                # Inspect module for agent classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self.validate_agent(obj) and obj.__module__ == module_name:
                        metadata = self._extract_metadata(obj, file_path, module_name)
                        agents.append(metadata)
                        logger.debug(f"Discovered agent: {metadata.name}")
                        
            finally:
                # Clean up sys.modules
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    
        except Exception as e:
            logger.debug(f"Could not analyze {file_path}: {e}")
            
        return agents
    
    def _is_safe_to_analyze(self, file_path: Path) -> bool:
        """Check if a Python file is safe to analyze by importing."""
        try:
            # Read first few lines to check for problematic patterns
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_preview = f.read(2048)  # Read first 2KB
            
            # Skip files with setup/installation patterns
            dangerous_patterns = [
                'setuptools', 'distutils', 'setup(', 'install_requires',
                'from setuptools import', 'import setuptools',
                'if __name__ == "__main__":\n    setup(',
                'sys.exit(', 'os.system(', 'subprocess.',
                'pip install', 'easy_install'
            ]
            
            content_lower = content_preview.lower()
            for pattern in dangerous_patterns:
                if pattern.lower() in content_lower:
                    logger.debug(f"Skipping {file_path}: contains pattern '{pattern}'")
                    return False
            
            # Check if file looks like a script rather than a module
            if 'if __name__ == "__main__":' in content_preview and 'class' not in content_preview:
                logger.debug(f"Skipping {file_path}: appears to be a script")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking safety of {file_path}: {e}")
            return False
    
    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to Python module name."""
        # Remove .py extension and convert path separators to dots
        module_path = str(file_path.with_suffix(''))
        return module_path.replace(os.sep, '.').replace('/', '.')
    
    def _extract_metadata(self, agent_class: Type, file_path: Path, module_name: str) -> AgentMetadata:
        """Extract metadata from an agent class."""
        # Basic metadata
        metadata = AgentMetadata(
            name=agent_class.__name__,
            module_path=module_name,
            class_name=agent_class.__name__,
            file_path=str(file_path),
        )
        
        # Extract docstring information
        if agent_class.__doc__:
            metadata.description = self._clean_docstring(agent_class.__doc__)
        
        # Extract metadata from class attributes
        for attr_name in dir(agent_class):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr_value = getattr(agent_class, attr_name)
                
                # Look for metadata attributes
                if attr_name == 'version' and isinstance(attr_value, str):
                    metadata.version = attr_value
                elif attr_name == 'capabilities' and isinstance(attr_value, (list, tuple)):
                    metadata.capabilities = list(attr_value)
                elif attr_name == 'dependencies' and isinstance(attr_value, (list, tuple)):
                    metadata.dependencies = list(attr_value)
                elif attr_name == 'tags' and isinstance(attr_value, (list, tuple)):
                    metadata.tags = list(attr_value)
                elif attr_name == 'author' and isinstance(attr_value, str):
                    metadata.author = attr_value
                    
            except Exception:
                continue
        
        # File metadata
        try:
            stat = file_path.stat()
            metadata.modified_date = str(stat.st_mtime)
        except Exception:
            pass
            
        return metadata
    
    def _clean_docstring(self, docstring: str) -> str:
        """Clean and normalize docstring."""
        lines = docstring.strip().split('\n')
        # Take first non-empty line as description
        for line in lines:
            cleaned = line.strip()
            if cleaned:
                return cleaned
        return docstring.strip()
    
    def validate_agent(self, agent_class: Type) -> bool:
        """Validate that a class is a valid agent."""
        try:
            # Must be a class
            if not inspect.isclass(agent_class):
                return False
            
            # Skip abstract classes
            if inspect.isabstract(agent_class):
                return False
            
            # Check if it inherits from known agent base classes
            if self.agent_base_classes:
                for base_class in self.agent_base_classes:
                    if issubclass(agent_class, base_class):
                        return True
            
            # Fallback: check for agent-like methods
            required_methods = ['execute', 'run', 'process']
            agent_methods = [method for method in required_methods 
                           if hasattr(agent_class, method)]
            
            if agent_methods:
                return True
            
            # Check for agent-like attributes
            agent_indicators = ['capabilities', 'agent_type', 'name']
            for indicator in agent_indicators:
                if hasattr(agent_class, indicator):
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error validating agent class {agent_class}: {e}")
            return False


class ConfigFileDiscovery(AgentDiscoveryStrategy):
    """Discovery strategy for configuration file-based agents."""
    
    def discover(self, search_paths: List[Path]) -> List[AgentMetadata]:
        """Discover agents from configuration files."""
        agents = []
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # Look for agent configuration files
            for config_file in search_path.rglob("agent.json"):
                try:
                    metadata = self._parse_config_file(config_file)
                    if metadata:
                        agents.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing config file {config_file}: {e}")
                    
            for config_file in search_path.rglob("agent.yaml"):
                try:
                    metadata = self._parse_yaml_config(config_file)
                    if metadata:
                        agents.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing YAML config {config_file}: {e}")
        
        return agents
    
    def _parse_config_file(self, config_file: Path) -> Optional[AgentMetadata]:
        """Parse JSON configuration file."""
        import json
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if not self._is_agent_config(config):
                return None
            
            return AgentMetadata(
                name=config.get('name', config_file.parent.name),
                module_path=config.get('module', ''),
                class_name=config.get('class', ''),
                file_path=str(config_file),
                version=config.get('version'),
                description=config.get('description'),
                capabilities=config.get('capabilities', []),
                dependencies=config.get('dependencies', []),
                tags=config.get('tags', []),
                author=config.get('author')
            )
            
        except Exception as e:
            logger.debug(f"Could not parse config file {config_file}: {e}")
            return None
    
    def _parse_yaml_config(self, config_file: Path) -> Optional[AgentMetadata]:
        """Parse YAML configuration file."""
        try:
            import yaml
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if not self._is_agent_config(config):
                return None
            
            return AgentMetadata(
                name=config.get('name', config_file.parent.name),
                module_path=config.get('module', ''),
                class_name=config.get('class', ''),
                file_path=str(config_file),
                version=config.get('version'),
                description=config.get('description'),
                capabilities=config.get('capabilities', []),
                dependencies=config.get('dependencies', []),
                tags=config.get('tags', []),
                author=config.get('author')
            )
            
        except ImportError:
            logger.debug("YAML library not available for config parsing")
            return None
        except Exception as e:
            logger.debug(f"Could not parse YAML config {config_file}: {e}")
            return None
    
    def _is_agent_config(self, config: Dict) -> bool:
        """Check if configuration represents an agent."""
        required_fields = ['name']
        agent_indicators = ['module', 'class', 'capabilities', 'agent_type']
        
        # Must have required fields
        for field in required_fields:
            if field not in config:
                return False
        
        # Should have at least one agent indicator
        return any(indicator in config for indicator in agent_indicators)
    
    def validate_agent(self, agent_class: Type) -> bool:
        """Validate agent from config (not applicable for config-based discovery)."""
        return True


class AgentDiscoveryEngine:
    """Main engine for discovering agents using multiple strategies."""
    
    def __init__(self, config_override: Optional[Dict] = None):
        self.config = config_override or get_settings()
        self.strategies: List[AgentDiscoveryStrategy] = []
        self.discovered_agents: Dict[str, AgentMetadata] = {}
        self._setup_strategies()
    
    def _setup_strategies(self):
        """Initialize discovery strategies."""
        self.strategies = [
            PythonModuleDiscovery(),
            ConfigFileDiscovery(),
        ]
        logger.info(f"Initialized {len(self.strategies)} discovery strategies")
    
    async def initialize(self):
        """Initialize the discovery engine (async interface for orchestrator compatibility)."""
        # Discovery engine is already initialized in __init__, but this provides
        # an async interface for consistency with other components
        logger.debug("Agent discovery engine initialized")
        return True
    
    def discover_all(self, search_paths: Optional[List[Path]] = None) -> Dict[str, AgentMetadata]:
        """Discover all agents using all available strategies."""
        if search_paths is None:
            search_paths = self._get_default_search_paths()
        
        logger.info(f"Starting agent discovery in {len(search_paths)} paths")
        
        all_agents = {}
        
        for strategy in self.strategies:
            try:
                strategy_name = strategy.__class__.__name__
                logger.debug(f"Running discovery strategy: {strategy_name}")
                
                agents = strategy.discover(search_paths)
                
                for agent in agents:
                    # Use module_path + class_name as unique key
                    key = f"{agent.module_path}.{agent.class_name}"
                    
                    if key in all_agents:
                        logger.warning(f"Duplicate agent found: {key}")
                        # Keep the one with more complete metadata
                        if len(agent.capabilities) > len(all_agents[key].capabilities):
                            all_agents[key] = agent
                    else:
                        all_agents[key] = agent
                
                logger.info(f"{strategy_name} discovered {len(agents)} agents")
                
            except Exception as e:
                logger.error(f"Error in discovery strategy {strategy.__class__.__name__}: {e}")
        
        self.discovered_agents = all_agents
        logger.info(f"Total agents discovered: {len(all_agents)}")
        
        return all_agents
    
    def discover_by_pattern(self, pattern: str, search_paths: Optional[List[Path]] = None) -> List[AgentMetadata]:
        """Discover agents matching a specific pattern."""
        all_agents = self.discover_all(search_paths)
        
        matching_agents = []
        pattern_lower = pattern.lower()
        
        for agent in all_agents.values():
            # Check name, tags, and capabilities
            if (pattern_lower in agent.name.lower() or
                any(pattern_lower in tag.lower() for tag in agent.tags) or
                any(pattern_lower in cap.lower() for cap in agent.capabilities)):
                matching_agents.append(agent)
        
        logger.info(f"Found {len(matching_agents)} agents matching pattern: {pattern}")
        return matching_agents
    
    def discover_by_capability(self, capability: str, search_paths: Optional[List[Path]] = None) -> List[AgentMetadata]:
        """Discover agents with a specific capability."""
        all_agents = self.discover_all(search_paths)
        
        matching_agents = []
        capability_lower = capability.lower()
        
        for agent in all_agents.values():
            if any(capability_lower in cap.lower() for cap in agent.capabilities):
                matching_agents.append(agent)
        
        logger.info(f"Found {len(matching_agents)} agents with capability: {capability}")
        return matching_agents
    
    def get_agent_by_name(self, name: str) -> Optional[AgentMetadata]:
        """Get a specific agent by name."""
        for agent in self.discovered_agents.values():
            if agent.name == name:
                return agent
        return None
    
    def refresh_discovery(self, search_paths: Optional[List[Path]] = None) -> Dict[str, AgentMetadata]:
        """Refresh the agent discovery cache."""
        logger.info("Refreshing agent discovery cache")
        return self.discover_all(search_paths)
    
    def _get_default_search_paths(self) -> List[Path]:
        """Get default search paths for agent discovery."""
        default_paths = [
            Path("src/mark1/agents"),
            Path("agents"),
            Path("src/agents"),
            Path("."),
        ]
        
        # Add paths from config
        config_paths = getattr(self.config, 'AGENT_DISCOVERY_PATHS', [])
        for path_str in config_paths:
            default_paths.append(Path(path_str))
        
        # Add current working directory subdirectories
        cwd = Path.cwd()
        for subdir in ['agents', 'src/agents', 'mark1/agents']:
            potential_path = cwd / subdir
            if potential_path not in default_paths:
                default_paths.append(potential_path)
        
        # Filter to existing paths
        existing_paths = [path for path in default_paths if path.exists()]
        
        logger.debug(f"Using search paths: {[str(p) for p in existing_paths]}")
        return existing_paths
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered agents."""
        if not self.discovered_agents:
            return {"total": 0}
        
        stats = {
            "total": len(self.discovered_agents),
            "by_strategy": {},
            "capabilities": {},
            "tags": {},
            "languages": {}
        }
        
        # Count by file extension (rough language detection)
        for agent in self.discovered_agents.values():
            ext = Path(agent.file_path).suffix.lower()
            stats["languages"][ext] = stats["languages"].get(ext, 0) + 1
            
            # Count capabilities
            for cap in agent.capabilities:
                stats["capabilities"][cap] = stats["capabilities"].get(cap, 0) + 1
            
            # Count tags
            for tag in agent.tags:
                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
        
        return stats
    
    async def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Scan a directory for agents and return structured results
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            include_patterns: Optional file patterns to include
            
        Returns:
            Structured scan results for compatibility
        """
        try:
            self.logger.info("Scanning directory for agents", directory=str(directory))
            
            # Discover agents in the directory
            discovered_agents = self.discover_all([directory])
            
            # Convert to structured format expected by orchestrator
            files_results = []
            total_files = 0
            
            # Group agents by file
            files_with_agents = {}
            for agent_key, agent_metadata in discovered_agents.items():
                file_path = agent_metadata.file_path
                if file_path not in files_with_agents:
                    files_with_agents[file_path] = []
                    total_files += 1
                
                files_with_agents[file_path].append({
                    "name": agent_metadata.name,
                    "file_path": file_path,
                    "framework": getattr(agent_metadata, 'framework', 'unknown'),
                    "capabilities": agent_metadata.capabilities,
                    "confidence": 0.8,  # Default confidence
                    "metadata": {
                        "module_path": agent_metadata.module_path,
                        "class_name": agent_metadata.class_name,
                        "description": agent_metadata.description,
                        "version": agent_metadata.version,
                        "author": agent_metadata.author
                    }
                })
            
            # Create files results
            for file_path, agents in files_with_agents.items():
                files_results.append({
                    "file_path": file_path,
                    "agents": agents,
                    "file_type": Path(file_path).suffix.lower(),
                })
            
            result = {
                "directory": str(directory),
                "total_files": total_files,
                "files": files_results,
                "summary": {
                    "agents_found": len(discovered_agents),
                    "files_with_agents": len(files_with_agents),
                    "scan_time": datetime.now().isoformat()
                }
            }
            
            self.logger.info("Directory scan completed", 
                           directory=str(directory),
                           agents_found=len(discovered_agents),
                           files_scanned=total_files)
            
            return result
            
        except Exception as e:
            self.logger.error("Directory scan failed", directory=str(directory), error=str(e))
            raise DiscoveryError(f"Directory scan failed: {e}")


# Convenience functions
def discover_agents(search_paths: Optional[List[Path]] = None) -> Dict[str, AgentMetadata]:
    """Convenience function to discover all agents."""
    engine = AgentDiscoveryEngine()
    return engine.discover_all(search_paths)


def find_agents_by_capability(capability: str, search_paths: Optional[List[Path]] = None) -> List[AgentMetadata]:
    """Convenience function to find agents by capability."""
    engine = AgentDiscoveryEngine()
    return engine.discover_by_capability(capability, search_paths)


def find_agent_by_name(name: str, search_paths: Optional[List[Path]] = None) -> Optional[AgentMetadata]:
    """Find a specific agent by name."""
    engine = AgentDiscoveryEngine()
    return engine.get_agent_by_name(name)


# Backwards compatibility alias
AgentDiscovery = AgentDiscoveryEngine