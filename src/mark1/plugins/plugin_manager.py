#!/usr/bin/env python3
"""
Plugin Manager for Mark-1 Universal Plugin System

Manages the complete plugin lifecycle including:
- Plugin discovery and installation from GitHub repositories
- Plugin configuration and environment setup
- Plugin registration and metadata management
- Plugin execution and monitoring
"""

import asyncio
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import git
import structlog

from .base_plugin import PluginMetadata, PluginStatus, PluginType, ExecutionMode, PluginException
from .repository_analyzer import RepositoryAnalyzer, RepositoryProfile
from .plugin_adapter import UniversalPluginAdapter


logger = structlog.get_logger(__name__)


@dataclass
class PluginInstallationResult:
    """Result of plugin installation"""
    success: bool
    plugin_id: Optional[str] = None
    plugin_metadata: Optional[PluginMetadata] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    installation_time: float = 0.0


@dataclass
class PluginValidationResult:
    """Result of plugin validation"""
    valid: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PluginManager:
    """
    Manages the complete plugin lifecycle
    """
    
    def __init__(self, plugins_directory: Path, temp_directory: Optional[Path] = None):
        self.plugins_directory = Path(plugins_directory)
        self.temp_directory = Path(temp_directory) if temp_directory else Path(tempfile.gettempdir()) / "mark1_plugins"
        self.logger = structlog.get_logger(__name__)
        
        # Create directories
        self.plugins_directory.mkdir(parents=True, exist_ok=True)
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.repository_analyzer = RepositoryAnalyzer()
        
        # Plugin registry
        self._installed_plugins: Dict[str, PluginMetadata] = {}
        self._active_adapters: Dict[str, UniversalPluginAdapter] = {}
        
        # Installation tracking
        self._installation_locks: Set[str] = set()
    
    async def install_plugin_from_repository(
        self,
        repository_url: str,
        branch: str = "main",
        force_reinstall: bool = False
    ) -> PluginInstallationResult:
        """
        Install a plugin from a GitHub repository
        
        Args:
            repository_url: GitHub repository URL
            branch: Branch to clone (default: main)
            force_reinstall: Force reinstallation if plugin exists
            
        Returns:
            PluginInstallationResult with installation outcome
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting plugin installation", 
                           repository_url=repository_url, branch=branch)
            
            # Generate plugin ID from repository URL
            plugin_id = self._generate_plugin_id(repository_url)
            
            # Check if already installing
            if plugin_id in self._installation_locks:
                return PluginInstallationResult(
                    success=False,
                    error="Plugin installation already in progress"
                )
            
            # Check if already installed
            if plugin_id in self._installed_plugins and not force_reinstall:
                return PluginInstallationResult(
                    success=True,
                    plugin_id=plugin_id,
                    plugin_metadata=self._installed_plugins[plugin_id],
                    warnings=["Plugin already installed"]
                )
            
            # Lock installation
            self._installation_locks.add(plugin_id)
            
            try:
                # Step 1: Clone repository
                repo_path = await self._clone_repository(repository_url, branch, plugin_id)
                
                # Step 2: Analyze repository
                self.logger.info("Analyzing repository", plugin_id=plugin_id)
                repo_profile = await self.repository_analyzer.analyze_repository(repo_path, repository_url)
                
                # Step 3: Convert to plugin metadata
                plugin_metadata = await self._create_plugin_metadata(plugin_id, repo_profile, repo_path)
                
                # Step 4: Validate plugin
                validation_result = await self._validate_plugin(plugin_metadata, repo_path)
                if not validation_result.valid:
                    return PluginInstallationResult(
                        success=False,
                        error=f"Plugin validation failed: {'; '.join(validation_result.issues)}"
                    )
                
                # Step 5: Install plugin
                final_plugin_path = await self._install_plugin_files(repo_path, plugin_id)
                plugin_metadata.configuration["plugin_path"] = str(final_plugin_path)
                
                # Step 6: Register plugin
                await self._register_plugin(plugin_metadata)
                
                installation_time = (datetime.now() - start_time).total_seconds()
                
                self.logger.info("Plugin installation completed", 
                               plugin_id=plugin_id,
                               installation_time=installation_time)
                
                return PluginInstallationResult(
                    success=True,
                    plugin_id=plugin_id,
                    plugin_metadata=plugin_metadata,
                    installation_time=installation_time,
                    warnings=validation_result.recommendations
                )
            
            finally:
                # Release lock
                self._installation_locks.discard(plugin_id)
        
        except Exception as e:
            installation_time = (datetime.now() - start_time).total_seconds()
            self.logger.error("Plugin installation failed", 
                            repository_url=repository_url,
                            error=str(e),
                            installation_time=installation_time)
            
            return PluginInstallationResult(
                success=False,
                error=str(e),
                installation_time=installation_time
            )
    
    async def _clone_repository(self, repository_url: str, branch: str, plugin_id: str) -> Path:
        """Clone repository to temporary location"""
        try:
            clone_path = self.temp_directory / f"clone_{plugin_id}"
            
            # Remove existing clone if present
            if clone_path.exists():
                shutil.rmtree(clone_path)
            
            self.logger.info("Cloning repository", url=repository_url, path=str(clone_path))
            
            # Clone repository
            repo = git.Repo.clone_from(
                repository_url,
                clone_path,
                branch=branch,
                depth=1  # Shallow clone for faster download
            )
            
            self.logger.info("Repository cloned successfully", path=str(clone_path))
            return clone_path
        
        except Exception as e:
            self.logger.error("Failed to clone repository", url=repository_url, error=str(e))
            raise PluginException(f"Repository clone failed: {e}")
    
    async def _create_plugin_metadata(
        self,
        plugin_id: str,
        repo_profile: RepositoryProfile,
        repo_path: Path
    ) -> PluginMetadata:
        """Create plugin metadata from repository profile"""
        try:
            # Convert repository profile to plugin metadata
            metadata = PluginMetadata(
                plugin_id=plugin_id,
                name=repo_profile.name,
                description=repo_profile.description,
                version="1.0.0",  # Default version, could be extracted from tags
                author="Unknown",  # Could be extracted from git config
                repository_url=repo_profile.repository_url,
                plugin_type=repo_profile.plugin_type,
                capabilities=repo_profile.capabilities,
                dependencies=[dep.name for dep in repo_profile.dependencies],
                environment_variables={var.name: var.default_value or "" for var in repo_profile.environment_variables},
                execution_mode=repo_profile.execution_mode,
                entry_points=repo_profile.entry_points,
                status=PluginStatus.INSTALLING
            )
            
            # Add additional configuration
            metadata.configuration.update({
                "has_dockerfile": repo_profile.has_dockerfile,
                "has_tests": repo_profile.has_tests,
                "estimated_setup_time": repo_profile.estimated_setup_time,
                "primary_language": repo_profile.primary_language,
                "service_requirements": [
                    {"name": svc.name, "type": svc.type, "required": svc.required}
                    for svc in repo_profile.service_requirements
                ]
            })
            
            return metadata
        
        except Exception as e:
            self.logger.error("Failed to create plugin metadata", error=str(e))
            raise PluginException(f"Plugin metadata creation failed: {e}")
    
    async def _validate_plugin(self, metadata: PluginMetadata, repo_path: Path) -> PluginValidationResult:
        """Validate plugin before installation"""
        issues = []
        recommendations = []
        
        try:
            # Check if entry points exist
            if not metadata.entry_points:
                issues.append("No entry points found")
            else:
                for entry_name, entry_path in metadata.entry_points.items():
                    full_path = repo_path / entry_path
                    if not full_path.exists():
                        issues.append(f"Entry point not found: {entry_path}")
            
            # Check for required files based on plugin type
            if metadata.plugin_type == PluginType.PYTHON_LIBRARY:
                if not any((repo_path / f).exists() for f in ["setup.py", "pyproject.toml"]):
                    recommendations.append("Consider adding setup.py or pyproject.toml for better Python package management")
            
            # Check for documentation
            if not metadata.description or len(metadata.description) < 10:
                recommendations.append("Consider adding a more detailed description")
            
            # Check for tests
            if not metadata.configuration.get("has_tests", False):
                recommendations.append("Consider adding tests for better reliability")
            
            # Validate dependencies
            if len(metadata.dependencies) > 50:
                recommendations.append("Large number of dependencies may increase installation time")
            
            return PluginValidationResult(
                valid=len(issues) == 0,
                issues=issues,
                recommendations=recommendations
            )
        
        except Exception as e:
            self.logger.error("Plugin validation failed", error=str(e))
            return PluginValidationResult(
                valid=False,
                issues=[f"Validation error: {e}"]
            )

    async def _install_plugin_files(self, repo_path: Path, plugin_id: str) -> Path:
        """Install plugin files to permanent location"""
        try:
            plugin_path = self.plugins_directory / plugin_id

            # Remove existing installation
            if plugin_path.exists():
                shutil.rmtree(plugin_path)

            # Copy repository to plugin directory
            shutil.copytree(repo_path, plugin_path)

            self.logger.info("Plugin files installed", plugin_id=plugin_id, path=str(plugin_path))
            return plugin_path

        except Exception as e:
            self.logger.error("Failed to install plugin files", plugin_id=plugin_id, error=str(e))
            raise PluginException(f"Plugin file installation failed: {e}")

    async def _register_plugin(self, metadata: PluginMetadata) -> None:
        """Register plugin in the system"""
        try:
            metadata.status = PluginStatus.READY
            metadata.updated_at = datetime.now(timezone.utc)

            self._installed_plugins[metadata.plugin_id] = metadata

            self.logger.info("Plugin registered", plugin_id=metadata.plugin_id)

        except Exception as e:
            self.logger.error("Failed to register plugin", plugin_id=metadata.plugin_id, error=str(e))
            raise PluginException(f"Plugin registration failed: {e}")

    async def get_plugin_adapter(self, plugin_id: str) -> Optional[UniversalPluginAdapter]:
        """Get or create plugin adapter"""
        try:
            # Return existing adapter if available
            if plugin_id in self._active_adapters:
                return self._active_adapters[plugin_id]

            # Check if plugin is installed
            if plugin_id not in self._installed_plugins:
                self.logger.warning("Plugin not found", plugin_id=plugin_id)
                return None

            metadata = self._installed_plugins[plugin_id]
            plugin_path = Path(metadata.configuration["plugin_path"])

            # Create new adapter
            adapter = UniversalPluginAdapter(metadata, plugin_path)

            # Initialize adapter
            if await adapter.initialize():
                self._active_adapters[plugin_id] = adapter
                return adapter
            else:
                self.logger.error("Failed to initialize plugin adapter", plugin_id=plugin_id)
                return None

        except Exception as e:
            self.logger.error("Failed to get plugin adapter", plugin_id=plugin_id, error=str(e))
            return None

    async def list_installed_plugins(self) -> List[PluginMetadata]:
        """List all installed plugins"""
        return list(self._installed_plugins.values())

    async def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get metadata for a specific plugin"""
        return self._installed_plugins.get(plugin_id)

    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin"""
        try:
            if plugin_id not in self._installed_plugins:
                self.logger.warning("Plugin not found for uninstallation", plugin_id=plugin_id)
                return False

            # Cleanup active adapter
            if plugin_id in self._active_adapters:
                await self._active_adapters[plugin_id].cleanup()
                del self._active_adapters[plugin_id]

            # Remove plugin files
            metadata = self._installed_plugins[plugin_id]
            plugin_path = Path(metadata.configuration["plugin_path"])
            if plugin_path.exists():
                shutil.rmtree(plugin_path)

            # Remove from registry
            del self._installed_plugins[plugin_id]

            self.logger.info("Plugin uninstalled", plugin_id=plugin_id)
            return True

        except Exception as e:
            self.logger.error("Failed to uninstall plugin", plugin_id=plugin_id, error=str(e))
            return False

    async def cleanup(self) -> None:
        """Cleanup plugin manager resources"""
        try:
            self.logger.info("Cleaning up plugin manager")

            # Cleanup all active adapters
            for adapter in self._active_adapters.values():
                try:
                    await adapter.cleanup()
                except:
                    pass

            self._active_adapters.clear()

            # Clean up temp directory
            if self.temp_directory.exists():
                try:
                    shutil.rmtree(self.temp_directory)
                except:
                    pass

            self.logger.info("Plugin manager cleanup completed")

        except Exception as e:
            self.logger.error("Plugin manager cleanup failed", error=str(e))

    def _generate_plugin_id(self, repository_url: str) -> str:
        """Generate unique plugin ID from repository URL"""
        # Extract repository name from URL
        repo_name = repository_url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]

        # Create unique ID
        return f"{repo_name}_{uuid.uuid4().hex[:8]}"

    def _create_plugin_metadata_from_profile(self, repo_profile: RepositoryProfile, plugin_directory: Path) -> PluginMetadata:
        """Create plugin metadata from repository profile for local installation"""
        try:
            # Convert repository profile to plugin metadata
            metadata = PluginMetadata(
                plugin_id="",  # Will be set later
                name=repo_profile.name,
                description=repo_profile.description,
                version="1.0.0",  # Default version
                author="Local Plugin",
                repository_url=f"file://{plugin_directory}",
                plugin_type=repo_profile.plugin_type,
                capabilities=repo_profile.capabilities,
                dependencies=[dep.name for dep in repo_profile.dependencies],
                environment_variables={var.name: var.default_value or "" for var in repo_profile.environment_variables},
                execution_mode=repo_profile.execution_mode,
                entry_points=repo_profile.entry_points,
                status=PluginStatus.INSTALLING
            )

            # Add additional configuration
            metadata.configuration.update({
                "has_dockerfile": repo_profile.has_dockerfile,
                "has_tests": repo_profile.has_tests,
                "estimated_setup_time": repo_profile.estimated_setup_time,
                "primary_language": repo_profile.primary_language,
                "service_requirements": [
                    {"name": svc.name, "type": svc.type, "required": svc.required}
                    for svc in repo_profile.service_requirements
                ]
            })

            return metadata

        except Exception as e:
            self.logger.error("Failed to create plugin metadata from profile", error=str(e))
            raise PluginException(f"Plugin metadata creation failed: {e}")

    async def install_plugin_from_local_directory(
        self,
        plugin_directory: Path,
        plugin_name: Optional[str] = None
    ) -> PluginInstallationResult:
        """
        Install a plugin from a local directory

        Args:
            plugin_directory: Path to the local plugin directory
            plugin_name: Optional custom name for the plugin

        Returns:
            PluginInstallationResult with installation details
        """
        start_time = time.time()

        try:
            self.logger.info("Installing plugin from local directory",
                           directory=str(plugin_directory))

            if not plugin_directory.exists():
                return PluginInstallationResult(
                    success=False,
                    error=f"Plugin directory does not exist: {plugin_directory}",
                    installation_time=time.time() - start_time
                )

            if not plugin_directory.is_dir():
                return PluginInstallationResult(
                    success=False,
                    error=f"Path is not a directory: {plugin_directory}",
                    installation_time=time.time() - start_time
                )

            # We'll analyze the directory directly

            # Analyze the plugin directory
            analyzer = RepositoryAnalyzer()
            repo_profile = await analyzer.analyze_repository(plugin_directory, f"file://{plugin_directory}")

            if not repo_profile:
                return PluginInstallationResult(
                    success=False,
                    error="Plugin analysis failed: Could not analyze repository",
                    installation_time=time.time() - start_time
                )

            # Convert repository profile to plugin metadata
            plugin_metadata = self._create_plugin_metadata_from_profile(repo_profile, plugin_directory)

            # Generate unique plugin ID
            plugin_id = f"{plugin_metadata.name.lower().replace(' ', '_')}_{int(time.time())}"
            plugin_metadata.plugin_id = plugin_id

            # Set up plugin directory in our plugins folder
            plugin_install_dir = self.plugins_directory / plugin_id
            plugin_install_dir.mkdir(parents=True, exist_ok=True)

            # Copy plugin files
            import shutil
            for item in plugin_directory.iterdir():
                if item.is_file():
                    shutil.copy2(item, plugin_install_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, plugin_install_dir / item.name, dirs_exist_ok=True)

            # Update plugin metadata with installation path
            plugin_metadata.configuration["plugin_path"] = str(plugin_install_dir)
            plugin_metadata.configuration["source_type"] = "local_directory"
            plugin_metadata.configuration["source_path"] = str(plugin_directory)

            # Register the plugin
            await self._register_plugin(plugin_metadata)

            installation_time = time.time() - start_time

            self.logger.info("Plugin installed successfully from local directory",
                           plugin_id=plugin_id,
                           name=plugin_metadata.name,
                           installation_time=installation_time)

            return PluginInstallationResult(
                success=True,
                plugin_id=plugin_id,
                plugin_metadata=plugin_metadata,
                installation_time=installation_time
            )

        except Exception as e:
            installation_time = time.time() - start_time
            self.logger.error("Failed to install plugin from local directory",
                            directory=str(plugin_directory),
                            error=str(e))

            return PluginInstallationResult(
                success=False,
                error=f"Installation failed: {e}",
                installation_time=installation_time
            )

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, '_background_task') and self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
