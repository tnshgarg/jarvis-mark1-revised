#!/usr/bin/env python3
"""
Repository Analyzer for Mark-1 Universal Plugin System

Analyzes GitHub repositories to extract plugin metadata, capabilities,
dependencies, and configuration requirements.
"""

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import toml
import structlog

from .base_plugin import PluginType, PluginCapability, PluginMetadata, ExecutionMode


logger = structlog.get_logger(__name__)


@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    version: Optional[str] = None
    source: str = ""  # requirements.txt, package.json, etc.
    optional: bool = False


@dataclass
class EnvironmentVariable:
    """Environment variable requirement"""
    name: str
    description: str = ""
    required: bool = True
    default_value: Optional[str] = None
    validation_pattern: Optional[str] = None


@dataclass
class ServiceRequirement:
    """External service requirement"""
    name: str
    type: str  # database, api, storage, etc.
    description: str = ""
    required: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryProfile:
    """Complete profile of a repository for plugin conversion"""
    name: str
    description: str
    repository_url: str
    primary_language: str
    plugin_type: PluginType
    execution_mode: ExecutionMode
    capabilities: List[PluginCapability] = field(default_factory=list)
    dependencies: List[DependencyInfo] = field(default_factory=list)
    environment_variables: List[EnvironmentVariable] = field(default_factory=list)
    service_requirements: List[ServiceRequirement] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    configuration_files: List[str] = field(default_factory=list)
    readme_content: str = ""
    has_dockerfile: bool = False
    has_tests: bool = False
    estimated_setup_time: int = 300  # seconds


class RepositoryAnalyzer:
    """
    Analyzes repositories to extract plugin metadata and requirements
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # File patterns for different languages and frameworks
        self.language_patterns = {
            "python": ["*.py", "requirements.txt", "setup.py", "pyproject.toml"],
            "javascript": ["*.js", "package.json", "*.ts"],
            "go": ["*.go", "go.mod", "go.sum"],
            "rust": ["*.rs", "Cargo.toml"],
            "java": ["*.java", "pom.xml", "build.gradle"],
        }
        
        # Common entry point patterns
        self.entry_point_patterns = {
            "python": ["main.py", "app.py", "__main__.py", "cli.py"],
            "javascript": ["index.js", "main.js", "app.js", "cli.js"],
            "go": ["main.go", "cmd/main.go"],
        }
    
    async def analyze_repository(self, repo_path: Path, repo_url: str = "") -> RepositoryProfile:
        """
        Perform comprehensive analysis of a repository
        
        Args:
            repo_path: Path to the cloned repository
            repo_url: Original repository URL
            
        Returns:
            RepositoryProfile with extracted metadata
        """
        try:
            self.logger.info("Starting repository analysis", path=str(repo_path))
            
            # Basic repository information
            name = repo_path.name
            description = await self._extract_description(repo_path)
            
            # Detect primary language and type
            primary_language = await self._detect_primary_language(repo_path)
            plugin_type = await self._infer_plugin_type(repo_path, primary_language)
            execution_mode = await self._determine_execution_mode(repo_path, plugin_type)
            
            # Extract capabilities
            capabilities = await self._extract_capabilities(repo_path, primary_language)
            
            # Analyze dependencies
            dependencies = await self._analyze_dependencies(repo_path, primary_language)
            
            # Detect environment variables
            env_vars = await self._detect_environment_variables(repo_path)
            
            # Detect service requirements
            services = await self._detect_service_requirements(repo_path)
            
            # Find entry points
            entry_points = await self._find_entry_points(repo_path, primary_language)
            
            # Additional metadata
            config_files = await self._find_configuration_files(repo_path)
            readme_content = await self._read_readme(repo_path)
            has_dockerfile = (repo_path / "Dockerfile").exists()
            has_tests = await self._detect_tests(repo_path)
            
            # Estimate setup complexity
            setup_time = await self._estimate_setup_time(
                dependencies, env_vars, services, has_dockerfile
            )
            
            profile = RepositoryProfile(
                name=name,
                description=description,
                repository_url=repo_url,
                primary_language=primary_language,
                plugin_type=plugin_type,
                execution_mode=execution_mode,
                capabilities=capabilities,
                dependencies=dependencies,
                environment_variables=env_vars,
                service_requirements=services,
                entry_points=entry_points,
                configuration_files=config_files,
                readme_content=readme_content,
                has_dockerfile=has_dockerfile,
                has_tests=has_tests,
                estimated_setup_time=setup_time
            )
            
            self.logger.info("Repository analysis completed", 
                           name=name, 
                           type=plugin_type.value,
                           capabilities_count=len(capabilities))
            
            return profile
            
        except Exception as e:
            self.logger.error("Repository analysis failed", path=str(repo_path), error=str(e))
            raise
    
    async def _extract_description(self, repo_path: Path) -> str:
        """Extract repository description from README or other sources"""
        try:
            # Try README files
            readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]
            for readme_file in readme_files:
                readme_path = repo_path / readme_file
                if readme_path.exists():
                    content = readme_path.read_text(encoding='utf-8', errors='ignore')
                    # Extract first paragraph or first line
                    lines = content.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            return line[:200]  # Limit description length
            
            # Try package.json for JavaScript projects
            package_json = repo_path / "package.json"
            if package_json.exists():
                try:
                    data = json.loads(package_json.read_text())
                    if "description" in data:
                        return data["description"]
                except:
                    pass
            
            # Try setup.py for Python projects
            setup_py = repo_path / "setup.py"
            if setup_py.exists():
                try:
                    content = setup_py.read_text()
                    # Simple regex to find description
                    match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
                except:
                    pass
            
            return f"Plugin from {repo_path.name}"
            
        except Exception as e:
            self.logger.warning("Failed to extract description", error=str(e))
            return f"Plugin from {repo_path.name}"
    
    async def _detect_primary_language(self, repo_path: Path) -> str:
        """Detect the primary programming language"""
        try:
            language_counts = {}
            
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix == ".py":
                        language_counts["python"] = language_counts.get("python", 0) + 1
                    elif suffix in [".js", ".ts"]:
                        language_counts["javascript"] = language_counts.get("javascript", 0) + 1
                    elif suffix == ".go":
                        language_counts["go"] = language_counts.get("go", 0) + 1
                    elif suffix == ".rs":
                        language_counts["rust"] = language_counts.get("rust", 0) + 1
                    elif suffix == ".java":
                        language_counts["java"] = language_counts.get("java", 0) + 1
            
            if language_counts:
                return max(language_counts, key=language_counts.get)
            
            # Check for specific files
            if (repo_path / "package.json").exists():
                return "javascript"
            elif (repo_path / "requirements.txt").exists() or (repo_path / "setup.py").exists():
                return "python"
            elif (repo_path / "go.mod").exists():
                return "go"
            elif (repo_path / "Cargo.toml").exists():
                return "rust"
            
            return "unknown"
            
        except Exception as e:
            self.logger.warning("Failed to detect primary language", error=str(e))
            return "unknown"

    async def _infer_plugin_type(self, repo_path: Path, language: str) -> PluginType:
        """Infer the plugin type based on repository structure"""
        try:
            # Check for specific indicators
            if (repo_path / "Dockerfile").exists():
                return PluginType.CONTAINER

            # Check for web service indicators
            web_indicators = ["app.py", "server.py", "main.py", "index.js", "server.js"]
            for indicator in web_indicators:
                if (repo_path / indicator).exists():
                    content = (repo_path / indicator).read_text(errors='ignore')
                    if any(framework in content.lower() for framework in
                          ["flask", "fastapi", "express", "django", "gin"]):
                        return PluginType.WEB_SERVICE

            # Check for CLI tool indicators
            cli_indicators = ["cli.py", "main.py", "__main__.py", "bin/"]
            for indicator in cli_indicators:
                path = repo_path / indicator
                if path.exists():
                    if path.is_file():
                        content = path.read_text(errors='ignore')
                        if any(cli_lib in content.lower() for cli_lib in
                              ["argparse", "click", "typer", "commander"]):
                            return PluginType.CLI_TOOL
                    elif path.is_dir():
                        return PluginType.CLI_TOOL

            # Check for AI agent indicators
            ai_indicators = ["agent", "langchain", "autogpt", "crew"]
            for file_path in repo_path.rglob("*.py"):
                try:
                    content = file_path.read_text(errors='ignore').lower()
                    if any(indicator in content for indicator in ai_indicators):
                        return PluginType.AI_AGENT
                except:
                    continue

            # Default based on language
            if language == "python":
                return PluginType.PYTHON_LIBRARY
            else:
                return PluginType.CLI_TOOL

        except Exception as e:
            self.logger.warning("Failed to infer plugin type", error=str(e))
            return PluginType.UNKNOWN

    async def _determine_execution_mode(self, repo_path: Path, plugin_type: PluginType) -> ExecutionMode:
        """Determine the best execution mode for the plugin"""
        if plugin_type == PluginType.CONTAINER:
            return ExecutionMode.CONTAINER
        elif plugin_type == PluginType.WEB_SERVICE:
            return ExecutionMode.HTTP_API
        elif plugin_type == PluginType.PYTHON_LIBRARY:
            return ExecutionMode.PYTHON_FUNCTION
        else:
            return ExecutionMode.SUBPROCESS

    async def _extract_capabilities(self, repo_path: Path, language: str) -> List[PluginCapability]:
        """Extract capabilities from repository analysis"""
        capabilities = []

        try:
            # Analyze Python files for function definitions
            if language == "python":
                capabilities.extend(await self._extract_python_capabilities(repo_path))

            # Analyze CLI help text
            capabilities.extend(await self._extract_cli_capabilities(repo_path))

            # Analyze README for documented features
            capabilities.extend(await self._extract_readme_capabilities(repo_path))

        except Exception as e:
            self.logger.warning("Failed to extract capabilities", error=str(e))

        return capabilities

    async def _extract_python_capabilities(self, repo_path: Path) -> List[PluginCapability]:
        """Extract capabilities from Python code analysis"""
        capabilities = []

        try:
            for py_file in repo_path.rglob("*.py"):
                if py_file.name.startswith("test_") or "/test" in str(py_file):
                    continue

                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Skip private functions
                            if node.name.startswith("_"):
                                continue

                            # Extract function info
                            docstring = ast.get_docstring(node)
                            if docstring:
                                capability = PluginCapability(
                                    name=node.name,
                                    description=docstring.split('\n')[0][:100],
                                    input_types=["any"],
                                    output_types=["any"]
                                )
                                capabilities.append(capability)

                except Exception as e:
                    self.logger.debug("Failed to parse Python file", file=str(py_file), error=str(e))
                    continue

        except Exception as e:
            self.logger.warning("Failed to extract Python capabilities", error=str(e))

        return capabilities[:10]  # Limit to prevent overwhelming

    async def _extract_cli_capabilities(self, repo_path: Path) -> List[PluginCapability]:
        """Extract capabilities from CLI analysis"""
        capabilities = []

        # This would involve running --help commands and parsing output
        # For now, return basic capability
        main_files = ["main.py", "cli.py", "app.py"]
        for main_file in main_files:
            if (repo_path / main_file).exists():
                capabilities.append(PluginCapability(
                    name="execute",
                    description="Execute the main functionality",
                    input_types=["text", "file"],
                    output_types=["text", "file"]
                ))
                break

        return capabilities

    async def _extract_readme_capabilities(self, repo_path: Path) -> List[PluginCapability]:
        """Extract capabilities from README documentation"""
        capabilities = []

        try:
            readme_content = await self._read_readme(repo_path)
            if not readme_content:
                return capabilities

            # Look for usage examples or feature lists
            lines = readme_content.lower().split('\n')
            for line in lines:
                if any(keyword in line for keyword in ["usage:", "features:", "commands:"]):
                    # Extract next few lines as potential capabilities
                    break

            # For now, add a generic capability based on README
            if readme_content:
                capabilities.append(PluginCapability(
                    name="process",
                    description="Process data according to repository functionality",
                    input_types=["any"],
                    output_types=["any"]
                ))

        except Exception as e:
            self.logger.warning("Failed to extract README capabilities", error=str(e))

        return capabilities

    async def _analyze_dependencies(self, repo_path: Path, language: str) -> List[DependencyInfo]:
        """Analyze repository dependencies"""
        dependencies = []

        try:
            if language == "python":
                dependencies.extend(await self._analyze_python_dependencies(repo_path))
            elif language == "javascript":
                dependencies.extend(await self._analyze_javascript_dependencies(repo_path))
            elif language == "go":
                dependencies.extend(await self._analyze_go_dependencies(repo_path))

        except Exception as e:
            self.logger.warning("Failed to analyze dependencies", error=str(e))

        return dependencies

    async def _analyze_python_dependencies(self, repo_path: Path) -> List[DependencyInfo]:
        """Analyze Python dependencies"""
        dependencies = []

        # Check requirements.txt
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text()
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse requirement line
                        if '>=' in line:
                            name, version = line.split('>=', 1)
                            dependencies.append(DependencyInfo(
                                name=name.strip(),
                                version=f">={version.strip()}",
                                source="requirements.txt"
                            ))
                        elif '==' in line:
                            name, version = line.split('==', 1)
                            dependencies.append(DependencyInfo(
                                name=name.strip(),
                                version=f"=={version.strip()}",
                                source="requirements.txt"
                            ))
                        else:
                            dependencies.append(DependencyInfo(
                                name=line,
                                source="requirements.txt"
                            ))
            except Exception as e:
                self.logger.warning("Failed to parse requirements.txt", error=str(e))

        # Check pyproject.toml
        pyproject_file = repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                content = toml.load(pyproject_file)
                if "tool" in content and "poetry" in content["tool"]:
                    deps = content["tool"]["poetry"].get("dependencies", {})
                    for name, version in deps.items():
                        if name != "python":
                            dependencies.append(DependencyInfo(
                                name=name,
                                version=str(version) if version else None,
                                source="pyproject.toml"
                            ))
            except Exception as e:
                self.logger.warning("Failed to parse pyproject.toml", error=str(e))

        return dependencies

    async def _analyze_javascript_dependencies(self, repo_path: Path) -> List[DependencyInfo]:
        """Analyze JavaScript dependencies"""
        dependencies = []

        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text())

                # Regular dependencies
                for name, version in data.get("dependencies", {}).items():
                    dependencies.append(DependencyInfo(
                        name=name,
                        version=version,
                        source="package.json"
                    ))

                # Dev dependencies
                for name, version in data.get("devDependencies", {}).items():
                    dependencies.append(DependencyInfo(
                        name=name,
                        version=version,
                        source="package.json",
                        optional=True
                    ))

            except Exception as e:
                self.logger.warning("Failed to parse package.json", error=str(e))

        return dependencies

    async def _analyze_go_dependencies(self, repo_path: Path) -> List[DependencyInfo]:
        """Analyze Go dependencies"""
        dependencies = []

        go_mod = repo_path / "go.mod"
        if go_mod.exists():
            try:
                content = go_mod.read_text()
                in_require = False

                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith("require"):
                        in_require = True
                        continue
                    elif line == ")":
                        in_require = False
                        continue
                    elif in_require and line:
                        parts = line.split()
                        if len(parts) >= 2:
                            dependencies.append(DependencyInfo(
                                name=parts[0],
                                version=parts[1],
                                source="go.mod"
                            ))

            except Exception as e:
                self.logger.warning("Failed to parse go.mod", error=str(e))

        return dependencies

    async def _detect_environment_variables(self, repo_path: Path) -> List[EnvironmentVariable]:
        """Detect required environment variables"""
        env_vars = []

        try:
            # Check .env.example files
            env_example_files = [".env.example", ".env.sample", "env.example"]
            for env_file in env_example_files:
                env_path = repo_path / env_file
                if env_path.exists():
                    try:
                        content = env_path.read_text()
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                name, default = line.split('=', 1)
                                env_vars.append(EnvironmentVariable(
                                    name=name.strip(),
                                    description=f"Environment variable from {env_file}",
                                    default_value=default.strip() if default.strip() else None
                                ))
                    except Exception as e:
                        self.logger.warning("Failed to parse env file", file=env_file, error=str(e))

            # Scan code for os.environ usage
            for py_file in repo_path.rglob("*.py"):
                try:
                    content = py_file.read_text(errors='ignore')
                    # Simple regex to find os.environ usage
                    env_matches = re.findall(r'os\.environ\.get\(["\']([^"\']+)["\']', content)
                    for match in env_matches:
                        if not any(ev.name == match for ev in env_vars):
                            env_vars.append(EnvironmentVariable(
                                name=match,
                                description=f"Environment variable found in {py_file.name}",
                                required=True
                            ))
                except:
                    continue

        except Exception as e:
            self.logger.warning("Failed to detect environment variables", error=str(e))

        return env_vars[:20]  # Limit to prevent overwhelming

    async def _detect_service_requirements(self, repo_path: Path) -> List[ServiceRequirement]:
        """Detect external service requirements"""
        services = []

        try:
            # Check for database usage
            db_patterns = {
                "postgresql": ["psycopg2", "asyncpg", "postgresql"],
                "mysql": ["mysql", "pymysql", "aiomysql"],
                "mongodb": ["pymongo", "motor", "mongodb"],
                "redis": ["redis", "aioredis"],
                "sqlite": ["sqlite3", "sqlite"]
            }

            # Scan Python files for database imports
            for py_file in repo_path.rglob("*.py"):
                try:
                    content = py_file.read_text(errors='ignore').lower()
                    for db_type, patterns in db_patterns.items():
                        if any(pattern in content for pattern in patterns):
                            if not any(s.name == db_type for s in services):
                                services.append(ServiceRequirement(
                                    name=db_type,
                                    type="database",
                                    description=f"{db_type.title()} database",
                                    required=True
                                ))
                except:
                    continue

            # Check for API service usage
            api_patterns = ["openai", "anthropic", "google", "aws", "azure"]
            for py_file in repo_path.rglob("*.py"):
                try:
                    content = py_file.read_text(errors='ignore').lower()
                    for api in api_patterns:
                        if api in content and "import" in content:
                            if not any(s.name == api for s in services):
                                services.append(ServiceRequirement(
                                    name=api,
                                    type="api",
                                    description=f"{api.title()} API service",
                                    required=True
                                ))
                except:
                    continue

        except Exception as e:
            self.logger.warning("Failed to detect service requirements", error=str(e))

        return services

    async def _find_entry_points(self, repo_path: Path, language: str) -> Dict[str, str]:
        """Find entry points for the plugin"""
        entry_points = {}

        try:
            if language in self.entry_point_patterns:
                for pattern in self.entry_point_patterns[language]:
                    entry_path = repo_path / pattern
                    if entry_path.exists():
                        entry_points["main"] = pattern
                        break

            # Check for CLI entry points
            cli_files = ["cli.py", "main.py", "__main__.py"]
            for cli_file in cli_files:
                if (repo_path / cli_file).exists():
                    entry_points["cli"] = cli_file
                    break

        except Exception as e:
            self.logger.warning("Failed to find entry points", error=str(e))

        return entry_points

    async def _find_configuration_files(self, repo_path: Path) -> List[str]:
        """Find configuration files"""
        config_files = []

        config_patterns = [
            "*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.cfg",
            ".env*", "config.*", "settings.*"
        ]

        try:
            for pattern in config_patterns:
                for config_file in repo_path.glob(pattern):
                    if config_file.is_file():
                        config_files.append(str(config_file.relative_to(repo_path)))

        except Exception as e:
            self.logger.warning("Failed to find configuration files", error=str(e))

        return config_files

    async def _read_readme(self, repo_path: Path) -> str:
        """Read README content"""
        readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]

        for readme_file in readme_files:
            readme_path = repo_path / readme_file
            if readme_path.exists():
                try:
                    return readme_path.read_text(encoding='utf-8', errors='ignore')
                except:
                    continue

        return ""

    async def _detect_tests(self, repo_path: Path) -> bool:
        """Check if repository has tests"""
        test_indicators = [
            "test/", "tests/", "spec/", "__tests__/",
            "test_*.py", "*_test.py", "*.test.js", "*.spec.js"
        ]

        for indicator in test_indicators:
            if "*" in indicator:
                if list(repo_path.rglob(indicator)):
                    return True
            else:
                if (repo_path / indicator).exists():
                    return True

        return False

    async def _estimate_setup_time(
        self,
        dependencies: List[DependencyInfo],
        env_vars: List[EnvironmentVariable],
        services: List[ServiceRequirement],
        has_dockerfile: bool
    ) -> int:
        """Estimate setup time in seconds"""
        base_time = 60  # 1 minute base

        # Add time for dependencies
        base_time += len(dependencies) * 10

        # Add time for environment variables
        base_time += len(env_vars) * 30

        # Add time for services
        base_time += len(services) * 120  # 2 minutes per service

        # Docker reduces setup time
        if has_dockerfile:
            base_time = max(base_time // 2, 120)  # At least 2 minutes

        return min(base_time, 1800)  # Cap at 30 minutes
