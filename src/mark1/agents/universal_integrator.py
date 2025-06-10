#!/usr/bin/env python3
"""
Universal AI Agent Integrator

Automatically analyzes and integrates any AI agent repository into Mark-1 orchestrator.
Supports multiple frameworks, architectures, and agent types.
"""

import asyncio
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re
import ast
import importlib.util
import yaml

logger = logging.getLogger(__name__)


class AgentFramework(Enum):
    """Supported AI agent frameworks"""
    CREWAI = "crewai"
    AUTOGPT = "autogpt"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    OPENAI_ASSISTANT = "openai_assistant"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class AgentCapability(Enum):
    """Standard agent capabilities"""
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    WEB_SCRAPING = "web_scraping"
    FILE_ANALYSIS = "file_analysis"
    API_INTERACTION = "api_interaction"
    MULTI_AGENT = "multi_agent"
    MEMORY = "memory"
    TOOLS = "tools"
    PLANNING = "planning"
    REASONING = "reasoning"


@dataclass
class AgentMetadata:
    """Metadata extracted from agent repository"""
    name: str
    framework: AgentFramework
    version: str
    description: str
    capabilities: List[AgentCapability]
    dependencies: List[str]
    entry_points: List[str]
    api_endpoints: List[str]
    config_files: List[str]
    documentation: List[str]
    examples: List[str]
    tests: List[str]


@dataclass
class IntegrationPlan:
    """Plan for integrating an agent into Mark-1"""
    agent_metadata: AgentMetadata
    integration_strategy: str
    wrapper_class: str
    api_adapter: str
    config_mapping: Dict[str, str]
    dependency_resolution: List[str]
    test_commands: List[str]
    health_check: str


class UniversalAgentIntegrator:
    """Universal system for integrating any AI agent repository"""
    
    def __init__(self, mark1_root: Path):
        self.mark1_root = Path(mark1_root)
        self.agents_dir = self.mark1_root / "agents"
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Framework detection patterns
        self.framework_patterns = {
            AgentFramework.CREWAI: [
                r"from crewai import",
                r"import crewai",
                r"@agent",
                r"Crew\(",
                r"Task\(",
            ],
            AgentFramework.AUTOGPT: [
                r"from autogpt",
                r"import autogpt",
                r"AutoGPTAgent",
                r"forge",
            ],
            AgentFramework.LANGCHAIN: [
                r"from langchain",
                r"import langchain",
                r"BaseAgent",
                r"AgentExecutor",
            ],
            AgentFramework.LLAMAINDEX: [
                r"from llama_index",
                r"import llama_index",
                r"ServiceContext",
            ],
            AgentFramework.OPENAI_ASSISTANT: [
                r"openai\.beta\.assistants",
                r"Assistant\(",
                r"client\.beta\.assistants",
            ],
        }
        
        # Capability detection patterns
        self.capability_patterns = {
            AgentCapability.CHAT: [
                r"chat", r"conversation", r"dialogue", r"message"
            ],
            AgentCapability.CODE_GENERATION: [
                r"code", r"python", r"javascript", r"programming", r"generate"
            ],
            AgentCapability.WEB_SCRAPING: [
                r"scrape", r"crawl", r"web", r"requests", r"beautifulsoup"
            ],
            AgentCapability.FILE_ANALYSIS: [
                r"file", r"document", r"pdf", r"analysis", r"parse"
            ],
            AgentCapability.API_INTERACTION: [
                r"api", r"rest", r"http", r"endpoint", r"request"
            ],
            AgentCapability.MULTI_AGENT: [
                r"multi.*agent", r"team", r"crew", r"swarm", r"collaborate"
            ],
            AgentCapability.MEMORY: [
                r"memory", r"remember", r"context", r"history", r"store"
            ],
            AgentCapability.TOOLS: [
                r"tool", r"function", r"action", r"plugin", r"extension"
            ],
            AgentCapability.PLANNING: [
                r"plan", r"strategy", r"goal", r"task.*planning", r"workflow"
            ],
            AgentCapability.REASONING: [
                r"reason", r"logic", r"think", r"analyze", r"infer"
            ],
        }
        
        logger.info("Universal Agent Integrator initialized")

    async def integrate_repository(self, repo_url: str, clone_to: Optional[str] = None) -> IntegrationPlan:
        """Integrate any AI agent repository into Mark-1"""
        try:
            # Step 1: Clone repository
            clone_path = await self._clone_repository(repo_url, clone_to)
            
            # Step 2: Analyze repository
            metadata = await self._analyze_repository(clone_path)
            
            # Step 3: Create integration plan
            plan = await self._create_integration_plan(metadata, clone_path)
            
            # Step 4: Execute integration
            await self._execute_integration(plan, clone_path)
            
            # Step 5: Test integration
            test_results = await self._test_integration(plan)
            
            logger.info(f"Successfully integrated {metadata.name} ({metadata.framework.value})")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to integrate repository {repo_url}: {e}")
            raise

    async def _clone_repository(self, repo_url: str, clone_to: Optional[str] = None) -> Path:
        """Clone repository to local directory"""
        if clone_to:
            target_path = self.agents_dir / clone_to
        else:
            # Extract repo name from URL
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            target_path = self.agents_dir / repo_name
        
        # Remove existing directory if it exists
        if target_path.exists():
            shutil.rmtree(target_path)
        
        # Clone repository with timeout and depth limit for faster cloning
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout
        except subprocess.TimeoutExpired:
            raise Exception(f"Repository cloning timed out after 2 minutes. Repository may be too large.")
        
        if result.returncode != 0:
            raise Exception(f"Failed to clone repository: {result.stderr}")
        
        logger.info(f"Repository cloned to {target_path}")
        return target_path

    async def _analyze_repository(self, repo_path: Path) -> AgentMetadata:
        """Analyze repository structure and extract metadata"""
        try:
            # Detect framework
            framework = await self._detect_framework(repo_path)
            
            # Extract basic info
            name = repo_path.name
            description = await self._extract_description(repo_path)
            version = await self._extract_version(repo_path)
            
            # Detect capabilities
            capabilities = await self._detect_capabilities(repo_path)
            
            # Extract dependencies
            dependencies = await self._extract_dependencies(repo_path)
            
            # Find entry points
            entry_points = await self._find_entry_points(repo_path)
            
            # Find API endpoints
            api_endpoints = await self._find_api_endpoints(repo_path)
            
            # Find configuration files
            config_files = await self._find_config_files(repo_path)
            
            # Find documentation
            documentation = await self._find_documentation(repo_path)
            
            # Find examples
            examples = await self._find_examples(repo_path)
            
            # Find tests
            tests = await self._find_tests(repo_path)
            
            metadata = AgentMetadata(
                name=name,
                framework=framework,
                version=version,
                description=description,
                capabilities=capabilities,
                dependencies=dependencies,
                entry_points=entry_points,
                api_endpoints=api_endpoints,
                config_files=config_files,
                documentation=documentation,
                examples=examples,
                tests=tests
            )
            
            logger.info(f"Repository analysis complete: {name} ({framework.value})")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze repository {repo_path}: {e}")
            raise

    async def _detect_framework(self, repo_path: Path) -> AgentFramework:
        """Detect the AI framework used by the repository"""
        file_contents = []
        
        # Read Python files
        for py_file in repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    file_contents.append(f.read())
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Read requirements files
        for req_file in ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]:
            req_path = repo_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r', encoding='utf-8') as f:
                        file_contents.append(f.read())
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        # Check patterns
        all_content = "\n".join(file_contents)
        
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    return framework
        
        return AgentFramework.UNKNOWN

    async def _detect_capabilities(self, repo_path: Path) -> List[AgentCapability]:
        """Detect agent capabilities from code analysis"""
        capabilities = set()
        
        # Read all text files
        file_contents = []
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.py', '.md', '.txt', '.yml', '.yaml', '.json']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_contents.append(f.read().lower())
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        all_content = "\n".join(file_contents)
        
        # Check capability patterns
        for capability, patterns in self.capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    capabilities.add(capability)
        
        return list(capabilities)

    async def _extract_dependencies(self, repo_path: Path) -> List[str]:
        """Extract dependencies from various dependency files"""
        dependencies = []
        
        # Check requirements.txt
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dependencies.append(line)
        
        # Check pyproject.toml
        pyproject_file = repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)
                    deps = data.get('project', {}).get('dependencies', [])
                    dependencies.extend(deps)
            except ImportError:
                pass
        
        # Check setup.py (basic parsing)
        setup_file = repo_path / "setup.py"
        if setup_file.exists():
            with open(setup_file, 'r') as f:
                content = f.read()
                # Look for install_requires
                match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if match:
                    deps_str = match.group(1)
                    deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
                    dependencies.extend(deps)
        
        return dependencies

    async def _find_entry_points(self, repo_path: Path) -> List[str]:
        """Find main entry points in the repository"""
        entry_points = []
        
        # Common entry point patterns
        entry_patterns = [
            "main.py", "app.py", "run.py", "start.py", "server.py",
            "agent.py", "bot.py", "cli.py", "__main__.py"
        ]
        
        for pattern in entry_patterns:
            for file_path in repo_path.rglob(pattern):
                entry_points.append(str(file_path.relative_to(repo_path)))
        
        # Check for if __name__ == "__main__" blocks
        for py_file in repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'if __name__ == "__main__"' in content:
                        entry_points.append(str(py_file.relative_to(repo_path)))
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return list(set(entry_points))

    async def _find_api_endpoints(self, repo_path: Path) -> List[str]:
        """Find API endpoints in the repository"""
        api_endpoints = []
        
        # Look for Flask/FastAPI/Django patterns
        for py_file in repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Flask routes
                    flask_routes = re.findall(r'@app\.route\(["\']([^"\']+)["\']', content)
                    api_endpoints.extend(flask_routes)
                    
                    # FastAPI routes
                    fastapi_routes = re.findall(r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']', content)
                    api_endpoints.extend([route[1] for route in fastapi_routes])
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return list(set(api_endpoints))

    async def _extract_description(self, repo_path: Path) -> str:
        """Extract description from README or docstrings"""
        # Check README files
        for readme_name in ["README.md", "README.txt", "README.rst", "readme.md"]:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract first paragraph
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('!'):
                            return line[:200] + "..." if len(line) > 200 else line
        
        return "AI Agent Repository"

    async def _extract_version(self, repo_path: Path) -> str:
        """Extract version from various sources"""
        # Check pyproject.toml
        pyproject_file = repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)
                    version = data.get('project', {}).get('version')
                    if version:
                        return version
            except ImportError:
                pass
        
        # Check __version__ in __init__.py
        for init_file in repo_path.rglob("__init__.py"):
            try:
                with open(init_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return "unknown"

    async def _find_config_files(self, repo_path: Path) -> List[str]:
        """Find configuration files"""
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.cfg", ".env*", "config.*"]
        config_files = []
        
        for pattern in config_patterns:
            for file_path in repo_path.rglob(pattern):
                if file_path.is_file():
                    config_files.append(str(file_path.relative_to(repo_path)))
        
        return config_files

    async def _find_documentation(self, repo_path: Path) -> List[str]:
        """Find documentation files"""
        doc_patterns = ["*.md", "*.rst", "*.txt"]
        doc_files = []
        
        for pattern in doc_patterns:
            for file_path in repo_path.rglob(pattern):
                if file_path.is_file() and any(keyword in file_path.name.lower() 
                                             for keyword in ['readme', 'doc', 'guide', 'manual']):
                    doc_files.append(str(file_path.relative_to(repo_path)))
        
        return doc_files

    async def _find_examples(self, repo_path: Path) -> List[str]:
        """Find example files"""
        example_dirs = ["examples", "samples", "demo", "demos"]
        examples = []
        
        for dir_name in example_dirs:
            example_dir = repo_path / dir_name
            if example_dir.exists() and example_dir.is_dir():
                for file_path in example_dir.rglob("*"):
                    if file_path.is_file():
                        examples.append(str(file_path.relative_to(repo_path)))
        
        return examples

    async def _find_tests(self, repo_path: Path) -> List[str]:
        """Find test files"""
        test_patterns = ["test_*.py", "*_test.py", "tests.py"]
        test_dirs = ["tests", "test", "testing"]
        tests = []
        
        # Find test files by pattern
        for pattern in test_patterns:
            for file_path in repo_path.rglob(pattern):
                if file_path.is_file():
                    tests.append(str(file_path.relative_to(repo_path)))
        
        # Find test directories
        for dir_name in test_dirs:
            test_dir = repo_path / dir_name
            if test_dir.exists() and test_dir.is_dir():
                for file_path in test_dir.rglob("*.py"):
                    if file_path.is_file():
                        tests.append(str(file_path.relative_to(repo_path)))
        
        return tests

    async def _create_integration_plan(self, metadata: AgentMetadata, repo_path: Path) -> IntegrationPlan:
        """Create integration plan based on metadata"""
        
        # Determine integration strategy
        if metadata.framework == AgentFramework.CREWAI:
            strategy = "crewai_wrapper"
            wrapper_class = "CrewAIAgentWrapper"
        elif metadata.framework == AgentFramework.AUTOGPT:
            strategy = "autogpt_wrapper"
            wrapper_class = "AutoGPTAgentWrapper"
        elif metadata.framework == AgentFramework.LANGCHAIN:
            strategy = "langchain_wrapper"
            wrapper_class = "LangChainAgentWrapper"
        else:
            strategy = "generic_wrapper"
            wrapper_class = "GenericAgentWrapper"
        
        # Create API adapter based on detected endpoints
        api_adapter = "RestAPIAdapter" if metadata.api_endpoints else "DirectCallAdapter"
        
        # Map configuration files
        config_mapping = {}
        for config_file in metadata.config_files:
            config_mapping[config_file] = f"mark1_config_{Path(config_file).stem}.yml"
        
        # Resolve dependencies
        dependency_resolution = [
            f"pip install {dep}" for dep in metadata.dependencies
        ]
        
        # Create test commands
        test_commands = []
        if metadata.tests:
            test_commands.append("python -m pytest " + " ".join(metadata.tests))
        if metadata.examples:
            test_commands.extend([f"python {example}" for example in metadata.examples[:3]])
        
        # Health check
        health_check = "basic_health_check"
        if metadata.api_endpoints:
            health_check = "api_health_check"
        
        plan = IntegrationPlan(
            agent_metadata=metadata,
            integration_strategy=strategy,
            wrapper_class=wrapper_class,
            api_adapter=api_adapter,
            config_mapping=config_mapping,
            dependency_resolution=dependency_resolution,
            test_commands=test_commands,
            health_check=health_check
        )
        
        return plan

    async def _execute_integration(self, plan: IntegrationPlan, repo_path: Path):
        """Execute the integration plan"""
        try:
            # Create agent wrapper
            await self._create_agent_wrapper(plan, repo_path)
            
            # Create API adapter
            await self._create_api_adapter(plan, repo_path)
            
            # Install dependencies
            await self._install_dependencies(plan)
            
            # Map configurations
            await self._map_configurations(plan, repo_path)
            
            # Register with Mark-1
            await self._register_with_mark1(plan)
            
            logger.info(f"Integration executed for {plan.agent_metadata.name}")
            
        except Exception as e:
            logger.error(f"Failed to execute integration: {e}")
            raise

    async def _create_agent_wrapper(self, plan: IntegrationPlan, repo_path: Path):
        """Create agent wrapper class"""
        wrapper_code = f'''#!/usr/bin/env python3
"""
Auto-generated wrapper for {plan.agent_metadata.name}
Framework: {plan.agent_metadata.framework.value}
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add agent repository to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mark1.agents.base import BaseAgent, AgentResponse
from mark1.utils.exceptions import AgentException

class {plan.wrapper_class}(BaseAgent):
    """Wrapper for {plan.agent_metadata.name} agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            agent_id="{plan.agent_metadata.name.lower()}",
            name="{plan.agent_metadata.name}",
            description="{plan.agent_metadata.description}",
            capabilities={[cap.value for cap in plan.agent_metadata.capabilities]},
            framework="{plan.agent_metadata.framework.value}"
        )
        self.config = config
        self.agent_instance = None
        
    async def initialize(self) -> bool:
        """Initialize the wrapped agent"""
        try:
            # Framework-specific initialization
            {self._get_framework_init_code(plan.agent_metadata.framework)}
            return True
        except Exception as e:
            raise AgentException(f"Failed to initialize {plan.agent_metadata.name}: {{e}}")
    
    async def process_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a prompt using the wrapped agent"""
        try:
            # Framework-specific processing
            {self._get_framework_process_code(plan.agent_metadata.framework)}
        except Exception as e:
            raise AgentException(f"Failed to process prompt: {{e}}")
    
    async def shutdown(self):
        """Shutdown the wrapped agent"""
        if self.agent_instance:
            # Cleanup if needed
            pass
        await super().shutdown()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {{
            "status": "healthy" if self.agent_instance else "unhealthy",
            "framework": "{plan.agent_metadata.framework.value}",
            "capabilities": {[cap.value for cap in plan.agent_metadata.capabilities]},
            "initialized": self.agent_instance is not None
        }}
'''
        
        # Write wrapper file
        wrapper_file = repo_path / f"{plan.wrapper_class.lower()}.py"
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        logger.info(f"Created wrapper: {wrapper_file}")

    def _get_framework_init_code(self, framework: AgentFramework) -> str:
        """Get framework-specific initialization code"""
        if framework == AgentFramework.CREWAI:
            return '''
            from crewai import Agent, Crew, Task
            # Initialize CrewAI agent
            self.agent_instance = Agent(
                role="Assistant",
                goal="Help with tasks",
                backstory="AI assistant created via Mark-1 integration"
            )
            '''
        elif framework == AgentFramework.LANGCHAIN:
            return '''
            from langchain.agents import initialize_agent
            from langchain.llms import OpenAI
            # Initialize LangChain agent
            llm = OpenAI()
            self.agent_instance = initialize_agent([], llm)
            '''
        else:
            return '''
            # Generic initialization
            self.agent_instance = {"initialized": True, "type": "generic"}
            '''

    def _get_framework_process_code(self, framework: AgentFramework) -> str:
        """Get framework-specific processing code"""
        if framework == AgentFramework.CREWAI:
            return '''
            task = Task(description=prompt, agent=self.agent_instance)
            crew = Crew(agents=[self.agent_instance], tasks=[task])
            result = crew.kickoff()
            
            return AgentResponse(
                agent_id=self.agent_id,
                response=str(result),
                metadata={"framework": "crewai", "context": context}
            )
            '''
        elif framework == AgentFramework.LANGCHAIN:
            return '''
            result = self.agent_instance.run(prompt)
            
            return AgentResponse(
                agent_id=self.agent_id,
                response=result,
                metadata={"framework": "langchain", "context": context}
            )
            '''
        else:
            return '''
            # Generic processing
            response = f"Processed: {prompt}"
            
            return AgentResponse(
                agent_id=self.agent_id,
                response=response,
                metadata={"framework": "generic", "context": context}
            )
            '''

    async def _create_api_adapter(self, plan: IntegrationPlan, repo_path: Path):
        """Create API adapter if needed"""
        if plan.api_adapter == "RestAPIAdapter":
            # Create REST API adapter
            adapter_code = f'''#!/usr/bin/env python3
"""
REST API Adapter for {plan.agent_metadata.name}
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional

class RestAPIAdapter:
    """REST API adapter for {plan.agent_metadata.name}"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def call_endpoint(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call API endpoint"""
        if not self.session:
            await self.initialize()
        
        url = f"{{self.base_url}}{{endpoint}}"
        async with self.session.post(url, json=data) as response:
            return await response.json()
    
    async def shutdown(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
'''
            
            adapter_file = repo_path / "api_adapter.py"
            with open(adapter_file, 'w') as f:
                f.write(adapter_code)

    async def _install_dependencies(self, plan: IntegrationPlan):
        """Install agent dependencies"""
        for cmd in plan.dependency_resolution:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to install dependency: {cmd}")
            except Exception as e:
                logger.warning(f"Error installing dependency {cmd}: {e}")

    async def _map_configurations(self, plan: IntegrationPlan, repo_path: Path):
        """Map agent configurations to Mark-1 format"""
        for original_config, mark1_config in plan.config_mapping.items():
            original_path = repo_path / original_config
            mark1_path = self.mark1_root / "config" / mark1_config
            
            if original_path.exists():
                # Copy and potentially transform config
                shutil.copy2(original_path, mark1_path)
                logger.info(f"Mapped config: {original_config} -> {mark1_config}")

    async def _register_with_mark1(self, plan: IntegrationPlan):
        """Register agent with Mark-1 orchestrator"""
        # Create agent registry entry
        registry_entry = {
            "agent_id": plan.agent_metadata.name.lower(),
            "name": plan.agent_metadata.name,
            "framework": plan.agent_metadata.framework.value,
            "capabilities": [cap.value for cap in plan.agent_metadata.capabilities],
            "wrapper_class": plan.wrapper_class,
            "integration_strategy": plan.integration_strategy,
            "health_check": plan.health_check,
            "metadata": {
                "name": plan.agent_metadata.name,
                "framework": plan.agent_metadata.framework.value,
                "version": plan.agent_metadata.version,
                "description": plan.agent_metadata.description,
                "capabilities": [cap.value for cap in plan.agent_metadata.capabilities],
                "dependencies": plan.agent_metadata.dependencies,
                "entry_points": plan.agent_metadata.entry_points,
                "api_endpoints": plan.agent_metadata.api_endpoints,
                "config_files": plan.agent_metadata.config_files,
                "documentation": plan.agent_metadata.documentation,
                "examples": plan.agent_metadata.examples,
                "tests": plan.agent_metadata.tests
            }
        }
        
        # Save to registry
        registry_file = self.mark1_root / "config" / "agent_registry.json"
        registry = {}
        
        # Create config directory if it doesn't exist
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only parse if file has content
                        registry = json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                registry = {}  # Start fresh if file is corrupted
        
        registry[plan.agent_metadata.name.lower()] = registry_entry
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Registered {plan.agent_metadata.name} with Mark-1")

    async def _test_integration(self, plan: IntegrationPlan) -> Dict[str, Any]:
        """Test the integrated agent"""
        test_results = {
            "agent": plan.agent_metadata.name,
            "framework": plan.agent_metadata.framework.value,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        # Run test commands
        for test_cmd in plan.test_commands:
            try:
                result = subprocess.run(
                    test_cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    test_results["tests_passed"] += 1
                    test_results["test_details"].append({
                        "command": test_cmd,
                        "status": "passed",
                        "output": result.stdout[:500]
                    })
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"].append({
                        "command": test_cmd,
                        "status": "failed",
                        "error": result.stderr[:500]
                    })
                    
            except subprocess.TimeoutExpired:
                test_results["tests_failed"] += 1
                test_results["test_details"].append({
                    "command": test_cmd,
                    "status": "timeout",
                    "error": "Test command timed out"
                })
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"].append({
                    "command": test_cmd,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"Test results for {plan.agent_metadata.name}: "
                   f"{test_results['tests_passed']} passed, {test_results['tests_failed']} failed")
        
        return test_results

    async def list_integrated_agents(self) -> List[Dict[str, Any]]:
        """List all integrated agents"""
        registry_file = self.mark1_root / "config" / "agent_registry.json"
        
        if not registry_file.exists():
            return []
        
        try:
            with open(registry_file, 'r') as f:
                content = f.read().strip()
                if not content:  # Empty file
                    return []
                registry = json.loads(content)
            
            return list(registry.values())
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Registry file {registry_file} is corrupted or unreadable")
            return []

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an integrated agent"""
        try:
            # Remove from registry
            registry_file = self.mark1_root / "config" / "agent_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                
                if agent_id in registry:
                    del registry[agent_id]
                    
                    with open(registry_file, 'w') as f:
                        json.dump(registry, f, indent=2)
            
            # Remove agent directory
            agent_dir = self.agents_dir / agent_id
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            
            logger.info(f"Removed agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False

    def cleanup(self):
        """Cleanup temporary resources"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir) 