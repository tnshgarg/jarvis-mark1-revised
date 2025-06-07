"""
Codebase Scanner for Mark-1 Orchestrator

Advanced codebase analysis system that discovers and analyzes AI agents
across multiple programming languages and frameworks.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.config.settings import get_settings
from mark1.utils.exceptions import ScanningException, InvalidCodebaseException, ParseException
from mark1.utils.constants import AGENT_DISCOVERY_PATHS, AGENT_FILE_PATTERNS, AGENT_KEYWORDS


class ScanStatus(Enum):
    """Scanning status enumeration"""
    PENDING = "pending"
    SCANNING = "scanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentDiscoveryInfo:
    """Information about a discovered agent"""
    name: str
    framework: str
    file_path: Path
    capabilities: List[str]
    confidence: float
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class ScanResult:
    """Result of a codebase scan"""
    scan_id: str
    scan_path: Path
    status: ScanStatus
    discovered_agents: List[AgentDiscoveryInfo]
    total_files_scanned: int
    scan_duration: float
    framework_distribution: Dict[str, int]
    error_count: int
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file"""
    file_path: Path
    language: str
    agents_found: List[Dict[str, Any]]
    has_llm_calls: bool
    framework_indicators: List[str]
    confidence_score: float
    analysis_time: float
    errors: List[str]


class CodebaseScanner:
    """
    Advanced codebase scanning system for agent discovery
    
    Scans codebases to discover AI agents, analyze their capabilities,
    and extract metadata for integration into the Mark-1 system.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Scanner state
        self._initialized = False
        self._active_scans: Dict[str, ScanResult] = {}
        self._scan_counter = 0
        
        # Language support
        self._supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp'
        }
        
        # Framework patterns
        self._framework_patterns = {
            'langchain': [
                'from langchain',
                'import langchain',
                'LangChain',
                'langchain_core',
                'langchain_community'
            ],
            'autogpt': [
                'from autogpt',
                'import autogpt',
                'AutoGPT',
                'auto_gpt'
            ],
            'crewai': [
                'from crewai',
                'import crewai',
                'CrewAI',
                'crew_ai'
            ],
            'openai': [
                'openai.ChatCompletion',
                'openai.Completion',
                'from openai',
                'import openai'
            ],
            'anthropic': [
                'from anthropic',
                'import anthropic',
                'anthropic.messages'
            ]
        }
        
        # LLM call patterns
        self._llm_patterns = [
            'openai.',
            'anthropic.',
            'ChatCompletion',
            'chat.completions',
            'llm.invoke',
            'llm.ainvoke',
            'messages.create',
            'completions.create'
        ]
    
    async def initialize(self) -> None:
        """Initialize the codebase scanner"""
        try:
            self.logger.info("Initializing codebase scanner...")
            
            # Validate configuration
            if not self.settings.scanning:
                raise ScanningException("Scanning configuration not found")
            
            self._initialized = True
            self.logger.info("Codebase scanner initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize codebase scanner", error=str(e))
            raise ScanningException(f"Scanner initialization failed: {e}")
    
    async def scan_directory(
        self,
        path: Path,
        recursive: bool = True,
        framework_filter: Optional[List[str]] = None,
        max_files: int = 10000,
        exclude_patterns: Optional[List[str]] = None
    ) -> ScanResult:
        """
        Scan a directory for AI agents
        
        Args:
            path: Directory path to scan
            recursive: Whether to scan recursively
            framework_filter: Optional framework filter
            max_files: Maximum files to scan
            exclude_patterns: Patterns to exclude
            
        Returns:
            ScanResult with discovered agents and statistics
        """
        if not self._initialized:
            raise ScanningException("Scanner not initialized")
        
        # Generate scan ID
        self._scan_counter += 1
        scan_id = f"scan_{self._scan_counter}_{int(datetime.now().timestamp())}"
        
        self.logger.info("Starting codebase scan", scan_id=scan_id, path=str(path))
        start_time = asyncio.get_event_loop().time()
        
        # Initialize scan result
        scan_result = ScanResult(
            scan_id=scan_id,
            scan_path=path,
            status=ScanStatus.SCANNING,
            discovered_agents=[],
            total_files_scanned=0,
            scan_duration=0.0,
            framework_distribution={},
            error_count=0,
            errors=[],
            started_at=datetime.now(timezone.utc)
        )
        
        self._active_scans[scan_id] = scan_result
        
        try:
            # Validate scan path
            if not path.exists():
                raise InvalidCodebaseException(str(path), "Path does not exist")
            
            if not path.is_dir():
                raise InvalidCodebaseException(str(path), "Path is not a directory")
            
            # Collect files to scan
            files_to_scan = await self._collect_files(
                path, recursive, exclude_patterns, max_files
            )
            
            self.logger.info("Files collected for scanning", 
                           count=len(files_to_scan), scan_id=scan_id)
            
            # Scan files in batches
            batch_size = 50
            for i in range(0, len(files_to_scan), batch_size):
                batch = files_to_scan[i:i + batch_size]
                await self._scan_file_batch(batch, scan_result, framework_filter)
                
                # Update progress
                scan_result.total_files_scanned = min(i + batch_size, len(files_to_scan))
            
            # Finalize scan
            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.scan_duration = asyncio.get_event_loop().time() - start_time
            
            self.logger.info("Codebase scan completed",
                           scan_id=scan_id,
                           agents_found=len(scan_result.discovered_agents),
                           files_scanned=scan_result.total_files_scanned,
                           duration=scan_result.scan_duration)
            
            return scan_result
            
        except Exception as e:
            scan_result.status = ScanStatus.FAILED
            scan_result.errors.append(str(e))
            scan_result.error_count += 1
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.scan_duration = asyncio.get_event_loop().time() - start_time
            
            self.logger.error("Codebase scan failed", scan_id=scan_id, error=str(e))
            raise ScanningException(f"Scan failed: {e}")
    
    async def _collect_files(
        self,
        path: Path,
        recursive: bool,
        exclude_patterns: Optional[List[str]],
        max_files: int
    ) -> List[Path]:
        """Collect files to scan"""
        files = []
        exclude_patterns = exclude_patterns or self.settings.scanning.excluded_directories
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in path.glob(pattern):
                if len(files) >= max_files:
                    break
                
                # Skip directories
                if file_path.is_dir():
                    continue
                
                # Skip excluded patterns
                if self._should_exclude_file(file_path, exclude_patterns):
                    continue
                
                # Check file extension
                if file_path.suffix not in self._supported_extensions:
                    continue
                
                # Check file size
                try:
                    file_size = file_path.stat().st_size
                    max_size_bytes = self.settings.scanning.max_file_size_mb * 1024 * 1024
                    if file_size > max_size_bytes:
                        continue
                except (OSError, PermissionError):
                    continue
                
                files.append(file_path)
            
            return files
            
        except Exception as e:
            self.logger.error("Failed to collect files", path=str(path), error=str(e))
            return []
    
    def _should_exclude_file(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded"""
        path_str = str(file_path)
        
        for pattern in exclude_patterns:
            if pattern in path_str:
                return True
        
        # Check against default excluded extensions
        if file_path.suffix in self.settings.scanning.excluded_extensions:
            return True
        
        return False
    
    async def _scan_file_batch(
        self,
        files: List[Path],
        scan_result: ScanResult,
        framework_filter: Optional[List[str]]
    ) -> None:
        """Scan a batch of files"""
        tasks = []
        
        for file_path in files:
            task = asyncio.create_task(self._analyze_file(file_path))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            file_path = files[i]
            
            if isinstance(result, Exception):
                scan_result.errors.append(f"Error analyzing {file_path}: {result}")
                scan_result.error_count += 1
                continue
            
            if not isinstance(result, FileAnalysisResult):
                continue
            
            # Apply framework filter
            if framework_filter:
                if not any(fw in result.framework_indicators for fw in framework_filter):
                    continue
            
            # Process discovered agents
            for agent_data in result.agents_found:
                agent_info = AgentDiscoveryInfo(
                    name=agent_data.get('name', file_path.stem),
                    framework=agent_data.get('framework', 'unknown'),
                    file_path=file_path,
                    capabilities=agent_data.get('capabilities', []),
                    confidence=agent_data.get('confidence', result.confidence_score),
                    metadata={
                        'language': result.language,
                        'has_llm_calls': result.has_llm_calls,
                        'framework_indicators': result.framework_indicators,
                        'file_size': file_path.stat().st_size,
                        'analysis_time': result.analysis_time
                    },
                    created_at=datetime.now(timezone.utc)
                )
                
                scan_result.discovered_agents.append(agent_info)
                
                # Update framework distribution
                framework = agent_info.framework
                scan_result.framework_distribution[framework] = (
                    scan_result.framework_distribution.get(framework, 0) + 1
                )
    
    async def _analyze_file(self, file_path: Path) -> FileAnalysisResult:
        """Analyze a single file for agent patterns"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Determine language
            language = self._supported_extensions.get(file_path.suffix, 'unknown')
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                raise ParseException(str(file_path), language, e)
            
            # Analyze content
            framework_indicators = self._detect_frameworks(content)
            has_llm_calls = self._detect_llm_calls(content)
            agents_found = []
            confidence_score = 0.0
            
            # Check for agent patterns
            if framework_indicators or has_llm_calls:
                agent_data = await self._extract_agent_info(
                    file_path, content, language, framework_indicators
                )
                
                if agent_data:
                    agents_found.append(agent_data)
                    confidence_score = agent_data.get('confidence', 0.5)
            
            analysis_time = asyncio.get_event_loop().time() - start_time
            
            return FileAnalysisResult(
                file_path=file_path,
                language=language,
                agents_found=agents_found,
                has_llm_calls=has_llm_calls,
                framework_indicators=framework_indicators,
                confidence_score=confidence_score,
                analysis_time=analysis_time,
                errors=[]
            )
            
        except Exception as e:
            analysis_time = asyncio.get_event_loop().time() - start_time
            
            return FileAnalysisResult(
                file_path=file_path,
                language=self._supported_extensions.get(file_path.suffix, 'unknown'),
                agents_found=[],
                has_llm_calls=False,
                framework_indicators=[],
                confidence_score=0.0,
                analysis_time=analysis_time,
                errors=[str(e)]
            )
    
    def _detect_frameworks(self, content: str) -> List[str]:
        """Detect framework usage in content"""
        detected_frameworks = []
        
        for framework, patterns in self._framework_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    detected_frameworks.append(framework)
                    break
        
        return detected_frameworks
    
    def _detect_llm_calls(self, content: str) -> bool:
        """Detect LLM API calls in content"""
        for pattern in self._llm_patterns:
            if pattern in content:
                return True
        return False
    
    async def _extract_agent_info(
        self,
        file_path: Path,
        content: str,
        language: str,
        framework_indicators: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract agent information from file content"""
        try:
            # Basic agent info
            agent_info = {
                'name': file_path.stem,
                'framework': framework_indicators[0] if framework_indicators else 'custom',
                'capabilities': [],
                'confidence': 0.5,
                'entry_point': None,
                'dependencies': []
            }
            
            # Language-specific analysis
            if language == 'python':
                await self._analyze_python_agent(content, agent_info)
            elif language in ['javascript', 'typescript']:
                await self._analyze_js_agent(content, agent_info)
            
            # Calculate confidence score
            confidence = 0.3  # Base confidence
            
            if framework_indicators:
                confidence += 0.3
            
            if self._detect_llm_calls(content):
                confidence += 0.2
            
            if agent_info['capabilities']:
                confidence += 0.2
            
            agent_info['confidence'] = min(confidence, 1.0)
            
            return agent_info
            
        except Exception as e:
            self.logger.error("Failed to extract agent info", 
                            file_path=str(file_path), error=str(e))
            return None
    
    async def _analyze_python_agent(self, content: str, agent_info: Dict[str, Any]) -> None:
        """Analyze Python agent code"""
        try:
            # Look for common agent patterns
            if 'class' in content and 'Agent' in content:
                agent_info['capabilities'].append('agent_class')
            
            if 'def run' in content or 'def execute' in content:
                agent_info['capabilities'].append('execution')
            
            if 'def chat' in content or 'def invoke' in content:
                agent_info['capabilities'].append('conversation')
            
            # Look for imports to determine entry point
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('def main'):
                    agent_info['entry_point'] = 'main'
                    break
                elif line.startswith('if __name__ == "__main__"'):
                    agent_info['entry_point'] = '__main__'
                    break
            
            # Extract dependencies from imports
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    # Simple dependency extraction
                    if 'langchain' in line:
                        agent_info['dependencies'].append('langchain')
                    elif 'openai' in line:
                        agent_info['dependencies'].append('openai')
                    elif 'anthropic' in line:
                        agent_info['dependencies'].append('anthropic')
            
        except Exception as e:
            self.logger.debug("Python analysis failed", error=str(e))
    
    async def _analyze_js_agent(self, content: str, agent_info: Dict[str, Any]) -> None:
        """Analyze JavaScript/TypeScript agent code"""
        try:
            # Look for common patterns
            if 'class' in content and ('Agent' in content or 'Bot' in content):
                agent_info['capabilities'].append('agent_class')
            
            if 'function run' in content or 'function execute' in content:
                agent_info['capabilities'].append('execution')
            
            if 'async function' in content:
                agent_info['capabilities'].append('async_execution')
            
            # Look for exports to determine entry point
            if 'module.exports' in content or 'export default' in content:
                agent_info['entry_point'] = 'export'
            
        except Exception as e:
            self.logger.debug("JavaScript analysis failed", error=str(e))
    
    async def get_scan_result(self, scan_id: str) -> Optional[ScanResult]:
        """Get scan result by ID"""
        return self._active_scans.get(scan_id)
    
    async def list_active_scans(self) -> List[ScanResult]:
        """List all active scans"""
        return list(self._active_scans.values())
    
    async def cancel_scan(self, scan_id: str) -> bool:
        """Cancel an active scan"""
        if scan_id in self._active_scans:
            scan_result = self._active_scans[scan_id]
            scan_result.status = ScanStatus.CANCELLED
            scan_result.completed_at = datetime.now(timezone.utc)
            return True
        return False
    
    async def clear_completed_scans(self) -> int:
        """Clear completed scans from memory"""
        completed_scan_ids = [
            scan_id for scan_id, result in self._active_scans.items()
            if result.status in [ScanStatus.COMPLETED, ScanStatus.FAILED, ScanStatus.CANCELLED]
        ]
        
        for scan_id in completed_scan_ids:
            del self._active_scans[scan_id]
        
        return len(completed_scan_ids)
    
    @property
    def is_initialized(self) -> bool:
        """Check if scanner is initialized"""
        return self._initialized
    
    @property
    def active_scan_count(self) -> int:
        """Get number of active scans"""
        return len([s for s in self._active_scans.values() if s.status == ScanStatus.SCANNING])
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return list(set(self._supported_extensions.values()))
    
    async def shutdown(self) -> None:
        """Shutdown the scanner"""
        try:
            self.logger.info("Shutting down codebase scanner...")
            
            # Cancel all active scans
            for scan_id in list(self._active_scans.keys()):
                await self.cancel_scan(scan_id)
            
            self._active_scans.clear()
            self._initialized = False
            
            self.logger.info("Codebase scanner shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during scanner shutdown", error=str(e))
