"""
Enhanced Codebase Scanner for Mark-1 Orchestrator

Comprehensive codebase analysis including agent detection, LLM call detection,
and multi-language AST analysis.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import structlog

from mark1.utils.exceptions import ScanException
from mark1.scanning.ast_analyzer import MultiLanguageASTAnalyzer, AnalysisResult
from mark1.scanning.llm_call_detector import LLMCallDetector, DetectionResult


@dataclass 
class AgentInfo:
    """Information about a detected agent"""
    name: str
    file_path: Path
    framework: str
    capabilities: List[str]
    confidence: float
    metadata: Dict[str, Any]
    module_path: Optional[str] = None
    class_name: Optional[str] = None


@dataclass
class ScanResults:
    """Comprehensive scan results"""
    scan_path: Path
    discovered_agents: List[AgentInfo]
    total_files_scanned: int
    scan_duration: float
    framework_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    ast_analysis_results: List[AnalysisResult]
    llm_detection_results: List[DetectionResult]
    code_quality_metrics: Dict[str, Any]
    migration_opportunities: Dict[str, Any]
    errors: List[str]


class CodebaseScanner:
    """
    Enhanced codebase scanner with multi-language AST analysis and LLM detection
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self._initialized = False
        
        # Initialize analyzers
        self.ast_analyzer = MultiLanguageASTAnalyzer()
        self.llm_detector = LLMCallDetector()
        
        # Supported file extensions
        self.code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c'}
        self.config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini'}
        self.doc_extensions = {'.md', '.rst', '.txt'}
        
    async def initialize(self) -> None:
        """Initialize the scanner"""
        if self._initialized:
            return
            
        try:
            self.logger.info("Initializing codebase scanner...")
            self._initialized = True
            self.logger.info("Codebase scanner initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize codebase scanner", error=str(e))
            raise ScanException(f"Scanner initialization failed: {e}")
    
    async def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        framework_filter: Optional[List[str]] = None,
        include_ast_analysis: bool = True,
        include_llm_detection: bool = True
    ) -> ScanResults:
        """
        Comprehensive directory scan with multiple analysis types
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            framework_filter: Optional framework filter
            include_ast_analysis: Whether to include AST analysis
            include_llm_detection: Whether to include LLM call detection
            
        Returns:
            Comprehensive scan results
        """
        start_time = datetime.now(timezone.utc)
        errors = []
        
        try:
            self.logger.info("Starting comprehensive codebase scan", 
                           directory=str(directory),
                           recursive=recursive,
                           include_ast=include_ast_analysis,
                           include_llm=include_llm_detection)
            
            if not directory.exists():
                raise ScanException(f"Directory not found: {directory}")
            
            # Collect all files
            all_files = await self._collect_files(directory, recursive)
            
            # Initialize results
            discovered_agents = []
            ast_analysis_results = []
            llm_detection_results = []
            framework_distribution = {}
            language_distribution = {}
            
            # Perform AST analysis
            if include_ast_analysis:
                self.logger.info("Performing AST analysis...")
                ast_analysis_results = await self.ast_analyzer.analyze_directory(directory, recursive)
                
                # Extract agents from AST analysis
                for result in ast_analysis_results:
                    # Track language distribution
                    lang = result.language.value
                    language_distribution[lang] = language_distribution.get(lang, 0) + 1
                    
                    # Extract agent information from patterns
                    for pattern in result.agent_patterns:
                        agent = await self._pattern_to_agent_info(pattern, result)
                        if agent:
                            discovered_agents.append(agent)
                            
                            # Track framework distribution
                            framework = agent.framework
                            framework_distribution[framework] = framework_distribution.get(framework, 0) + 1
            
            # Perform LLM call detection
            if include_llm_detection:
                self.logger.info("Performing LLM call detection...")
                llm_detection_results = await self.llm_detector.detect_directory(directory, recursive)
            
            # Apply framework filter if specified
            if framework_filter:
                discovered_agents = [
                    agent for agent in discovered_agents 
                    if agent.framework.lower() in [f.lower() for f in framework_filter]
                ]
            
            # Calculate metrics and opportunities
            code_quality_metrics = await self._calculate_quality_metrics(ast_analysis_results)
            migration_opportunities = await self._analyze_migration_opportunities(llm_detection_results)
            
            scan_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            results = ScanResults(
                scan_path=directory,
                discovered_agents=discovered_agents,
                total_files_scanned=len(all_files),
                scan_duration=scan_duration,
                framework_distribution=framework_distribution,
                language_distribution=language_distribution,
                ast_analysis_results=ast_analysis_results,
                llm_detection_results=llm_detection_results,
                code_quality_metrics=code_quality_metrics,
                migration_opportunities=migration_opportunities,
                errors=errors
            )
            
            self.logger.info("Codebase scan completed",
                           directory=str(directory),
                           agents_found=len(discovered_agents),
                           files_scanned=len(all_files),
                           duration=scan_duration)
            
            return results
            
        except Exception as e:
            errors.append(f"Scan error: {str(e)}")
            self.logger.error("Codebase scan failed", directory=str(directory), error=str(e))
            
            scan_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return ScanResults(
                scan_path=directory,
                discovered_agents=[],
                total_files_scanned=0,
                scan_duration=scan_duration,
                framework_distribution={},
                language_distribution={},
                ast_analysis_results=[],
                llm_detection_results=[],
                code_quality_metrics={},
                migration_opportunities={},
                errors=errors
            )
    
    async def _collect_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Collect all relevant files from directory"""
        files = []
        
        try:
            if recursive:
                all_paths = list(directory.rglob("*"))
            else:
                all_paths = list(directory.iterdir())
            
            # Filter for relevant files
            relevant_extensions = self.code_extensions | self.config_extensions | self.doc_extensions
            
            for path in all_paths:
                if path.is_file() and path.suffix.lower() in relevant_extensions:
                    # Skip hidden files and common ignore patterns
                    if not any(part.startswith('.') for part in path.parts):
                        if not any(ignore in str(path) for ignore in ['__pycache__', 'node_modules', '.git']):
                            files.append(path)
            
            return files
            
        except Exception as e:
            self.logger.error("File collection failed", directory=str(directory), error=str(e))
            return []
    
    async def _pattern_to_agent_info(self, pattern: Dict[str, Any], analysis_result: AnalysisResult) -> Optional[AgentInfo]:
        """Convert detected pattern to AgentInfo"""
        try:
            # Extract basic information
            name = pattern.get('name', 'Unknown')
            framework = pattern.get('framework', 'unknown')
            capabilities = pattern.get('capabilities', [])
            confidence = pattern.get('confidence', 0.0)
            
            # Build metadata
            metadata = {
                'pattern_type': pattern.get('type'),
                'base_classes': pattern.get('base_classes', []),
                'methods': pattern.get('methods', []),
                'language': analysis_result.language.value,
                'analysis_time': analysis_result.analysis_time,
                'complexity_score': analysis_result.complexity_score
            }
            
            # Add quality metrics if available
            if analysis_result.quality_metrics:
                metadata['quality_metrics'] = analysis_result.quality_metrics
            
            return AgentInfo(
                name=name,
                file_path=analysis_result.file_path,
                framework=framework,
                capabilities=capabilities,
                confidence=confidence,
                metadata=metadata,
                module_path=str(analysis_result.file_path.with_suffix('')).replace('/', '.'),
                class_name=name
            )
            
        except Exception as e:
            self.logger.warning("Failed to convert pattern to agent info", error=str(e))
            return None
    
    async def _calculate_quality_metrics(self, ast_results: List[AnalysisResult]) -> Dict[str, Any]:
        """Calculate overall code quality metrics"""
        if not ast_results:
            return {}
        
        try:
            total_files = len(ast_results)
            total_elements = sum(len(r.elements) for r in ast_results)
            total_lines = sum(r.quality_metrics.get('total_lines', 0) for r in ast_results)
            
            # Calculate averages
            avg_complexity = sum(r.complexity_score for r in ast_results) / total_files
            avg_documentation_ratio = sum(
                r.quality_metrics.get('documentation_ratio', 0) for r in ast_results
            ) / total_files
            
            # Count patterns and capabilities
            total_patterns = sum(len(r.agent_patterns) for r in ast_results)
            all_capabilities = set()
            for result in ast_results:
                all_capabilities.update(result.capabilities)
            
            # Error analysis
            files_with_errors = len([r for r in ast_results if r.errors])
            total_errors = sum(len(r.errors) for r in ast_results)
            
            return {
                'total_files_analyzed': total_files,
                'total_code_elements': total_elements,
                'total_lines_of_code': total_lines,
                'average_complexity': round(avg_complexity, 2),
                'average_documentation_ratio': round(avg_documentation_ratio, 2),
                'total_agent_patterns': total_patterns,
                'unique_capabilities': len(all_capabilities),
                'files_with_errors': files_with_errors,
                'total_analysis_errors': total_errors,
                'analysis_success_rate': round((total_files - files_with_errors) / total_files, 2) if total_files > 0 else 0
            }
            
        except Exception as e:
            self.logger.error("Quality metrics calculation failed", error=str(e))
            return {'error': str(e)}
    
    async def _analyze_migration_opportunities(self, llm_results: List[DetectionResult]) -> Dict[str, Any]:
        """Analyze opportunities for migrating to local LLM"""
        if not llm_results:
            return {'total_opportunities': 0}
        
        try:
            # Generate migration report
            migration_report = self.llm_detector.generate_migration_report(llm_results)
            
            # Add additional analysis
            total_files_with_llm = len(llm_results)
            total_llm_calls = sum(r.total_calls for r in llm_results)
            
            # Analyze complexity distribution
            complexity_distribution = {}
            for result in llm_results:
                providers = [p.value for p in result.providers_found]
                complexity = 'low' if len(providers) == 1 else 'medium' if len(providers) == 2 else 'high'
                complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
            
            # Calculate potential impact
            estimated_savings = migration_report['summary'].get('potential_savings', 0)
            impact_level = 'low' if estimated_savings < 100 else 'medium' if estimated_savings < 500 else 'high'
            
            migration_opportunities = {
                **migration_report,
                'analysis': {
                    'files_with_llm_calls': total_files_with_llm,
                    'total_llm_calls': total_llm_calls,
                    'complexity_distribution': complexity_distribution,
                    'estimated_impact': impact_level,
                    'recommended_priority': self._calculate_migration_priority(llm_results)
                }
            }
            
            return migration_opportunities
            
        except Exception as e:
            self.logger.error("Migration opportunity analysis failed", error=str(e))
            return {'error': str(e)}
    
    def _calculate_migration_priority(self, llm_results: List[DetectionResult]) -> str:
        """Calculate migration priority based on usage patterns"""
        if not llm_results:
            return 'none'
        
        total_calls = sum(r.total_calls for r in llm_results)
        total_cost = sum(r.estimated_monthly_cost for r in llm_results)
        unique_providers = set()
        
        for result in llm_results:
            unique_providers.update(result.providers_found)
        
        # Priority calculation
        if total_cost > 500 or total_calls > 50:
            return 'high'
        elif total_cost > 100 or total_calls > 20:
            return 'medium'
        elif total_calls > 0:
            return 'low'
        else:
            return 'none'
    
    async def analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive analysis on a single file"""
        try:
            self.logger.info("Analyzing single file", file_path=str(file_path))
            
            results = {}
            
            # AST Analysis
            ast_result = await self.ast_analyzer.analyze_file(file_path)
            if ast_result:
                results['ast_analysis'] = {
                    'language': ast_result.language.value,
                    'elements': len(ast_result.elements),
                    'imports': ast_result.imports,
                    'framework_indicators': ast_result.framework_indicators,
                    'agent_patterns': ast_result.agent_patterns,
                    'capabilities': ast_result.capabilities,
                    'complexity_score': ast_result.complexity_score,
                    'quality_metrics': ast_result.quality_metrics,
                    'analysis_time': ast_result.analysis_time
                }
            
            # LLM Detection
            llm_result = await self.llm_detector.detect_file(file_path)
            if llm_result.calls:
                results['llm_detection'] = {
                    'total_calls': llm_result.total_calls,
                    'providers_found': [p.value for p in llm_result.providers_found],
                    'call_types': [c.value for c in llm_result.call_types_found],
                    'estimated_monthly_cost': llm_result.estimated_monthly_cost,
                    'replacement_suggestions': len(llm_result.replacement_suggestions),
                    'analysis_time': llm_result.analysis_time
                }
            
            return results
            
        except Exception as e:
            self.logger.error("Single file analysis failed", file_path=str(file_path), error=str(e))
            return {'error': str(e)}
    
    def get_scan_summary(self, results: ScanResults) -> Dict[str, Any]:
        """Generate a summary of scan results"""
        return {
            'scan_overview': {
                'path': str(results.scan_path),
                'duration': results.scan_duration,
                'files_scanned': results.total_files_scanned,
                'agents_discovered': len(results.discovered_agents)
            },
            'distribution': {
                'frameworks': results.framework_distribution,
                'languages': results.language_distribution
            },
            'quality': results.code_quality_metrics,
            'migration': {
                'total_opportunities': len(results.llm_detection_results),
                'complexity': results.migration_opportunities.get('migration_complexity', 'unknown')
            },
            'errors': len(results.errors)
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if scanner is initialized"""
        return self._initialized
