"""
Advanced AST Analysis Engine for Mark-1 Orchestrator

Provides comprehensive Abstract Syntax Tree analysis across multiple
programming languages to detect AI agents, extract capabilities,
and understand code structure.
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import structlog

from mark1.utils.exceptions import ParseException, AnalysisException
from mark1.utils.constants import AGENT_KEYWORDS, CAPABILITY_KEYWORDS


class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"


class NodeType(Enum):
    """AST node types we're interested in"""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    DECORATOR = "decorator"
    ASSIGNMENT = "assignment"
    CALL = "call"
    ASYNC_FUNCTION = "async_function"


@dataclass
class CodeElement:
    """Represents a code element found during AST analysis"""
    name: str
    node_type: NodeType
    line_number: int
    column: int
    end_line: int
    docstring: Optional[str] = None
    parameters: List[str] = None
    return_type: Optional[str] = None
    decorators: List[str] = None
    base_classes: List[str] = None
    imports: List[str] = None
    is_async: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.decorators is None:
            self.decorators = []
        if self.base_classes is None:
            self.base_classes = []
        if self.imports is None:
            self.imports = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalysisResult:
    """Result of AST analysis on a file"""
    file_path: Path
    language: LanguageType
    elements: List[CodeElement]
    imports: List[str]
    framework_indicators: List[str]
    agent_patterns: List[Dict[str, Any]]
    capabilities: List[str]
    llm_calls: List[Dict[str, Any]]
    complexity_score: float
    quality_metrics: Dict[str, Any]
    errors: List[str]
    analysis_time: float


class BaseASTAnalyzer(ABC):
    """Abstract base class for language-specific AST analyzers"""
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def parse_file(self, file_path: Path) -> AnalysisResult:
        """Parse a file and return analysis result"""
        pass
    
    @abstractmethod
    def extract_elements(self, tree: Any) -> List[CodeElement]:
        """Extract code elements from AST"""
        pass
    
    @abstractmethod
    def detect_patterns(self, elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Detect agent patterns in code elements"""
        pass


class PythonASTAnalyzer(BaseASTAnalyzer):
    """Advanced Python AST analyzer for agent detection and code analysis"""
    
    def __init__(self):
        super().__init__()
        self.framework_patterns = {
            'langchain': [
                'langchain', 'LangChain', 'from langchain', 'import langchain',
                'BaseLLM', 'BaseMemory', 'BaseRetriever', 'BaseAgent',
                'LLMChain', 'ConversationChain', 'AgentExecutor'
            ],
            'autogpt': [
                'autogpt', 'AutoGPT', 'from autogpt', 'Agent',
                'execute_command', 'memory.add', 'goals', 'constraints'
            ],
            'crewai': [
                'crewai', 'CrewAI', 'from crewai', 'import crewai',
                'Agent', 'Task', 'Crew', 'role', 'goal', 'backstory'
            ],
            'openai': [
                'openai', 'ChatCompletion', 'openai.Completion',
                'chat.completions.create', 'completions.create'
            ],
            'anthropic': [
                'anthropic', 'from anthropic', 'messages.create',
                'claude', 'anthropic.messages'
            ]
        }
        
        self.llm_call_patterns = [
            'openai.ChatCompletion.create',
            'openai.Completion.create',
            'client.chat.completions.create',
            'anthropic.messages.create',
            'llm.invoke',
            'llm.ainvoke',
            'llm.predict',
            'llm.apredict',
            'generate_response',
            'chat_completion'
        ]
        
        self.agent_indicators = [
            'BaseAgent', 'Agent', 'AIAgent', 'ChatAgent',
            'execute', 'run', 'process', 'handle',
            'memory', 'tools', 'capabilities'
        ]
    
    async def parse_file(self, file_path: Path) -> AnalysisResult:
        """Parse Python file and extract comprehensive analysis"""
        start_time = __import__('time').time()
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                errors.append(f"Syntax error: {e}")
                return self._create_error_result(file_path, errors, start_time)
            
            # Extract code elements
            elements = self.extract_elements(tree)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Detect framework indicators
            framework_indicators = self._detect_frameworks(content, imports)
            
            # Detect agent patterns
            agent_patterns = self.detect_patterns(elements)
            
            # Extract capabilities
            capabilities = self._extract_capabilities(elements, content)
            
            # Detect LLM calls
            llm_calls = self._detect_llm_calls(tree, content)
            
            # Calculate complexity and quality metrics
            complexity_score = self._calculate_complexity(elements)
            quality_metrics = self._calculate_quality_metrics(elements, content)
            
            analysis_time = __import__('time').time() - start_time
            
            return AnalysisResult(
                file_path=file_path,
                language=LanguageType.PYTHON,
                elements=elements,
                imports=imports,
                framework_indicators=framework_indicators,
                agent_patterns=agent_patterns,
                capabilities=capabilities,
                llm_calls=llm_calls,
                complexity_score=complexity_score,
                quality_metrics=quality_metrics,
                errors=errors,
                analysis_time=analysis_time
            )
            
        except Exception as e:
            errors.append(f"Analysis error: {str(e)}")
            return self._create_error_result(file_path, errors, start_time)
    
    def extract_elements(self, tree: ast.AST) -> List[CodeElement]:
        """Extract code elements from Python AST"""
        elements = []
        
        class ElementVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.elements = []
                self.current_class = None
            
            def visit_ClassDef(self, node):
                # Extract class information
                element = CodeElement(
                    name=node.name,
                    node_type=NodeType.CLASS,
                    line_number=node.lineno,
                    column=node.col_offset,
                    end_line=getattr(node, 'end_lineno', node.lineno),
                    docstring=ast.get_docstring(node),
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    base_classes=[self._get_base_name(base) for base in node.bases],
                    metadata={'is_agent': self._is_agent_class(node)}
                )
                self.elements.append(element)
                
                # Visit class methods
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class
            
            def visit_FunctionDef(self, node):
                self._visit_function(node, is_async=False)
            
            def visit_AsyncFunctionDef(self, node):
                self._visit_function(node, is_async=True)
            
            def _visit_function(self, node, is_async=False):
                # Extract function/method information
                node_type = NodeType.METHOD if self.current_class else NodeType.FUNCTION
                if is_async:
                    node_type = NodeType.ASYNC_FUNCTION
                
                # Extract parameters
                parameters = []
                for arg in node.args.args:
                    parameters.append(arg.arg)
                
                # Extract return type
                return_type = None
                if node.returns:
                    return_type = self._get_node_name(node.returns)
                
                element = CodeElement(
                    name=node.name,
                    node_type=node_type,
                    line_number=node.lineno,
                    column=node.col_offset,
                    end_line=getattr(node, 'end_lineno', node.lineno),
                    docstring=ast.get_docstring(node),
                    parameters=parameters,
                    return_type=return_type,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    is_async=is_async,
                    metadata={
                        'class': self.current_class,
                        'is_agent_method': self._is_agent_method(node)
                    }
                )
                self.elements.append(element)
            
            def _get_decorator_name(self, decorator):
                """Extract decorator name"""
                if isinstance(decorator, ast.Name):
                    return decorator.id
                elif isinstance(decorator, ast.Attribute):
                    return f"{self._get_node_name(decorator.value)}.{decorator.attr}"
                elif isinstance(decorator, ast.Call):
                    return self._get_node_name(decorator.func)
                return str(decorator)
            
            def _get_base_name(self, base):
                """Extract base class name"""
                return self._get_node_name(base)
            
            def _get_node_name(self, node):
                """Get name from AST node"""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self._get_node_name(node.value)}.{node.attr}"
                elif isinstance(node, ast.Constant):
                    return str(node.value)
                return str(node)
            
            def _is_agent_class(self, node):
                """Check if class appears to be an agent"""
                # Check class name
                if any(indicator in node.name for indicator in self.analyzer.agent_indicators):
                    return True
                
                # Check base classes
                for base in node.bases:
                    base_name = self._get_base_name(base)
                    if any(indicator in base_name for indicator in self.analyzer.agent_indicators):
                        return True
                
                # Check methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name in ['execute', 'run', 'process', 'handle']:
                            return True
                
                return False
            
            def _is_agent_method(self, node):
                """Check if method appears to be agent-related"""
                return node.name in ['execute', 'run', 'process', 'handle', 'invoke', 'call']
        
        visitor = ElementVisitor(self)
        visitor.visit(tree)
        return visitor.elements
    
    def detect_patterns(self, elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Detect agent patterns in code elements"""
        patterns = []
        
        for element in elements:
            if element.node_type == NodeType.CLASS and element.metadata.get('is_agent', False):
                pattern = {
                    'type': 'agent_class',
                    'name': element.name,
                    'confidence': 0.8,
                    'framework': self._identify_framework(element),
                    'capabilities': self._extract_class_capabilities(element),
                    'base_classes': element.base_classes,
                    'methods': [e.name for e in elements 
                              if e.node_type in [NodeType.METHOD, NodeType.ASYNC_FUNCTION] 
                              and e.metadata.get('class') == element.name]
                }
                patterns.append(pattern)
        
        return patterns
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _detect_frameworks(self, content: str, imports: List[str]) -> List[str]:
        """Detect framework usage"""
        frameworks = []
        
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                if pattern in content or any(pattern in imp for imp in imports):
                    if framework not in frameworks:
                        frameworks.append(framework)
                    break
        
        return frameworks
    
    def _extract_capabilities(self, elements: List[CodeElement], content: str) -> List[str]:
        """Extract capabilities from code analysis"""
        capabilities = set()
        
        # Extract from method names and docstrings
        for element in elements:
            if element.node_type in [NodeType.FUNCTION, NodeType.METHOD, NodeType.ASYNC_FUNCTION]:
                # Check method name
                for keyword in CAPABILITY_KEYWORDS:
                    if keyword.lower() in element.name.lower():
                        capabilities.add(keyword)
                
                # Check docstring
                if element.docstring:
                    for keyword in CAPABILITY_KEYWORDS:
                        if keyword.lower() in element.docstring.lower():
                            capabilities.add(keyword)
        
        # Extract from content analysis
        for keyword in CAPABILITY_KEYWORDS:
            if keyword.lower() in content.lower():
                capabilities.add(keyword)
        
        return list(capabilities)
    
    def _detect_llm_calls(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect LLM API calls"""
        llm_calls = []
        
        class LLMCallVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.calls = []
            
            def visit_Call(self, node):
                call_name = self._get_call_name(node)
                
                for pattern in self.analyzer.llm_call_patterns:
                    if pattern in call_name:
                        call_info = {
                            'pattern': pattern,
                            'line': node.lineno,
                            'call_name': call_name,
                            'arguments': self._extract_arguments(node)
                        }
                        self.calls.append(call_info)
                        break
                
                self.generic_visit(node)
            
            def _get_call_name(self, node):
                """Get the full call name"""
                if isinstance(node.func, ast.Name):
                    return node.func.id
                elif isinstance(node.func, ast.Attribute):
                    return f"{self._get_node_name(node.func.value)}.{node.func.attr}"
                return str(node.func)
            
            def _get_node_name(self, node):
                """Get name from AST node"""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self._get_node_name(node.value)}.{node.attr}"
                return str(node)
            
            def _extract_arguments(self, node):
                """Extract call arguments"""
                args = []
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        args.append(arg.value)
                    else:
                        args.append(str(arg))
                return args
        
        visitor = LLMCallVisitor(self)
        visitor.visit(tree)
        return visitor.calls
    
    def _calculate_complexity(self, elements: List[CodeElement]) -> float:
        """Calculate code complexity score"""
        base_score = len(elements) * 0.1
        
        # Add complexity for classes and methods
        for element in elements:
            if element.node_type == NodeType.CLASS:
                base_score += 2.0
            elif element.node_type in [NodeType.FUNCTION, NodeType.METHOD]:
                base_score += 1.0
                base_score += len(element.parameters) * 0.1
        
        return min(base_score, 10.0)  # Cap at 10.0
    
    def _calculate_quality_metrics(self, elements: List[CodeElement], content: str) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'empty_lines': len([line for line in lines if not line.strip()]),
            'classes': len([e for e in elements if e.node_type == NodeType.CLASS]),
            'functions': len([e for e in elements if e.node_type in [NodeType.FUNCTION, NodeType.METHOD]]),
            'documented_elements': len([e for e in elements if e.docstring]),
            'documentation_ratio': len([e for e in elements if e.docstring]) / max(len(elements), 1)
        }
    
    def _identify_framework(self, element: CodeElement) -> Optional[str]:
        """Identify which framework a class belongs to"""
        # Check base classes
        for base in element.base_classes:
            for framework, patterns in self.framework_patterns.items():
                if any(pattern in base for pattern in patterns):
                    return framework
        
        # Check decorators
        for decorator in element.decorators:
            for framework, patterns in self.framework_patterns.items():
                if any(pattern in decorator for pattern in patterns):
                    return framework
        
        return None
    
    def _extract_class_capabilities(self, element: CodeElement) -> List[str]:
        """Extract capabilities from a class element"""
        capabilities = []
        
        # Check class name
        for keyword in CAPABILITY_KEYWORDS:
            if keyword.lower() in element.name.lower():
                capabilities.append(keyword)
        
        # Check docstring
        if element.docstring:
            for keyword in CAPABILITY_KEYWORDS:
                if keyword.lower() in element.docstring.lower():
                    capabilities.append(keyword)
        
        return capabilities
    
    def _create_error_result(self, file_path: Path, errors: List[str], start_time: float) -> AnalysisResult:
        """Create an error result"""
        return AnalysisResult(
            file_path=file_path,
            language=LanguageType.PYTHON,
            elements=[],
            imports=[],
            framework_indicators=[],
            agent_patterns=[],
            capabilities=[],
            llm_calls=[],
            complexity_score=0.0,
            quality_metrics={},
            errors=errors,
            analysis_time=__import__('time').time() - start_time
        )


class JavaScriptASTAnalyzer(BaseASTAnalyzer):
    """JavaScript/TypeScript AST analyzer for agent detection"""
    
    def __init__(self):
        super().__init__()
        self.framework_patterns = {
            'nodejs_ai': ['require("openai")', 'import.*openai', 'chatgpt', 'gpt-3', 'gpt-4'],
            'langchain_js': ['langchain', '@langchain', 'LangChain'],
            'vercel_ai': ['@vercel/ai', 'ai/react', 'useChat', 'useCompletion'],
            'openai_js': ['openai', 'OpenAI', 'createCompletion', 'createChatCompletion']
        }
    
    async def parse_file(self, file_path: Path) -> AnalysisResult:
        """Parse JavaScript/TypeScript file (basic implementation)"""
        start_time = __import__('time').time()
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic regex-based analysis (could be enhanced with proper JS parser)
            elements = self._extract_elements_regex(content)
            imports = self._extract_imports_regex(content)
            framework_indicators = self._detect_frameworks(content)
            agent_patterns = self.detect_patterns(elements)
            capabilities = self._extract_capabilities_regex(content)
            llm_calls = self._detect_llm_calls_regex(content)
            
            analysis_time = __import__('time').time() - start_time
            
            return AnalysisResult(
                file_path=file_path,
                language=LanguageType.JAVASCRIPT if file_path.suffix == '.js' else LanguageType.TYPESCRIPT,
                elements=elements,
                imports=imports,
                framework_indicators=framework_indicators,
                agent_patterns=agent_patterns,
                capabilities=capabilities,
                llm_calls=llm_calls,
                complexity_score=len(elements) * 0.5,
                quality_metrics=self._calculate_quality_metrics_regex(content),
                errors=errors,
                analysis_time=analysis_time
            )
            
        except Exception as e:
            errors.append(f"Analysis error: {str(e)}")
            return AnalysisResult(
                file_path=file_path,
                language=LanguageType.JAVASCRIPT,
                elements=[],
                imports=[],
                framework_indicators=[],
                agent_patterns=[],
                capabilities=[],
                llm_calls=[],
                complexity_score=0.0,
                quality_metrics={},
                errors=errors,
                analysis_time=__import__('time').time() - start_time
            )
    
    def extract_elements(self, tree: Any) -> List[CodeElement]:
        """Extract elements using regex (placeholder)"""
        return []
    
    def detect_patterns(self, elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Detect agent patterns in JavaScript/TypeScript"""
        # Basic pattern detection
        patterns = []
        for element in elements:
            if 'agent' in element.name.lower() or 'chat' in element.name.lower():
                patterns.append({
                    'type': 'potential_agent',
                    'name': element.name,
                    'confidence': 0.6,
                    'framework': 'javascript'
                })
        return patterns
    
    def _extract_elements_regex(self, content: str) -> List[CodeElement]:
        """Extract code elements using regex"""
        elements = []
        
        # Extract classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            elements.append(CodeElement(
                name=match.group(1),
                node_type=NodeType.CLASS,
                line_number=line_num,
                column=match.start(),
                end_line=line_num
            ))
        
        # Extract functions
        func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=|(\w+)\s*:\s*(?:async\s+)?function)'
        for match in re.finditer(func_pattern, content):
            name = match.group(1) or match.group(2) or match.group(3)
            if name:
                line_num = content[:match.start()].count('\n') + 1
                elements.append(CodeElement(
                    name=name,
                    node_type=NodeType.FUNCTION,
                    line_number=line_num,
                    column=match.start(),
                    end_line=line_num
                ))
        
        return elements
    
    def _extract_imports_regex(self, content: str) -> List[str]:
        """Extract imports using regex"""
        imports = []
        
        # ES6 imports
        import_pattern = r'import.*?from\s+["\']([^"\']+)["\']'
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1))
        
        # CommonJS requires
        require_pattern = r'require\(["\']([^"\']+)["\']\)'
        for match in re.finditer(require_pattern, content):
            imports.append(match.group(1))
        
        return imports
    
    def _detect_frameworks(self, content: str) -> List[str]:
        """Detect JavaScript frameworks"""
        frameworks = []
        
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    frameworks.append(framework)
                    break
        
        return frameworks
    
    def _extract_capabilities_regex(self, content: str) -> List[str]:
        """Extract capabilities using regex"""
        capabilities = []
        
        # Look for common AI/ML capability indicators
        capability_patterns = [
            'chat', 'completion', 'generate', 'analyze', 'process',
            'translate', 'summarize', 'classify', 'predict'
        ]
        
        for pattern in capability_patterns:
            if pattern in content.lower():
                capabilities.append(pattern)
        
        return capabilities
    
    def _detect_llm_calls_regex(self, content: str) -> List[Dict[str, Any]]:
        """Detect LLM calls using regex"""
        calls = []
        
        # OpenAI patterns
        openai_patterns = [
            r'openai\.createCompletion',
            r'openai\.createChatCompletion',
            r'client\.chat\.completions\.create'
        ]
        
        for pattern in openai_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                calls.append({
                    'pattern': pattern,
                    'line': line_num,
                    'call_name': match.group(0)
                })
        
        return calls
    
    def _calculate_quality_metrics_regex(self, content: str) -> Dict[str, Any]:
        """Calculate basic quality metrics"""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('//')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('//')]),
            'empty_lines': len([line for line in lines if not line.strip()])
        }


class MultiLanguageASTAnalyzer:
    """
    Main AST analyzer that coordinates multiple language analyzers
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Initialize language-specific analyzers
        self.analyzers = {
            LanguageType.PYTHON: PythonASTAnalyzer(),
            LanguageType.JAVASCRIPT: JavaScriptASTAnalyzer(),
            LanguageType.TYPESCRIPT: JavaScriptASTAnalyzer(),
        }
        
        # Language detection mapping
        self.language_map = {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
        }
    
    async def analyze_file(self, file_path: Path) -> Optional[AnalysisResult]:
        """
        Analyze a file using the appropriate language analyzer
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            AnalysisResult or None if language not supported
        """
        try:
            # Detect language
            language = self._detect_language(file_path)
            if not language:
                self.logger.debug("Unsupported file type", file_path=str(file_path))
                return None
            
            # Get appropriate analyzer
            analyzer = self.analyzers.get(language)
            if not analyzer:
                self.logger.warning("No analyzer for language", language=language.value)
                return None
            
            # Perform analysis
            self.logger.debug("Analyzing file", file_path=str(file_path), language=language.value)
            result = await analyzer.parse_file(file_path)
            
            self.logger.info("File analysis completed", 
                           file_path=str(file_path),
                           language=language.value,
                           elements=len(result.elements),
                           patterns=len(result.agent_patterns),
                           analysis_time=result.analysis_time)
            
            return result
            
        except Exception as e:
            self.logger.error("File analysis failed", file_path=str(file_path), error=str(e))
            return None
    
    def _detect_language(self, file_path: Path) -> Optional[LanguageType]:
        """Detect programming language from file extension"""
        suffix = file_path.suffix.lower()
        return self.language_map.get(suffix)
    
    async def analyze_directory(self, directory: Path, recursive: bool = True) -> List[AnalysisResult]:
        """
        Analyze all supported files in a directory
        
        Args:
            directory: Directory to analyze
            recursive: Whether to scan recursively
            
        Returns:
            List of analysis results
        """
        results = []
        
        try:
            # Collect files
            if recursive:
                files = list(directory.rglob("*"))
            else:
                files = list(directory.iterdir())
            
            # Filter supported files
            supported_files = [f for f in files if f.is_file() and self._detect_language(f)]
            
            self.logger.info("Starting directory analysis", 
                           directory=str(directory),
                           total_files=len(supported_files))
            
            # Analyze each file
            for file_path in supported_files:
                result = await self.analyze_file(file_path)
                if result:
                    results.append(result)
            
            self.logger.info("Directory analysis completed",
                           directory=str(directory),
                           files_analyzed=len(results))
            
            return results
            
        except Exception as e:
            self.logger.error("Directory analysis failed", directory=str(directory), error=str(e))
            return []
    
    def get_supported_languages(self) -> List[LanguageType]:
        """Get list of supported programming languages"""
        return list(self.analyzers.keys())
    
    def get_language_extensions(self) -> Dict[str, LanguageType]:
        """Get mapping of file extensions to languages"""
        return self.language_map.copy()
