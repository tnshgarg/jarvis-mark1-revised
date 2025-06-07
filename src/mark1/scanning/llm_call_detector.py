"""
LLM API Call Detection and Replacement Engine

Detects API calls to various LLM providers (OpenAI, Anthropic, etc.)
and provides suggestions for replacing them with local alternatives.
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from mark1.utils.exceptions import DetectionException, ReplacementException
from mark1.utils.constants import LLM_CALL_PATTERNS


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    PALM = "palm"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    LOCAL = "local"
    UNKNOWN = "unknown"


class CallType(Enum):
    """Types of LLM API calls"""
    CHAT_COMPLETION = "chat_completion"
    TEXT_COMPLETION = "text_completion"
    EMBEDDING = "embedding"
    FINE_TUNING = "fine_tuning"
    MODERATION = "moderation"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    UNKNOWN = "unknown"


@dataclass
class LLMCall:
    """Represents a detected LLM API call"""
    provider: LLMProvider
    call_type: CallType
    method_name: str
    line_number: int
    column: int
    file_path: Path
    code_snippet: str
    parameters: Dict[str, Any]
    model_name: Optional[str] = None
    api_key_usage: Optional[str] = None
    cost_estimate: Optional[float] = None
    confidence: float = 0.0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class ReplacementSuggestion:
    """Suggestion for replacing an LLM call with local alternative"""
    original_call: LLMCall
    replacement_code: str
    replacement_provider: LLMProvider
    compatibility_score: float
    required_changes: List[str]
    migration_notes: str
    estimated_performance: Dict[str, Any]


@dataclass
class DetectionResult:
    """Result of LLM call detection analysis"""
    file_path: Path
    calls: List[LLMCall]
    total_calls: int
    providers_found: Set[LLMProvider]
    call_types_found: Set[CallType]
    estimated_monthly_cost: float
    replacement_suggestions: List[ReplacementSuggestion]
    analysis_time: float
    errors: List[str]


class LLMCallDetector:
    """
    Advanced LLM API call detector and analyzer
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Provider detection patterns
        self.provider_patterns = {
            LLMProvider.OPENAI: [
                r'openai\.ChatCompletion\.create',
                r'openai\.Completion\.create',
                r'openai\.Embedding\.create',
                r'client\.chat\.completions\.create',
                r'client\.completions\.create',
                r'client\.embeddings\.create',
                r'OpenAI\(\)',
                r'from openai import',
                r'import openai'
            ],
            LLMProvider.ANTHROPIC: [
                r'anthropic\.messages\.create',
                r'anthropic\.completions\.create',
                r'client\.messages\.create',
                r'Anthropic\(\)',
                r'from anthropic import',
                r'import anthropic'
            ],
            LLMProvider.HUGGINGFACE: [
                r'transformers\.pipeline',
                r'AutoModel\.from_pretrained',
                r'AutoTokenizer\.from_pretrained',
                r'from transformers import',
                r'huggingface_hub\.InferenceClient',
                r'pipeline\('
            ],
            LLMProvider.OLLAMA: [
                r'ollama\.chat',
                r'ollama\.generate',
                r'ollama\.Client',
                r'from ollama import',
                r'import ollama'
            ]
        }
        
        # Call type detection patterns
        self.call_type_patterns = {
            CallType.CHAT_COMPLETION: [
                'chat.completions.create',
                'ChatCompletion.create',
                'messages.create',
                'chat(',
                'messages='
            ],
            CallType.TEXT_COMPLETION: [
                'completions.create',
                'Completion.create',
                'generate(',
                'completion(',
                'prompt='
            ],
            CallType.EMBEDDING: [
                'embeddings.create',
                'Embedding.create',
                'encode(',
                'embed('
            ],
            CallType.STREAMING: [
                'stream=True',
                'streaming=True',
                'stream_chat',
                'stream_generate'
            ]
        }
        
        # Cost estimation (rough estimates per 1K tokens)
        self.cost_estimates = {
            LLMProvider.OPENAI: {
                'gpt-4': {'input': 0.03, 'output': 0.06},
                'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
                'text-davinci-003': {'input': 0.02, 'output': 0.02}
            },
            LLMProvider.ANTHROPIC: {
                'claude-3': {'input': 0.015, 'output': 0.075},
                'claude-2': {'input': 0.008, 'output': 0.024}
            }
        }
    
    async def detect_file(self, file_path: Path) -> DetectionResult:
        """
        Detect LLM API calls in a single file
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Detection result with all found calls
        """
        start_time = __import__('time').time()
        errors = []
        calls = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect calls based on file type
            if file_path.suffix == '.py':
                calls = await self._detect_python_calls(content, file_path)
            elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
                calls = await self._detect_javascript_calls(content, file_path)
            else:
                # Generic text-based detection
                calls = await self._detect_generic_calls(content, file_path)
            
            # Analyze detected calls
            providers_found = {call.provider for call in calls}
            call_types_found = {call.call_type for call in calls}
            
            # Estimate costs
            estimated_cost = self._estimate_monthly_cost(calls)
            
            # Generate replacement suggestions
            replacement_suggestions = await self._generate_replacements(calls)
            
            analysis_time = __import__('time').time() - start_time
            
            return DetectionResult(
                file_path=file_path,
                calls=calls,
                total_calls=len(calls),
                providers_found=providers_found,
                call_types_found=call_types_found,
                estimated_monthly_cost=estimated_cost,
                replacement_suggestions=replacement_suggestions,
                analysis_time=analysis_time,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Detection error: {str(e)}")
            self.logger.error("LLM call detection failed", file_path=str(file_path), error=str(e))
            
            return DetectionResult(
                file_path=file_path,
                calls=[],
                total_calls=0,
                providers_found=set(),
                call_types_found=set(),
                estimated_monthly_cost=0.0,
                replacement_suggestions=[],
                analysis_time=__import__('time').time() - start_time,
                errors=errors
            )
    
    async def _detect_python_calls(self, content: str, file_path: Path) -> List[LLMCall]:
        """Detect LLM calls in Python code using AST analysis"""
        calls = []
        
        try:
            tree = ast.parse(content)
            
            class LLMCallVisitor(ast.NodeVisitor):
                def __init__(self, detector):
                    self.detector = detector
                    self.calls = []
                    self.imports = {}
                
                def visit_Import(self, node):
                    """Track imports for context"""
                    for alias in node.names:
                        self.imports[alias.asname or alias.name] = alias.name
                
                def visit_ImportFrom(self, node):
                    """Track from imports for context"""
                    if node.module:
                        for alias in node.names:
                            full_name = f"{node.module}.{alias.name}"
                            self.imports[alias.asname or alias.name] = full_name
                
                def visit_Call(self, node):
                    """Visit function calls to detect LLM API calls"""
                    call_name = self._get_call_name(node)
                    
                    # Check against known patterns
                    for provider, patterns in self.detector.provider_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, call_name):
                                llm_call = self._create_llm_call(node, call_name, provider, file_path)
                                if llm_call:
                                    self.calls.append(llm_call)
                                break
                    
                    self.generic_visit(node)
                
                def _get_call_name(self, node):
                    """Extract the full call name"""
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
                
                def _create_llm_call(self, node, call_name, provider, file_path):
                    """Create LLMCall object from AST node"""
                    try:
                        # Extract parameters
                        parameters = self._extract_call_parameters(node)
                        
                        # Determine call type
                        call_type = self.detector._determine_call_type(call_name, parameters)
                        
                        # Extract model name
                        model_name = parameters.get('model') or parameters.get('engine')
                        
                        # Get code snippet
                        code_snippet = call_name  # Simplified for now
                        
                        return LLMCall(
                            provider=provider,
                            call_type=call_type,
                            method_name=call_name,
                            line_number=node.lineno,
                            column=node.col_offset,
                            file_path=file_path,
                            code_snippet=code_snippet,
                            parameters=parameters,
                            model_name=model_name,
                            confidence=0.9
                        )
                    except Exception as e:
                        self.detector.logger.warning("Failed to create LLM call", error=str(e))
                        return None
                
                def _extract_call_parameters(self, node):
                    """Extract parameters from function call"""
                    parameters = {}
                    
                    # Extract keyword arguments
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Constant):
                            parameters[keyword.arg] = keyword.value.value
                        elif isinstance(keyword.value, ast.Str):  # For older Python versions
                            parameters[keyword.arg] = keyword.value.s
                        else:
                            parameters[keyword.arg] = str(keyword.value)
                    
                    return parameters
            
            visitor = LLMCallVisitor(self)
            visitor.visit(tree)
            calls = visitor.calls
            
        except SyntaxError as e:
            self.logger.warning("Python syntax error", file_path=str(file_path), error=str(e))
        except Exception as e:
            self.logger.error("Python AST analysis failed", file_path=str(file_path), error=str(e))
        
        return calls
    
    async def _detect_javascript_calls(self, content: str, file_path: Path) -> List[LLMCall]:
        """Detect LLM calls in JavaScript/TypeScript code using regex"""
        calls = []
        
        # JavaScript/TypeScript patterns
        js_patterns = {
            LLMProvider.OPENAI: [
                r'openai\.createCompletion',
                r'openai\.createChatCompletion',
                r'client\.chat\.completions\.create',
                r'new OpenAI\(',
                r'import.*openai'
            ],
            LLMProvider.ANTHROPIC: [
                r'anthropic\.messages\.create',
                r'new Anthropic\(',
                r'import.*anthropic'
            ]
        }
        
        for provider, patterns in js_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    call_name = match.group(0)
                    
                    # Determine call type
                    call_type = self._determine_call_type(call_name, {})
                    
                    call = LLMCall(
                        provider=provider,
                        call_type=call_type,
                        method_name=call_name,
                        line_number=line_num,
                        column=match.start(),
                        file_path=file_path,
                        code_snippet=call_name,
                        parameters={},
                        confidence=0.7
                    )
                    calls.append(call)
        
        return calls
    
    async def _detect_generic_calls(self, content: str, file_path: Path) -> List[LLMCall]:
        """Generic text-based detection for any file type"""
        calls = []
        
        # Generic patterns that might indicate LLM usage
        generic_patterns = [
            (r'openai', LLMProvider.OPENAI),
            (r'anthropic', LLMProvider.ANTHROPIC),
            (r'gpt-[34]', LLMProvider.OPENAI),
            (r'claude', LLMProvider.ANTHROPIC),
            (r'ChatCompletion', LLMProvider.OPENAI),
            (r'completions\.create', LLMProvider.OPENAI)
        ]
        
        for pattern, provider in generic_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count('\n') + 1
                
                call = LLMCall(
                    provider=provider,
                    call_type=CallType.UNKNOWN,
                    method_name=match.group(0),
                    line_number=line_num,
                    column=match.start(),
                    file_path=file_path,
                    code_snippet=match.group(0),
                    parameters={},
                    confidence=0.5
                )
                calls.append(call)
        
        return calls
    
    def _determine_call_type(self, call_name: str, parameters: Dict[str, Any]) -> CallType:
        """Determine the type of LLM call based on method name and parameters"""
        call_name_lower = call_name.lower()
        
        for call_type, patterns in self.call_type_patterns.items():
            for pattern in patterns:
                if pattern.lower() in call_name_lower:
                    return call_type
        
        # Check parameters for additional clues
        if 'messages' in parameters:
            return CallType.CHAT_COMPLETION
        elif 'prompt' in parameters:
            return CallType.TEXT_COMPLETION
        elif 'input' in parameters and 'embed' in call_name_lower:
            return CallType.EMBEDDING
        
        return CallType.UNKNOWN
    
    def _estimate_monthly_cost(self, calls: List[LLMCall]) -> float:
        """Estimate monthly cost based on detected calls"""
        total_cost = 0.0
        
        for call in calls:
            if call.provider in self.cost_estimates and call.model_name:
                model_costs = self.cost_estimates[call.provider].get(call.model_name, {})
                if model_costs:
                    # Rough estimate: assume 1000 tokens per call, 100 calls per month
                    estimated_tokens = 1000
                    monthly_calls = 100
                    
                    input_cost = model_costs.get('input', 0) * estimated_tokens / 1000
                    output_cost = model_costs.get('output', 0) * estimated_tokens / 1000
                    
                    call_cost = (input_cost + output_cost) * monthly_calls
                    total_cost += call_cost
        
        return total_cost
    
    async def _generate_replacements(self, calls: List[LLMCall]) -> List[ReplacementSuggestion]:
        """Generate replacement suggestions for detected LLM calls"""
        suggestions = []
        
        for call in calls:
            suggestion = await self._create_replacement_suggestion(call)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _create_replacement_suggestion(self, call: LLMCall) -> Optional[ReplacementSuggestion]:
        """Create a replacement suggestion for a specific LLM call"""
        try:
            if call.provider == LLMProvider.OPENAI:
                return self._suggest_openai_replacement(call)
            elif call.provider == LLMProvider.ANTHROPIC:
                return self._suggest_anthropic_replacement(call)
            else:
                return None
                
        except Exception as e:
            self.logger.error("Failed to create replacement suggestion", error=str(e))
            return None
    
    def _suggest_openai_replacement(self, call: LLMCall) -> ReplacementSuggestion:
        """Suggest Ollama replacement for OpenAI calls"""
        if call.call_type == CallType.CHAT_COMPLETION:
            replacement_code = """
# Replace OpenAI chat completion with Ollama
import ollama

response = ollama.chat(
    model='llama2',  # or your preferred local model
    messages=messages,
    stream=False
)
result = response['message']['content']
"""
        elif call.call_type == CallType.TEXT_COMPLETION:
            replacement_code = """
# Replace OpenAI text completion with Ollama
import ollama

response = ollama.generate(
    model='llama2',  # or your preferred local model
    prompt=prompt
)
result = response['response']
"""
        else:
            replacement_code = "# Generic Ollama replacement needed"
        
        return ReplacementSuggestion(
            original_call=call,
            replacement_code=replacement_code,
            replacement_provider=LLMProvider.OLLAMA,
            compatibility_score=0.8,
            required_changes=[
                "Install Ollama",
                "Download appropriate model (e.g., llama2)",
                "Update import statements",
                "Modify response parsing"
            ],
            migration_notes="Ollama provides local LLM inference with good compatibility",
            estimated_performance={
                'speed': 'Depends on hardware',
                'cost': 'Free after setup',
                'privacy': 'Complete local control'
            }
        )
    
    def _suggest_anthropic_replacement(self, call: LLMCall) -> ReplacementSuggestion:
        """Suggest Ollama replacement for Anthropic calls"""
        replacement_code = """
# Replace Anthropic with Ollama
import ollama

response = ollama.chat(
    model='llama2',  # or your preferred local model
    messages=[{'role': 'user', 'content': prompt}]
)
result = response['message']['content']
"""
        
        return ReplacementSuggestion(
            original_call=call,
            replacement_code=replacement_code,
            replacement_provider=LLMProvider.OLLAMA,
            compatibility_score=0.7,
            required_changes=[
                "Install Ollama",
                "Download appropriate model",
                "Convert message format",
                "Update response handling"
            ],
            migration_notes="Anthropic message format needs conversion to Ollama format",
            estimated_performance={
                'speed': 'Depends on hardware',
                'cost': 'Free after setup',
                'privacy': 'Complete local control'
            }
        )
    
    async def detect_directory(self, directory: Path, recursive: bool = True) -> List[DetectionResult]:
        """
        Detect LLM calls in all files in a directory
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of detection results
        """
        results = []
        
        try:
            # Collect files
            if recursive:
                files = list(directory.rglob("*"))
            else:
                files = list(directory.iterdir())
            
            # Filter supported files
            supported_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.txt', '.md'}
            supported_files = [
                f for f in files 
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            
            self.logger.info("Starting LLM call detection", 
                           directory=str(directory),
                           total_files=len(supported_files))
            
            # Analyze each file
            for file_path in supported_files:
                result = await self.detect_file(file_path)
                if result.calls:  # Only include files with detected calls
                    results.append(result)
            
            self.logger.info("LLM call detection completed",
                           directory=str(directory),
                           files_with_calls=len(results),
                           total_calls=sum(r.total_calls for r in results))
            
            return results
            
        except Exception as e:
            self.logger.error("Directory LLM detection failed", directory=str(directory), error=str(e))
            return []
    
    def generate_migration_report(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """Generate a comprehensive migration report"""
        total_calls = sum(r.total_calls for r in results)
        total_cost = sum(r.estimated_monthly_cost for r in results)
        
        provider_summary = {}
        call_type_summary = {}
        
        for result in results:
            for provider in result.providers_found:
                provider_summary[provider.value] = provider_summary.get(provider.value, 0) + 1
            
            for call_type in result.call_types_found:
                call_type_summary[call_type.value] = call_type_summary.get(call_type.value, 0) + 1
        
        return {
            'summary': {
                'total_files_analyzed': len(results),
                'total_llm_calls': total_calls,
                'estimated_monthly_cost': total_cost,
                'potential_savings': total_cost,  # Assuming free local alternatives
            },
            'providers': provider_summary,
            'call_types': call_type_summary,
            'files': [
                {
                    'path': str(r.file_path),
                    'calls': r.total_calls,
                    'cost': r.estimated_monthly_cost,
                    'providers': [p.value for p in r.providers_found]
                }
                for r in results
            ],
            'migration_complexity': self._assess_migration_complexity(results)
        }
    
    def _assess_migration_complexity(self, results: List[DetectionResult]) -> str:
        """Assess the overall complexity of migrating to local LLM"""
        total_calls = sum(r.total_calls for r in results)
        unique_providers = set()
        unique_call_types = set()
        
        for result in results:
            unique_providers.update(result.providers_found)
            unique_call_types.update(result.call_types_found)
        
        if total_calls == 0:
            return "none"
        elif total_calls <= 5 and len(unique_providers) <= 1:
            return "low"
        elif total_calls <= 20 and len(unique_providers) <= 2:
            return "medium"
        else:
            return "high"
