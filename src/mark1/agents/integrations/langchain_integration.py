"""
LangChain Integration for Mark-1 Agent Orchestrator

Provides comprehensive integration with LangChain agents, chains, and tools.
Supports traditional LangChain agents, LangGraph workflows, and tool ecosystems.
"""

import ast
import re
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass
import structlog

from mark1.agents.integrations.base_integration import (
    BaseIntegration, BaseAgentAdapter, IntegrationType, AgentCapability,
    IntegratedAgent, IntegrationError
)
from mark1.agents.discovery import DiscoveredAgent
from mark1.utils.exceptions import IntegrationError


@dataclass
class LangChainAgentInfo:
    """Information about a detected LangChain agent"""
    agent_type: str  # 'react', 'plan-execute', 'langgraph', 'custom'
    tools: List[Dict[str, Any]]
    memory_type: Optional[str]
    llm_info: Dict[str, Any]
    prompt_template: Optional[str]
    state_schema: Optional[Dict[str, Any]]  # For LangGraph
    nodes: List[str] = None  # For LangGraph
    edges: List[Dict[str, Any]] = None  # For LangGraph
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.edges is None:
            self.edges = []


class LangChainAgentAdapter(BaseAgentAdapter):
    """
    Adapter for LangChain agents to provide unified interface
    """
    
    def __init__(self, agent_instance: Any, metadata: Dict[str, Any]):
        super().__init__(agent_instance, metadata)
        self.agent_info = metadata.get('langchain_info', {})
        self.agent_type = self.agent_info.get('agent_type', 'unknown')
    
    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the LangChain agent
        
        Args:
            input_data: Input data containing 'input', 'chat_history', etc.
            
        Returns:
            Standardized response with 'output', 'intermediate_steps', etc.
        """
        try:
            # Handle different LangChain agent types
            if self.agent_type == 'langgraph':
                return await self._invoke_langgraph_agent(input_data)
            elif self.agent_type in ['react', 'plan-execute']:
                return await self._invoke_traditional_agent(input_data)
            else:
                return await self._invoke_generic_agent(input_data)
                
        except Exception as e:
            self.logger.error("Agent invocation failed", error=str(e))
            return {
                "output": None,
                "error": str(e),
                "success": False
            }
    
    async def _invoke_langgraph_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke LangGraph agent"""
        try:
            # LangGraph agents typically use .invoke() or .ainvoke()
            if hasattr(self.agent_instance, 'ainvoke'):
                result = await self.agent_instance.ainvoke(input_data)
            elif hasattr(self.agent_instance, 'invoke'):
                result = self.agent_instance.invoke(input_data)
            else:
                raise IntegrationError("LangGraph agent missing invoke method")
            
            return {
                "output": result,
                "agent_type": "langgraph",
                "success": True
            }
        except Exception as e:
            raise IntegrationError(f"LangGraph invocation failed: {str(e)}")
    
    async def _invoke_traditional_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke traditional LangChain agent (ReAct, etc.)"""
        try:
            # Traditional agents use AgentExecutor
            if hasattr(self.agent_instance, 'ainvoke'):
                result = await self.agent_instance.ainvoke(input_data)
            elif hasattr(self.agent_instance, 'invoke'):
                result = self.agent_instance.invoke(input_data)
            elif hasattr(self.agent_instance, 'run'):
                # Legacy API
                result = {"output": self.agent_instance.run(input_data.get('input', ''))}
            else:
                raise IntegrationError("Agent missing invocation method")
            
            return {
                "output": result.get('output', result),
                "intermediate_steps": result.get('intermediate_steps', []),
                "agent_type": self.agent_type,
                "success": True
            }
        except Exception as e:
            raise IntegrationError(f"Traditional agent invocation failed: {str(e)}")
    
    async def _invoke_generic_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke generic LangChain component"""
        try:
            # Try common invocation patterns
            input_text = input_data.get('input', input_data.get('query', ''))
            
            if hasattr(self.agent_instance, 'ainvoke'):
                result = await self.agent_instance.ainvoke(input_text)
            elif hasattr(self.agent_instance, 'invoke'):
                result = self.agent_instance.invoke(input_text)
            elif hasattr(self.agent_instance, 'run'):
                result = self.agent_instance.run(input_text)
            elif hasattr(self.agent_instance, '__call__'):
                result = self.agent_instance(input_text)
            else:
                raise IntegrationError("No known invocation method found")
            
            return {
                "output": result,
                "agent_type": "generic",
                "success": True
            }
        except Exception as e:
            raise IntegrationError(f"Generic agent invocation failed: {str(e)}")
    
    async def stream(self, input_data: Dict[str, Any]):
        """
        Stream responses from LangChain agent
        """
        try:
            if hasattr(self.agent_instance, 'astream'):
                async for chunk in self.agent_instance.astream(input_data):
                    yield {
                        "chunk": chunk,
                        "agent_type": self.agent_type,
                        "timestamp": __import__('time').time()
                    }
            elif hasattr(self.agent_instance, 'stream'):
                for chunk in self.agent_instance.stream(input_data):
                    yield {
                        "chunk": chunk,
                        "agent_type": self.agent_type,
                        "timestamp": __import__('time').time()
                    }
            else:
                # Fallback to regular invoke
                result = await self.invoke(input_data)
                yield {
                    "chunk": result,
                    "agent_type": self.agent_type,
                    "final": True,
                    "timestamp": __import__('time').time()
                }
        except Exception as e:
            yield {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": __import__('time').time()
            }
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Extract capabilities from LangChain agent"""
        capabilities = []
        
        # Analyze based on agent type
        if self.agent_type == 'react':
            capabilities.extend([
                AgentCapability.CHAT,
                AgentCapability.TOOL_USE,
                AgentCapability.PLANNING,
                AgentCapability.EXECUTION
            ])
        elif self.agent_type == 'langgraph':
            capabilities.extend([
                AgentCapability.PLANNING,
                AgentCapability.EXECUTION,
                AgentCapability.MEMORY
            ])
        elif self.agent_type == 'plan-execute':
            capabilities.extend([
                AgentCapability.PLANNING,
                AgentCapability.EXECUTION,
                AgentCapability.ANALYSIS
            ])
        
        # Check for memory
        if self.agent_info.get('memory_type'):
            capabilities.append(AgentCapability.MEMORY)
        
        # Check for tools
        if self.agent_info.get('tools'):
            capabilities.append(AgentCapability.TOOL_USE)
        
        # Check for multimodal capabilities
        llm_info = self.agent_info.get('llm_info', {})
        if 'vision' in str(llm_info).lower() or 'multimodal' in str(llm_info).lower():
            capabilities.append(AgentCapability.MULTIMODAL)
        
        return list(set(capabilities))  # Remove duplicates
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tools used by the agent"""
        return self.agent_info.get('tools', [])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get LLM model information"""
        return self.agent_info.get('llm_info', {})
    
    async def health_check(self) -> bool:
        """Check if the LangChain agent is healthy"""
        try:
            # Try a simple test invocation
            test_input = {"input": "Hello, are you working?"}
            result = await self.invoke(test_input)
            return result.get('success', False)
        except Exception:
            return False


class LangChainIntegration(BaseIntegration):
    """
    LangChain framework integration for Mark-1
    
    Supports:
    - Traditional LangChain agents (ReAct, Plan-and-Execute)
    - LangGraph workflow agents
    - LangChain chains and tools
    - Memory systems
    """
    
    def __init__(self):
        super().__init__()
        
        # LangChain detection patterns
        self.langchain_patterns = {
            'imports': [
                r'from langchain\.agents import',
                r'from langchain\.chains import',
                r'from langchain\.tools import',
                r'from langchain_.*? import',
                r'import langchain',
                r'from langgraph import'
            ],
            'agent_types': {
                'react': [
                    'create_react_agent',
                    'ReActSingleInputOutputParser',
                    'ReActDocstoreAgent'
                ],
                'plan_execute': [
                    'PlanAndExecute',
                    'load_agent_executor',
                    'PlanningOutputParser'
                ],
                'langgraph': [
                    'StateGraph',
                    'CompiledGraph',
                    'langgraph.graph',
                    'add_node',
                    'add_edge'
                ],
                'tools': [
                    'Tool(',
                    '@tool',
                    'BaseTool',
                    'StructuredTool'
                ]
            },
            'memory': [
                'ConversationBufferMemory',
                'ConversationSummaryMemory',
                'VectorStoreRetrieverMemory',
                'memory='
            ],
            'llm': [
                'ChatOpenAI',
                'OpenAI(',
                'ChatAnthropic',
                'Ollama(',
                'ChatOllama'
            ]
        }
    
    def get_integration_type(self) -> IntegrationType:
        """Return LangChain integration type"""
        return IntegrationType.LANGCHAIN
    
    async def detect_agents(self, scan_path: Path) -> List[DiscoveredAgent]:
        """
        Detect LangChain agents in the given path
        """
        discovered_agents = []
        
        try:
            # Scan Python files for LangChain patterns
            python_files = list(scan_path.rglob("*.py"))
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if self._detect_framework_markers(content):
                        agent_info = await self._analyze_langchain_file(file_path, content)
                        
                        if agent_info:
                            discovered_agent = DiscoveredAgent(
                                name=file_path.stem,
                                file_path=file_path,
                                framework='langchain',
                                class_name=agent_info.get('class_name'),
                                confidence=agent_info.get('confidence', 0.8),
                                capabilities=agent_info.get('capabilities', []),
                                metadata={
                                    'langchain_info': agent_info,
                                    'agent_type': agent_info.get('agent_type', 'unknown')
                                }
                            )
                            discovered_agents.append(discovered_agent)
                            
                            self.logger.info("LangChain agent detected",
                                           file_path=str(file_path),
                                           agent_type=agent_info.get('agent_type'))
                
                except Exception as e:
                    self.logger.warning("Failed to analyze file",
                                      file_path=str(file_path),
                                      error=str(e))
            
            return discovered_agents
            
        except Exception as e:
            self.logger.error("LangChain agent detection failed", error=str(e))
            return []
    
    async def _analyze_langchain_file(self, file_path: Path, content: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a LangChain file to extract agent information
        """
        try:
            # Parse AST for detailed analysis
            tree = ast.parse(content)
            
            agent_info = {
                'file_path': str(file_path),
                'agent_type': 'unknown',
                'tools': [],
                'memory_type': None,
                'llm_info': {},
                'confidence': 0.5,
                'capabilities': []
            }
            
            # Analyze imports and identify agent type
            agent_type = self._identify_agent_type(content)
            agent_info['agent_type'] = agent_type
            
            # Extract tools
            tools = self._extract_tools_from_ast(tree, content)
            agent_info['tools'] = tools
            
            # Extract memory information
            memory_info = self._extract_memory_info(content)
            agent_info['memory_type'] = memory_info
            
            # Extract LLM information
            llm_info = self._extract_llm_info(content)
            agent_info['llm_info'] = llm_info
            
            # Extract capabilities
            capabilities = self.extract_capabilities(content)
            agent_info['capabilities'] = [cap.value for cap in capabilities]
            
            # Find main agent class
            class_name = self._find_agent_class(tree)
            agent_info['class_name'] = class_name
            
            # Calculate confidence score
            confidence = self._calculate_confidence(agent_info, content)
            agent_info['confidence'] = confidence
            
            # Only return if we found meaningful agent patterns
            if confidence > 0.6:
                return agent_info
            
            return None
            
        except Exception as e:
            self.logger.error("LangChain file analysis failed",
                            file_path=str(file_path),
                            error=str(e))
            return None
    
    def _identify_agent_type(self, content: str) -> str:
        """Identify the type of LangChain agent"""
        
        # Check for LangGraph patterns
        langgraph_patterns = self.langchain_patterns['agent_types']['langgraph']
        if any(pattern in content for pattern in langgraph_patterns):
            return 'langgraph'
        
        # Check for ReAct patterns
        react_patterns = self.langchain_patterns['agent_types']['react']
        if any(pattern in content for pattern in react_patterns):
            return 'react'
        
        # Check for Plan-and-Execute patterns
        plan_execute_patterns = self.langchain_patterns['agent_types']['plan_execute']
        if any(pattern in content for pattern in plan_execute_patterns):
            return 'plan_execute'
        
        # Check for tool patterns
        tool_patterns = self.langchain_patterns['agent_types']['tools']
        if any(pattern in content for pattern in tool_patterns):
            return 'tools'
        
        # Check for general agent patterns
        if 'AgentExecutor' in content:
            return 'executor'
        elif 'Chain' in content:
            return 'chain'
        
        return 'unknown'
    
    def _extract_tools_from_ast(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract tool definitions from AST"""
        tools = []
        
        class ToolVisitor(ast.NodeVisitor):
            def __init__(self):
                self.tools = []
            
            def visit_FunctionDef(self, node):
                # Check for @tool decorator
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Name) and decorator.id == 'tool') or \
                       (isinstance(decorator, ast.Call) and 
                        isinstance(decorator.func, ast.Name) and decorator.func.id == 'tool'):
                        
                        tool_info = {
                            'name': node.name,
                            'type': 'function_tool',
                            'docstring': ast.get_docstring(node),
                            'parameters': [arg.arg for arg in node.args.args]
                        }
                        self.tools.append(tool_info)
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for Tool() instantiation
                if isinstance(node.func, ast.Name) and node.func.id == 'Tool':
                    tool_info = {'type': 'tool_instance'}
                    
                    # Extract tool name and description from arguments
                    for keyword in node.keywords:
                        if keyword.arg == 'name' and isinstance(keyword.value, ast.Constant):
                            tool_info['name'] = keyword.value.value
                        elif keyword.arg == 'description' and isinstance(keyword.value, ast.Constant):
                            tool_info['description'] = keyword.value.value
                    
                    if 'name' in tool_info:
                        self.tools.append(tool_info)
                
                self.generic_visit(node)
        
        visitor = ToolVisitor()
        visitor.visit(tree)
        
        return visitor.tools
    
    def _extract_memory_info(self, content: str) -> Optional[str]:
        """Extract memory type information"""
        memory_patterns = self.langchain_patterns['memory']
        
        for pattern in memory_patterns:
            if pattern in content:
                if 'ConversationBufferMemory' in content:
                    return 'buffer'
                elif 'ConversationSummaryMemory' in content:
                    return 'summary'
                elif 'VectorStoreRetrieverMemory' in content:
                    return 'vector'
                else:
                    return 'unknown'
        
        return None
    
    def _extract_llm_info(self, content: str) -> Dict[str, Any]:
        """Extract LLM information"""
        llm_info = {}
        
        llm_patterns = self.langchain_patterns['llm']
        
        for pattern in llm_patterns:
            if pattern in content:
                if 'ChatOpenAI' in content:
                    llm_info['provider'] = 'openai'
                    llm_info['type'] = 'chat'
                elif 'OpenAI(' in content:
                    llm_info['provider'] = 'openai'
                    llm_info['type'] = 'completion'
                elif 'ChatAnthropic' in content:
                    llm_info['provider'] = 'anthropic'
                    llm_info['type'] = 'chat'
                elif 'Ollama' in content:
                    llm_info['provider'] = 'ollama'
                    llm_info['type'] = 'local'
                
                # Extract model name if possible
                model_match = re.search(r'model[_-]?name?["\']\s*:\s*["\']([^"\']+)["\']', content)
                if model_match:
                    llm_info['model'] = model_match.group(1)
                
                break
        
        return llm_info
    
    def _find_agent_class(self, tree: ast.AST) -> Optional[str]:
        """Find the main agent class in the file"""
        
        class ClassVisitor(ast.NodeVisitor):
            def __init__(self):
                self.agent_classes = []
            
            def visit_ClassDef(self, node):
                # Look for classes that might be agents
                if any(keyword in node.name.lower() for keyword in ['agent', 'executor', 'graph', 'chain']):
                    self.agent_classes.append(node.name)
                
                self.generic_visit(node)
        
        visitor = ClassVisitor()
        visitor.visit(tree)
        
        # Return the first agent class found
        return visitor.agent_classes[0] if visitor.agent_classes else None
    
    def _calculate_confidence(self, agent_info: Dict[str, Any], content: str) -> float:
        """Calculate confidence score for agent detection"""
        confidence = 0.0
        
        # Base score for having LangChain imports
        if any(pattern in content for pattern in self.langchain_patterns['imports']):
            confidence += 0.3
        
        # Agent type specific scores
        agent_type = agent_info.get('agent_type', 'unknown')
        if agent_type != 'unknown':
            confidence += 0.3
        
        # Tool usage
        if agent_info.get('tools'):
            confidence += 0.2
        
        # Memory usage
        if agent_info.get('memory_type'):
            confidence += 0.1
        
        # LLM configuration
        if agent_info.get('llm_info'):
            confidence += 0.1
        
        # Class definition
        if agent_info.get('class_name'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def integrate_agent(self, discovered_agent: DiscoveredAgent) -> IntegratedAgent:
        """
        Integrate a discovered LangChain agent
        """
        try:
            # Load the agent module
            agent_instance = await self._load_agent_instance(discovered_agent)
            
            # Create adapter
            adapter = self.create_adapter(agent_instance, discovered_agent.metadata)
            
            # Extract capabilities
            capabilities = adapter.get_capabilities()
            
            # Create integrated agent
            integrated_agent = IntegratedAgent(
                id=f"langchain_{discovered_agent.name}_{hash(str(discovered_agent.file_path))}",
                name=discovered_agent.name,
                framework=IntegrationType.LANGCHAIN,
                original_path=discovered_agent.file_path,
                adapter=adapter,
                capabilities=capabilities,
                metadata=discovered_agent.metadata,
                tools=[tool.get('name', 'unknown') for tool in adapter.get_tools()],
                model_info=adapter.get_model_info()
            )
            
            self.logger.info("LangChain agent integrated successfully",
                           agent_name=integrated_agent.name,
                           capabilities=len(capabilities),
                           tools=len(integrated_agent.tools))
            
            return integrated_agent
            
        except Exception as e:
            raise IntegrationError(f"Failed to integrate LangChain agent: {str(e)}")
    
    async def _load_agent_instance(self, discovered_agent: DiscoveredAgent) -> Any:
        """
        Load the actual agent instance from the discovered agent
        """
        try:
            # For now, return a mock instance
            # In a real implementation, you would dynamically import and instantiate the agent
            mock_instance = {
                'name': discovered_agent.name,
                'file_path': str(discovered_agent.file_path),
                'metadata': discovered_agent.metadata
            }
            
            return mock_instance
            
        except Exception as e:
            raise IntegrationError(f"Failed to load agent instance: {str(e)}")
    
    def create_adapter(self, agent_instance: Any, metadata: Dict[str, Any]) -> LangChainAgentAdapter:
        """Create LangChain agent adapter"""
        return LangChainAgentAdapter(agent_instance, metadata)
    
    def extract_capabilities(self, agent_code: str, agent_instance: Any = None) -> List[AgentCapability]:
        """Extract capabilities from LangChain agent code"""
        capabilities = []
        
        # Check for chat capabilities
        if any(pattern in agent_code.lower() for pattern in ['chat', 'conversation', 'message']):
            capabilities.append(AgentCapability.CHAT)
        
        # Check for tool usage
        if any(pattern in agent_code for pattern in ['Tool', '@tool', 'tools=']):
            capabilities.append(AgentCapability.TOOL_USE)
        
        # Check for memory
        if any(pattern in agent_code for pattern in ['Memory', 'memory=']):
            capabilities.append(AgentCapability.MEMORY)
        
        # Check for planning capabilities
        if any(pattern in agent_code for pattern in ['plan', 'execute', 'ReAct']):
            capabilities.append(AgentCapability.PLANNING)
            capabilities.append(AgentCapability.EXECUTION)
        
        # Check for analysis capabilities
        if any(pattern in agent_code.lower() for pattern in ['analyze', 'analysis', 'summarize']):
            capabilities.append(AgentCapability.ANALYSIS)
        
        # Check for generation capabilities
        if any(pattern in agent_code.lower() for pattern in ['generate', 'create', 'write']):
            capabilities.append(AgentCapability.GENERATION)
        
        return list(set(capabilities))  # Remove duplicates
    
    def extract_tools(self, agent_code: str, agent_instance: Any = None) -> List[Dict[str, Any]]:
        """Extract tools from LangChain agent code"""
        tools = []
        
        # Use AST to extract tools
        try:
            tree = ast.parse(agent_code)
            tools = self._extract_tools_from_ast(tree, agent_code)
        except:
            # Fallback to regex-based extraction
            tool_pattern = r'Tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']'
            matches = re.findall(tool_pattern, agent_code)
            for match in matches:
                tools.append({'name': match, 'type': 'extracted'})
        
        return tools
    
    def _detect_framework_markers(self, content: str) -> bool:
        """Detect LangChain framework markers"""
        import_patterns = self.langchain_patterns['imports']
        
        # Check for LangChain imports
        for pattern in import_patterns:
            if re.search(pattern, content):
                return True
        
        return False
