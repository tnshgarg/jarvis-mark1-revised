"""
Advanced LangChain Integration for Mark-1 Agent Orchestrator

Session 14: Advanced LangChain & LangGraph Integration
Extends the basic LangChain integration with sophisticated features:
- Advanced LangGraph state management
- Multi-agent LangChain coordination
- Complex workflow adaptation
- Comprehensive tool ecosystem integration
"""

import ast
import re
import json
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict

from mark1.agents.integrations.base_integration import (
    BaseIntegration, BaseAgentAdapter, IntegrationType, AgentCapability,
    IntegratedAgent, IntegrationError
)
from mark1.agents.integrations.langchain_integration import (
    LangChainIntegration, LangChainAgentAdapter, LangChainAgentInfo
)
from mark1.agents.discovery import DiscoveredAgent
from mark1.utils.exceptions import IntegrationError


class LangGraphNodeType(Enum):
    """Types of LangGraph nodes"""
    CONDITIONAL = "conditional"
    STANDARD = "standard"
    START = "start"
    END = "end"
    TOOL = "tool"
    AGENT = "agent"
    PARALLEL = "parallel"
    LOOP = "loop"


class WorkflowComplexity(Enum):
    """Workflow complexity levels"""
    SIMPLE = "simple"          # Linear workflow, no conditionals
    MODERATE = "moderate"      # Some conditionals, basic branching
    COMPLEX = "complex"        # Multiple branches, loops
    ADVANCED = "advanced"      # Nested workflows, dynamic routing


@dataclass
class LangGraphNode:
    """Represents a node in a LangGraph workflow"""
    node_id: str
    name: str
    node_type: LangGraphNodeType
    function_name: Optional[str] = None
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LangGraphEdge:
    """Represents an edge in a LangGraph workflow"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    condition_function: Optional[str] = None
    edge_type: str = "standard"  # standard, conditional, loop
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LangGraphStateSchema:
    """Represents the state schema of a LangGraph workflow"""
    schema_name: str
    fields: Dict[str, Dict[str, Any]]  # field_name -> {type, description, required}
    base_classes: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LangGraphWorkflow:
    """Complete representation of a LangGraph workflow"""
    workflow_id: str
    name: str
    state_schema: LangGraphStateSchema
    nodes: List[LangGraphNode]
    edges: List[LangGraphEdge]
    entry_point: Optional[str] = None
    complexity: WorkflowComplexity = WorkflowComplexity.SIMPLE
    tools: List[Dict[str, Any]] = field(default_factory=list)
    memory_config: Optional[Dict[str, Any]] = None
    llm_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentConfiguration:
    """Configuration for multi-agent LangChain systems"""
    coordinator_agent: Optional[str] = None
    agent_roles: Dict[str, str] = field(default_factory=dict)  # agent_id -> role
    communication_protocol: str = "sequential"  # sequential, parallel, hierarchical
    shared_memory: Optional[str] = None
    conflict_resolution: str = "priority"  # priority, voting, coordinator
    agent_dependencies: Dict[str, List[str]] = field(default_factory=dict)


class AdvancedLangChainAgentAdapter(LangChainAgentAdapter):
    """
    Advanced adapter for complex LangChain agents with enhanced capabilities
    """
    
    def __init__(self, agent_instance: Any, metadata: Dict[str, Any]):
        super().__init__(agent_instance, metadata)
        self.workflow_info = metadata.get('langgraph_workflow')
        self.multi_agent_config = metadata.get('multi_agent_config')
        self.state_manager = None
        self._initialize_advanced_features()
    
    def _initialize_advanced_features(self):
        """Initialize advanced adapter features"""
        if self.workflow_info:
            self.state_manager = LangGraphStateManager(self.workflow_info)
    
    async def invoke_with_state(self, input_data: Dict[str, Any], initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced invocation with explicit state management
        """
        try:
            if self.agent_type == 'langgraph' and self.state_manager:
                return await self._invoke_with_state_management(input_data, initial_state)
            else:
                return await super().invoke(input_data)
        except Exception as e:
            self.logger.error("State-aware invocation failed", error=str(e))
            return {
                "output": None,
                "error": str(e),
                "success": False,
                "state": initial_state
            }
    
    async def _invoke_with_state_management(self, input_data: Dict[str, Any], initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Invoke LangGraph agent with state management"""
        try:
            # Prepare state
            if initial_state:
                state = initial_state
            else:
                state = self.state_manager.create_initial_state(input_data)
            
            # Validate state schema
            validation_result = self.state_manager.validate_state(state)
            if not validation_result.get('valid', False):
                return {
                    "output": None,
                    "error": f"Invalid state: {validation_result.get('errors', [])}",
                    "success": False,
                    "state": state
                }
            
            # Execute workflow
            if hasattr(self.agent_instance, 'ainvoke'):
                result = await self.agent_instance.ainvoke(state)
            elif hasattr(self.agent_instance, 'invoke'):
                result = self.agent_instance.invoke(state)
            else:
                raise IntegrationError("LangGraph agent missing invoke method")
            
            # Extract execution trace
            execution_trace = self.state_manager.extract_execution_trace(result)
            
            return {
                "output": result,
                "agent_type": "langgraph",
                "success": True,
                "state": result if isinstance(result, dict) else state,
                "execution_trace": execution_trace,
                "workflow_info": self.workflow_info.workflow_id if self.workflow_info else None
            }
            
        except Exception as e:
            raise IntegrationError(f"LangGraph state management failed: {str(e)}")
    
    async def stream_with_state(self, input_data: Dict[str, Any], initial_state: Optional[Dict[str, Any]] = None):
        """Stream responses with state tracking"""
        try:
            state = initial_state or {}
            
            if self.agent_type == 'langgraph' and hasattr(self.agent_instance, 'astream'):
                async for chunk in self.agent_instance.astream(state):
                    # Track state changes
                    if isinstance(chunk, dict):
                        state.update(chunk)
                    
                    yield {
                        "chunk": chunk,
                        "agent_type": self.agent_type,
                        "state": state.copy(),
                        "timestamp": __import__('time').time()
                    }
            else:
                # Fallback to regular streaming
                async for chunk in super().stream(input_data):
                    yield chunk
                    
        except Exception as e:
            yield {
                "error": str(e),
                "agent_type": self.agent_type,
                "state": initial_state,
                "timestamp": __import__('time').time()
            }
    
    def get_workflow_info(self) -> Optional[LangGraphWorkflow]:
        """Get detailed workflow information"""
        return self.workflow_info
    
    def get_multi_agent_config(self) -> Optional[MultiAgentConfiguration]:
        """Get multi-agent configuration"""
        return self.multi_agent_config
    
    async def coordinate_with_agents(self, agent_adapters: List['AdvancedLangChainAgentAdapter'], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate execution with other agents"""
        if not self.multi_agent_config:
            return await self.invoke(input_data)
        
        try:
            protocol = self.multi_agent_config.communication_protocol
            
            if protocol == "sequential":
                return await self._sequential_coordination(agent_adapters, input_data)
            elif protocol == "parallel":
                return await self._parallel_coordination(agent_adapters, input_data)
            elif protocol == "hierarchical":
                return await self._hierarchical_coordination(agent_adapters, input_data)
            else:
                # Default to single agent execution
                return await self.invoke(input_data)
                
        except Exception as e:
            self.logger.error("Multi-agent coordination failed", error=str(e))
            return {
                "output": None,
                "error": f"Coordination failed: {str(e)}",
                "success": False
            }
    
    async def _sequential_coordination(self, agent_adapters: List['AdvancedLangChainAgentAdapter'], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agents sequentially"""
        results = []
        current_input = input_data
        
        for adapter in agent_adapters:
            result = await adapter.invoke(current_input)
            results.append(result)
            
            # Use output as input for next agent
            if result.get('success', False):
                current_input = {"input": result.get('output', '')}
        
        return {
            "output": results[-1].get('output') if results else None,
            "success": all(r.get('success', False) for r in results),
            "coordination_type": "sequential",
            "agent_results": results
        }
    
    async def _parallel_coordination(self, agent_adapters: List['AdvancedLangChainAgentAdapter'], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agents in parallel"""
        tasks = [adapter.invoke(input_data) for adapter in agent_adapters]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif result.get('success', False):
                successful_results.append(result)
            else:
                errors.append(result.get('error', 'Unknown error'))
        
        return {
            "output": successful_results,
            "success": len(errors) == 0,
            "coordination_type": "parallel",
            "successful_agents": len(successful_results),
            "errors": errors
        }
    
    async def _hierarchical_coordination(self, agent_adapters: List['AdvancedLangChainAgentAdapter'], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agents in hierarchical order"""
        if not self.multi_agent_config.coordinator_agent:
            return await self._sequential_coordination(agent_adapters, input_data)
        
        # Execute coordinator first
        coordinator_result = await self.invoke(input_data)
        
        if not coordinator_result.get('success', False):
            return coordinator_result
        
        # Distribute tasks to subordinate agents
        subordinate_tasks = []
        for adapter in agent_adapters:
            if adapter != self:  # Don't include coordinator
                subordinate_tasks.append(adapter.invoke({
                    "input": coordinator_result.get('output', ''),
                    "coordinator_instructions": input_data
                }))
        
        subordinate_results = await asyncio.gather(*subordinate_tasks, return_exceptions=True)
        
        return {
            "output": coordinator_result.get('output'),
            "success": coordinator_result.get('success', False),
            "coordination_type": "hierarchical",
            "coordinator_result": coordinator_result,
            "subordinate_results": subordinate_results
        }


class LangGraphStateManager:
    """
    Advanced state management for LangGraph workflows
    """
    
    def __init__(self, workflow: LangGraphWorkflow):
        self.workflow = workflow
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def create_initial_state(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial state based on schema and input"""
        state = {}
        
        # Initialize required fields
        for field_name, field_info in self.workflow.state_schema.fields.items():
            if field_info.get('required', False):
                default_value = self._get_default_value(field_info.get('type', 'str'))
                state[field_name] = input_data.get(field_name, default_value)
        
        # Add input data
        state.update(input_data)
        
        return state
    
    def validate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state against schema"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        schema_fields = self.workflow.state_schema.fields
        
        # Check required fields
        for field_name, field_info in schema_fields.items():
            if field_info.get('required', False) and field_name not in state:
                validation_result["errors"].append(f"Missing required field: {field_name}")
                validation_result["valid"] = False
        
        # Check field types
        for field_name, value in state.items():
            if field_name in schema_fields:
                expected_type = schema_fields[field_name].get('type', 'any')
                if not self._validate_type(value, expected_type):
                    validation_result["warnings"].append(f"Type mismatch for {field_name}: expected {expected_type}")
        
        return validation_result
    
    def extract_execution_trace(self, result: Any) -> List[Dict[str, Any]]:
        """Extract execution trace from workflow result"""
        trace = []
        
        if isinstance(result, dict):
            # Look for common trace patterns
            if 'steps' in result:
                trace = result['steps']
            elif 'intermediate_steps' in result:
                trace = result['intermediate_steps']
            elif 'execution_log' in result:
                trace = result['execution_log']
        
        return trace
    
    def _get_default_value(self, type_str: str) -> Any:
        """Get default value for a type"""
        type_defaults = {
            'str': '',
            'int': 0,
            'float': 0.0,
            'bool': False,
            'list': [],
            'dict': {},
            'any': None
        }
        return type_defaults.get(type_str.lower(), None)
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_checks = {
            'str': lambda v: isinstance(v, str),
            'int': lambda v: isinstance(v, int),
            'float': lambda v: isinstance(v, (int, float)),
            'bool': lambda v: isinstance(v, bool),
            'list': lambda v: isinstance(v, list),
            'dict': lambda v: isinstance(v, dict),
            'any': lambda v: True
        }
        
        check_func = type_checks.get(expected_type.lower())
        return check_func(value) if check_func else True


class AdvancedLangChainIntegration(LangChainIntegration):
    """
    Advanced LangChain integration with sophisticated features
    Session 14: Enhanced capabilities for complex workflows and multi-agent systems
    """
    
    def __init__(self):
        super().__init__()
        self.workflow_analyzer = LangGraphWorkflowAnalyzer()
        self.multi_agent_detector = MultiAgentDetector()
        self.tool_ecosystem_mapper = ToolEcosystemMapper()
        
        # Enhanced patterns for advanced detection
        self.advanced_patterns = {
            'conditional_logic': [
                r'add_conditional_edges',
                r'if.*state\[',
                r'def.*should_.*\(',
                r'Condition\(',
                r'branch_on'
            ],
            'parallel_execution': [
                r'ParallelExecution',
                r'parallel_tool_executor',
                r'asyncio\.gather',
                r'concurrent\.futures'
            ],
            'dynamic_routing': [
                r'dynamic_route',
                r'route_by_',
                r'DynamicRouter',
                r'RouteDecision'
            ],
            'multi_agent': [
                r'MultiAgentExecutor',
                r'AgentTeam',
                r'CoordinatorAgent',
                r'agent_coordination',
                r'CrewAI.*import'
            ],
            'state_persistence': [
                r'StatePersistence',
                r'checkpoint',
                r'save_state',
                r'load_state',
                r'MemoryStore'
            ]
        }
    
    async def detect_agents(self, scan_path: Path) -> List[DiscoveredAgent]:
        """Enhanced agent detection with advanced patterns"""
        discovered_agents = await super().detect_agents(scan_path)
        
        # Enhance discovered agents with advanced analysis
        enhanced_agents = []
        
        for agent in discovered_agents:
            try:
                enhanced_info = await self._enhance_agent_analysis(agent)
                if enhanced_info:
                    agent.metadata.update(enhanced_info)
                enhanced_agents.append(agent)
            except Exception as e:
                self.logger.warning("Failed to enhance agent analysis",
                                  agent_name=agent.name,
                                  error=str(e))
                enhanced_agents.append(agent)
        
        return enhanced_agents
    
    async def _enhance_agent_analysis(self, agent: DiscoveredAgent) -> Dict[str, Any]:
        """Enhance agent analysis with advanced features"""
        enhanced_info = {}
        
        try:
            with open(agent.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze LangGraph workflows
            if agent.metadata.get('agent_type') == 'langgraph':
                workflow_info = await self.workflow_analyzer.analyze_workflow(content)
                if workflow_info:
                    enhanced_info['langgraph_workflow'] = workflow_info
            
            # Detect multi-agent patterns
            multi_agent_config = self.multi_agent_detector.detect_configuration(content)
            if multi_agent_config:
                enhanced_info['multi_agent_config'] = multi_agent_config
            
            # Map tool ecosystem
            tool_ecosystem = self.tool_ecosystem_mapper.map_tools(content)
            if tool_ecosystem:
                enhanced_info['tool_ecosystem'] = tool_ecosystem
            
            # Analyze complexity
            complexity = self._analyze_complexity(content)
            enhanced_info['complexity'] = complexity
            
            return enhanced_info
            
        except Exception as e:
            self.logger.error("Enhanced analysis failed",
                            agent_name=agent.name,
                            error=str(e))
            return {}
    
    def _analyze_complexity(self, content: str) -> WorkflowComplexity:
        """Analyze workflow complexity"""
        complexity_score = 0
        
        # Check for advanced patterns
        for pattern_type, patterns in self.advanced_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 8:
            return WorkflowComplexity.ADVANCED
        elif complexity_score >= 5:
            return WorkflowComplexity.COMPLEX
        elif complexity_score >= 2:
            return WorkflowComplexity.MODERATE
        else:
            return WorkflowComplexity.SIMPLE
    
    def create_adapter(self, agent_instance: Any, metadata: Dict[str, Any]) -> AdvancedLangChainAgentAdapter:
        """Create advanced LangChain agent adapter"""
        return AdvancedLangChainAgentAdapter(agent_instance, metadata)
    
    async def integrate_multi_agent_system(self, discovered_agents: List[DiscoveredAgent]) -> Dict[str, Any]:
        """Integrate a multi-agent system"""
        try:
            # Group agents by multi-agent configuration
            agent_groups = self._group_agents_by_system(discovered_agents)
            
            integrated_systems = []
            
            for group_id, agents in agent_groups.items():
                system_result = await self._integrate_agent_group(group_id, agents)
                integrated_systems.append(system_result)
            
            return {
                "success": True,
                "integrated_systems": integrated_systems,
                "total_systems": len(integrated_systems),
                "total_agents": len(discovered_agents)
            }
            
        except Exception as e:
            self.logger.error("Multi-agent system integration failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "integrated_systems": [],
                "total_systems": 0,
                "total_agents": 0
            }
    
    def _group_agents_by_system(self, agents: List[DiscoveredAgent]) -> Dict[str, List[DiscoveredAgent]]:
        """Group agents by multi-agent system"""
        groups = defaultdict(list)
        
        for agent in agents:
            multi_agent_config = agent.metadata.get('multi_agent_config')
            if multi_agent_config:
                # Group by coordinator or system identifier
                group_id = multi_agent_config.get('system_id', 'default_system')
            else:
                # Single agent system
                group_id = f"single_agent_{agent.name}"
            
            groups[group_id].append(agent)
        
        return dict(groups)
    
    async def _integrate_agent_group(self, group_id: str, agents: List[DiscoveredAgent]) -> Dict[str, Any]:
        """Integrate a group of agents as a system"""
        try:
            integrated_agents = []
            
            for agent in agents:
                integrated_agent = await self.integrate_agent(agent)
                integrated_agents.append(integrated_agent)
            
            # Create system coordination
            coordination_config = self._create_coordination_config(agents)
            
            return {
                "group_id": group_id,
                "agents": integrated_agents,
                "coordination_config": coordination_config,
                "success": True
            }
            
        except Exception as e:
            return {
                "group_id": group_id,
                "agents": [],
                "coordination_config": None,
                "success": False,
                "error": str(e)
            }
    
    def _create_coordination_config(self, agents: List[DiscoveredAgent]) -> Dict[str, Any]:
        """Create coordination configuration for agent group"""
        # Analyze agents to determine best coordination strategy
        has_coordinator = any(
            agent.metadata.get('multi_agent_config', {}).get('coordinator_agent')
            for agent in agents
        )
        
        if has_coordinator:
            protocol = "hierarchical"
        elif len(agents) > 3:
            protocol = "parallel"
        else:
            protocol = "sequential"
        
        return {
            "communication_protocol": protocol,
            "conflict_resolution": "priority",
            "shared_memory": True,
            "coordination_timeout": 300,
            "retry_strategy": "exponential_backoff"
        }


class LangGraphWorkflowAnalyzer:
    """
    Analyzes LangGraph workflows to extract detailed information
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def analyze_workflow(self, content: str) -> Optional[LangGraphWorkflow]:
        """Analyze LangGraph workflow from code"""
        try:
            tree = ast.parse(content)
            
            # Extract workflow components
            state_schema = self._extract_state_schema(tree, content)
            nodes = self._extract_nodes(tree, content)
            edges = self._extract_edges(tree, content)
            entry_point = self._find_entry_point(tree, content)
            tools = self._extract_workflow_tools(tree, content)
            
            if not nodes and not edges:
                return None
            
            workflow = LangGraphWorkflow(
                workflow_id=f"workflow_{hash(content[:1000])}",
                name=self._extract_workflow_name(tree) or "Unknown Workflow",
                state_schema=state_schema,
                nodes=nodes,
                edges=edges,
                entry_point=entry_point,
                complexity=self._determine_complexity(nodes, edges),
                tools=tools
            )
            
            return workflow
            
        except Exception as e:
            self.logger.error("Workflow analysis failed", error=str(e))
            return None
    
    def _extract_state_schema(self, tree: ast.AST, content: str) -> LangGraphStateSchema:
        """Extract state schema from workflow"""
        schema = LangGraphStateSchema(
            schema_name="WorkflowState",
            fields={}
        )
        
        class StateVisitor(ast.NodeVisitor):
            def __init__(self):
                self.state_classes = []
            
            def visit_ClassDef(self, node):
                # Look for TypedDict classes
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'TypedDict':
                        self.state_classes.append(node)
                self.generic_visit(node)
        
        visitor = StateVisitor()
        visitor.visit(tree)
        
        if visitor.state_classes:
            state_class = visitor.state_classes[0]  # Use first found
            schema.schema_name = state_class.name
            
            # Extract fields from class annotations
            for node in state_class.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    field_name = node.target.id
                    field_type = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation)
                    
                    schema.fields[field_name] = {
                        'type': field_type,
                        'required': True,  # Default assumption
                        'description': ''
                    }
        
        return schema
    
    def _extract_nodes(self, tree: ast.AST, content: str) -> List[LangGraphNode]:
        """Extract nodes from workflow"""
        nodes = []
        
        # Pattern matching for add_node calls
        node_pattern = r'\.add_node\(\s*["\']([^"\']+)["\'](?:\s*,\s*([^)]+))?\)'
        matches = re.findall(node_pattern, content)
        
        for match in matches:
            node_name = match[0]
            function_name = match[1].strip() if match[1] else None
            
            # Remove 'self.' prefix if present
            if function_name and function_name.startswith('self.'):
                function_name = function_name[5:]
            
            node = LangGraphNode(
                node_id=node_name,
                name=node_name,
                node_type=LangGraphNodeType.STANDARD,
                function_name=function_name
            )
            nodes.append(node)
        
        return nodes
    
    def _extract_edges(self, tree: ast.AST, content: str) -> List[LangGraphEdge]:
        """Extract edges from workflow"""
        edges = []
        
        # Pattern matching for add_edge calls
        edge_pattern = r'\.add_edge\(\s*["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])?\)'
        matches = re.findall(edge_pattern, content)
        
        for match in matches:
            from_node = match[0]
            to_node = match[1] if match[1] else "END"
            
            edge = LangGraphEdge(
                from_node=from_node,
                to_node=to_node,
                edge_type="standard"
            )
            edges.append(edge)
        
        # Pattern matching for conditional edges
        conditional_pattern = r'\.add_conditional_edges\(\s*["\']([^"\']+)["\'](?:\s*,\s*([^,)]+))?'
        conditional_matches = re.findall(conditional_pattern, content)
        
        for match in conditional_matches:
            from_node = match[0]
            condition_func = match[1].strip() if match[1] else None
            
            edge = LangGraphEdge(
                from_node=from_node,
                to_node="CONDITIONAL",
                edge_type="conditional",
                condition_function=condition_func
            )
            edges.append(edge)
        
        return edges
    
    def _find_entry_point(self, tree: ast.AST, content: str) -> Optional[str]:
        """Find workflow entry point"""
        entry_pattern = r'\.set_entry_point\(\s*["\']([^"\']+)["\']'
        match = re.search(entry_pattern, content)
        return match.group(1) if match else None
    
    def _extract_workflow_tools(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract tools used in workflow"""
        tools = []
        
        # Look for tool definitions and usage
        tool_pattern = r'@tool\s*\ndef\s+(\w+)'
        matches = re.findall(tool_pattern, content)
        
        for tool_name in matches:
            tools.append({
                'name': tool_name,
                'type': 'function_tool',
                'usage_context': 'workflow'
            })
        
        return tools
    
    def _extract_workflow_name(self, tree: ast.AST) -> Optional[str]:
        """Extract workflow name from class or function"""
        class NameVisitor(ast.NodeVisitor):
            def __init__(self):
                self.workflow_names = []
            
            def visit_ClassDef(self, node):
                if any(keyword in node.name.lower() for keyword in ['workflow', 'graph', 'chain']):
                    self.workflow_names.append(node.name)
                self.generic_visit(node)
        
        visitor = NameVisitor()
        visitor.visit(tree)
        
        return visitor.workflow_names[0] if visitor.workflow_names else None
    
    def _determine_complexity(self, nodes: List[LangGraphNode], edges: List[LangGraphEdge]) -> WorkflowComplexity:
        """Determine workflow complexity based on structure"""
        node_count = len(nodes)
        edge_count = len(edges)
        conditional_edges = sum(1 for edge in edges if edge.edge_type == "conditional")
        
        if conditional_edges > 2 or node_count > 8:
            return WorkflowComplexity.ADVANCED
        elif conditional_edges > 0 or node_count > 5:
            return WorkflowComplexity.COMPLEX
        elif node_count > 3:
            return WorkflowComplexity.MODERATE
        else:
            return WorkflowComplexity.SIMPLE


class MultiAgentDetector:
    """
    Detects multi-agent configurations in LangChain code
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def detect_configuration(self, content: str) -> Optional[MultiAgentConfiguration]:
        """Detect multi-agent configuration"""
        try:
            config = MultiAgentConfiguration()
            
            # Detect coordinator pattern
            coordinator_patterns = [
                r'coordinator\s*=',
                r'CoordinatorAgent',
                r'class.*Coordinator.*Agent'
            ]
            
            for pattern in coordinator_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    config.coordinator_agent = "detected"
                    break
            
            # Detect communication protocol
            if 'parallel' in content.lower():
                config.communication_protocol = "parallel"
            elif 'hierarchical' in content.lower() or 'coordinator' in content.lower():
                config.communication_protocol = "hierarchical"
            else:
                config.communication_protocol = "sequential"
            
            # Detect shared memory
            memory_patterns = ['shared_memory', 'global_memory', 'team_memory']
            for pattern in memory_patterns:
                if pattern in content.lower():
                    config.shared_memory = "detected"
                    break
            
            # Only return config if we found multi-agent indicators
            multi_agent_indicators = [
                'multi.*agent', 'agent.*team', 'crew', 'coordinator',
                'multiple.*agents', 'agent.*collaboration'
            ]
            
            has_multi_agent = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in multi_agent_indicators
            )
            
            return config if has_multi_agent else None
            
        except Exception as e:
            self.logger.error("Multi-agent detection failed", error=str(e))
            return None


class ToolEcosystemMapper:
    """
    Maps and analyzes the tool ecosystem in LangChain agents
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def map_tools(self, content: str) -> Dict[str, Any]:
        """Map the tool ecosystem"""
        try:
            ecosystem = {
                "tools": [],
                "tool_chains": [],
                "custom_tools": [],
                "external_apis": [],
                "tool_categories": set()
            }
            
            # Extract tool definitions
            ecosystem["tools"] = self._extract_tools(content)
            ecosystem["custom_tools"] = self._extract_custom_tools(content)
            ecosystem["external_apis"] = self._extract_external_apis(content)
            ecosystem["tool_chains"] = self._extract_tool_chains(content)
            
            # Categorize tools
            for tool in ecosystem["tools"]:
                category = self._categorize_tool(tool)
                ecosystem["tool_categories"].add(category)
            
            ecosystem["tool_categories"] = list(ecosystem["tool_categories"])
            
            return ecosystem
            
        except Exception as e:
            self.logger.error("Tool ecosystem mapping failed", error=str(e))
            return {}
    
    def _extract_tools(self, content: str) -> List[Dict[str, Any]]:
        """Extract standard tools"""
        tools = []
        
        # Pattern for Tool() instantiation
        tool_pattern = r'Tool\(\s*name\s*=\s*["\']([^"\']+)["\'](?:\s*,\s*description\s*=\s*["\']([^"\']*)["\'])?'
        matches = re.findall(tool_pattern, content)
        
        for match in matches:
            tools.append({
                'name': match[0],
                'description': match[1] if match[1] else '',
                'type': 'standard_tool'
            })
        
        return tools
    
    def _extract_custom_tools(self, content: str) -> List[Dict[str, Any]]:
        """Extract custom tool definitions"""
        tools = []
        
        # Pattern for @tool decorator
        tool_pattern = r'@tool\s*\ndef\s+(\w+)\([^)]*\)(?:\s*->\s*[^:]+)?:'
        matches = re.findall(tool_pattern, content)
        
        for tool_name in matches:
            tools.append({
                'name': tool_name,
                'type': 'custom_function_tool',
                'defined_locally': True
            })
        
        return tools
    
    def _extract_external_apis(self, content: str) -> List[str]:
        """Extract external API usage"""
        apis = []
        
        api_patterns = [
            r'requests\.get\(',
            r'requests\.post\(',
            r'httpx\.',
            r'aiohttp\.',
            r'urllib\.',
            r'api\..*\(',
            r'client\..*\('
        ]
        
        for pattern in api_patterns:
            if re.search(pattern, content):
                apis.append(pattern.replace(r'\.', '.').replace(r'\(', ''))
        
        return list(set(apis))
    
    def _extract_tool_chains(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool chains and pipelines"""
        chains = []
        
        # Look for sequential tool usage patterns
        chain_patterns = [
            r'chain\s*=.*Tool',
            r'pipeline\s*=',
            r'SequentialChain',
            r'SimpleSequentialChain'
        ]
        
        for pattern in chain_patterns:
            if re.search(pattern, content):
                chains.append({
                    'type': 'tool_chain',
                    'pattern': pattern
                })
        
        return chains
    
    def _categorize_tool(self, tool: Dict[str, Any]) -> str:
        """Categorize a tool based on its name and description"""
        name = tool.get('name', '').lower()
        description = tool.get('description', '').lower()
        combined = f"{name} {description}"
        
        if any(keyword in combined for keyword in ['search', 'google', 'bing', 'web']):
            return "search"
        elif any(keyword in combined for keyword in ['calculate', 'math', 'compute']):
            return "computation"
        elif any(keyword in combined for keyword in ['file', 'read', 'write', 'storage']):
            return "file_operations"
        elif any(keyword in combined for keyword in ['api', 'request', 'http']):
            return "api_integration"
        elif any(keyword in combined for keyword in ['database', 'sql', 'query']):
            return "database"
        else:
            return "other" 