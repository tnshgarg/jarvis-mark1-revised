#!/usr/bin/env python3
"""
Intelligent Task Orchestrator for Mark-1 Universal Plugin System

Uses OLLAMA for AI-powered task planning, plugin selection, and workflow orchestration.
Takes natural language prompts and orchestrates the complete plugin ecosystem.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import structlog

from ..llm.ollama_client import OllamaClient, ChatMessage
from ..plugins import PluginManager, PluginMetadata
from ..core.context_manager import ContextManager, ContextType, ContextScope, ContextPriority
from ..core.workflow_engine import WorkflowEngine, Workflow, WorkflowStep
from ..storage.database import get_db_session
from ..storage.repositories.plugin_repository import PluginRepository
from ..utils.exceptions import OrchestrationException


logger = structlog.get_logger(__name__)


class IntelligentOrchestrator:
    """
    AI-powered orchestrator that takes natural language prompts and orchestrates
    the complete plugin ecosystem using OLLAMA for intelligent planning.
    """
    
    def __init__(self, ollama_url: str):
        self.ollama_client = OllamaClient(base_url=ollama_url)
        self.plugin_manager: Optional[PluginManager] = None
        self.context_manager: Optional[ContextManager] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.logger = structlog.get_logger(__name__)
        
        # Orchestration state
        self._active_orchestrations: Dict[str, Dict[str, Any]] = {}
        self._orchestration_history: List[Dict[str, Any]] = []
    
    async def initialize(self, plugin_manager: PluginManager, context_manager: ContextManager, workflow_engine: WorkflowEngine):
        """Initialize the intelligent orchestrator"""
        self.plugin_manager = plugin_manager
        self.context_manager = context_manager
        self.workflow_engine = workflow_engine
        
        # Test OLLAMA connection (with fallback mode)
        ollama_available = await self.ollama_client.health_check()
        if not ollama_available:
            self.logger.warning("OLLAMA is not accessible, using fallback mode")
            self.fallback_mode = True
        else:
            self.fallback_mode = False
        
        self.logger.info("Intelligent orchestrator initialized")
    
    async def orchestrate_from_prompt(
        self,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_plugins: int = 5,
        timeout: int = 600
    ) -> Dict[str, Any]:
        """
        Main orchestration method - takes a natural language prompt and orchestrates plugins
        
        Args:
            user_prompt: Natural language description of what to do
            context: Optional context information
            max_plugins: Maximum number of plugins to use
            timeout: Total timeout for orchestration
            
        Returns:
            Orchestration result with execution details and outputs
        """
        orchestration_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info("Starting intelligent orchestration", 
                           orchestration_id=orchestration_id,
                           prompt=user_prompt)
            
            # Store orchestration state
            self._active_orchestrations[orchestration_id] = {
                "prompt": user_prompt,
                "context": context or {},
                "start_time": start_time,
                "status": "planning"
            }
            
            # Step 1: Analyze the prompt and understand intent
            if self.fallback_mode:
                intent_analysis = self._fallback_intent_analysis(user_prompt)
            else:
                intent_analysis = await self._analyze_user_intent(user_prompt, context)
            
            # Step 2: Get available plugins
            available_plugins = await self.plugin_manager.list_installed_plugins()
            
            # Step 3: AI-powered task planning
            if self.fallback_mode:
                execution_plan = await self._create_fallback_plan(available_plugins, intent_analysis)
            else:
                execution_plan = await self._create_intelligent_plan(
                    user_prompt, intent_analysis, available_plugins, max_plugins
                )
            
            # Step 4: Create workflow from plan
            workflow = await self._create_workflow_from_plan(execution_plan, orchestration_id)
            
            # Step 5: Execute the workflow
            self._active_orchestrations[orchestration_id]["status"] = "executing"
            execution_result = await self._execute_intelligent_workflow(workflow, orchestration_id)
            
            # Step 6: Process and store results
            final_result = await self._process_orchestration_results(
                orchestration_id, execution_result, start_time
            )
            
            self.logger.info("Intelligent orchestration completed", 
                           orchestration_id=orchestration_id,
                           success=final_result["success"],
                           duration=final_result["execution_time"])
            
            return final_result
            
        except Exception as e:
            self.logger.error("Intelligent orchestration failed", 
                            orchestration_id=orchestration_id, error=str(e))
            
            return {
                "orchestration_id": orchestration_id,
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "results": {}
            }
        
        finally:
            # Cleanup
            if orchestration_id in self._active_orchestrations:
                del self._active_orchestrations[orchestration_id]

    def _fallback_intent_analysis(self, user_prompt: str) -> Dict[str, Any]:
        """Simple fallback intent analysis when OLLAMA is not available"""
        # Simple keyword-based analysis
        prompt_lower = user_prompt.lower()

        # Determine task type based on keywords
        if any(word in prompt_lower for word in ["analyze", "analysis", "examine", "study"]):
            task_type = "analysis"
        elif any(word in prompt_lower for word in ["convert", "transform", "change", "format"]):
            task_type = "conversion"
        elif any(word in prompt_lower for word in ["process", "handle", "manage", "work"]):
            task_type = "processing"
        elif any(word in prompt_lower for word in ["create", "generate", "make", "build"]):
            task_type = "creation"
        else:
            task_type = "general"

        # Extract keywords
        keywords = [word for word in user_prompt.split() if len(word) > 3][:5]

        return {
            "task_type": task_type,
            "complexity": "medium",
            "required_capabilities": [task_type, "general"],
            "input_types": ["text"],
            "output_types": ["text"],
            "estimated_duration": 120,
            "priority": "medium",
            "keywords": keywords
        }

    async def _analyze_user_intent(self, user_prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Use OLLAMA to analyze user intent and extract key information"""
        try:
            # Select best model for analysis
            model = await self.ollama_client.select_best_model("analysis", "medium")
            
            analysis_prompt = f"""
Analyze the following user request and extract key information:

User Request: "{user_prompt}"

Context: {json.dumps(context or {}, indent=2)}

Please analyze and provide a JSON response with:
1. task_type: The type of task (e.g., "data_processing", "file_manipulation", "analysis", "automation", "development")
2. complexity: Task complexity ("low", "medium", "high")
3. required_capabilities: List of capabilities needed (e.g., ["file_processing", "data_analysis", "visualization"])
4. input_types: Expected input types (e.g., ["file", "text", "url"])
5. output_types: Expected output types (e.g., ["file", "report", "visualization"])
6. estimated_steps: Number of estimated steps (1-10)
7. priority: Task priority ("low", "medium", "high")
8. keywords: Important keywords from the request

Respond only with valid JSON.
"""
            
            response = await self.ollama_client.generate(
                model=model,
                prompt=analysis_prompt,
                format="json"
            )
            
            try:
                intent_analysis = json.loads(response.response)
                self.logger.info("Intent analysis completed", analysis=intent_analysis)
                return intent_analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return {
                    "task_type": "general",
                    "complexity": "medium",
                    "required_capabilities": ["general"],
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "estimated_steps": 3,
                    "priority": "medium",
                    "keywords": user_prompt.split()[:5]
                }
                
        except Exception as e:
            self.logger.error("Intent analysis failed", error=str(e))
            return {
                "task_type": "general",
                "complexity": "medium",
                "required_capabilities": ["general"],
                "input_types": ["text"],
                "output_types": ["text"],
                "estimated_steps": 3,
                "priority": "medium",
                "keywords": []
            }
    
    async def _create_intelligent_plan(
        self,
        user_prompt: str,
        intent_analysis: Dict[str, Any],
        available_plugins: List[PluginMetadata],
        max_plugins: int
    ) -> Dict[str, Any]:
        """Use OLLAMA to create an intelligent execution plan"""
        try:
            # Select best model for planning
            model = await self.ollama_client.select_best_model("planning", "high")
            
            # Prepare plugin information for the AI
            plugin_info = []
            for plugin in available_plugins:
                capabilities = [cap.name for cap in plugin.capabilities]
                plugin_info.append({
                    "id": plugin.plugin_id,
                    "name": plugin.name,
                    "description": plugin.description,
                    "type": plugin.plugin_type.value,
                    "capabilities": capabilities,
                    "execution_mode": plugin.execution_mode.value
                })
            
            planning_prompt = f"""
Create an execution plan for the following user request:

User Request: "{user_prompt}"

Intent Analysis: {json.dumps(intent_analysis, indent=2)}

Available Plugins: {json.dumps(plugin_info, indent=2)}

Create a detailed execution plan with the following JSON structure:
{{
    "plan_summary": "Brief description of the plan",
    "total_steps": number_of_steps,
    "estimated_duration": estimated_duration_in_seconds,
    "execution_mode": "sequential" or "parallel",
    "steps": [
        {{
            "step_id": "step_1",
            "description": "What this step does",
            "plugin_id": "plugin_to_use",
            "capability": "capability_to_execute",
            "inputs": {{"key": "value"}},
            "parameters": {{"key": "value"}},
            "depends_on": ["previous_step_ids"],
            "estimated_duration": duration_in_seconds,
            "output_key": "key_for_storing_output"
        }}
    ],
    "data_flow": [
        {{
            "from_step": "step_1",
            "to_step": "step_2", 
            "data_key": "output_key"
        }}
    ],
    "success_criteria": "How to determine if the plan succeeded"
}}

Guidelines:
- Use only the available plugins
- Ensure proper data flow between steps
- Consider dependencies between steps
- Limit to {max_plugins} plugins maximum
- Be specific about inputs and outputs
- Include error handling considerations

Respond only with valid JSON.
"""
            
            response = await self.ollama_client.generate(
                model=model,
                prompt=planning_prompt,
                format="json"
            )
            
            try:
                execution_plan = json.loads(response.response)
                self.logger.info("Intelligent plan created", 
                               steps=execution_plan.get("total_steps", 0),
                               duration=execution_plan.get("estimated_duration", 0))
                return execution_plan
            except json.JSONDecodeError:
                # Fallback to simple plan
                return await self._create_fallback_plan(available_plugins, intent_analysis)
                
        except Exception as e:
            self.logger.error("Intelligent planning failed", error=str(e))
            return await self._create_fallback_plan(available_plugins, intent_analysis)

    async def _resolve_plugin_id(self, plugin_id: str) -> str:
        """Resolve plugin ID from AI-generated name to actual plugin ID"""
        if not plugin_id:
            return plugin_id

        # Get all available plugins
        available_plugins = await self.plugin_manager.list_installed_plugins()

        # First try exact match
        for plugin in available_plugins:
            if plugin.plugin_id == plugin_id:
                return plugin_id

        # Try partial match (AI might use short names)
        for plugin in available_plugins:
            if plugin_id in plugin.plugin_id or plugin.plugin_id.startswith(plugin_id):
                self.logger.info("Resolved plugin ID",
                               ai_name=plugin_id,
                               actual_id=plugin.plugin_id)
                return plugin.plugin_id

        # Try name-based matching
        for plugin in available_plugins:
            if plugin_id.lower() in plugin.name.lower() or plugin.name.lower() in plugin_id.lower():
                self.logger.info("Resolved plugin by name",
                               ai_name=plugin_id,
                               actual_id=plugin.plugin_id)
                return plugin.plugin_id

        # If no match found, return original (will fail gracefully)
        self.logger.warning("Could not resolve plugin ID", plugin_id=plugin_id)
        return plugin_id

    async def _create_fallback_plan(self, available_plugins: List[PluginMetadata], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple fallback plan when AI planning fails"""
        if not available_plugins:
            return {
                "plan_summary": "No plugins available",
                "total_steps": 0,
                "estimated_duration": 0,
                "execution_mode": "sequential",
                "steps": [],
                "data_flow": [],
                "success_criteria": "No execution possible"
            }
        
        # Use first available plugin
        plugin = available_plugins[0]
        capability = plugin.capabilities[0] if plugin.capabilities else None
        
        return {
            "plan_summary": f"Simple execution using {plugin.name}",
            "total_steps": 1,
            "estimated_duration": 60,
            "execution_mode": "sequential",
            "steps": [{
                "step_id": "step_1",
                "description": f"Execute {plugin.name}",
                "plugin_id": plugin.plugin_id,
                "capability": capability.name if capability else "execute",
                "inputs": {"prompt": intent_analysis.get("keywords", [])},
                "parameters": {},
                "depends_on": [],
                "estimated_duration": 60,
                "output_key": "result"
            }],
            "data_flow": [],
            "success_criteria": "Plugin execution completes successfully"
        }

    async def _create_workflow_from_plan(self, execution_plan: Dict[str, Any], orchestration_id: str) -> Workflow:
        """Create a workflow from the AI-generated execution plan"""
        try:
            workflow_steps = []

            for step_data in execution_plan.get("steps", []):
                # Resolve plugin ID (AI might use short names)
                plugin_id = await self._resolve_plugin_id(step_data.get("plugin_id"))

                step = WorkflowStep(
                    step_id=step_data.get("step_id", f"step_{len(workflow_steps)}"),
                    name=step_data.get("description", "Unnamed step"),
                    description=step_data.get("description", ""),
                    agent_id=plugin_id,  # Using agent_id for plugin_id
                    dependencies=step_data.get("depends_on", []),
                    parameters={
                        "capability": step_data.get("capability"),
                        "inputs": step_data.get("inputs", {}),
                        "plugin_parameters": step_data.get("parameters", {}),
                        "output_key": step_data.get("output_key", "result")
                    },
                    timeout=step_data.get("estimated_duration", 300)
                )
                workflow_steps.append(step)

            # Convert workflow steps to dictionaries
            steps_data = []
            for step in workflow_steps:
                steps_data.append({
                    "step_id": step.step_id,
                    "name": step.name,
                    "description": step.description,
                    "agent_id": step.agent_id,
                    "dependencies": step.dependencies,
                    "parameters": step.parameters,
                    "timeout": step.timeout
                })

            workflow = await self.workflow_engine.create_workflow(
                name=f"Intelligent Orchestration {orchestration_id[:8]}",
                description=execution_plan.get("plan_summary", "AI-generated workflow"),
                steps=steps_data
            )

            return workflow

        except Exception as e:
            self.logger.error("Failed to create workflow from plan", error=str(e))
            raise OrchestrationException(f"Workflow creation failed: {e}")

    async def _execute_intelligent_workflow(self, workflow: Workflow, orchestration_id: str) -> Dict[str, Any]:
        """Execute the workflow with intelligent monitoring and context management"""
        try:
            # Create execution context
            execution_context = {
                "orchestration_id": orchestration_id,
                "workflow_id": workflow.workflow_id,
                "step_results": {},
                "shared_data": {},
                "errors": []
            }

            # Store context
            context_result = await self.context_manager.create_context(
                key=f"orchestration_{orchestration_id}",
                content=execution_context,
                context_type=ContextType.TASK,
                scope=ContextScope.TASK,
                priority=ContextPriority.HIGH,
                task_id=orchestration_id,
                expires_in_hours=24
            )

            # Store the actual context_id for later updates
            actual_context_id = context_result.context_id if context_result.success else None

            # Start workflow execution in the workflow engine
            await self.workflow_engine.execute_workflow(workflow.workflow_id)

            # Execute workflow steps
            for step in workflow.steps:
                try:
                    self.logger.info("Executing workflow step",
                                   step_id=step.step_id,
                                   orchestration_id=orchestration_id)

                    # Get plugin adapter
                    plugin_id = step.agent_id  # agent_id contains plugin_id
                    adapter = await self.plugin_manager.get_plugin_adapter(plugin_id)

                    if not adapter:
                        raise OrchestrationException(f"Plugin adapter not found: {plugin_id}")

                    # Prepare inputs with context
                    step_inputs = step.parameters.get("inputs", {})
                    step_inputs = await self._resolve_step_inputs(step_inputs, execution_context)

                    # Execute plugin capability
                    capability = step.parameters.get("capability", "execute")
                    plugin_parameters = step.parameters.get("plugin_parameters", {})

                    result = await adapter.execute(
                        capability=capability,
                        inputs=step_inputs,
                        parameters=plugin_parameters
                    )

                    # Store step result
                    output_key = step.parameters.get("output_key", step.step_id)
                    execution_context["step_results"][step.step_id] = result
                    execution_context["shared_data"][output_key] = result.data if result.success else None

                    self.logger.info("Workflow step completed successfully",
                                   step_id=step.step_id,
                                   orchestration_id=orchestration_id)

                    # Record execution in database (non-blocking)
                    try:
                        async with get_db_session() as session:
                            plugin_repo = PluginRepository(session)
                            await plugin_repo.record_plugin_execution(
                                plugin_id=plugin_id,
                                capability_name=capability,
                                execution_id=f"{orchestration_id}_{step.step_id}",
                                inputs=step_inputs,
                                parameters=plugin_parameters,
                                status="success" if result.success else "error",
                                execution_time=result.execution_time,
                                outputs=result.data if result.success else None,
                                error_message=result.error if not result.success else None
                            )
                    except Exception as db_error:
                        # Database recording failed, but don't fail the workflow
                        self.logger.warning("Database recording failed",
                                          step_id=step.step_id,
                                          error=str(db_error))

                    if not result.success:
                        execution_context["errors"].append({
                            "step_id": step.step_id,
                            "error": result.error,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        self.logger.warning("Step execution failed",
                                          step_id=step.step_id,
                                          error=result.error)

                    # Update context (serialize PluginResult objects)
                    try:
                        # Convert PluginResult objects to dictionaries for JSON serialization
                        serializable_context = {}
                        for key, value in execution_context.items():
                            if key == "step_results":
                                serializable_context[key] = {}
                                for step_id, result in value.items():
                                    if hasattr(result, 'to_dict'):
                                        serializable_context[key][step_id] = result.to_dict()
                                    else:
                                        serializable_context[key][step_id] = result
                            else:
                                serializable_context[key] = value

                        if actual_context_id:
                            await self.context_manager.update_context(
                                context_id=actual_context_id,
                                content=serializable_context
                            )
                    except Exception as ctx_error:
                        # Context update failed, but don't fail the workflow
                        self.logger.warning("Context update failed",
                                          step_id=step.step_id,
                                          error=str(ctx_error))

                except Exception as e:
                    error_msg = f"Step execution failed: {e}"

                    # Check if this is a database/context error but plugin executed successfully
                    if any(keyword in str(e).lower() for keyword in ["asyncsession", "query", "database", "context", "json serializable"]):
                        # This is likely a database/context error, but plugin may have executed
                        self.logger.warning("Database/context error during step execution",
                                          step_id=step.step_id,
                                          error=str(e))
                        # Don't add to errors if it's just a database/context issue
                    else:
                        execution_context["errors"].append({
                            "step_id": step.step_id,
                            "error": error_msg,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        self.logger.error("Step execution error",
                                        step_id=step.step_id,
                                        error=str(e))

            return execution_context

        except Exception as e:
            self.logger.error("Workflow execution failed", error=str(e))
            # Return a proper error context instead of raising exception
            return {
                "orchestration_id": orchestration_id,
                "step_results": {},
                "shared_data": {},
                "errors": [{"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}],
                "execution_time": 0.0
            }

    async def _resolve_step_inputs(self, step_inputs: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve step inputs by substituting references to previous step outputs"""
        resolved_inputs = {}

        for key, value in step_inputs.items():
            if isinstance(value, str) and value.startswith("${"):
                # Extract reference (e.g., "${step_1.result}" -> "step_1.result")
                ref = value[2:-1]  # Remove ${ and }

                if "." in ref:
                    step_ref, data_key = ref.split(".", 1)
                    if step_ref in execution_context["step_results"]:
                        step_result = execution_context["step_results"][step_ref]
                        if hasattr(step_result, 'data') and isinstance(step_result.data, dict):
                            resolved_inputs[key] = step_result.data.get(data_key, value)
                        else:
                            resolved_inputs[key] = step_result.data
                    else:
                        resolved_inputs[key] = execution_context["shared_data"].get(data_key, value)
                else:
                    # Direct reference to shared data
                    resolved_inputs[key] = execution_context["shared_data"].get(ref, value)
            else:
                resolved_inputs[key] = value

        return resolved_inputs

    async def _process_orchestration_results(
        self,
        orchestration_id: str,
        execution_result: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Process and format the final orchestration results"""
        try:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            # Determine overall success
            errors = execution_result.get("errors", [])
            step_results = execution_result.get("step_results", {})

            # Count successful steps (including those with database errors but successful execution)
            successful_count = 0
            for result in step_results.values():
                if hasattr(result, 'success') and result.success:
                    successful_count += 1
                elif isinstance(result, dict) and result.get("success"):
                    successful_count += 1

            # Success if we have successful steps and no critical errors (ignore database/context errors)
            critical_errors = [e for e in errors if not any(keyword in e.get("error", "").lower()
                                                           for keyword in ["database", "asyncsession", "query", "greenlet", "context", "json serializable"])]
            success = successful_count > 0 and len(critical_errors) == 0

            # Collect all outputs
            outputs = {}
            for step_id, result in execution_result.get("step_results", {}).items():
                if hasattr(result, 'data') and result.data:
                    outputs[step_id] = result.data

            # Create final result
            final_result = {
                "orchestration_id": orchestration_id,
                "success": success,
                "execution_time": execution_time,
                "total_steps": len(execution_result.get("step_results", {})),
                "successful_steps": successful_count,
                "errors": errors,
                "outputs": outputs,
                "shared_data": execution_result.get("shared_data", {}),
                "workflow_id": execution_result.get("workflow_id"),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

            # Store in orchestration history
            self._orchestration_history.append(final_result)

            # Store final context
            await self.context_manager.create_context(
                key=f"orchestration_result_{orchestration_id}",
                content=final_result,
                context_type=ContextType.SYSTEM,
                scope=ContextScope.GLOBAL,
                priority=ContextPriority.HIGH,
                task_id=orchestration_id,
                expires_in_hours=168  # Keep results for a week
            )

            return final_result

        except Exception as e:
            self.logger.error("Failed to process orchestration results", error=str(e))
            return {
                "orchestration_id": orchestration_id,
                "success": False,
                "error": f"Result processing failed: {e}",
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds()
            }

    async def get_orchestration_status(self, orchestration_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an active orchestration"""
        return self._active_orchestrations.get(orchestration_id)

    async def get_orchestration_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent orchestration history"""
        return self._orchestration_history[-limit:]

    async def cleanup(self):
        """Cleanup resources"""
        await self.ollama_client.close()
