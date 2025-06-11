#!/usr/bin/env python3
"""
Universal Plugin Adapter for Mark-1 Plugin System

Provides adapters for different plugin execution modes including:
- Subprocess execution for CLI tools
- Python function calls for libraries
- HTTP API calls for web services
- Container execution for Docker-based tools
"""

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import structlog

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from .base_plugin import (
    BasePlugin, PluginMetadata, PluginResult, PluginException,
    ExecutionMode, PluginStatus
)


logger = structlog.get_logger(__name__)


@dataclass
class PluginExecutionContext:
    """Context for plugin execution"""
    plugin_id: str
    working_directory: Path
    environment_variables: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300
    max_memory_mb: int = 1024
    temp_directory: Optional[Path] = None
    input_files: List[Path] = field(default_factory=list)
    output_directory: Optional[Path] = None


class AdapterException(PluginException):
    """Exception raised by plugin adapters"""
    pass


class UniversalPluginAdapter(BasePlugin):
    """
    Universal adapter that can execute plugins in different modes
    """
    
    def __init__(self, metadata: PluginMetadata, plugin_path: Path):
        super().__init__(metadata)
        self.plugin_path = plugin_path
        self.logger = structlog.get_logger(__name__)
        self._execution_context: Optional[PluginExecutionContext] = None
        self._process: Optional[subprocess.Popen] = None
        self._http_client: Optional[Any] = None  # httpx.AsyncClient when available
    
    async def initialize(self) -> bool:
        """Initialize the plugin adapter"""
        try:
            self.logger.info("Initializing plugin adapter", 
                           plugin_id=self.plugin_id,
                           execution_mode=self.metadata.execution_mode.value)
            
            # Set up execution context
            self._execution_context = PluginExecutionContext(
                plugin_id=self.plugin_id,
                working_directory=self.plugin_path,
                environment_variables=self.metadata.environment_variables.copy(),
                temp_directory=Path(tempfile.mkdtemp(prefix=f"mark1_plugin_{self.plugin_id}_"))
            )
            
            # Initialize based on execution mode
            if self.metadata.execution_mode == ExecutionMode.HTTP_API:
                if HTTPX_AVAILABLE:
                    self._http_client = httpx.AsyncClient(timeout=30.0)
                else:
                    raise AdapterException("httpx not available for HTTP API execution", self.plugin_id)
            
            self.metadata.status = PluginStatus.READY
            self._initialized = True
            
            self.logger.info("Plugin adapter initialized", plugin_id=self.plugin_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize plugin adapter", 
                            plugin_id=self.plugin_id, error=str(e))
            self.metadata.status = PluginStatus.ERROR
            raise AdapterException(f"Initialization failed: {e}", self.plugin_id)
    
    async def execute(
        self,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> PluginResult:
        """Execute a plugin capability"""
        if not self._initialized:
            raise AdapterException("Plugin adapter not initialized", self.plugin_id)
        
        start_time = time.time()
        
        try:
            self.logger.info("Executing plugin capability", 
                           plugin_id=self.plugin_id,
                           capability=capability)
            
            # Validate inputs
            if not await self.validate_inputs(capability, inputs):
                raise AdapterException(f"Invalid inputs for capability: {capability}", self.plugin_id)
            
            # Execute based on mode
            result = None
            if self.metadata.execution_mode == ExecutionMode.SUBPROCESS:
                result = await self._execute_subprocess(capability, inputs, parameters)
            elif self.metadata.execution_mode == ExecutionMode.PYTHON_FUNCTION:
                result = await self._execute_python_function(capability, inputs, parameters)
            elif self.metadata.execution_mode == ExecutionMode.HTTP_API:
                result = await self._execute_http_api(capability, inputs, parameters)
            elif self.metadata.execution_mode == ExecutionMode.CONTAINER:
                result = await self._execute_container(capability, inputs, parameters)
            else:
                raise AdapterException(f"Unsupported execution mode: {self.metadata.execution_mode}", self.plugin_id)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info("Plugin capability executed successfully", 
                           plugin_id=self.plugin_id,
                           capability=capability,
                           execution_time=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("Plugin capability execution failed", 
                            plugin_id=self.plugin_id,
                            capability=capability,
                            error=str(e),
                            execution_time=execution_time)
            
            return PluginResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_subprocess(
        self,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> PluginResult:
        """Execute plugin as subprocess"""
        try:
            # Determine command to run
            entry_point = self.metadata.entry_points.get("main", "main.py")
            command = self._build_subprocess_command(entry_point, capability, inputs, parameters)
            
            # Set up environment
            env = self._execution_context.environment_variables.copy()
            
            # Execute subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.plugin_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._execution_context.timeout
                )
                
                return_code = process.returncode
                
                if return_code == 0:
                    # Try to parse output as JSON, fallback to text
                    try:
                        output_data = json.loads(stdout.decode())
                    except:
                        output_data = stdout.decode()
                    
                    return PluginResult(
                        success=True,
                        data=output_data,
                        metadata={"return_code": return_code}
                    )
                else:
                    return PluginResult(
                        success=False,
                        error=stderr.decode(),
                        metadata={"return_code": return_code}
                    )
            
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return PluginResult(
                    success=False,
                    error=f"Execution timeout after {self._execution_context.timeout} seconds"
                )
        
        except Exception as e:
            return PluginResult(
                success=False,
                error=f"Subprocess execution failed: {e}"
            )
    
    def _build_subprocess_command(
        self,
        entry_point: str,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Build subprocess command"""
        command = []
        
        # Determine interpreter
        if entry_point.endswith(".py"):
            # Try different Python executables
            import sys
            python_exe = sys.executable or "python3"
            command.extend([python_exe, entry_point])
        elif entry_point.endswith(".js"):
            command.extend(["node", entry_point])
        else:
            command.append(entry_point)
        
        # Add capability as argument
        command.append(capability)
        
        # Add inputs as JSON argument
        if inputs:
            command.extend(["--input", json.dumps(inputs)])
        
        # Add parameters
        if parameters:
            for key, value in parameters.items():
                command.extend([f"--{key}", str(value)])
        
        return command
    
    async def _execute_python_function(
        self,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> PluginResult:
        """Execute plugin as Python function"""
        try:
            # This would involve dynamic import and function execution
            # For now, return a placeholder result
            return PluginResult(
                success=True,
                data={"message": "Python function execution not yet implemented"},
                metadata={"capability": capability}
            )
        
        except Exception as e:
            return PluginResult(
                success=False,
                error=f"Python function execution failed: {e}"
            )
    
    async def _execute_http_api(
        self,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> PluginResult:
        """Execute plugin via HTTP API"""
        try:
            if not self._http_client:
                raise AdapterException("HTTP client not initialized", self.plugin_id)
            
            # Build API request
            url = self.metadata.configuration.get("api_url", "http://localhost:8000")
            endpoint = f"{url}/{capability}"
            
            payload = {
                "inputs": inputs,
                "parameters": parameters or {}
            }
            
            response = await self._http_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            result_data = response.json()
            
            return PluginResult(
                success=True,
                data=result_data,
                metadata={"status_code": response.status_code}
            )
        
        except Exception as e:
            return PluginResult(
                success=False,
                error=f"HTTP API execution failed: {e}"
            )
    
    async def _execute_container(
        self,
        capability: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> PluginResult:
        """Execute plugin in container"""
        try:
            # This would involve Docker API calls
            # For now, return a placeholder result
            return PluginResult(
                success=True,
                data={"message": "Container execution not yet implemented"},
                metadata={"capability": capability}
            )
        
        except Exception as e:
            return PluginResult(
                success=False,
                error=f"Container execution failed: {e}"
            )

    async def validate_inputs(
        self,
        capability: str,
        inputs: Dict[str, Any]
    ) -> bool:
        """Validate inputs for a capability"""
        try:
            capability_obj = self.get_capability(capability)
            if not capability_obj:
                self.logger.warning("Unknown capability", capability=capability)
                return False

            # Basic validation - check if required input types are present
            # This could be enhanced with schema validation
            return True

        except Exception as e:
            self.logger.error("Input validation failed", error=str(e))
            return False

    async def get_progress(self) -> Dict[str, Any]:
        """Get current execution progress"""
        return {
            "status": self.metadata.status.value,
            "initialized": self._initialized,
            "execution_mode": self.metadata.execution_mode.value,
            "active_process": self._process is not None if hasattr(self, '_process') else False
        }

    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        try:
            self.logger.info("Cleaning up plugin adapter", plugin_id=self.plugin_id)

            # Kill any running processes
            if hasattr(self, '_process') and self._process:
                try:
                    self._process.kill()
                    await self._process.wait()
                except:
                    pass

            # Close HTTP client
            if self._http_client:
                await self._http_client.aclose()

            # Clean up temp directory
            if self._execution_context and self._execution_context.temp_directory:
                try:
                    import shutil
                    shutil.rmtree(self._execution_context.temp_directory)
                except:
                    pass

            self.logger.info("Plugin adapter cleanup completed", plugin_id=self.plugin_id)

        except Exception as e:
            self.logger.error("Plugin adapter cleanup failed",
                            plugin_id=self.plugin_id, error=str(e))
