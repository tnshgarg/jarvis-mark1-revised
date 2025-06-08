#!/usr/bin/env python3
"""
Mark-1 CLI Utilities

This module provides utility classes for the CLI interface including
formatting, validation, configuration management, and helper functions.
"""

import json
import yaml
import sys
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Optional dependencies for enhanced formatting
try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Mock colorama classes for when not available
    class Fore:
        GREEN = RED = YELLOW = CYAN = BLUE = MAGENTA = ""
    class Style:
        RESET_ALL = ""

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

logger = logging.getLogger(__name__)


class CLIFormatter:
    """Output formatting utilities for CLI"""
    
    def __init__(self):
        self.format_type = 'table'
        self.use_color = True
        
    def set_format(self, format_type: str):
        """Set output format type"""
        self.format_type = format_type
        
    def set_color(self, use_color: bool):
        """Enable or disable colored output"""
        self.use_color = use_color
        
    def output(self, data: Any, format_override: Optional[str] = None):
        """Format and output data"""
        format_type = format_override or self.format_type
        
        if format_type == 'json':
            self._output_json(data)
        elif format_type == 'yaml':
            self._output_yaml(data)
        elif format_type == 'table':
            self._output_table(data)
        else:  # text
            self._output_text(data)
    
    def _output_json(self, data: Any):
        """Output data as JSON"""
        try:
            formatted = json.dumps(data, indent=2, default=str)
            if self.use_color:
                # Simple JSON syntax highlighting
                formatted = self._colorize_json(formatted)
            print(formatted)
        except Exception as e:
            logger.error(f"JSON formatting error: {e}")
            print(json.dumps({"error": "Failed to format as JSON"}, indent=2))
    
    def _output_yaml(self, data: Any):
        """Output data as YAML"""
        try:
            formatted = yaml.dump(data, default_flow_style=False, indent=2)
            if self.use_color:
                formatted = self._colorize_yaml(formatted)
            print(formatted)
        except Exception as e:
            logger.error(f"YAML formatting error: {e}")
            print("error: Failed to format as YAML")
    
    def _output_table(self, data: Any):
        """Output data as table"""
        try:
            if isinstance(data, dict):
                if 'agents' in data:
                    self._output_agents_table(data['agents'])
                elif 'tasks' in data:
                    self._output_tasks_table(data['tasks'])
                elif 'workflows' in data:
                    self._output_workflows_table(data['workflows'])
                elif 'configuration' in data:
                    self._output_config_table(data['configuration'])
                else:
                    self._output_generic_table(data)
            elif isinstance(data, list):
                self._output_list_table(data)
            else:
                self._output_text(data)
        except Exception as e:
            logger.error(f"Table formatting error: {e}")
            self._output_text(data)
    
    def _output_text(self, data: Any):
        """Output data as plain text"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{key}:")
                    self._print_nested(value, indent=2)
                else:
                    print(f"{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                print(f"- {item}")
        else:
            print(str(data))
    
    def _output_agents_table(self, agents: List[Dict[str, Any]]):
        """Output agents as formatted table"""
        if not agents:
            print("No agents found.")
            return
        
        if not HAS_TABULATE:
            # Fallback to simple text format
            print("Agents:")
            for agent in agents:
                capabilities = ", ".join(agent.get('capabilities', []))
                print(f"  ID: {agent.get('id', '')}")
                print(f"  Name: {agent.get('name', '')}")
                print(f"  Type: {agent.get('type', '')}")
                print(f"  Status: {agent.get('status', '')}")
                print(f"  Capabilities: {capabilities}")
                print(f"  Last Activity: {agent.get('last_activity', '')[:19] if agent.get('last_activity') else ''}")
                print("  ---")
            print(f"Total agents: {len(agents)}")
            return
        
        headers = ["ID", "Name", "Type", "Status", "Capabilities", "Last Activity"]
        rows = []
        
        for agent in agents:
            capabilities = ", ".join(agent.get('capabilities', []))
            if len(capabilities) > 30:
                capabilities = capabilities[:27] + "..."
            
            row = [
                agent.get('id', ''),
                agent.get('name', ''),
                agent.get('type', ''),
                self._colorize_status(agent.get('status', '')),
                capabilities,
                agent.get('last_activity', '')[:19] if agent.get('last_activity') else ''
            ]
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print(f"\nTotal agents: {len(agents)}")
    
    def _output_tasks_table(self, tasks: List[Dict[str, Any]]):
        """Output tasks as formatted table"""
        if not tasks:
            print("No tasks found.")
            return
        
        if not HAS_TABULATE:
            # Fallback to simple text format
            print("Tasks:")
            for task in tasks:
                print(f"  ID: {task.get('id', '')}")
                print(f"  Name: {task.get('name', '')}")
                print(f"  Status: {task.get('status', '')}")
                print(f"  Agent: {task.get('agent_id', 'Unassigned')}")
                print(f"  Priority: {task.get('priority', '')}")
                print(f"  Progress: {task.get('progress', 0)}%")
                print(f"  Created: {task.get('created_at', '')[:19] if task.get('created_at') else ''}")
                print("  ---")
            print(f"Total tasks: {len(tasks)}")
            return
        
        headers = ["ID", "Name", "Status", "Agent", "Priority", "Progress", "Created"]
        rows = []
        
        for task in tasks:
            row = [
                task.get('id', ''),
                task.get('name', '')[:30] + "..." if len(task.get('name', '')) > 30 else task.get('name', ''),
                self._colorize_status(task.get('status', '')),
                task.get('agent_id', 'Unassigned'),
                task.get('priority', ''),
                f"{task.get('progress', 0)}%",
                task.get('created_at', '')[:19] if task.get('created_at') else ''
            ]
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print(f"\nTotal tasks: {len(tasks)}")
    
    def _output_workflows_table(self, workflows: List[Dict[str, Any]]):
        """Output workflows as formatted table"""
        if not workflows:
            print("No workflows found.")
            return
        
        if not HAS_TABULATE:
            # Fallback to simple text format
            print("Workflows:")
            for workflow in workflows:
                completed_steps = workflow.get('completed_steps', 0)
                total_steps = workflow.get('steps', 0)
                progress = workflow.get('progress', 0)
                print(f"  ID: {workflow.get('id', '')}")
                print(f"  Name: {workflow.get('name', '')}")
                print(f"  Status: {workflow.get('status', '')}")
                print(f"  Steps: {completed_steps}/{total_steps}")
                print(f"  Progress: {progress}%")
                print(f"  Created: {workflow.get('created_at', '')[:19] if workflow.get('created_at') else ''}")
                print("  ---")
            print(f"Total workflows: {len(workflows)}")
            return
        
        headers = ["ID", "Name", "Status", "Steps", "Progress", "Created"]
        rows = []
        
        for workflow in workflows:
            completed_steps = workflow.get('completed_steps', 0)
            total_steps = workflow.get('steps', 0)
            progress = workflow.get('progress', 0)
            
            row = [
                workflow.get('id', ''),
                workflow.get('name', '')[:40] + "..." if len(workflow.get('name', '')) > 40 else workflow.get('name', ''),
                self._colorize_status(workflow.get('status', '')),
                f"{completed_steps}/{total_steps}",
                f"{progress}%",
                workflow.get('created_at', '')[:19] if workflow.get('created_at') else ''
            ]
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print(f"\nTotal workflows: {len(workflows)}")
    
    def _output_config_table(self, config: Dict[str, Any]):
        """Output configuration as formatted table"""
        if not HAS_TABULATE:
            # Fallback to simple text format
            print("Configuration:")
            def print_config(data, prefix=""):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        print_config(value, full_key)
                    else:
                        print(f"  {full_key}: {value}")
            print_config(config)
            return
        
        rows = []
        
        def flatten_config(data, prefix=""):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_config(value, full_key)
                else:
                    rows.append([full_key, str(value)])
        
        flatten_config(config)
        
        headers = ["Configuration Key", "Value"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    def _output_generic_table(self, data: Dict[str, Any]):
        """Output generic dictionary as table"""
        if not HAS_TABULATE:
            # Fallback to simple text format
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, default=str)[:50] + "..." if len(json.dumps(value, default=str)) > 50 else json.dumps(value, default=str)
                else:
                    value_str = str(value)
                print(f"  {key}: {value_str}")
            return
        
        rows = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, default=str)[:50] + "..." if len(json.dumps(value, default=str)) > 50 else json.dumps(value, default=str)
            else:
                value_str = str(value)
            rows.append([key, value_str])
        
        headers = ["Key", "Value"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    def _output_list_table(self, data: List[Any]):
        """Output list as table"""
        if not data:
            print("No data found.")
            return
        
        if not HAS_TABULATE:
            # Fallback to simple text format
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    print(f"  Item {i}:")
                    for key, value in item.items():
                        print(f"    {key}: {value}")
                else:
                    print(f"  {i}: {item}")
                print("  ---")
            return
        
        if isinstance(data[0], dict):
            # Extract headers from first item
            headers = list(data[0].keys())
            rows = []
            for item in data:
                row = [str(item.get(header, '')) for header in headers]
                rows.append(row)
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            # Simple list
            rows = [[i, str(item)] for i, item in enumerate(data)]
            headers = ["Index", "Value"]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    def _colorize_status(self, status: str) -> str:
        """Colorize status based on value"""
        if not self.use_color or not HAS_COLORAMA:
            return status
        
        status_lower = status.lower()
        if status_lower in ['active', 'running', 'healthy', 'connected']:
            return f"{Fore.GREEN}{status}{Style.RESET_ALL}"
        elif status_lower in ['idle', 'pending', 'created']:
            return f"{Fore.YELLOW}{status}{Style.RESET_ALL}"
        elif status_lower in ['failed', 'error', 'unhealthy', 'disconnected']:
            return f"{Fore.RED}{status}{Style.RESET_ALL}"
        elif status_lower in ['completed', 'success']:
            return f"{Fore.CYAN}{status}{Style.RESET_ALL}"
        else:
            return status
    
    def _colorize_json(self, json_str: str) -> str:
        """Simple JSON syntax highlighting"""
        if not self.use_color or not HAS_COLORAMA:
            return json_str
        
        # This is a simple implementation - in production, use a proper syntax highlighter
        json_str = json_str.replace('":', f'"{Fore.BLUE}:{Style.RESET_ALL}')
        json_str = json_str.replace('true', f'{Fore.GREEN}true{Style.RESET_ALL}')
        json_str = json_str.replace('false', f'{Fore.RED}false{Style.RESET_ALL}')
        json_str = json_str.replace('null', f'{Fore.MAGENTA}null{Style.RESET_ALL}')
        return json_str
    
    def _colorize_yaml(self, yaml_str: str) -> str:
        """Simple YAML syntax highlighting"""
        if not self.use_color or not HAS_COLORAMA:
            return yaml_str
        
        lines = yaml_str.split('\n')
        colored_lines = []
        
        for line in lines:
            if ':' in line and not line.strip().startswith('#'):
                key, value = line.split(':', 1)
                colored_line = f"{Fore.BLUE}{key}{Style.RESET_ALL}:{value}"
                colored_lines.append(colored_line)
            elif line.strip().startswith('#'):
                colored_lines.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
            else:
                colored_lines.append(line)
        
        return '\n'.join(colored_lines)
    
    def _print_nested(self, data: Any, indent: int = 0):
        """Print nested data with indentation"""
        prefix = " " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}:")
                    self._print_nested(value, indent + 2)
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._print_nested(item, indent + 2)
                else:
                    print(f"{prefix}- {item}")
    
    def success(self, message: str):
        """Print success message"""
        if self.use_color:
            print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
        else:
            print(f"✓ {message}")
    
    def error(self, message: str):
        """Print error message"""
        if self.use_color:
            print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}", file=sys.stderr)
        else:
            print(f"✗ {message}", file=sys.stderr)
    
    def warning(self, message: str):
        """Print warning message"""
        if self.use_color:
            print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")
        else:
            print(f"⚠ {message}")
    
    def info(self, message: str):
        """Print info message"""
        if self.use_color:
            print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")
        else:
            print(f"ℹ {message}")


class CLIValidator:
    """Validation utilities for CLI inputs"""
    
    @staticmethod
    def validate_agent_id(agent_id: str) -> bool:
        """Validate agent ID format"""
        return bool(agent_id and len(agent_id) > 0 and not agent_id.isspace())
    
    @staticmethod
    def validate_task_id(task_id: str) -> bool:
        """Validate task ID format"""
        return bool(task_id and len(task_id) > 0 and not task_id.isspace())
    
    @staticmethod
    def validate_workflow_id(workflow_id: str) -> bool:
        """Validate workflow ID format"""
        return bool(workflow_id and len(workflow_id) > 0 and not workflow_id.isspace())
    
    @staticmethod
    def validate_config_key(key: str) -> bool:
        """Validate configuration key format"""
        return bool(key and '.' in key and not key.startswith('.') and not key.endswith('.'))
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """Validate JSON string"""
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def validate_yaml(yaml_str: str) -> bool:
        """Validate YAML string"""
        try:
            yaml.safe_load(yaml_str)
            return True
        except yaml.YAMLError:
            return False
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path exists and is readable"""
        try:
            path = Path(file_path)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except (OSError, PermissionError):
            return False
    
    @staticmethod
    def validate_priority(priority: int) -> bool:
        """Validate task priority range"""
        return 1 <= priority <= 10
    
    @staticmethod
    def validate_timeout(timeout: int) -> bool:
        """Validate timeout value"""
        return timeout > 0 and timeout <= 3600  # Max 1 hour
    
    @staticmethod
    def validate_port(port: int) -> bool:
        """Validate port number"""
        return 1024 <= port <= 65535
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate name format (alphanumeric with underscores/hyphens)"""
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))


class CLIConfig:
    """Configuration management for CLI"""
    
    def __init__(self):
        self.config_data = {}
        self.config_file = None
        self.default_config_paths = [
            Path.home() / '.mark1' / 'config.yaml',
            Path.cwd() / 'mark1.yaml',
            Path.cwd() / 'config' / 'mark1.yaml'
        ]
        
        # Load default configuration
        self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        self.config_data = {
            'cli': {
                'default_output_format': 'table',
                'use_color': True,
                'pager': 'auto',
                'editor': 'nano'
            },
            'api': {
                'base_url': 'http://localhost:8000',
                'timeout': 30,
                'retry_attempts': 3
            },
            'websocket': {
                'url': 'ws://localhost:8765',
                'reconnect_attempts': 5,
                'reconnect_delay': 1
            },
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'file': None
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """Load configuration from file"""
        if config_path:
            paths_to_try = [Path(config_path)]
        else:
            paths_to_try = self.default_config_paths
        
        for path in paths_to_try:
            try:
                if path.exists():
                    with open(path, 'r') as f:
                        if path.suffix.lower() in ['.yaml', '.yml']:
                            loaded_config = yaml.safe_load(f)
                        else:  # Assume JSON
                            loaded_config = json.load(f)
                    
                    # Merge with default config
                    self._merge_config(loaded_config)
                    self.config_file = path
                    logger.info(f"Loaded configuration from {path}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
                continue
        
        logger.info("Using default configuration")
        return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        if config_path:
            path = Path(config_path)
        elif self.config_file:
            path = self.config_file
        else:
            path = self.default_config_paths[0]
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            
            self.config_file = path
            logger.info(f"Saved configuration to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by dot-notation key"""
        keys = key.split('.')
        config = self.config_data
        
        try:
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set final value
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config key {key}: {e}")
            return False
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing"""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config_data, new_config)
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of errors"""
        errors = []
        
        # Validate API configuration
        if 'api' in self.config_data:
            api_config = self.config_data['api']
            if 'base_url' not in api_config:
                errors.append("API base_url is required")
            if 'timeout' in api_config and not isinstance(api_config['timeout'], int):
                errors.append("API timeout must be an integer")
        
        # Validate WebSocket configuration
        if 'websocket' in self.config_data:
            ws_config = self.config_data['websocket']
            if 'url' not in ws_config:
                errors.append("WebSocket URL is required")
        
        # Validate logging configuration
        if 'logging' in self.config_data:
            log_config = self.config_data['logging']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if 'level' in log_config and log_config['level'] not in valid_levels:
                errors.append(f"Logging level must be one of: {valid_levels}")
        
        return errors
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self._load_default_config()
        self.config_file = None
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data"""
        return self.config_data.copy()


class CLIHelper:
    """Helper utilities for CLI operations"""
    
    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """Prompt user for confirmation"""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{message} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        
        return response in ['y', 'yes', 'true', '1']
    
    @staticmethod
    def prompt_for_input(message: str, required: bool = True, validator: Optional[callable] = None) -> Optional[str]:
        """Prompt user for input with optional validation"""
        while True:
            response = input(f"{message}: ").strip()
            
            if not response and required:
                print("This field is required. Please provide a value.")
                continue
            
            if not response and not required:
                return None
            
            if validator and not validator(response):
                print("Invalid input. Please try again.")
                continue
            
            return response
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}m {remaining_seconds:.1f}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(remaining_minutes)}m"
    
    @staticmethod
    def format_size(bytes_size: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}PB"
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def parse_key_value_pairs(pairs: List[str]) -> Dict[str, str]:
        """Parse key=value pairs from command line arguments"""
        result = {}
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                result[key.strip()] = value.strip()
            else:
                result[pair.strip()] = ''
        return result 