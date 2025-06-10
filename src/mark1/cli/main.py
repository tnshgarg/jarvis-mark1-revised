#!/usr/bin/env python3
"""
Mark-1 CLI Interface - Main Application

This module provides the main command-line interface for the Mark-1 AI Orchestrator,
including command parsing, routing, and execution.

Features:
- Comprehensive command structure
- Argument parsing and validation
- Interactive and batch modes
- Integration with all Mark-1 components
- Developer tools and utilities
- Configuration management
- Error handling and help system
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .commands import (
    AgentCommands, TaskCommands, WorkflowCommands, 
    ConfigCommands, SystemCommands, DevCommands
)
from .agent_manager import app as agent_manager_app
from .utils import CLIFormatter, CLIValidator, CLIConfig
from ..core.logging import setup_logging

logger = logging.getLogger(__name__)


class Mark1CLI:
    """Main Mark-1 CLI Application"""
    
    def __init__(self):
        self.config = CLIConfig()
        self.formatter = CLIFormatter()
        self.validator = CLIValidator()
        
        # Initialize command handlers
        self.agent_commands = AgentCommands()
        self.task_commands = TaskCommands()
        self.workflow_commands = WorkflowCommands()
        self.config_commands = ConfigCommands()
        self.system_commands = SystemCommands()
        self.dev_commands = DevCommands()
        
        # Command registry
        self.commands = {
            'agent': self.agent_commands,
            'task': self.task_commands,
            'workflow': self.workflow_commands,
            'config': self.config_commands,
            'system': self.system_commands,
            'dev': self.dev_commands
        }
        
        self.parser = self._create_parser()
        logger.info("Mark-1 CLI initialized")
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all commands and options"""
        parser = argparse.ArgumentParser(
            prog='mark1',
            description='Mark-1 AI Orchestrator Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  mark1 agent list                         # List all agents
  mark1 agent integrate <repo-url>         # Integrate AI agent repository
  mark1 agent remove <agent-id>            # Remove integrated agent
  mark1 task create --name "My Task"       # Create a new task
  mark1 workflow run --id workflow-123     # Run a workflow
  mark1 system status                      # Show system status
  mark1 dev generate-agent TestAgent       # Generate agent template
  
For more help on a specific command, use:
  mark1 <command> --help
            """
        )
        
        # Global options
        parser.add_argument(
            '--version', 
            action='version', 
            version='Mark-1 CLI v1.0.0'
        )
        parser.add_argument(
            '--config', '-c',
            help='Path to configuration file',
            default=None
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress output except errors'
        )
        parser.add_argument(
            '--output', '-o',
            choices=['json', 'yaml', 'table', 'text'],
            default='table',
            help='Output format'
        )
        parser.add_argument(
            '--no-color',
            action='store_true',
            help='Disable colored output'
        )
        
        # Create subparsers for main commands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='<command>'
        )
        
        # Agent commands
        agent_parser = subparsers.add_parser(
            'agent',
            help='Agent management commands',
            description='Manage AI agents in the Mark-1 system'
        )
        self._add_agent_commands(agent_parser)
        
        # Task commands
        task_parser = subparsers.add_parser(
            'task',
            help='Task management commands',
            description='Manage tasks and executions'
        )
        self._add_task_commands(task_parser)
        
        # Workflow commands
        workflow_parser = subparsers.add_parser(
            'workflow',
            help='Workflow management commands',
            description='Manage workflows and orchestration'
        )
        self._add_workflow_commands(workflow_parser)
        
        # Config commands
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management commands',
            description='Manage Mark-1 configuration'
        )
        self._add_config_commands(config_parser)
        
        # System commands
        system_parser = subparsers.add_parser(
            'system',
            help='System management commands',
            description='System status, health, and administration'
        )
        self._add_system_commands(system_parser)
        
        # Developer commands
        dev_parser = subparsers.add_parser(
            'dev',
            help='Developer tools and utilities',
            description='Development tools for Mark-1 system'
        )
        self._add_dev_commands(dev_parser)
        
        return parser
    
    def _add_agent_commands(self, parser: argparse.ArgumentParser):
        """Add agent-related commands"""
        agent_subparsers = parser.add_subparsers(dest='agent_action', help='Agent actions')
        
        # List agents
        list_parser = agent_subparsers.add_parser('list', help='List all agents')
        list_parser.add_argument('--status', choices=['active', 'idle', 'working'], help='Filter by status')
        list_parser.add_argument('--type', help='Filter by agent type')
        
        # Show agent details
        show_parser = agent_subparsers.add_parser('show', help='Show agent details')
        show_parser.add_argument('agent_id', help='Agent ID to show')
        show_parser.add_argument('--include-logs', action='store_true', help='Include recent logs')
        
        # Create agent
        create_parser = agent_subparsers.add_parser('create', help='Create new agent')
        create_parser.add_argument('--name', required=True, help='Agent name')
        create_parser.add_argument('--type', required=True, help='Agent type')
        create_parser.add_argument('--capabilities', nargs='+', help='Agent capabilities')
        create_parser.add_argument('--config', help='Path to agent config file')
        
        # Universal Integration Commands
        integrate_parser = agent_subparsers.add_parser('integrate', help='Integrate AI agent repository')
        integrate_parser.add_argument('repo_url', help='Git repository URL to integrate')
        integrate_parser.add_argument('--name', '-n', help='Custom name for the agent')
        integrate_parser.add_argument('--test/--no-test', dest='auto_test', default=True, help='Run tests after integration')
        integrate_parser.add_argument('--force', '-f', action='store_true', help='Force integration even if agent exists')
        
        # Remove integrated agent
        remove_parser = agent_subparsers.add_parser('remove', help='Remove integrated agent')
        remove_parser.add_argument('agent_id', help='Agent ID to remove')
        remove_parser.add_argument('--force', '-f', action='store_true', help='Force removal without confirmation')
        
        # Test agents
        test_parser = agent_subparsers.add_parser('test', help='Test integrated agents')
        test_parser.add_argument('agent_id', nargs='?', help='Specific agent ID to test (optional)')
        test_parser.add_argument('--prompt', '-p', default='Hello, can you help me?', help='Test prompt')
        test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # Analyze repository
        analyze_parser = agent_subparsers.add_parser('analyze', help='Analyze repository without integrating')
        analyze_parser.add_argument('repo_url', help='Repository URL to analyze')
        analyze_parser.add_argument('--output', '-o', help='Save analysis to file')
        
        # Integration status
        status_parser = agent_subparsers.add_parser('status', help='Show agent integration status')
        
        # Delete agent
        delete_parser = agent_subparsers.add_parser('delete', help='Delete agent')
        delete_parser.add_argument('agent_id', help='Agent ID to delete')
        delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
        
        # Start/stop agent
        start_parser = agent_subparsers.add_parser('start', help='Start agent')
        start_parser.add_argument('agent_id', help='Agent ID to start')
        
        stop_parser = agent_subparsers.add_parser('stop', help='Stop agent')
        stop_parser.add_argument('agent_id', help='Agent ID to stop')
        stop_parser.add_argument('--graceful', action='store_true', help='Graceful shutdown')
        
        # Agent logs
        logs_parser = agent_subparsers.add_parser('logs', help='Show agent logs')
        logs_parser.add_argument('agent_id', help='Agent ID')
        logs_parser.add_argument('--lines', '-n', type=int, default=50, help='Number of lines to show')
        logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow logs in real-time')
    
    def _add_task_commands(self, parser: argparse.ArgumentParser):
        """Add task-related commands"""
        task_subparsers = parser.add_subparsers(dest='task_action', help='Task actions')
        
        # List tasks
        list_parser = task_subparsers.add_parser('list', help='List tasks')
        list_parser.add_argument('--status', choices=['pending', 'running', 'completed', 'failed'], help='Filter by status')
        list_parser.add_argument('--agent', help='Filter by agent ID')
        list_parser.add_argument('--limit', type=int, default=50, help='Limit number of results')
        
        # Show task details
        show_parser = task_subparsers.add_parser('show', help='Show task details')
        show_parser.add_argument('task_id', help='Task ID to show')
        show_parser.add_argument('--include-logs', action='store_true', help='Include execution logs')
        
        # Create task
        create_parser = task_subparsers.add_parser('create', help='Create new task')
        create_parser.add_argument('--name', required=True, help='Task name')
        create_parser.add_argument('--description', help='Task description')
        create_parser.add_argument('--agent', help='Assign to specific agent')
        create_parser.add_argument('--priority', type=int, default=5, help='Task priority (1-10)')
        create_parser.add_argument('--config', help='Path to task config file')
        
        # Execute task
        exec_parser = task_subparsers.add_parser('execute', help='Execute task')
        exec_parser.add_argument('task_id', help='Task ID to execute')
        exec_parser.add_argument('--wait', action='store_true', help='Wait for completion')
        exec_parser.add_argument('--timeout', type=int, help='Execution timeout in seconds')
        
        # Cancel task
        cancel_parser = task_subparsers.add_parser('cancel', help='Cancel task')
        cancel_parser.add_argument('task_id', help='Task ID to cancel')
        cancel_parser.add_argument('--force', action='store_true', help='Force cancellation')
        
        # Task logs
        logs_parser = task_subparsers.add_parser('logs', help='Show task logs')
        logs_parser.add_argument('task_id', help='Task ID')
        logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow logs in real-time')
    
    def _add_workflow_commands(self, parser: argparse.ArgumentParser):
        """Add workflow-related commands"""
        workflow_subparsers = parser.add_subparsers(dest='workflow_action', help='Workflow actions')
        
        # List workflows
        list_parser = workflow_subparsers.add_parser('list', help='List workflows')
        list_parser.add_argument('--status', choices=['pending', 'running', 'completed', 'failed'], help='Filter by status')
        
        # Show workflow
        show_parser = workflow_subparsers.add_parser('show', help='Show workflow details')
        show_parser.add_argument('workflow_id', help='Workflow ID to show')
        show_parser.add_argument('--include-steps', action='store_true', help='Include step details')
        
        # Create workflow
        create_parser = workflow_subparsers.add_parser('create', help='Create workflow')
        create_parser.add_argument('--name', required=True, help='Workflow name')
        create_parser.add_argument('--description', help='Workflow description')
        create_parser.add_argument('--config', help='Path to workflow config file')
        
        # Run workflow
        run_parser = workflow_subparsers.add_parser('run', help='Run workflow')
        run_parser.add_argument('workflow_id', help='Workflow ID to run')
        run_parser.add_argument('--wait', action='store_true', help='Wait for completion')
        run_parser.add_argument('--params', help='Workflow parameters (JSON)')
        
        # Stop workflow
        stop_parser = workflow_subparsers.add_parser('stop', help='Stop workflow')
        stop_parser.add_argument('workflow_id', help='Workflow ID to stop')
        stop_parser.add_argument('--graceful', action='store_true', help='Graceful stop')
    
    def _add_config_commands(self, parser: argparse.ArgumentParser):
        """Add configuration commands"""
        config_subparsers = parser.add_subparsers(dest='config_action', help='Configuration actions')
        
        # Show config
        show_parser = config_subparsers.add_parser('show', help='Show configuration')
        show_parser.add_argument('--section', help='Show specific configuration section')
        
        # Set config
        set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key')
        set_parser.add_argument('value', help='Configuration value')
        
        # Get config
        get_parser = config_subparsers.add_parser('get', help='Get configuration value')
        get_parser.add_argument('key', help='Configuration key')
        
        # Validate config
        validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
        validate_parser.add_argument('--file', help='Configuration file to validate')
        
        # Reset config
        reset_parser = config_subparsers.add_parser('reset', help='Reset configuration')
        reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset')
    
    def _add_system_commands(self, parser: argparse.ArgumentParser):
        """Add system management commands"""
        system_subparsers = parser.add_subparsers(dest='system_action', help='System actions')
        
        # System status
        status_parser = system_subparsers.add_parser('status', help='Show system status')
        status_parser.add_argument('--detailed', action='store_true', help='Show detailed status')
        
        # Health check
        health_parser = system_subparsers.add_parser('health', help='System health check')
        health_parser.add_argument('--component', help='Check specific component')
        
        # System info
        info_parser = system_subparsers.add_parser('info', help='Show system information')
        
        # Start/stop system
        start_parser = system_subparsers.add_parser('start', help='Start Mark-1 system')
        start_parser.add_argument('--component', help='Start specific component')
        
        stop_parser = system_subparsers.add_parser('stop', help='Stop Mark-1 system')
        stop_parser.add_argument('--component', help='Stop specific component')
        stop_parser.add_argument('--graceful', action='store_true', help='Graceful shutdown')
        
        # System logs
        logs_parser = system_subparsers.add_parser('logs', help='Show system logs')
        logs_parser.add_argument('--component', help='Show logs for specific component')
        logs_parser.add_argument('--lines', '-n', type=int, default=100, help='Number of lines')
        logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow logs')
    
    def _add_dev_commands(self, parser: argparse.ArgumentParser):
        """Add developer tools and utilities"""
        dev_subparsers = parser.add_subparsers(dest='dev_action', help='Developer actions')
        
        # Generate code
        generate_parser = dev_subparsers.add_parser('generate', help='Generate code templates')
        generate_subparsers = generate_parser.add_subparsers(dest='generate_type', help='Generation type')
        
        # Generate agent
        agent_gen = generate_subparsers.add_parser('agent', help='Generate agent template')
        agent_gen.add_argument('name', help='Agent name')
        agent_gen.add_argument('--type', default='utility', help='Agent type')
        agent_gen.add_argument('--output', help='Output directory')
        
        # Generate workflow
        workflow_gen = generate_subparsers.add_parser('workflow', help='Generate workflow template')
        workflow_gen.add_argument('name', help='Workflow name')
        workflow_gen.add_argument('--steps', type=int, default=3, help='Number of steps')
        
        # Generate API
        api_gen = generate_subparsers.add_parser('api', help='Generate API endpoint')
        api_gen.add_argument('name', help='API endpoint name')
        api_gen.add_argument('--method', choices=['GET', 'POST', 'PUT', 'DELETE'], default='GET')
        
        # Debug tools
        debug_parser = dev_subparsers.add_parser('debug', help='Debug tools')
        debug_subparsers = debug_parser.add_subparsers(dest='debug_type', help='Debug type')
        
        # Debug agent
        agent_debug = debug_subparsers.add_parser('agent', help='Debug agent')
        agent_debug.add_argument('agent_id', help='Agent ID to debug')
        
        # Debug task
        task_debug = debug_subparsers.add_parser('task', help='Debug task')
        task_debug.add_argument('task_id', help='Task ID to debug')
        
        # Debug system
        system_debug = debug_subparsers.add_parser('system', help='Debug system')
        
        # Performance profiling
        profile_parser = dev_subparsers.add_parser('profile', help='Performance profiling')
        profile_parser.add_argument('target', choices=['tasks', 'agents', 'api'], help='Profiling target')
        profile_parser.add_argument('--duration', type=int, default=60, help='Profiling duration (seconds)')
        
        # Test utilities
        test_parser = dev_subparsers.add_parser('test', help='Test utilities')
        test_subparsers = test_parser.add_subparsers(dest='test_type', help='Test type')
        
        # Run tests
        run_tests = test_subparsers.add_parser('run', help='Run tests')
        run_tests.add_argument('--scope', choices=['unit', 'integration', 'all'], default='all')
        run_tests.add_argument('--coverage', action='store_true', help='Generate coverage report')
        
        # Generate test data
        gen_data = test_subparsers.add_parser('data', help='Generate test data')
        gen_data.add_argument('type', choices=['tasks', 'agents', 'workflows'], help='Data type')
        gen_data.add_argument('--count', type=int, default=10, help='Number of items to generate')
    
    async def execute_command(self, args: argparse.Namespace) -> int:
        """Execute the parsed command"""
        try:
            # Set up logging level based on verbosity
            if args.quiet:
                logging.getLogger().setLevel(logging.ERROR)
            elif args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            
            # Load configuration
            if args.config:
                self.config.load_config(args.config)
            
            # Set output format
            self.formatter.set_format(args.output)
            self.formatter.set_color(not args.no_color)
            
            # Route to appropriate command handler
            if not hasattr(args, 'command') or args.command is None:
                self.parser.print_help()
                return 1
            
            command_handler = self.commands.get(args.command)
            if not command_handler:
                self.formatter.error(f"Unknown command: {args.command}")
                return 1
            
            # Execute the command
            result = await command_handler.execute(args)
            
            if result.get('success', False):
                if result.get('output'):
                    self.formatter.output(result['output'], args.output)
                return 0
            else:
                self.formatter.error(result.get('error', 'Command failed'))
                return 1
                
        except Exception as e:
            logger.exception("Command execution failed")
            self.formatter.error(f"Error: {str(e)}")
            return 1
    
    def parse_args(self, argv: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments"""
        return self.parser.parse_args(argv)


def create_cli_app() -> Mark1CLI:
    """Create a new CLI application instance"""
    return Mark1CLI()


async def main_cli(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point"""
    # Set up logging
    setup_logging()
    
    # Create CLI app
    cli = create_cli_app()
    
    # Parse arguments
    try:
        args = cli.parse_args(argv)
    except SystemExit as e:
        return e.code
    
    # Execute command
    return await cli.execute_command(args)


def cli_entry_point():
    """Entry point for console scripts"""
    try:
        exit_code = asyncio.run(main_cli())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point() 