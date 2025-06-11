#!/usr/bin/env python3
"""
Plugin Management CLI Commands for Mark-1 Universal Plugin System

Command-line interface for managing plugins including installation,
listing, execution, and monitoring.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import structlog

from ..core.orchestrator import Mark1Orchestrator
from ..plugins.base_plugin import PluginStatus


logger = structlog.get_logger(__name__)
console = Console()

# Create plugin CLI app
plugin_app = typer.Typer(
    name="plugin",
    help="Plugin management commands for Mark-1 Universal Plugin System",
    rich_markup_mode="rich"
)


@plugin_app.command("install")
def install_plugin(
    repository_url: str = typer.Argument(..., help="GitHub repository URL to install as plugin"),
    branch: str = typer.Option("main", "--branch", "-b", help="Branch to clone"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstallation if plugin exists"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Install a plugin from a GitHub repository
    
    Examples:
        mark1 plugin install https://github.com/user/awesome-tool
        mark1 plugin install https://github.com/user/data-processor --branch develop
        mark1 plugin install https://github.com/user/image-tool --force
    """
    async def _install():
        try:
            orchestrator = Mark1Orchestrator()
            await orchestrator.initialize()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Installing plugin...", total=None)
                
                result = await orchestrator.install_plugin_from_repository(
                    repository_url=repository_url,
                    branch=branch,
                    force_reinstall=force
                )
                
                progress.update(task, completed=True)
            
            if result.success:
                console.print(Panel(
                    f"✅ Plugin installed successfully!\n\n"
                    f"Plugin ID: {result.plugin_id}\n"
                    f"Name: {result.plugin_metadata.name if result.plugin_metadata else 'Unknown'}\n"
                    f"Installation time: {result.installation_time:.2f}s",
                    title="Installation Complete",
                    border_style="green"
                ))
                
                if result.warnings:
                    console.print("\n⚠️  Warnings:")
                    for warning in result.warnings:
                        console.print(f"  • {warning}")
            else:
                console.print(Panel(
                    f"❌ Plugin installation failed!\n\n"
                    f"Error: {result.error}",
                    title="Installation Failed",
                    border_style="red"
                ))
                raise typer.Exit(1)
            
            await orchestrator.shutdown()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_install())


@plugin_app.command("list")
def list_plugins(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """
    List all installed plugins
    
    Examples:
        mark1 plugin list
        mark1 plugin list --status ready
        mark1 plugin list --verbose
    """
    async def _list():
        try:
            orchestrator = Mark1Orchestrator()
            await orchestrator.initialize()
            
            plugins = await orchestrator.list_installed_plugins()
            
            if status:
                plugins = [p for p in plugins if p.status.value == status]
            
            if not plugins:
                console.print("No plugins found.")
                return
            
            # Create table
            table = Table(title="Installed Plugins")
            table.add_column("Plugin ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Status", style="blue")
            table.add_column("Capabilities", style="magenta")
            
            if verbose:
                table.add_column("Repository", style="dim")
                table.add_column("Version", style="dim")
            
            for plugin in plugins:
                capabilities_str = ", ".join([cap.name for cap in plugin.capabilities[:3]])
                if len(plugin.capabilities) > 3:
                    capabilities_str += f" (+{len(plugin.capabilities) - 3} more)"
                
                row = [
                    plugin.plugin_id,
                    plugin.name,
                    plugin.plugin_type.value,
                    plugin.status.value,
                    capabilities_str or "None"
                ]
                
                if verbose:
                    row.extend([
                        plugin.repository_url,
                        plugin.version
                    ])
                
                table.add_row(*row)
            
            console.print(table)
            
            await orchestrator.shutdown()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_list())


@plugin_app.command("info")
def plugin_info(
    plugin_id: str = typer.Argument(..., help="Plugin ID to get information about")
):
    """
    Get detailed information about a specific plugin
    
    Examples:
        mark1 plugin info awesome-tool_abc123
    """
    async def _info():
        try:
            orchestrator = Mark1Orchestrator()
            await orchestrator.initialize()
            
            metadata = await orchestrator.get_plugin_metadata(plugin_id)
            
            if not metadata:
                console.print(f"[red]Plugin not found: {plugin_id}[/red]")
                raise typer.Exit(1)
            
            # Display plugin information
            info_text = f"""
[bold]Plugin Information[/bold]

[cyan]Basic Info:[/cyan]
  • ID: {metadata.plugin_id}
  • Name: {metadata.name}
  • Description: {metadata.description}
  • Version: {metadata.version}
  • Author: {metadata.author}
  • Type: {metadata.plugin_type.value}
  • Status: {metadata.status.value}
  • Execution Mode: {metadata.execution_mode.value}

[cyan]Repository:[/cyan]
  • URL: {metadata.repository_url}

[cyan]Capabilities:[/cyan]
"""
            
            for cap in metadata.capabilities:
                info_text += f"  • {cap.name}: {cap.description}\n"
                info_text += f"    Inputs: {', '.join(cap.input_types)}\n"
                info_text += f"    Outputs: {', '.join(cap.output_types)}\n"
            
            if metadata.dependencies:
                info_text += f"\n[cyan]Dependencies:[/cyan]\n"
                for dep in metadata.dependencies[:10]:  # Show first 10
                    info_text += f"  • {dep}\n"
                if len(metadata.dependencies) > 10:
                    info_text += f"  ... and {len(metadata.dependencies) - 10} more\n"
            
            if metadata.environment_variables:
                info_text += f"\n[cyan]Environment Variables:[/cyan]\n"
                for var_name, var_value in metadata.environment_variables.items():
                    info_text += f"  • {var_name}: {var_value or '(not set)'}\n"
            
            console.print(Panel(info_text, title=f"Plugin: {metadata.name}", border_style="blue"))
            
            await orchestrator.shutdown()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_info())


@plugin_app.command("execute")
def execute_plugin(
    task_description: str = typer.Argument(..., help="Task description to execute"),
    plugins: Optional[str] = typer.Option(None, "--plugins", "-p", help="Comma-separated list of plugin IDs to use"),
    max_plugins: int = typer.Option(3, "--max-plugins", "-m", help="Maximum number of plugins to use"),
    timeout: int = typer.Option(300, "--timeout", "-t", help="Execution timeout in seconds"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to file")
):
    """
    Execute a task using available plugins
    
    Examples:
        mark1 plugin execute "Process this image and resize it"
        mark1 plugin execute "Analyze this CSV file" --plugins image-tool_abc123,data-analyzer_def456
        mark1 plugin execute "Generate a report" --max-plugins 2 --timeout 600
    """
    async def _execute():
        try:
            orchestrator = Mark1Orchestrator()
            await orchestrator.initialize()
            
            # Parse plugin filter
            plugin_filter = None
            if plugins:
                plugin_filter = [p.strip() for p in plugins.split(",")]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing plugin task...", total=None)
                
                # Note: We need to implement orchestrate_plugin_task method
                # For now, let's use the existing orchestrate_task method
                result = await orchestrator.orchestrate_task(
                    task_description=task_description,
                    agent_filter=plugin_filter,  # Using agent_filter for now
                    max_agents=max_plugins,
                    timeout=timeout
                )
                
                progress.update(task, completed=True)
            
            # Display results
            if result.status.value == "completed":
                console.print(Panel(
                    f"✅ Task completed successfully!\n\n"
                    f"Task ID: {result.task_id}\n"
                    f"Summary: {result.summary}\n"
                    f"Execution time: {result.execution_time:.2f}s\n"
                    f"Plugins used: {', '.join(result.agents_used)}",
                    title="Execution Complete",
                    border_style="green"
                ))
                
                # Save results if requested
                if output_file:
                    output_path = Path(output_file)
                    with open(output_path, 'w') as f:
                        json.dump({
                            "task_id": result.task_id,
                            "task_description": task_description,
                            "status": result.status.value,
                            "summary": result.summary,
                            "execution_time": result.execution_time,
                            "plugins_used": result.agents_used,
                            "result_data": result.result_data
                        }, f, indent=2)
                    console.print(f"Results saved to: {output_path}")
            else:
                console.print(Panel(
                    f"❌ Task execution failed!\n\n"
                    f"Task ID: {result.task_id}\n"
                    f"Summary: {result.summary}",
                    title="Execution Failed",
                    border_style="red"
                ))
                raise typer.Exit(1)
            
            await orchestrator.shutdown()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_execute())


@plugin_app.command("uninstall")
def uninstall_plugin(
    plugin_id: str = typer.Argument(..., help="Plugin ID to uninstall"),
    force: bool = typer.Option(False, "--force", "-f", help="Force uninstallation without confirmation")
):
    """
    Uninstall a plugin
    
    Examples:
        mark1 plugin uninstall awesome-tool_abc123
        mark1 plugin uninstall data-processor_def456 --force
    """
    async def _uninstall():
        try:
            orchestrator = Mark1Orchestrator()
            await orchestrator.initialize()
            
            # Check if plugin exists
            metadata = await orchestrator.get_plugin_metadata(plugin_id)
            if not metadata:
                console.print(f"[red]Plugin not found: {plugin_id}[/red]")
                raise typer.Exit(1)
            
            # Confirm uninstallation
            if not force:
                confirm = typer.confirm(f"Are you sure you want to uninstall plugin '{metadata.name}' ({plugin_id})?")
                if not confirm:
                    console.print("Uninstallation cancelled.")
                    return
            
            # Uninstall plugin
            success = await orchestrator.plugin_manager.uninstall_plugin(plugin_id)
            
            if success:
                console.print(f"✅ Plugin '{metadata.name}' uninstalled successfully!")
            else:
                console.print(f"❌ Failed to uninstall plugin '{metadata.name}'")
                raise typer.Exit(1)
            
            await orchestrator.shutdown()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_uninstall())
