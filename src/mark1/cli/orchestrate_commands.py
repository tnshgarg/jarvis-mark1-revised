#!/usr/bin/env python3
"""
Orchestration CLI Commands for Mark-1 Universal Plugin System

Provides CLI commands for intelligent task orchestration using natural language prompts.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import structlog

from ..core.intelligent_orchestrator import IntelligentOrchestrator
from ..plugins import PluginManager
from ..core.context_manager import ContextManager
from ..core.workflow_engine import WorkflowEngine


logger = structlog.get_logger(__name__)
console = Console()

# Create orchestrate CLI app
orchestrate_app = typer.Typer(
    name="orchestrate",
    help="Intelligent task orchestration using natural language prompts",
    rich_markup_mode="rich"
)

# OLLAMA URL from environment or default
OLLAMA_URL = "https://f6da-103-167-213-208.ngrok-free.app"


@orchestrate_app.command("run")
def orchestrate_task(
    prompt: str = typer.Argument(..., help="Natural language description of what you want to accomplish"),
    max_plugins: int = typer.Option(5, "--max-plugins", "-m", help="Maximum number of plugins to use"),
    timeout: int = typer.Option(600, "--timeout", "-t", help="Total timeout in seconds"),
    context_file: Optional[str] = typer.Option(None, "--context", "-c", help="JSON file with additional context"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    real_time: bool = typer.Option(True, "--real-time", "-r", help="Show real-time progress")
):
    """
    Orchestrate a task using natural language prompt
    
    Examples:
        mark1 orchestrate run "Analyze the sales data and create a visualization"
        mark1 orchestrate run "Process all images in the folder and compress them" --max-plugins 3
        mark1 orchestrate run "Generate a report from the CSV file" --context context.json
    """
    async def _orchestrate():
        try:
            # Load context if provided
            context = {}
            if context_file:
                try:
                    with open(context_file, 'r') as f:
                        context = json.load(f)
                    console.print(f"✅ Loaded context from {context_file}")
                except Exception as e:
                    console.print(f"⚠️  Failed to load context file: {e}")
            
            # Initialize components
            console.print("🚀 Initializing Mark-1 Intelligent Orchestrator...")
            
            # Create plugin manager
            plugins_dir = Path.home() / ".mark1" / "plugins"
            plugins_dir.mkdir(parents=True, exist_ok=True)
            plugin_manager = PluginManager(plugins_directory=plugins_dir)
            
            # Create context manager
            context_manager = ContextManager()
            await context_manager.initialize()
            
            # Create workflow engine
            workflow_engine = WorkflowEngine()
            await workflow_engine.initialize()
            
            # Create intelligent orchestrator
            orchestrator = IntelligentOrchestrator(ollama_url=OLLAMA_URL)
            await orchestrator.initialize(plugin_manager, context_manager, workflow_engine)
            
            console.print("✅ All components initialized")
            
            # Check available plugins
            available_plugins = await plugin_manager.list_installed_plugins()
            console.print(f"📦 Found {len(available_plugins)} available plugins")
            
            if len(available_plugins) == 0:
                console.print(Panel(
                    "⚠️  No plugins installed!\n\n"
                    "Install plugins first using:\n"
                    "mark1 plugin install <github-repo-url>",
                    title="No Plugins Available",
                    border_style="yellow"
                ))
                return
            
            # Show available plugins
            if verbose:
                plugins_table = Table(title="Available Plugins")
                plugins_table.add_column("Name", style="cyan")
                plugins_table.add_column("Type", style="green")
                plugins_table.add_column("Capabilities", style="yellow")
                
                for plugin in available_plugins:
                    capabilities = ", ".join([cap.name for cap in plugin.capabilities[:3]])
                    if len(plugin.capabilities) > 3:
                        capabilities += f" (+{len(plugin.capabilities) - 3} more)"
                    
                    plugins_table.add_row(
                        plugin.name,
                        plugin.plugin_type.value,
                        capabilities or "None"
                    )
                
                console.print(plugins_table)
            
            # Start orchestration
            console.print(Panel(
                f"🎯 Starting intelligent orchestration\n\n"
                f"Prompt: {prompt}\n"
                f"Max plugins: {max_plugins}\n"
                f"Timeout: {timeout}s",
                title="Orchestration Started",
                border_style="blue"
            ))
            
            if real_time:
                # Real-time progress display
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    
                    main_task = progress.add_task("Orchestrating...", total=100)
                    
                    # Start orchestration
                    result = await orchestrator.orchestrate_from_prompt(
                        user_prompt=prompt,
                        context=context,
                        max_plugins=max_plugins,
                        timeout=timeout
                    )
                    
                    progress.update(main_task, completed=100)
            else:
                # Simple execution
                result = await orchestrator.orchestrate_from_prompt(
                    user_prompt=prompt,
                    context=context,
                    max_plugins=max_plugins,
                    timeout=timeout
                )
            
            # Display results
            if result["success"]:
                console.print(Panel(
                    f"✅ Orchestration completed successfully!\n\n"
                    f"Orchestration ID: {result['orchestration_id']}\n"
                    f"Execution time: {result['execution_time']:.2f}s\n"
                    f"Steps executed: {result['successful_steps']}/{result['total_steps']}\n"
                    f"Outputs generated: {len(result['outputs'])}",
                    title="Success",
                    border_style="green"
                ))
                
                # Show outputs
                if result["outputs"] and verbose:
                    console.print("\n📊 Outputs:")
                    for step_id, output in result["outputs"].items():
                        console.print(f"  {step_id}: {str(output)[:100]}...")
                
                # Show shared data
                if result["shared_data"] and verbose:
                    console.print("\n🔗 Shared Data:")
                    for key, value in result["shared_data"].items():
                        console.print(f"  {key}: {str(value)[:100]}...")
            
            else:
                console.print(Panel(
                    f"❌ Orchestration failed!\n\n"
                    f"Orchestration ID: {result['orchestration_id']}\n"
                    f"Execution time: {result['execution_time']:.2f}s\n"
                    f"Error: {result.get('error', 'Unknown error')}",
                    title="Failed",
                    border_style="red"
                ))
                
                # Show errors
                if result.get("errors"):
                    console.print("\n❌ Errors:")
                    for error in result["errors"]:
                        console.print(f"  Step {error.get('step_id', 'unknown')}: {error.get('error', 'Unknown error')}")
            
            # Save results if requested
            if output_file:
                try:
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    console.print(f"💾 Results saved to {output_file}")
                except Exception as e:
                    console.print(f"⚠️  Failed to save results: {e}")
            
            # Cleanup
            await orchestrator.cleanup()
            await context_manager.cleanup()
            
        except Exception as e:
            console.print(f"[red]❌ Orchestration failed: {e}[/red]")
            import traceback
            if verbose:
                console.print(f"[red]{traceback.format_exc()}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_orchestrate())


@orchestrate_app.command("status")
def orchestration_status(
    orchestration_id: Optional[str] = typer.Argument(None, help="Orchestration ID to check status"),
    list_active: bool = typer.Option(False, "--list", "-l", help="List all active orchestrations"),
    history: bool = typer.Option(False, "--history", "-h", help="Show orchestration history")
):
    """
    Check orchestration status or list active orchestrations
    
    Examples:
        mark1 orchestrate status abc123
        mark1 orchestrate status --list
        mark1 orchestrate status --history
    """
    async def _status():
        try:
            # Initialize orchestrator
            orchestrator = IntelligentOrchestrator(ollama_url=OLLAMA_URL)
            
            if history:
                # Show history
                history_data = await orchestrator.get_orchestration_history(limit=10)
                
                if not history_data:
                    console.print("No orchestration history found.")
                    return
                
                history_table = Table(title="Orchestration History")
                history_table.add_column("ID", style="cyan")
                history_table.add_column("Status", style="green")
                history_table.add_column("Duration", style="yellow")
                history_table.add_column("Steps", style="blue")
                history_table.add_column("Started", style="dim")
                
                for item in history_data:
                    status = "✅ Success" if item["success"] else "❌ Failed"
                    orchestration_id_short = item["orchestration_id"][:8]
                    duration = f"{item['execution_time']:.1f}s"
                    steps = f"{item['successful_steps']}/{item['total_steps']}"
                    started = item["start_time"][:19].replace("T", " ")
                    
                    history_table.add_row(
                        orchestration_id_short,
                        status,
                        duration,
                        steps,
                        started
                    )
                
                console.print(history_table)
            
            elif list_active:
                # List active orchestrations
                console.print("Active orchestrations feature not yet implemented")
            
            elif orchestration_id:
                # Check specific orchestration
                status = await orchestrator.get_orchestration_status(orchestration_id)
                
                if status:
                    console.print(Panel(
                        f"Orchestration ID: {orchestration_id}\n"
                        f"Status: {status['status']}\n"
                        f"Started: {status['start_time']}\n"
                        f"Prompt: {status['prompt']}",
                        title="Orchestration Status",
                        border_style="blue"
                    ))
                else:
                    console.print(f"Orchestration {orchestration_id} not found or completed.")
            
            else:
                console.print("Please specify an orchestration ID, use --list, or --history")
            
            await orchestrator.cleanup()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_status())


@orchestrate_app.command("test")
def test_orchestration():
    """
    Test the orchestration system with a simple example
    """
    async def _test():
        try:
            console.print("🧪 Testing Mark-1 Orchestration System")
            
            # Test OLLAMA connection
            from ..llm.ollama_client import OllamaClient
            ollama_client = OllamaClient(base_url=OLLAMA_URL)
            
            console.print("🔗 Testing OLLAMA connection...")
            if await ollama_client.health_check():
                console.print("✅ OLLAMA is accessible")
                
                # List available models
                models = await ollama_client.list_models()
                console.print(f"📋 Found {len(models)} OLLAMA models")
                for model in models[:3]:  # Show first 3
                    console.print(f"  • {model.name}")
            else:
                console.print("❌ OLLAMA is not accessible")
                return
            
            # Test plugin system
            console.print("\n📦 Testing plugin system...")
            plugins_dir = Path.home() / ".mark1" / "plugins"
            plugin_manager = PluginManager(plugins_directory=plugins_dir)
            
            plugins = await plugin_manager.list_installed_plugins()
            console.print(f"✅ Found {len(plugins)} installed plugins")
            
            await ollama_client.close()
            console.print("\n🎉 Orchestration system test completed!")
            
        except Exception as e:
            console.print(f"[red]❌ Test failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_test())
