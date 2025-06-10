#!/usr/bin/env python3
"""
Standalone Agent CLI

Independent command-line interface for AI agent integration and management.
No database dependencies - works out of the box.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import typer
from rich.console import Console
from rich.panel import Panel

# Import our agent manager
from mark1.cli.agent_manager import app as agent_app

console = Console()

# Create main app
app = typer.Typer(
    name="mark1-agent",
    help="Mark-1 AI Agent Integration & Orchestration CLI",
    rich_markup_mode="rich"
)

# Add agent commands
app.add_typer(agent_app, name="", help="Agent management commands")


@app.command()
def orchestrate(
    prompt: str = typer.Argument(..., help="Your prompt/task description"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed orchestration process"),
    agent: str = typer.Option(None, "--agent", "-a", help="Force specific agent (optional)")
):
    """🎯 Orchestrate a task - Give Mark-1 any prompt and it will auto-select agents and execute"""
    
    async def run_orchestration():
        from mark1.orchestration.smart_orchestrator import SmartOrchestrator
        
        try:
            orchestrator = SmartOrchestrator(project_root)
            
            if verbose:
                console.print(f"[blue]🔍 Analyzing prompt: '{prompt}'[/blue]")
            
            # Run smart orchestration
            result = await orchestrator.orchestrate_prompt(prompt, force_agent=agent, verbose=verbose)
            
            # Display results
            console.print(Panel(
                f"[green]✅ Task Completed![/green]\n\n"
                f"[bold]Agent Used:[/bold] {result['agent_used']}\n"
                f"[bold]Framework:[/bold] {result['framework']}\n"
                f"[bold]Execution Time:[/bold] {result['execution_time']:.2f}s\n\n"
                f"[bold]Response:[/bold]\n{result['response']}\n\n"
                f"[dim]Confidence: {result['confidence']:.1%} | "
                f"Capabilities: {', '.join(result['capabilities'][:3])}[/dim]",
                title="🎯 Orchestration Result",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]❌ Orchestration failed: {e}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(1)
    
    asyncio.run(run_orchestration())


@app.command()
def quick_start():
    """🚀 Quick start guide for Mark-1 Agent System"""
    
    console.print(Panel(
        """[bold cyan]🚀 Mark-1 Quick Start Guide[/bold cyan]

[bold]1. Integrate your first agent:[/bold]
   mark1-agent integrate https://github.com/yoheinakajima/babyagi.git --quick

[bold]2. List available agents:[/bold]
   mark1-agent list

[bold]3. Give Mark-1 any task:[/bold]
   mark1-agent orchestrate "Generate a Python script for web scraping"
   mark1-agent orchestrate "Analyze this data and create a report"
   mark1-agent orchestrate "Plan a software project"

[bold]4. Test specific agents:[/bold]
   mark1-agent test babyagi --prompt "Create a task list"

[bold]5. Advanced usage:[/bold]
   mark1-agent orchestrate "Complex task" --verbose --agent specific_agent

[green]🎯 That's it! Mark-1 will automatically choose the best agent for any task.[/green]
        """,
        title="Quick Start",
        border_style="cyan"
    ))


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main() 