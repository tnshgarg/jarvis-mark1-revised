"""
Mark-1 Orchestrator - Main Entry Point

Advanced AI Agent Orchestration System with comprehensive framework integration
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mark1.config.settings import get_settings
from mark1.config.logging_config import setup_logging
from mark1.storage.database import init_database, close_database
from mark1.core.orchestrator import Mark1Orchestrator
from mark1.utils.exceptions import Mark1BaseException, handle_exception
from mark1.api.rest_api import create_app
from mark1.api.cli import cli as cli_app

# Initialize Rich console for beautiful output
console = Console()

# Create Typer app
app = typer.Typer(
    name="mark1",
    help="Mark-1 AI Agent Orchestrator - Advanced multi-agent system management",
    rich_markup_mode="rich"
)

# Global orchestrator instance
orchestrator: Optional[Mark1Orchestrator] = None

# Add agent management commands
try:
    from mark1.cli.agent_manager import app as agent_app
    app.add_typer(agent_app, name="agent", help="AI Agent integration and management")
except ImportError as e:
    console.print(f"[yellow]Warning: Could not import agent manager: {e}[/yellow]")


def print_banner():
    """Print the Mark-1 banner"""
    banner_text = """
    ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗     ██╗
    ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝    ███║
    ██╔████╔██║███████║██████╔╝█████╔╝     ╚██║
    ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗      ██║
    ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗     ██║
    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝
    
    Advanced AI Agent Orchestration System
    """
    
    console.print(Panel(
        Text(banner_text, style="bold blue"),
        title="[bold white]Mark-1 Orchestrator[/bold white]",
        subtitle="[italic]Multi-Agent System Management[/italic]",
        border_style="blue"
    ))


async def setup_application() -> Mark1Orchestrator:
    """Initialize the Mark-1 application"""
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings)
    logger = structlog.get_logger(__name__)
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_database()
        
        # Create orchestrator
        logger.info("Initializing orchestrator...")
        orch = Mark1Orchestrator()
        await orch.initialize()
        
        # Create necessary directories
        settings.create_directories()
        
        logger.info("Mark-1 orchestrator initialized successfully")
        return orch
        
    except Exception as e:
        logger.error("Failed to initialize Mark-1", error=str(e))
        raise Mark1BaseException(f"Application initialization failed: {e}")


async def cleanup_application():
    """Cleanup application resources"""
    logger = structlog.get_logger(__name__)
    
    try:
        if orchestrator:
            logger.info("Shutting down orchestrator...")
            await orchestrator.shutdown()
        
        logger.info("Closing database connections...")
        await close_database()
        
        logger.info("Mark-1 orchestrator shutdown complete")
        
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        console.print("\n[yellow]Received shutdown signal, cleaning up...[/yellow]")
        asyncio.create_task(cleanup_application())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes")
):
    """Start the Mark-1 API server"""
    print_banner()
    
    try:
        import uvicorn
        from mark1.api.rest_api import create_app
        
        console.print(f"[green]Starting Mark-1 API server on {host}:{port}[/green]")
        
        # Create FastAPI app
        fastapi_app = create_app()
        
        # Run with uvicorn
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_config=None  # Use our custom logging
        )
        
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def scan(
    path: Path = typer.Argument(..., help="Path to scan for agents"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Recursive scan"),
    frameworks: Optional[str] = typer.Option(None, "--frameworks", "-f", help="Comma-separated list of frameworks to detect")
):
    """Scan codebase for AI agents"""
    print_banner()
    
    async def run_scan():
        try:
            orch = await setup_application()
            
            console.print(f"[blue]Scanning {path} for AI agents...[/blue]")
            
            # Parse frameworks filter
            framework_filter = None
            if frameworks:
                framework_filter = [f.strip() for f in frameworks.split(",")]
            
            # Run the scan
            results = await orch.scan_codebase(
                path=path,
                recursive=recursive,
                framework_filter=framework_filter
            )
            
            console.print(f"[green]Scan complete! Found {len(results.agents)} agents[/green]")
            
            # Display results
            for agent in results.agents:
                console.print(f"  • {agent.name} ({agent.framework}) - {agent.capabilities}")
            
            # Save results if output specified
            if output:
                await results.save_to_file(output)
                console.print(f"[blue]Results saved to {output}[/blue]")
                
        except Exception as e:
            handle_exception(e, reraise=False)
            console.print(f"[red]Scan failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await cleanup_application()
    
    asyncio.run(run_scan())


@app.command()
def orchestrate(
    task: str = typer.Argument(..., help="Task description to orchestrate"),
    agents: Optional[str] = typer.Option(None, "--agents", "-a", help="Comma-separated list of agent IDs"),
    max_agents: int = typer.Option(3, "--max-agents", help="Maximum number of agents to use"),
    timeout: int = typer.Option(300, "--timeout", "-t", help="Task timeout in seconds")
):
    """Orchestrate a task across multiple agents"""
    print_banner()
    
    async def run_orchestration():
        try:
            orch = await setup_application()
            
            console.print(f"[blue]Orchestrating task: {task}[/blue]")
            
            # Parse agent filter
            agent_filter = None
            if agents:
                agent_filter = [a.strip() for a in agents.split(",")]
            
            # Run orchestration
            result = await orch.orchestrate_task(
                task_description=task,
                agent_filter=agent_filter,
                max_agents=max_agents,
                timeout=timeout
            )
            
            console.print(f"[green]Task completed successfully![/green]")
            console.print(f"Result: {result.summary}")
            
        except Exception as e:
            handle_exception(e, reraise=False)
            console.print(f"[red]Orchestration failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await cleanup_application()
    
    asyncio.run(run_orchestration())


@app.command()
def interactive():
    """Start interactive Mark-1 shell"""
    print_banner()
    
    async def run_interactive():
        try:
            global orchestrator
            orchestrator = await setup_application()
            
            console.print("[green]Mark-1 interactive shell started[/green]")
            console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]")
            
            # Start CLI interface
            await cli_app.run_interactive(orchestrator)
            
        except Exception as e:
            handle_exception(e, reraise=False)
            console.print(f"[red]Interactive shell failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await cleanup_application()
    
    asyncio.run(run_interactive())


@app.command()
def status():
    """Show Mark-1 system status"""
    print_banner()
    
    async def check_status():
        try:
            orch = await setup_application()
            status = await orch.get_system_status()
            
            console.print("[green]Mark-1 System Status[/green]")
            console.print(f"Status: {status.overall_status}")
            console.print(f"Agents: {status.agent_count}")
            console.print(f"Active Tasks: {status.active_tasks}")
            console.print(f"Database: {status.database_status}")
            console.print(f"LLM Provider: {status.llm_status}")
            
        except Exception as e:
            handle_exception(e, reraise=False)
            console.print(f"[red]Status check failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await cleanup_application()
    
    asyncio.run(check_status())


@app.command()
def init(
    path: Path = typer.Argument(Path.cwd(), help="Path to initialize Mark-1 project"),
    template: str = typer.Option("basic", "--template", "-t", help="Project template to use")
):
    """Initialize a new Mark-1 project"""
    print_banner()
    
    try:
        from mark1.utils.project_template import create_project_template
        
        console.print(f"[blue]Initializing Mark-1 project at {path}[/blue]")
        
        create_project_template(path, template)
        
        console.print(f"[green]Mark-1 project initialized successfully![/green]")
        console.print(f"[dim]Next steps:[/dim]")
        console.print("  1. Configure .env file")
        console.print("  2. Run 'mark1 scan .' to discover agents")
        console.print("  3. Run 'mark1 serve' to start the API server")
        
    except Exception as e:
        handle_exception(e, reraise=False)
        console.print(f"[red]Project initialization failed: {e}[/red]")
        raise typer.Exit(1)


# Add CLI commands
app.add_typer(cli_app, name="cli", help="Interactive CLI commands")


def main():
    """Main entry point"""
    setup_signal_handlers()
    app()


if __name__ == "__main__":
    main()
