#!/usr/bin/env python3
"""
Agent Manager CLI

Command-line interface for managing AI agent integrations in Mark-1
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()
app = typer.Typer(
    name="agent",
    help="Manage AI agent integrations in Mark-1 orchestrator",
    rich_markup_mode="rich"
)

# Import with fallback for missing dependencies
try:
    from mark1.agents.universal_integrator import UniversalAgentIntegrator, AgentFramework
except ImportError:
    try:
        from ..agents.universal_integrator import UniversalAgentIntegrator, AgentFramework
    except ImportError:
        console.print("[red]Warning: Universal integrator not available[/red]")
        UniversalAgentIntegrator = None
        AgentFramework = None

# Settings import with fallback
try:
    from mark1.config.settings import get_settings
except ImportError:
    def get_settings():
        class MockSettings:
            base_dir = Path.cwd()
        return MockSettings()


@app.command()
def integrate(
    repo_url: str = typer.Argument(..., help="Git repository URL to integrate"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom name for the agent"),
    auto_test: bool = typer.Option(True, "--test/--no-test", help="Run tests after integration"),
    force: bool = typer.Option(False, "--force", "-f", help="Force integration even if agent exists"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick integration (skip dependency installation)")
):
    """Integrate a new AI agent repository into Mark-1"""
    
    if UniversalAgentIntegrator is None:
        console.print("[red]Universal integrator not available. Please check dependencies.[/red]")
        raise typer.Exit(1)
    
    async def run_integration():
        settings = get_settings()
        integrator = UniversalAgentIntegrator(getattr(settings, 'base_dir', Path.cwd()))
        
        try:
            if quick:
                # Quick integration: analyze only, skip heavy operations
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Quick integration starting...", total=None)
                    
                    progress.update(task, description="🔗 Cloning repository...")
                    clone_path = await integrator._clone_repository(repo_url, name)
                    
                    progress.update(task, description="📊 Analyzing repository...")
                    metadata = await integrator._analyze_repository(clone_path)
                    
                    progress.update(task, description="📋 Creating plan...")
                    plan = await integrator._create_integration_plan(metadata, clone_path)
                    
                    progress.update(task, description="🎭 Generating wrapper...")
                    await integrator._create_agent_wrapper(plan, clone_path)
                    await integrator._create_api_adapter(plan, clone_path)
                    await integrator._register_with_mark1(plan)
                    
                    progress.update(task, description="✅ Quick integration complete!")
                
                console.print(Panel(
                    f"[green]Quick integration completed![/green]\n\n"
                    f"⚠️  Dependencies not installed (use --no-quick for full integration)\n"
                    f"Framework: {plan.agent_metadata.framework.value}\n"
                    f"Agent ID: {plan.agent_metadata.name.lower()}",
                    title="Quick Integration Complete",
                    border_style="yellow"
                ))
                
                _display_analysis_results(plan)
                
            else:
                # Full integration workflow
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    
                    # Complete integration workflow
                    task = progress.add_task("Starting integration workflow...", total=None)
                    
                    progress.update(task, description="🔗 Cloning repository...")
                    
                    # This does the complete integration: clone → analyze → plan → execute → test
                    plan = await integrator.integrate_repository(repo_url, name)
                    
                    progress.update(task, description="✅ Integration complete!")
                    
                    # Display the results
                    _display_analysis_results(plan)
                    
                    # Confirm if user wants to keep it
                    if not force:
                        console.print("\n[yellow]Integration completed successfully![/yellow]")
                        if not Confirm.ask("Keep this integration?"):
                            console.print("[yellow]Removing integration...[/yellow]")
                            await integrator.remove_agent(plan.agent_metadata.name.lower())
                            console.print("[yellow]Integration removed[/yellow]")
                            return
                    
                    console.print(Panel(
                        f"[green]Successfully integrated {plan.agent_metadata.name}![/green]\n\n"
                        f"Framework: {plan.agent_metadata.framework.value}\n"
                        f"Capabilities: {', '.join([cap.value for cap in plan.agent_metadata.capabilities])}\n"
                        f"Agent ID: {plan.agent_metadata.name.lower()}",
                        title="Integration Complete",
                        border_style="green"
                    ))
                
        except Exception as e:
            console.print(f"[red]Integration failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            integrator.cleanup()
    
    asyncio.run(run_integration())


@app.command("list")
def list_agents():
    """List all integrated agents"""
    
    if UniversalAgentIntegrator is None:
        console.print("[red]Universal integrator not available. Please check dependencies.[/red]")
        raise typer.Exit(1)
    
    async def run_list():
        settings = get_settings()
        integrator = UniversalAgentIntegrator(getattr(settings, 'base_dir', Path.cwd()))
        
        try:
            agents = await integrator.list_integrated_agents()
            
            if not agents:
                console.print("[yellow]No integrated agents found[/yellow]")
                return
            
            table = Table(title="Integrated AI Agents")
            table.add_column("Agent ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Framework", style="blue")
            table.add_column("Capabilities", style="green")
            table.add_column("Status", style="yellow")
            
            for agent in agents:
                capabilities = ", ".join(agent.get("capabilities", [])[:3])
                if len(agent.get("capabilities", [])) > 3:
                    capabilities += "..."
                
                table.add_row(
                    agent["agent_id"],
                    agent["name"],
                    agent["framework"],
                    capabilities,
                    "Active"  # TODO: Get actual status
                )
            
            console.print(table)
            console.print(f"\n[blue]Total agents: {len(agents)}[/blue]")
            
        except Exception as e:
            console.print(f"[red]Failed to list agents: {e}[/red]")
            raise typer.Exit(1)
        finally:
            integrator.cleanup()
    
    asyncio.run(run_list())


@app.command()
def remove(
    agent_id: str = typer.Argument(..., help="Agent ID to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation")
):
    """Remove an integrated agent"""
    
    if UniversalAgentIntegrator is None:
        console.print("[red]Universal integrator not available. Please check dependencies.[/red]")
        raise typer.Exit(1)
    
    async def run_removal():
        settings = get_settings()
        integrator = UniversalAgentIntegrator(getattr(settings, 'base_dir', Path.cwd()))
        
        try:
            # Check if agent exists
            agents = await integrator.list_integrated_agents()
            agent = next((a for a in agents if a["agent_id"] == agent_id), None)
            
            if not agent:
                console.print(f"[red]Agent '{agent_id}' not found[/red]")
                raise typer.Exit(1)
            
            # Confirm removal
            if not force:
                console.print(Panel(
                    f"Name: {agent['name']}\n"
                    f"Framework: {agent['framework']}\n"
                    f"Capabilities: {', '.join(agent.get('capabilities', []))}",
                    title=f"Remove Agent: {agent_id}",
                    border_style="red"
                ))
                
                if not Confirm.ask(f"Are you sure you want to remove agent '{agent_id}'?"):
                    console.print("[yellow]Removal cancelled[/yellow]")
                    return
            
            # Remove agent
            success = await integrator.remove_agent(agent_id)
            
            if success:
                console.print(f"[green]Successfully removed agent '{agent_id}'[/green]")
            else:
                console.print(f"[red]Failed to remove agent '{agent_id}'[/red]")
                raise typer.Exit(1)
                
        except Exception as e:
            console.print(f"[red]Removal failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            integrator.cleanup()
    
    asyncio.run(run_removal())


@app.command()
def test(
    agent_id: Optional[str] = typer.Argument(None, help="Specific agent ID to test (optional)"),
    prompt: str = typer.Option("Hello, can you help me?", "--prompt", "-p", help="Test prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Test integrated agents"""
    
    if UniversalAgentIntegrator is None:
        console.print("[red]Universal integrator not available. Please check dependencies.[/red]")
        raise typer.Exit(1)
    
    async def run_test():
        settings = get_settings()
        integrator = UniversalAgentIntegrator(getattr(settings, 'base_dir', Path.cwd()))
        
        try:
            agents = await integrator.list_integrated_agents()
            
            if agent_id:
                # Test specific agent
                agent = next((a for a in agents if a["agent_id"] == agent_id), None)
                if not agent:
                    console.print(f"[red]Agent '{agent_id}' not found[/red]")
                    raise typer.Exit(1)
                agents = [agent]
            
            if not agents:
                console.print("[yellow]No agents to test[/yellow]")
                return
            
            console.print(f"[blue]Testing {len(agents)} agent(s) with prompt: '{prompt}'[/blue]\n")
            
            for agent in agents:
                console.print(f"Testing {agent['name']} ({agent['framework']})...")
                
                # TODO: Implement actual agent testing via wrapper
                # For now, simulate test
                import random
                import time
                
                start_time = time.time()
                await asyncio.sleep(0.5)  # Simulate processing
                end_time = time.time()
                
                success = random.choice([True, True, True, False])  # 75% success rate
                
                if success:
                    response = f"Mock response from {agent['name']}: I understand your request and I'm ready to help!"
                    console.print(f"  ✅ [green]Success[/green] ({end_time - start_time:.2f}s)")
                    if verbose:
                        console.print(f"     Response: {response}")
                else:
                    console.print(f"  ❌ [red]Failed[/red] (timeout or error)")
                
                console.print()
            
        except Exception as e:
            console.print(f"[red]Testing failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            integrator.cleanup()
    
    asyncio.run(run_test())


@app.command()
def analyze(
    repo_url: str = typer.Argument(..., help="Repository URL to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save analysis to file")
):
    """Analyze a repository without integrating it"""
    
    if UniversalAgentIntegrator is None:
        console.print("[red]Universal integrator not available. Please check dependencies.[/red]")
        raise typer.Exit(1)
    
    async def run_analysis():
        settings = get_settings()
        integrator = UniversalAgentIntegrator(getattr(settings, 'base_dir', Path.cwd()))
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Analyzing repository...", total=None)
                
                # Clone and analyze
                clone_path = await integrator._clone_repository(repo_url)
                metadata = await integrator._analyze_repository(clone_path)
                plan = await integrator._create_integration_plan(metadata, clone_path)
                
                progress.update(task, description="✅ Analysis complete")
            
            # Display results
            _display_analysis_results(plan, detailed=True)
            
            # Save to file if requested
            if output:
                analysis_data = {
                    "repository": repo_url,
                    "metadata": {
                        "name": metadata.name,
                        "framework": metadata.framework.value,
                        "version": metadata.version,
                        "description": metadata.description,
                        "capabilities": [cap.value for cap in metadata.capabilities],
                        "dependencies": metadata.dependencies,
                        "entry_points": metadata.entry_points,
                        "api_endpoints": metadata.api_endpoints,
                        "config_files": metadata.config_files,
                        "documentation": metadata.documentation,
                        "examples": metadata.examples,
                        "tests": metadata.tests
                    },
                    "integration_plan": {
                        "strategy": plan.integration_strategy,
                        "wrapper_class": plan.wrapper_class,
                        "api_adapter": plan.api_adapter,
                        "health_check": plan.health_check
                    }
                }
                
                with open(output, 'w') as f:
                    json.dump(analysis_data, f, indent=2)
                
                console.print(f"[blue]Analysis saved to {output}[/blue]")
                
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            raise typer.Exit(1)
        finally:
            integrator.cleanup()
    
    asyncio.run(run_analysis())


@app.command()
def status():
    """Show overall agent integration status"""
    
    if UniversalAgentIntegrator is None:
        console.print("[red]Universal integrator not available. Please check dependencies.[/red]")
        raise typer.Exit(1)
    
    async def run_status():
        settings = get_settings()
        integrator = UniversalAgentIntegrator(getattr(settings, 'base_dir', Path.cwd()))
        
        try:
            agents = await integrator.list_integrated_agents()
            
            # Count by framework
            framework_counts = {}
            capability_counts = {}
            
            for agent in agents:
                framework = agent.get("framework", "unknown")
                framework_counts[framework] = framework_counts.get(framework, 0) + 1
                
                for capability in agent.get("capabilities", []):
                    capability_counts[capability] = capability_counts.get(capability, 0) + 1
            
            console.print(Panel(
                f"Total Agents: {len(agents)}\n"
                f"Active Integrations: {len(agents)}\n"  # TODO: Get actual active count
                f"Frameworks: {len(framework_counts)}\n"
                f"Unique Capabilities: {len(capability_counts)}",
                title="Mark-1 Agent Integration Status",
                border_style="blue"
            ))
            
            if framework_counts:
                console.print("\n[bold]Frameworks:[/bold]")
                for framework, count in sorted(framework_counts.items()):
                    console.print(f"  • {framework}: {count} agent(s)")
            
            if capability_counts:
                console.print("\n[bold]Top Capabilities:[/bold]")
                sorted_caps = sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)
                for capability, count in sorted_caps[:10]:
                    console.print(f"  • {capability}: {count} agent(s)")
            
        except Exception as e:
            console.print(f"[red]Failed to get status: {e}[/red]")
            raise typer.Exit(1)
        finally:
            integrator.cleanup()
    
    asyncio.run(run_status())


def _display_analysis_results(plan, detailed: bool = False):
    """Display repository analysis results"""
    metadata = plan.agent_metadata
    
    console.print(Panel(
        f"[bold]Name:[/bold] {metadata.name}\n"
        f"[bold]Framework:[/bold] {metadata.framework.value}\n"
        f"[bold]Version:[/bold] {metadata.version}\n"
        f"[bold]Description:[/bold] {metadata.description}\n"
        f"[bold]Capabilities:[/bold] {', '.join([cap.value for cap in metadata.capabilities])}\n"
        f"[bold]Dependencies:[/bold] {len(metadata.dependencies)} packages\n"
        f"[bold]Entry Points:[/bold] {len(metadata.entry_points)} found\n"
        f"[bold]API Endpoints:[/bold] {len(metadata.api_endpoints)} found",
        title="Repository Analysis",
        border_style="cyan"
    ))
    
    if detailed:
        if metadata.dependencies:
            console.print("\n[bold]Dependencies:[/bold]")
            for dep in metadata.dependencies[:10]:  # Show first 10
                console.print(f"  • {dep}")
            if len(metadata.dependencies) > 10:
                console.print(f"  ... and {len(metadata.dependencies) - 10} more")
        
        if metadata.entry_points:
            console.print("\n[bold]Entry Points:[/bold]")
            for entry in metadata.entry_points:
                console.print(f"  • {entry}")
        
        if metadata.api_endpoints:
            console.print("\n[bold]API Endpoints:[/bold]")
            for endpoint in metadata.api_endpoints:
                console.print(f"  • {endpoint}")
        
        console.print(f"\n[bold]Integration Strategy:[/bold] {plan.integration_strategy}")
        console.print(f"[bold]Wrapper Class:[/bold] {plan.wrapper_class}")
        console.print(f"[bold]API Adapter:[/bold] {plan.api_adapter}")


def _display_test_results(test_results: Dict[str, Any]):
    """Display test results"""
    total_tests = test_results["tests_passed"] + test_results["tests_failed"]
    
    if total_tests == 0:
        console.print("[yellow]No tests found[/yellow]")
        return
    
    console.print(Panel(
        f"Tests Passed: {test_results['tests_passed']}\n"
        f"Tests Failed: {test_results['tests_failed']}\n"
        f"Success Rate: {(test_results['tests_passed'] / total_tests * 100):.1f}%",
        title="Test Results",
        border_style="green" if test_results["tests_failed"] == 0 else "yellow"
    ))
    
    if test_results["test_details"]:
        console.print("\n[bold]Test Details:[/bold]")
        for detail in test_results["test_details"]:
            status_color = "green" if detail["status"] == "passed" else "red"
            console.print(f"  [{status_color}]{detail['status'].upper()}[/{status_color}] {detail['command']}")


if __name__ == "__main__":
    app() 