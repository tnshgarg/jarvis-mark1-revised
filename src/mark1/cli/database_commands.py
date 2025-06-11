#!/usr/bin/env python3
"""
Database and Context CLI Commands for Mark-1 Universal Plugin System

Provides CLI commands for accessing database records, context storage,
and system state information.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.json import JSON
import structlog

from ..core.context_manager import ContextManager, ContextType, ContextScope, ContextPriority
from ..storage.database import get_db_session
from ..storage.repositories.plugin_repository import PluginRepository
from ..storage.repositories.context_repository import ContextRepository


logger = structlog.get_logger(__name__)
console = Console()

# Create database CLI app
database_app = typer.Typer(
    name="database",
    help="Database and context management commands",
    rich_markup_mode="rich"
)


@database_app.command("status")
def database_status():
    """
    Show database connection status and basic statistics
    """
    async def _status():
        try:
            console.print("🗄️  Database Status", style="bold blue")
            console.print("=" * 50)
            
            # Test database connection
            try:
                async with get_db_session() as session:
                    await session.execute("SELECT 1")
                    console.print("✅ Database connection: [green]Active[/green]")
                    
                    # Get plugin statistics
                    plugin_repo = PluginRepository(session)
                    plugins = await plugin_repo.get_all_plugins()
                    console.print(f"📦 Total plugins: [cyan]{len(plugins)}[/cyan]")
                    
                    # Get context statistics
                    context_repo = ContextRepository(session)
                    contexts = await context_repo.get_all_contexts(session, limit=1000)
                    console.print(f"🗃️  Total contexts: [cyan]{len(contexts)}[/cyan]")
                    
                    # Show recent activity
                    recent_plugins = await plugin_repo.get_recent_plugins(limit=5)
                    if recent_plugins:
                        console.print("\n📋 Recent Plugins:")
                        for plugin in recent_plugins:
                            console.print(f"  • {plugin.name} ({plugin.status.value})")
                    
            except Exception as e:
                console.print(f"❌ Database connection: [red]Failed[/red]")
                console.print(f"   Error: {e}")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_status())


@database_app.command("plugins")
def list_plugins(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of plugins to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """
    List all plugins in the database
    """
    async def _list_plugins():
        try:
            async with get_db_session() as session:
                plugin_repo = PluginRepository(session)
                plugins = await plugin_repo.get_all_plugins()
                
                if status:
                    plugins = [p for p in plugins if p.status.value == status]
                
                plugins = plugins[:limit]
                
                if not plugins:
                    console.print("No plugins found.")
                    return
                
                # Create table
                table = Table(title=f"Plugins Database ({len(plugins)} entries)")
                table.add_column("Name", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Version", style="blue")
                table.add_column("Created", style="dim")
                
                if verbose:
                    table.add_column("Capabilities", style="magenta")
                    table.add_column("Repository", style="dim")
                
                for plugin in plugins:
                    capabilities = ", ".join([cap.name for cap in plugin.capabilities[:3]])
                    if len(plugin.capabilities) > 3:
                        capabilities += f" (+{len(plugin.capabilities) - 3})"
                    
                    row = [
                        plugin.name,
                        plugin.plugin_type.value,
                        plugin.status.value,
                        plugin.version,
                        plugin.created_at.strftime("%Y-%m-%d %H:%M") if plugin.created_at else "Unknown"
                    ]
                    
                    if verbose:
                        row.extend([
                            capabilities or "None",
                            plugin.repository_url or "Unknown"
                        ])
                    
                    table.add_row(*row)
                
                console.print(table)
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_list_plugins())


@database_app.command("plugin")
def show_plugin(
    plugin_id: str = typer.Argument(..., help="Plugin ID to show details for")
):
    """
    Show detailed information about a specific plugin
    """
    async def _show_plugin():
        try:
            async with get_db_session() as session:
                plugin_repo = PluginRepository(session)
                plugin = await plugin_repo.get_plugin_by_id(plugin_id)
                
                if not plugin:
                    console.print(f"Plugin not found: {plugin_id}")
                    return
                
                # Create detailed view
                console.print(Panel(
                    f"[bold cyan]{plugin.name}[/bold cyan]\n"
                    f"ID: {plugin.plugin_id}\n"
                    f"Type: {plugin.plugin_type.value}\n"
                    f"Status: {plugin.status.value}\n"
                    f"Version: {plugin.version}\n"
                    f"Author: {plugin.author or 'Unknown'}\n"
                    f"Repository: {plugin.repository_url or 'Unknown'}\n"
                    f"Created: {plugin.created_at.strftime('%Y-%m-%d %H:%M:%S') if plugin.created_at else 'Unknown'}",
                    title="Plugin Details",
                    border_style="blue"
                ))
                
                # Show capabilities
                if plugin.capabilities:
                    console.print("\n🎯 Capabilities:")
                    for cap in plugin.capabilities:
                        console.print(f"  • [cyan]{cap.name}[/cyan]: {cap.description}")
                
                # Show dependencies
                if plugin.dependencies:
                    console.print("\n📦 Dependencies:")
                    for dep in plugin.dependencies:
                        console.print(f"  • {dep.name} {dep.version or ''}")
                
                # Show configuration
                if plugin.configuration:
                    console.print("\n⚙️  Configuration:")
                    config_json = JSON(json.dumps(plugin.configuration, indent=2))
                    console.print(config_json)
                
                # Show recent executions
                executions = await plugin_repo.get_plugin_executions(plugin_id, limit=5)
                if executions:
                    console.print("\n📊 Recent Executions:")
                    exec_table = Table()
                    exec_table.add_column("Capability", style="cyan")
                    exec_table.add_column("Status", style="green")
                    exec_table.add_column("Duration", style="yellow")
                    exec_table.add_column("Timestamp", style="dim")
                    
                    for exec_record in executions:
                        status_color = "green" if exec_record.status == "success" else "red"
                        exec_table.add_row(
                            exec_record.capability_name,
                            f"[{status_color}]{exec_record.status}[/{status_color}]",
                            f"{exec_record.execution_time:.2f}s" if exec_record.execution_time else "Unknown",
                            exec_record.created_at.strftime("%Y-%m-%d %H:%M:%S") if exec_record.created_at else "Unknown"
                        )
                    
                    console.print(exec_table)
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_show_plugin())


@database_app.command("contexts")
def list_contexts(
    context_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by context type"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Filter by scope"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of contexts to show"),
    active_only: bool = typer.Option(True, "--active-only", "-a", help="Show only active contexts")
):
    """
    List contexts stored in the database
    """
    async def _list_contexts():
        try:
            async with get_db_session() as session:
                context_repo = ContextRepository(session)
                contexts = await context_repo.get_all_contexts(
                    session, 
                    context_type=context_type,
                    context_scope=scope,
                    active_only=active_only,
                    limit=limit
                )
                
                if not contexts:
                    console.print("No contexts found.")
                    return
                
                # Create table
                table = Table(title=f"Contexts Database ({len(contexts)} entries)")
                table.add_column("Key", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Scope", style="yellow")
                table.add_column("Priority", style="blue")
                table.add_column("Size", style="magenta")
                table.add_column("Created", style="dim")
                
                for context in contexts:
                    size_str = f"{context.size_bytes or 0} bytes" if context.size_bytes else "Unknown"
                    
                    table.add_row(
                        context.context_key[:30] + "..." if len(context.context_key) > 30 else context.context_key,
                        context.context_type,
                        context.context_scope,
                        context.priority,
                        size_str,
                        context.created_at.strftime("%Y-%m-%d %H:%M") if context.created_at else "Unknown"
                    )
                
                console.print(table)
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_list_contexts())


@database_app.command("context")
def show_context(
    context_key: str = typer.Argument(..., help="Context key to show"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show context content")
):
    """
    Show detailed information about a specific context
    """
    async def _show_context():
        try:
            # Initialize context manager
            context_manager = ContextManager()
            await context_manager.initialize()
            
            # Get context
            result = await context_manager.get_context(key=context_key)
            
            if not result.success:
                console.print(f"Context not found: {context_key}")
                console.print(f"Error: {result.message}")
                return
            
            # Show context details
            console.print(Panel(
                f"[bold cyan]{context_key}[/bold cyan]\n"
                f"ID: {result.context_id}\n"
                f"Success: {result.success}\n"
                f"Cache Hit: {result.cache_hit}\n"
                f"Operation Time: {result.operation_time:.4f}s",
                title="Context Details",
                border_style="blue"
            ))
            
            # Show metadata
            if result.metadata:
                console.print("\n📊 Metadata:")
                metadata_json = JSON(json.dumps(result.metadata, indent=2, default=str))
                console.print(metadata_json)
            
            # Show content if requested
            if show_content and result.content:
                console.print("\n📄 Content:")
                content_json = JSON(json.dumps(result.content, indent=2, default=str))
                console.print(content_json)
            
            await context_manager.cleanup()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_show_context())


@database_app.command("cleanup")
def cleanup_database(
    dry_run: bool = typer.Option(True, "--dry-run", "-d", help="Show what would be cleaned without doing it"),
    expired_contexts: bool = typer.Option(True, "--expired-contexts", help="Clean expired contexts"),
    old_executions: bool = typer.Option(False, "--old-executions", help="Clean old execution records"),
    days: int = typer.Option(30, "--days", help="Clean records older than this many days")
):
    """
    Clean up old database records
    """
    async def _cleanup():
        try:
            console.print("🧹 Database Cleanup", style="bold blue")
            console.print("=" * 50)
            
            if dry_run:
                console.print("🔍 DRY RUN - No changes will be made")
            
            async with get_db_session() as session:
                if expired_contexts:
                    context_repo = ContextRepository(session)
                    expired = await context_repo.get_expired_contexts(session)
                    
                    console.print(f"📋 Found {len(expired)} expired contexts")
                    
                    if not dry_run and expired:
                        for context in expired:
                            await context_repo.delete_context(session, str(context.id))
                        await session.commit()
                        console.print(f"✅ Cleaned {len(expired)} expired contexts")
                
                if old_executions:
                    plugin_repo = PluginRepository(session)
                    # This would need to be implemented in the repository
                    console.print("🔄 Old execution cleanup not yet implemented")
            
            console.print("✅ Cleanup completed")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_cleanup())
