#!/usr/bin/env python3
"""
Database and Context Management Commands for Mark-1

This module provides CLI commands for database operations, context management,
and monitoring reports.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import select
from mark1.core.context_manager import ContextManager, ContextType, ContextScope, ContextPriority
from mark1.plugins import PluginManager
from mark1.storage.database import get_db_session

app = typer.Typer(name="db", help="Database and context management commands")
console = Console()


@app.command("status")
def db_status():
    """Show database connection status and statistics"""
    console.print("🗄️  Database Status Check", style="bold blue")
    console.print("-" * 50)
    
    try:
        # Test database connection
        async def check_db():
            try:
                async with get_db_session() as session:
                    from sqlalchemy import text
                    result = await session.execute(text("SELECT 1"))
                    return True, "Connected"
            except Exception as e:
                return False, str(e)
        
        connected, message = asyncio.run(check_db())
        
        if connected:
            console.print("✅ Database: Connected", style="green")
            console.print(f"📍 Status: {message}")
            
            # Get database file info
            db_path = Path("./data/mark1.db")
            if db_path.exists():
                stat = db_path.stat()
                console.print(f"📁 Database file: {db_path}")
                console.print(f"📊 Size: {stat.st_size / 1024:.2f} KB")
                console.print(f"🕒 Modified: {datetime.fromtimestamp(stat.st_mtime)}")
            else:
                console.print("⚠️  Database file not found (in-memory mode)")
        else:
            console.print("❌ Database: Disconnected", style="red")
            console.print(f"🚨 Error: {message}")
            console.print("💡 System will use memory-only mode")
            
    except Exception as e:
        console.print(f"❌ Database check failed: {e}", style="red")


@app.command("init")
def db_init():
    """Initialize database and create tables"""
    console.print("🚀 Initializing Database", style="bold blue")
    
    try:
        async def init_db():
            from mark1.storage.database import init_database
            await init_database()
            return True
        
        success = asyncio.run(init_db())
        
        if success:
            console.print("✅ Database initialized successfully", style="green")
            console.print("📊 All tables created")
        else:
            console.print("❌ Database initialization failed", style="red")
            
    except Exception as e:
        console.print(f"❌ Database initialization error: {e}", style="red")


@app.command("contexts")
def list_contexts(
    limit: int = typer.Option(10, help="Number of contexts to show"),
    context_type: Optional[str] = typer.Option(None, help="Filter by context type"),
    agent_id: Optional[str] = typer.Option(None, help="Filter by agent ID")
):
    """List stored contexts"""
    console.print("🗃️  Context Management", style="bold blue")
    console.print("-" * 50)
    
    try:
        async def get_contexts():
            context_manager = ContextManager()
            await context_manager.initialize()
            
            # Get cache statistics
            cache_stats = context_manager.cache.stats
            
            console.print("📊 Cache Statistics:")
            console.print(f"  Total entries: {cache_stats.total_entries}")
            console.print(f"  Total size: {cache_stats.total_size_bytes / 1024:.2f} KB")
            console.print(f"  Hit rate: {cache_stats.hit_rate:.2%}")
            console.print(f"  Miss rate: {cache_stats.miss_rate:.2%}")
            console.print(f"  Evictions: {cache_stats.eviction_count}")
            
            # Show cached contexts
            if context_manager.cache._cache:
                console.print(f"\n🗂️  Cached Contexts ({len(context_manager.cache._cache)}):")
                
                table = Table()
                table.add_column("Context ID", style="cyan")
                table.add_column("Key", style="green")
                table.add_column("Type", style="yellow")
                table.add_column("Size", style="magenta")
                table.add_column("Compressed", style="blue")
                table.add_column("Access Count", style="red")
                
                for ctx_id, entry in list(context_manager.cache._cache.items())[:limit]:
                    table.add_row(
                        ctx_id[:8] + "...",
                        entry.key or "N/A",
                        entry.context_type.value if entry.context_type else "N/A",
                        f"{entry.size_bytes} bytes",
                        "Yes" if entry.is_compressed else "No",
                        str(entry.access_count)
                    )
                
                console.print(table)
            else:
                console.print("📭 No contexts in cache")
            
            await context_manager.cleanup()
        
        asyncio.run(get_contexts())
        
    except Exception as e:
        console.print(f"❌ Context listing failed: {e}", style="red")


@app.command("create-context")
def create_context(
    key: str = typer.Argument(..., help="Context key"),
    content: str = typer.Argument(..., help="Context content (JSON string)"),
    context_type: str = typer.Option("task", help="Context type (task/session/conversation/result)"),
    scope: str = typer.Option("task", help="Context scope (global/session/task/agent)"),
    priority: str = typer.Option("medium", help="Priority (low/medium/high/critical)")
):
    """Create a new context entry"""
    console.print("➕ Creating Context", style="bold blue")
    
    try:
        # Parse content as JSON
        try:
            content_data = json.loads(content)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            content_data = {"text": content}
        
        # Convert string enums to actual enums
        type_map = {
            "task": ContextType.TASK,
            "session": ContextType.SESSION,
            "conversation": ContextType.CONVERSATION,
            "result": ContextType.RESULT
        }
        
        scope_map = {
            "global": ContextScope.GLOBAL,
            "session": ContextScope.SESSION,
            "task": ContextScope.TASK,
            "agent": ContextScope.AGENT
        }
        
        priority_map = {
            "low": ContextPriority.LOW,
            "medium": ContextPriority.MEDIUM,
            "high": ContextPriority.HIGH,
            "critical": ContextPriority.CRITICAL
        }
        
        async def create_ctx():
            context_manager = ContextManager()
            await context_manager.initialize()
            
            result = await context_manager.create_context(
                key=key,
                content=content_data,
                context_type=type_map.get(context_type, ContextType.TASK),
                scope=scope_map.get(scope, ContextScope.TASK),
                priority=priority_map.get(priority, ContextPriority.MEDIUM)
            )
            
            await context_manager.cleanup()
            return result
        
        result = asyncio.run(create_ctx())
        
        if result.success:
            console.print("✅ Context created successfully", style="green")
            console.print(f"🆔 Context ID: {result.context_id}")
            console.print(f"🔑 Key: {key}")
            console.print(f"📊 Size: {len(json.dumps(content_data))} bytes")
            console.print(f"🗜️  Compressed: {result.compressed}")
        else:
            console.print(f"❌ Context creation failed: {result.message}", style="red")
            
    except Exception as e:
        console.print(f"❌ Context creation error: {e}", style="red")


@app.command("get-context")
def get_context(
    context_id: Optional[str] = typer.Option(None, help="Context ID"),
    key: Optional[str] = typer.Option(None, help="Context key")
):
    """Retrieve a context by ID or key"""
    if not context_id and not key:
        console.print("❌ Either context_id or key must be provided", style="red")
        return
    
    console.print("🔍 Retrieving Context", style="bold blue")
    
    try:
        async def get_ctx():
            context_manager = ContextManager()
            await context_manager.initialize()
            
            if context_id:
                result = await context_manager.get_context(context_id=context_id)
            else:
                result = await context_manager.get_context(key=key)
            
            await context_manager.cleanup()
            return result
        
        result = asyncio.run(get_ctx())
        
        if result.success:
            console.print("✅ Context retrieved successfully", style="green")
            console.print(f"🆔 Context ID: {result.context_id}")
            console.print(f"💾 Cache hit: {result.cache_hit}")
            console.print(f"🗜️  Compressed: {result.compressed}")
            
            if result.data:
                console.print("\n📄 Content:")
                console.print(Panel(json.dumps(result.data, indent=2), title="Context Data"))
            else:
                console.print("📭 No content available")
        else:
            console.print(f"❌ Context retrieval failed: {result.message}", style="red")
            
    except Exception as e:
        console.print(f"❌ Context retrieval error: {e}", style="red")


@app.command("query")
def query_database(
    table: str = typer.Argument(..., help="Table name (plugins, plugin_executions, plugin_capabilities)"),
    limit: int = typer.Option(10, help="Number of records to show"),
    where: Optional[str] = typer.Option(None, help="WHERE clause (e.g., 'status=active')")
):
    """Query database tables directly"""
    console.print(f"🔍 Querying Database Table: {table}", style="bold blue")
    console.print("-" * 50)

    try:
        async def query_db():
            async with get_db_session() as session:
                # Build query based on table
                if table == "plugins":
                    from mark1.storage.models.plugin_model import Plugin
                    query = select(Plugin)
                elif table == "plugin_executions":
                    from mark1.storage.models.plugin_model import PluginExecution
                    query = select(PluginExecution)
                elif table == "plugin_capabilities":
                    from mark1.storage.models.plugin_model import PluginCapability
                    query = select(PluginCapability)
                else:
                    console.print(f"❌ Unknown table: {table}", style="red")
                    return

                # Add WHERE clause if provided
                if where:
                    # Simple WHERE parsing (extend as needed)
                    if "=" in where:
                        field, value = where.split("=", 1)
                        field = field.strip()
                        value = value.strip().strip("'\"")

                        if table == "plugins":
                            if field == "status":
                                query = query.filter(Plugin.status == value)
                            elif field == "name":
                                query = query.filter(Plugin.name.like(f"%{value}%"))
                        # Add more WHERE conditions as needed

                query = query.limit(limit)
                result = await session.execute(query)
                records = result.scalars().all()

                if records:
                    console.print(f"📊 Found {len(records)} records:")

                    # Create table for display
                    table_display = Table()

                    if table == "plugins":
                        table_display.add_column("ID", style="cyan")
                        table_display.add_column("Name", style="green")
                        table_display.add_column("Type", style="yellow")
                        table_display.add_column("Status", style="magenta")
                        table_display.add_column("Created", style="blue")

                        for record in records:
                            table_display.add_row(
                                record.plugin_id[:8] + "...",
                                record.name,
                                record.plugin_type,
                                record.status,
                                record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "N/A"
                            )

                    elif table == "plugin_executions":
                        table_display.add_column("ID", style="cyan")
                        table_display.add_column("Plugin", style="green")
                        table_display.add_column("Capability", style="yellow")
                        table_display.add_column("Status", style="magenta")
                        table_display.add_column("Duration", style="blue")
                        table_display.add_column("Started", style="red")

                        for record in records:
                            table_display.add_row(
                                record.execution_id[:8] + "...",
                                record.plugin_id[:8] + "...",
                                record.capability_name,
                                record.status,
                                f"{record.execution_time:.3f}s" if record.execution_time else "N/A",
                                record.started_at.strftime("%Y-%m-%d %H:%M") if record.started_at else "N/A"
                            )

                    elif table == "plugin_capabilities":
                        table_display.add_column("Plugin", style="cyan")
                        table_display.add_column("Name", style="green")
                        table_display.add_column("Description", style="yellow")
                        table_display.add_column("Input Types", style="magenta")
                        table_display.add_column("Output Types", style="blue")

                        for record in records:
                            table_display.add_row(
                                record.plugin_id[:8] + "...",
                                record.name,
                                record.description[:50] + "..." if len(record.description) > 50 else record.description,
                                ", ".join(record.input_types) if record.input_types else "N/A",
                                ", ".join(record.output_types) if record.output_types else "N/A"
                            )

                    console.print(table_display)
                else:
                    console.print("📭 No records found")

        asyncio.run(query_db())

    except Exception as e:
        console.print(f"❌ Database query failed: {e}", style="red")


@app.command("exec-sql")
def execute_sql(
    sql: str = typer.Argument(..., help="SQL query to execute"),
    params: Optional[str] = typer.Option(None, help="JSON parameters for the query")
):
    """Execute raw SQL query (read-only)"""
    console.print("⚡ Executing SQL Query", style="bold blue")
    console.print("-" * 50)
    console.print(f"📝 Query: {sql}")

    # Safety check - only allow SELECT queries
    if not sql.strip().upper().startswith("SELECT"):
        console.print("❌ Only SELECT queries are allowed for safety", style="red")
        return

    try:
        async def exec_sql():
            async with get_db_session() as session:
                # Parse parameters if provided
                query_params = {}
                if params:
                    try:
                        query_params = json.loads(params)
                    except json.JSONDecodeError:
                        console.print("❌ Invalid JSON parameters", style="red")
                        return

                # Execute query
                result = await session.execute(sql, query_params)
                rows = result.fetchall()

                if rows:
                    console.print(f"📊 Query returned {len(rows)} rows:")

                    # Get column names
                    columns = list(result.keys()) if hasattr(result, 'keys') else [f"col_{i}" for i in range(len(rows[0]))]

                    # Create table
                    table_display = Table()
                    for col in columns:
                        table_display.add_column(str(col), style="cyan")

                    # Add rows
                    for row in rows[:50]:  # Limit to 50 rows for display
                        table_display.add_row(*[str(val) for val in row])

                    console.print(table_display)

                    if len(rows) > 50:
                        console.print(f"... and {len(rows) - 50} more rows")
                else:
                    console.print("📭 Query returned no results")

        asyncio.run(exec_sql())

    except Exception as e:
        console.print(f"❌ SQL execution failed: {e}", style="red")


@app.command("context-search")
def search_contexts(
    query: str = typer.Argument(..., help="Search query for context content"),
    limit: int = typer.Option(10, help="Number of contexts to show"),
    context_type: Optional[str] = typer.Option(None, help="Filter by context type")
):
    """Search contexts by content"""
    console.print(f"🔍 Searching Contexts: '{query}'", style="bold blue")
    console.print("-" * 50)

    try:
        async def search_ctx():
            context_manager = ContextManager()
            await context_manager.initialize()

            # Get all cached contexts
            matching_contexts = []

            for ctx_id, entry in context_manager.cache._cache.items():
                # Check if query matches key or content
                if query.lower() in (entry.key or "").lower():
                    matching_contexts.append((ctx_id, entry, "key"))
                elif entry.data and query.lower() in str(entry.data).lower():
                    matching_contexts.append((ctx_id, entry, "content"))

                if len(matching_contexts) >= limit:
                    break

            if matching_contexts:
                console.print(f"📊 Found {len(matching_contexts)} matching contexts:")

                table = Table()
                table.add_column("Context ID", style="cyan")
                table.add_column("Key", style="green")
                table.add_column("Type", style="yellow")
                table.add_column("Match", style="magenta")
                table.add_column("Size", style="blue")
                table.add_column("Content Preview", style="white")

                for ctx_id, entry, match_type in matching_contexts:
                    content_preview = str(entry.data)[:100] + "..." if entry.data else "N/A"
                    table.add_row(
                        ctx_id[:8] + "...",
                        entry.key or "N/A",
                        entry.context_type.value if entry.context_type else "N/A",
                        match_type,
                        f"{entry.size_bytes} bytes",
                        content_preview
                    )

                console.print(table)
            else:
                console.print("📭 No matching contexts found")

            await context_manager.cleanup()

        asyncio.run(search_ctx())

    except Exception as e:
        console.print(f"❌ Context search failed: {e}", style="red")


@app.command("context-export")
def export_contexts(
    output_file: str = typer.Argument(..., help="Output file path (JSON)"),
    context_type: Optional[str] = typer.Option(None, help="Filter by context type")
):
    """Export contexts to JSON file"""
    console.print(f"📤 Exporting Contexts to: {output_file}", style="bold blue")
    console.print("-" * 50)

    try:
        async def export_ctx():
            context_manager = ContextManager()
            await context_manager.initialize()

            # Collect contexts
            contexts_data = []

            for ctx_id, entry in context_manager.cache._cache.items():
                if context_type and entry.context_type and entry.context_type.value != context_type:
                    continue

                context_data = {
                    "context_id": ctx_id,
                    "key": entry.key,
                    "context_type": entry.context_type.value if entry.context_type else None,
                    "scope": entry.scope.value if entry.scope else None,
                    "priority": entry.priority.value if entry.priority else None,
                    "size_bytes": entry.size_bytes,
                    "is_compressed": entry.is_compressed,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                    "data": entry.data
                }
                contexts_data.append(context_data)

            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(contexts_data, f, indent=2, default=str)

            console.print(f"✅ Exported {len(contexts_data)} contexts to {output_file}")
            console.print(f"📊 File size: {output_path.stat().st_size / 1024:.2f} KB")

            await context_manager.cleanup()

        asyncio.run(export_ctx())

    except Exception as e:
        console.print(f"❌ Context export failed: {e}", style="red")


@app.command("monitoring")
def monitoring_report():
    """Generate comprehensive monitoring report"""
    console.print("📊 Mark-1 System Monitoring Report", style="bold blue")
    console.print("=" * 60)
    
    try:
        async def generate_report():
            # System status
            console.print("\n🖥️  System Status", style="bold green")
            console.print("-" * 30)
            
            # Database status
            try:
                async with get_db_session() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))
                console.print("✅ Database: Connected")
            except Exception:
                console.print("⚠️  Database: Memory-only mode")
            
            # Context manager status
            context_manager = ContextManager()
            await context_manager.initialize()
            
            cache_stats = context_manager.cache.stats
            console.print(f"✅ Context Manager: Active ({cache_stats.total_entries} contexts)")
            
            # Plugin manager status
            plugins_dir = Path.home() / ".mark1" / "plugins"
            plugin_manager = PluginManager(plugins_directory=plugins_dir)
            
            try:
                plugins = await plugin_manager.list_installed_plugins()
                console.print(f"✅ Plugin Manager: {len(plugins)} plugins installed")
            except Exception:
                console.print("⚠️  Plugin Manager: Error loading plugins")
            
            # Performance metrics
            console.print("\n⚡ Performance Metrics", style="bold green")
            console.print("-" * 30)
            
            console.print(f"📊 Cache hit rate: {cache_stats.hit_rate:.2%}")
            console.print(f"📊 Cache miss rate: {cache_stats.miss_rate:.2%}")
            console.print(f"📊 Total cache size: {cache_stats.total_size_bytes / 1024:.2f} KB")
            console.print(f"📊 Cache entries: {cache_stats.total_entries}")
            console.print(f"📊 Cache evictions: {cache_stats.eviction_count}")
            
            # Plugin status
            if plugins:
                console.print("\n🔌 Plugin Status", style="bold green")
                console.print("-" * 30)
                
                plugin_table = Table()
                plugin_table.add_column("Plugin Name", style="cyan")
                plugin_table.add_column("Type", style="yellow")
                plugin_table.add_column("Status", style="green")
                plugin_table.add_column("Capabilities", style="magenta")
                
                for plugin in plugins:
                    status_color = "green" if plugin.status.value == "active" else "yellow"
                    plugin_table.add_row(
                        plugin.name,
                        plugin.plugin_type.value,
                        f"[{status_color}]{plugin.status.value}[/{status_color}]",
                        str(len(plugin.capabilities))
                    )
                
                console.print(plugin_table)
            
            # Resource usage
            console.print("\n💾 Resource Usage", style="bold green")
            console.print("-" * 30)

            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()

                console.print(f"🧠 Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                console.print(f"💽 Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB")
                console.print(f"🔄 CPU percent: {process.cpu_percent():.1f}%")
            except ImportError:
                console.print("⚠️  Resource monitoring unavailable (psutil not installed)")
                console.print("💡 System is running in lightweight mode")
            
            # Cleanup
            await context_manager.cleanup()
            await plugin_manager.cleanup()
        
        asyncio.run(generate_report())
        
    except Exception as e:
        console.print(f"❌ Monitoring report failed: {e}", style="red")


if __name__ == "__main__":
    app()
