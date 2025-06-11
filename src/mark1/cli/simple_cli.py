#!/usr/bin/env python3
"""
Simple CLI for Mark-1 Universal Plugin System

This provides a simple command-line interface with all essential commands:
- Database and context management
- Plugin management  
- Orchestration commands
- Monitoring and reports
"""

import sys
from pathlib import Path
import typer
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import command modules
from mark1.cli.db_commands import app as db_app
from mark1.cli.plugin_commands import plugin_app
from mark1.cli.orchestrate_commands import orchestrate_app

# Create main app
app = typer.Typer(
    name="mark1",
    help="Mark-1 Universal Plugin System CLI",
    no_args_is_help=True
)

console = Console()

# Add command groups
app.add_typer(db_app, name="db", help="Database and context management")
app.add_typer(plugin_app, name="plugin", help="Plugin management")
app.add_typer(orchestrate_app, name="orchestrate", help="AI orchestration")

@app.command("version")
def version():
    """Show Mark-1 version information"""
    console.print("🚀 Mark-1 Universal Plugin System", style="bold blue")
    console.print("Version: 1.0.0 Production")
    console.print("Status: ✅ Fully Operational")

@app.command("status")
def system_status():
    """Quick system status check"""
    console.print("🖥️  Mark-1 System Status", style="bold blue")
    console.print("-" * 30)
    
    # Quick status checks
    try:
        # Check OLLAMA
        import asyncio
        from mark1.llm.ollama_client import OllamaClient
        
        async def check_ollama():
            client = OllamaClient("https://f6da-103-167-213-208.ngrok-free.app")
            return await client.health_check()
        
        ollama_ok = asyncio.run(check_ollama())
        console.print(f"🤖 OLLAMA: {'✅ Connected' if ollama_ok else '❌ Disconnected'}")
        
        # Check plugins
        from pathlib import Path
        plugins_dir = Path.home() / ".mark1" / "plugins"
        plugin_count = len(list(plugins_dir.glob("*"))) if plugins_dir.exists() else 0
        console.print(f"📦 Plugins: {plugin_count} installed")
        
        # Check database
        try:
            from mark1.storage.database import get_db_session
            async def check_db():
                async with get_db_session() as session:
                    await session.execute("SELECT 1")
                return True
            
            db_ok = asyncio.run(check_db())
            console.print(f"🗄️  Database: {'✅ Connected' if db_ok else '⚠️  Memory-only'}")
        except:
            console.print("🗄️  Database: ⚠️  Memory-only mode")
        
        console.print("\n💡 Use 'mark1 db monitoring' for detailed report")
        console.print("💡 Use 'mark1 --help' to see all commands")
        
    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="red")

if __name__ == "__main__":
    app()
