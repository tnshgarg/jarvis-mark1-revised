"""
Command Line Interface for Mark-1 Orchestrator

This module provides additional CLI commands for the Mark-1 system.
"""

import typer
from rich.console import Console

# Create console for output
console = Console()

# Create CLI app
cli = typer.Typer(
    name="mark1-cli",
    help="Mark-1 Orchestrator CLI utilities",
    rich_markup_mode="rich"
)


@cli.command()
def version():
    """Show Mark-1 version information"""
    console.print("[blue]Mark-1 Orchestrator v0.1.0[/blue]")


@cli.command()
def test():
    """Test CLI functionality"""
    console.print("[green]CLI is working![/green]")


def main():
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
