#!/usr/bin/env python3
"""
Mark-1 Launcher Script

Simple launcher for Mark-1 AI Agent Integration & Orchestration system.
Usage: python mark1_launcher.py [command] [options]
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Main launcher function"""
    
    # Path to the standalone CLI
    cli_path = Path(__file__).parent / "src" / "mark1" / "cli" / "standalone_agent_cli.py"
    
    if not cli_path.exists():
        print("❌ Error: Mark-1 CLI not found!")
        print(f"Expected: {cli_path}")
        sys.exit(1)
    
    # Forward all arguments to the CLI
    cmd = [sys.executable, str(cli_path)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main() 