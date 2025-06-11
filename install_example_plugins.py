#!/usr/bin/env python3
"""
Install Example Plugins into Mark-1 System

This script installs the example plugins we created into the Mark-1 plugin system.
"""

import asyncio
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from mark1.plugins import PluginManager


async def install_example_plugins():
    """Install all example plugins"""
    print("📦 Installing Example Plugins into Mark-1")
    print("=" * 50)
    
    # Initialize plugin manager
    plugins_dir = Path.home() / ".mark1" / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_manager = PluginManager(plugins_directory=plugins_dir)
    
    # Example plugins directory
    example_plugins_dir = Path.home() / ".mark1" / "example_plugins"
    
    if not example_plugins_dir.exists():
        print("❌ Example plugins directory not found. Run create_example_plugins.py first.")
        return
    
    # List of plugins to install
    plugin_dirs = [
        example_plugins_dir / "text_analyzer_plugin",
        example_plugins_dir / "file_processor_plugin", 
        example_plugins_dir / "data_converter_plugin"
    ]
    
    installed_plugins = []
    
    for plugin_dir in plugin_dirs:
        if not plugin_dir.exists():
            print(f"⚠️  Plugin directory not found: {plugin_dir}")
            continue
        
        print(f"\n🔧 Installing {plugin_dir.name}...")
        
        try:
            # Simulate repository installation by analyzing the local directory
            result = await plugin_manager.install_plugin_from_local_directory(
                plugin_directory=plugin_dir,
                plugin_name=plugin_dir.name.replace("_plugin", "").replace("_", " ").title()
            )
            
            if result.success:
                print(f"✅ {plugin_dir.name} installed successfully!")
                print(f"   Plugin ID: {result.plugin_id}")
                print(f"   Name: {result.plugin_metadata.name}")
                print(f"   Type: {result.plugin_metadata.plugin_type.value}")
                print(f"   Capabilities: {len(result.plugin_metadata.capabilities)}")
                installed_plugins.append(result)
            else:
                print(f"❌ Failed to install {plugin_dir.name}: {result.error}")
                
        except Exception as e:
            print(f"❌ Error installing {plugin_dir.name}: {e}")
    
    # Show summary
    print(f"\n📊 Installation Summary")
    print("=" * 50)
    print(f"✅ Successfully installed: {len(installed_plugins)} plugins")
    
    if installed_plugins:
        print("\n📋 Installed Plugins:")
        for result in installed_plugins:
            print(f"  • {result.plugin_metadata.name} ({result.plugin_id[:8]}...)")
            for cap in result.plugin_metadata.capabilities[:3]:
                print(f"    - {cap.name}: {cap.description}")
            if len(result.plugin_metadata.capabilities) > 3:
                print(f"    - ... and {len(result.plugin_metadata.capabilities) - 3} more")
    
    # Test plugin listing
    print(f"\n🔍 Verifying Installation...")
    all_plugins = await plugin_manager.list_installed_plugins()
    print(f"📦 Total plugins in system: {len(all_plugins)}")
    
    for plugin in all_plugins:
        print(f"  • {plugin.name} ({plugin.plugin_type.value}) - {plugin.status.value}")
    
    await plugin_manager.cleanup()
    
    print(f"\n🎉 Plugin installation completed!")
    print("You can now use these plugins in orchestration workflows.")


if __name__ == "__main__":
    asyncio.run(install_example_plugins())
