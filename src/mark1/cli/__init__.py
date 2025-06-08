#!/usr/bin/env python3
"""
Mark-1 CLI Interface Module

This module provides command-line interface functionality for the Mark-1 AI Orchestrator.
"""

from .main import main_cli, create_cli_app
from .commands import *
from .utils import *

__version__ = "1.0.0"
__all__ = ["main_cli", "create_cli_app"] 