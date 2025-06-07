"""
Mark-1 API Package

Session 20: API Layer & REST Endpoints
Provides comprehensive REST API for Mark-1 orchestrator functionality
"""

from .rest_api import create_app, Mark1API
from .auth import AuthenticationManager, AuthorizationManager
from .middleware import SecurityMiddleware, RateLimitMiddleware

__all__ = [
    'create_app',
    'Mark1API',
    'AuthenticationManager', 
    'AuthorizationManager',
    'SecurityMiddleware',
    'RateLimitMiddleware'
]

__version__ = "1.0.0"
