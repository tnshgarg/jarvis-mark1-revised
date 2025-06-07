"""
REST API for Mark-1 Orchestrator

This module provides the FastAPI application and endpoints for the Mark-1 system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from mark1.config.settings import get_settings
from mark1.utils.exceptions import Mark1BaseException


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()
    logger = structlog.get_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title="Mark-1 Orchestrator API",
        description="Advanced AI Agent Orchestration System",
        version=settings.version,
        docs_url="/docs" if settings.api.enable_docs else None,
        redoc_url="/redoc" if settings.api.enable_docs else None
    )
    
    # Add CORS middleware
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Exception handlers
    @app.exception_handler(Mark1BaseException)
    async def mark1_exception_handler(request, exc: Mark1BaseException):
        logger.error("Mark-1 exception occurred", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": "mark1_error"}
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "type": "http_error"}
        )
    
    # Basic endpoints
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Mark-1 Orchestrator API",
            "version": settings.version,
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": settings.version
        }
    
    @app.get("/info")
    async def info():
        """System information endpoint"""
        return {
            "app_name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "debug": settings.debug
        }
    
    logger.info("FastAPI application created", app_name=settings.app_name)
    return app
