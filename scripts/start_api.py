#!/usr/bin/env python3
"""
Mark-1 API Local Development Server

Quick start script for testing the REST API locally
"""

import uvicorn
from src.mark1.api.rest_api import create_app

def main():
    """Start the API server with development settings"""
    print("🚀 Starting Mark-1 API Development Server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("📖 ReDoc Documentation: http://127.0.0.1:8000/redoc")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Create the FastAPI app
        app = create_app(
            enable_auth=False,  # Disable auth for easy testing
            cors_origins=["*"]  # Allow all origins for development
        )
        
        print("✅ FastAPI application created successfully")
        print("🎯 Starting server...")
        
        # Start the server (without reload to avoid import string issues)
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload to prevent import issues
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("🔧 Try running with: uvicorn src.mark1.api.rest_api:create_app --factory --host 127.0.0.1 --port 8000")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 