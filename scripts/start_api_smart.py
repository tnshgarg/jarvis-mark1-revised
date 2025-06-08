#!/usr/bin/env python3
"""
Smart Mark-1 API Development Server

Automatically finds available port and starts the API server
"""

import socket
import uvicorn
from src.mark1.api.rest_api import create_app

def find_free_port(start_port=8000, max_port=8010):
    """Find the first available port starting from start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Start the API server with development settings"""
    print("🚀 Starting Mark-1 API Development Server...")
    
    # Find available port
    port = find_free_port()
    if not port:
        print("❌ No available ports found between 8000-8010")
        return 1
    
    print(f"📍 Server will be available at: http://127.0.0.1:{port}")
    print(f"📚 API Documentation: http://127.0.0.1:{port}/docs")
    print(f"📖 ReDoc Documentation: http://127.0.0.1:{port}/redoc")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Create the FastAPI app
        app = create_app(
            enable_auth=False,  # Disable auth for easy testing
            cors_origins=["*"]  # Allow all origins for development
        )
        
        print("✅ FastAPI application created successfully")
        print(f"🎯 Starting server on port {port}...")
        
        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print(f"🔧 Try running manually: uvicorn src.mark1.api.rest_api:create_app --factory --host 127.0.0.1 --port {port}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 