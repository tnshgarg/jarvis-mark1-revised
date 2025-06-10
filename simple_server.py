#!/usr/bin/env python3
"""
Simple FastAPI Server for Mark-1 AI Orchestrator (Docker Deployment)

This simplified server provides a minimal API for interacting with
the Mark-1 AI Orchestrator. It's designed for use in Docker deployments.
"""

import os
import json
import sys
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the project root to sys.path to make imports work
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the adapter for our weather agent
from test_agents.simple_ai_agent.mark1_adapter import Mark1WeatherAgentAdapter

# Create FastAPI app
app = FastAPI(
    title="Mark-1 AI Orchestrator API",
    description="Simplified API for Mark-1 AI Orchestrator",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our weather agent adapter
weather_agent_adapter = None

class QueryModel(BaseModel):
    """Model for query requests."""
    query: str
    context: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global weather_agent_adapter
    
    try:
        # Initialize the weather agent adapter
        weather_agent_adapter = Mark1WeatherAgentAdapter()
        init_success = await weather_agent_adapter.initialize()
        
        if not init_success:
            print("❌ Failed to initialize Weather Agent Adapter")
        else:
            print("✅ Weather Agent Adapter initialized successfully")
            
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint providing basic information."""
    return {
        "name": "Mark-1 AI Orchestrator",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "online",
            "database": "online",
            "weather_agent": "online" if weather_agent_adapter else "offline"
        }
    }

@app.get("/agents")
async def list_agents():
    """List available agents."""
    agents = []
    
    # Add our weather agent if it's initialized
    if weather_agent_adapter:
        agent_info = weather_agent_adapter.get_info()
        agents.append({
            "id": "weather-agent-1",
            "name": agent_info["name"],
            "status": "ready",
            "capabilities": agent_info["capabilities"],
            "description": agent_info["description"]
        })
    
    return {
        "agents": agents,
        "total": len(agents),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/weather")
async def process_weather_query(query_data: QueryModel):
    """Process a weather query using our weather agent."""
    global weather_agent_adapter
    
    if not weather_agent_adapter:
        raise HTTPException(status_code=503, detail="Weather agent is not available")
    
    try:
        # Execute the query
        result = await weather_agent_adapter.execute({
            "query": query_data.query
        }, context=query_data.context)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Mark-1 API server on port {port}")
    uvicorn.run("simple_server:app", host="0.0.0.0", port=port, reload=False) 