#!/usr/bin/env python3
"""
OLLAMA Client for Mark-1 Universal Plugin System

Provides real OLLAMA integration for AI-powered task planning,
plugin orchestration, and intelligent workflow management.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import httpx
import structlog

from ..config.settings import get_settings


logger = structlog.get_logger(__name__)


@dataclass
class OllamaModel:
    """OLLAMA model information"""
    name: str
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OllamaResponse:
    """OLLAMA API response"""
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class ChatMessage:
    """Chat message for OLLAMA"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """OLLAMA chat response"""
    model: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    """
    OLLAMA client for local LLM inference
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama.base_url
        self.logger = structlog.get_logger(__name__)
        self.client = httpx.AsyncClient(timeout=self.settings.ollama.timeout)
        
        # Model capabilities mapping
        self.model_capabilities = {
            "llama2": ["general", "reasoning", "code"],
            "llama2:13b": ["general", "reasoning", "code", "complex"],
            "llama2:70b": ["general", "reasoning", "code", "complex", "analysis"],
            "codellama": ["code", "programming", "debugging"],
            "codellama:13b": ["code", "programming", "debugging", "complex"],
            "mistral": ["general", "reasoning", "fast"],
            "mixtral": ["general", "reasoning", "complex", "multilingual"],
            "neural-chat": ["conversation", "general"],
            "starcode": ["code", "programming"],
            "phi": ["general", "reasoning", "small"],
            "orca-mini": ["general", "reasoning", "small"],
            "vicuna": ["general", "conversation"],
            "wizard-coder": ["code", "programming", "debugging"]
        }
    
    async def list_models(self) -> List[OllamaModel]:
        """List available OLLAMA models"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                model = OllamaModel(
                    name=model_data.get("name", ""),
                    size=model_data.get("size", 0),
                    digest=model_data.get("digest", ""),
                    modified_at=model_data.get("modified_at", ""),
                    details=model_data.get("details", {})
                )
                models.append(model)
            
            self.logger.info("Listed OLLAMA models", count=len(models))
            return models
            
        except Exception as e:
            self.logger.error("Failed to list OLLAMA models", error=str(e))
            return []
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """Generate text using OLLAMA"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "raw": raw
            }
            
            if system:
                payload["system"] = system
            if template:
                payload["template"] = template
            if context:
                payload["context"] = context
            if format:
                payload["format"] = format
            if options:
                payload["options"] = options
            
            self.logger.debug("Generating with OLLAMA", model=model, prompt_length=len(prompt))
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"ngrok-skip-browser-warning": "true"}  # For ngrok URLs
            )
            response.raise_for_status()
            
            data = response.json()
            
            ollama_response = OllamaResponse(
                model=data.get("model", model),
                response=data.get("response", ""),
                done=data.get("done", True),
                context=data.get("context"),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                prompt_eval_duration=data.get("prompt_eval_duration"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration")
            )
            
            self.logger.info("OLLAMA generation completed", 
                           model=model, 
                           response_length=len(ollama_response.response),
                           duration=ollama_response.total_duration)
            
            return ollama_response
            
        except Exception as e:
            self.logger.error("OLLAMA generation failed", model=model, error=str(e))
            raise
    
    async def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Chat with OLLAMA using conversation format"""
        try:
            payload = {
                "model": model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "stream": stream
            }
            
            if format:
                payload["format"] = format
            if options:
                payload["options"] = options
            
            self.logger.debug("Chatting with OLLAMA", model=model, messages_count=len(messages))
            
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"ngrok-skip-browser-warning": "true"}  # For ngrok URLs
            )
            response.raise_for_status()
            
            data = response.json()
            message_data = data.get("message", {})
            
            chat_response = ChatResponse(
                model=data.get("model", model),
                message=ChatMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", "")
                ),
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                prompt_eval_duration=data.get("prompt_eval_duration"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration")
            )
            
            self.logger.info("OLLAMA chat completed", 
                           model=model, 
                           response_length=len(chat_response.message.content))
            
            return chat_response
            
        except Exception as e:
            self.logger.error("OLLAMA chat failed", model=model, error=str(e))
            raise
    
    async def select_best_model(self, task_type: str, complexity: str = "medium") -> str:
        """Select the best OLLAMA model for a specific task"""
        try:
            available_models = await self.list_models()
            model_names = [model.name for model in available_models]
            
            # Task type to capability mapping
            task_capabilities = {
                "code": ["code", "programming"],
                "analysis": ["reasoning", "analysis"],
                "planning": ["reasoning", "complex"],
                "conversation": ["conversation", "general"],
                "general": ["general"],
                "debugging": ["code", "debugging"],
                "reasoning": ["reasoning", "complex"]
            }
            
            required_capabilities = task_capabilities.get(task_type, ["general"])
            
            # Score models based on capabilities and complexity
            best_model = None
            best_score = 0
            
            for model_name in model_names:
                # Clean model name (remove tags)
                base_name = model_name.split(":")[0]
                capabilities = self.model_capabilities.get(base_name, ["general"])
                
                # Calculate score
                score = 0
                for req_cap in required_capabilities:
                    if req_cap in capabilities:
                        score += 2
                
                # Complexity bonus
                if complexity == "high" and "complex" in capabilities:
                    score += 3
                elif complexity == "medium" and "reasoning" in capabilities:
                    score += 2
                elif complexity == "low" and "fast" in capabilities:
                    score += 2
                
                # Size preference (larger models for complex tasks)
                if "13b" in model_name and complexity in ["medium", "high"]:
                    score += 1
                elif "70b" in model_name and complexity == "high":
                    score += 2
                elif "7b" in model_name or ":" not in model_name:
                    score += 1  # Default size bonus
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            # Fallback to default model
            if not best_model:
                best_model = self.settings.ollama.default_model
            
            self.logger.info("Selected OLLAMA model", 
                           task_type=task_type, 
                           complexity=complexity,
                           selected_model=best_model,
                           score=best_score)
            
            return best_model
            
        except Exception as e:
            self.logger.error("Model selection failed", error=str(e))
            return self.settings.ollama.default_model
    
    async def health_check(self) -> bool:
        """Check if OLLAMA is healthy and accessible"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/tags",
                headers={"ngrok-skip-browser-warning": "true"}
            )
            response.raise_for_status()
            
            self.logger.info("OLLAMA health check passed")
            return True
            
        except Exception as e:
            self.logger.error("OLLAMA health check failed", error=str(e))
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
