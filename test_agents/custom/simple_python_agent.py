#!/usr/bin/env python3
"""
Simple Python Custom Agent

A basic example of a custom agent that can be integrated with Mark-1.
This agent demonstrates text analysis and generation capabilities.
"""

__version__ = "1.0.0"
__author__ = "Mark-1 Team"
__license__ = "MIT"

import asyncio
import json
import re
from typing import Dict, Any, List


class SimpleAnalysisAgent:
    """
    A simple agent that performs text analysis and basic generation tasks.
    
    This agent demonstrates:
    - Text analysis capabilities
    - Simple generation features
    - Async operation support
    - Tool-like functions
    """
    
    def __init__(self, model_name: str = "simple-analyzer", debug: bool = False):
        self.model_name = model_name
        self.debug = debug
        self.processed_count = 0
        self.capabilities = [
            "text_analysis",
            "keyword_extraction", 
            "sentiment_analysis",
            "content_generation",
            "summarization"
        ]
        
        if self.debug:
            print(f"✅ SimpleAnalysisAgent initialized with model: {model_name}")
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method for the agent"""
        self.processed_count += 1
        
        task_type = input_data.get("task", "analyze")
        content = input_data.get("input", input_data.get("text", ""))
        
        if self.debug:
            print(f"🔍 Processing task: {task_type} with content length: {len(content)}")
        
        if task_type == "analyze":
            return await self.analyze_text(content)
        elif task_type == "generate":
            return await self.generate_content(input_data)
        elif task_type == "summarize":
            return await self.summarize_text(content)
        elif task_type == "extract_keywords":
            return await self.extract_keywords(content)
        else:
            return await self.analyze_text(content)
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and return insights"""
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Basic sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        positive_score = sum(1 for word in positive_words if word in text.lower())
        negative_score = sum(1 for word in negative_words if word in text.lower())
        
        sentiment = "neutral"
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        
        return {
            "task": "text_analysis",
            "analysis": {
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": sentence_count,
                "sentiment": sentiment,
                "sentiment_scores": {
                    "positive": positive_score,
                    "negative": negative_score
                }
            },
            "processed_by": self.model_name,
            "processing_time": 0.1
        }
    
    async def generate_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on input parameters"""
        await asyncio.sleep(0.15)
        
        topic = input_data.get("topic", "general")
        style = input_data.get("style", "informative")
        length = input_data.get("length", "medium")
        
        # Simple content generation templates
        templates = {
            "informative": f"Here is some informative content about {topic}. This provides detailed information and insights.",
            "creative": f"Let me tell you an interesting story about {topic}. It's a fascinating subject that sparks imagination.",
            "technical": f"Technical analysis of {topic}: This involves systematic examination and evaluation of key components."
        }
        
        content = templates.get(style, templates["informative"])
        
        if length == "short":
            content = content[:100] + "..."
        elif length == "long":
            content += f" Additionally, {topic} has many complex aspects that require careful consideration and analysis."
        
        return {
            "task": "content_generation",
            "generated_content": content,
            "parameters": {
                "topic": topic,
                "style": style,
                "length": length
            },
            "processed_by": self.model_name,
            "processing_time": 0.15
        }
    
    async def summarize_text(self, text: str) -> Dict[str, Any]:
        """Summarize the input text"""
        await asyncio.sleep(0.12)
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Simple summarization: take first and last sentences
        if len(sentences) <= 2:
            summary = text
        else:
            summary = f"{sentences[0]}. {sentences[-1]}."
        
        return {
            "task": "summarization", 
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0,
            "processed_by": self.model_name,
            "processing_time": 0.12
        }
    
    async def extract_keywords(self, text: str) -> Dict[str, Any]:
        """Extract keywords from text"""
        await asyncio.sleep(0.08)
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency
        keyword_freq = {}
        for word in keywords:
            keyword_freq[word] = keyword_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "task": "keyword_extraction",
            "keywords": [kw[0] for kw in top_keywords],
            "keyword_frequencies": dict(top_keywords),
            "total_unique_keywords": len(keyword_freq),
            "processed_by": self.model_name,
            "processing_time": 0.08
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous execution wrapper"""
        return asyncio.run(self.run(input_data))
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return self.capabilities.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "model_name": self.model_name,
            "processed_count": self.processed_count,
            "capabilities": self.capabilities,
            "status": "ready",
            "debug_mode": self.debug
        }


# Alternative function-based agent
async def simple_text_processor(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Function-based agent for simple text processing
    
    This demonstrates how function-based agents can be integrated.
    """
    text = input_data.get("input", input_data.get("text", ""))
    operation = input_data.get("operation", "count_words")
    
    await asyncio.sleep(0.05)
    
    if operation == "count_words":
        result = len(text.split())
        return {
            "operation": "count_words",
            "result": result,
            "input_length": len(text),
            "agent_type": "function_based"
        }
    elif operation == "reverse":
        result = text[::-1]
        return {
            "operation": "reverse",
            "result": result,
            "agent_type": "function_based"
        }
    elif operation == "uppercase":
        result = text.upper()
        return {
            "operation": "uppercase", 
            "result": result,
            "agent_type": "function_based"
        }
    else:
        return {
            "operation": operation,
            "result": f"Unknown operation: {operation}",
            "agent_type": "function_based",
            "error": True
        }


def process_text_sync(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous text processing function"""
    text = input_data.get("input", "")
    return {
        "processed_text": text.lower().strip(),
        "word_count": len(text.split()),
        "agent_type": "sync_function"
    }


if __name__ == "__main__":
    # Example usage
    agent = SimpleAnalysisAgent(debug=True)
    
    test_input = {
        "task": "analyze",
        "input": "This is a wonderful example of great text analysis. The content is excellent and provides amazing insights."
    }
    
    result = agent.execute(test_input)
    print(f"Analysis Result: {json.dumps(result, indent=2)}")
    
    # Test generation
    gen_input = {
        "task": "generate",
        "topic": "artificial intelligence",
        "style": "technical",
        "length": "medium"
    }
    
    gen_result = agent.execute(gen_input)
    print(f"\nGeneration Result: {json.dumps(gen_result, indent=2)}") 