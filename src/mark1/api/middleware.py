"""
API Middleware

Session 20: API Layer & REST Endpoints
Security, rate limiting, and logging middleware for the FastAPI application
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_403_FORBIDDEN
import structlog


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for adding security headers and protection"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = structlog.get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        """Process request and add security headers"""
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
        }
        
        # Block suspicious requests
        user_agent = request.headers.get("user-agent", "").lower()
        if any(suspicious in user_agent for suspicious in ["sqlmap", "nikto", "nmap", "masscan"]):
            self.logger.warning("Suspicious user agent blocked", user_agent=user_agent, 
                              client_ip=request.client.host)
            return JSONResponse(
                status_code=HTTP_403_FORBIDDEN,
                content={"error": "Forbidden", "message": "Suspicious activity detected"}
            )
        
        # Check for common attack patterns in path
        path = request.url.path.lower()
        if any(pattern in path for pattern in ["../", "..\\", "union select", "<script", "javascript:"]):
            self.logger.warning("Malicious path detected", path=path, 
                              client_ip=request.client.host)
            return JSONResponse(
                status_code=HTTP_403_FORBIDDEN,
                content={"error": "Forbidden", "message": "Invalid request path"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with per-IP and per-user limits"""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_limit: int = 20):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        
        # Storage for rate limiting (in production, use Redis)
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.burst_counts: Dict[str, int] = defaultdict(int)
        self.last_reset: Dict[str, float] = defaultdict(float)
        
        self.logger = structlog.get_logger(__name__)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from token if available
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, you'd decode the token to get user ID
            # For now, use the token as identifier
            return f"user:{auth_header[-20:]}"
        
        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _is_rate_limited(self, client_id: str) -> tuple[bool, Dict[str, Any]]:
        """Check if client is rate limited"""
        now = time.time()
        current_minute = int(now // 60)
        
        # Clean old entries (older than 1 minute)
        minute_ago = now - 60
        while self.request_counts[client_id] and self.request_counts[client_id][0] < minute_ago:
            self.request_counts[client_id].popleft()
        
        # Reset burst counter every minute
        if self.last_reset[client_id] < current_minute:
            self.burst_counts[client_id] = 0
            self.last_reset[client_id] = current_minute
        
        # Check burst limit (immediate)
        if self.burst_counts[client_id] >= self.burst_limit:
            return True, {
                "limit_type": "burst",
                "limit": self.burst_limit,
                "window": "immediate",
                "reset_time": (current_minute + 1) * 60
            }
        
        # Check per-minute limit
        if len(self.request_counts[client_id]) >= self.requests_per_minute:
            return True, {
                "limit_type": "rate",
                "limit": self.requests_per_minute,
                "window": "minute",
                "reset_time": (current_minute + 1) * 60
            }
        
        return False, {}
    
    def _record_request(self, client_id: str):
        """Record a request for rate limiting"""
        now = time.time()
        self.request_counts[client_id].append(now)
        self.burst_counts[client_id] += 1
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        is_limited, limit_info = self._is_rate_limited(client_id)
        
        if is_limited:
            self.logger.warning("Rate limit exceeded", client_id=client_id, 
                              limit_info=limit_info, path=request.url.path)
            
            headers = {
                "X-RateLimit-Limit": str(limit_info["limit"]),
                "X-RateLimit-Window": limit_info["window"],
                "X-RateLimit-Reset": str(int(limit_info["reset_time"])),
                "Retry-After": str(int(limit_info["reset_time"] - time.time()))
            }
            
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {limit_info['limit']} per {limit_info['window']}",
                    "retry_after": int(limit_info["reset_time"] - time.time())
                },
                headers=headers
            )
        
        # Record the request
        self._record_request(client_id)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - len(self.request_counts[client_id]))
        )
        response.headers["X-RateLimit-Window"] = "minute"
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging middleware for requests and responses"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = structlog.get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response information"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Get client info
        client_ip = request.client.host
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Log request
        self.logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent", ""),
            content_type=request.headers.get("content-type", ""),
            content_length=request.headers.get("content-length", "0")
        )
        
        # Add request ID to request state for access in endpoints
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            self.logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                response_size=response.headers.get("content-length", "unknown")
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            self.logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Re-raise the exception
            raise


class CORSMiddleware:
    """Simple CORS middleware (FastAPI has built-in CORS, this is for reference)"""
    
    def __init__(self, app, allow_origins=None, allow_methods=None, allow_headers=None):
        self.app = app
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
    
    async def __call__(self, scope, receive, send):
        """Process CORS headers"""
        if scope["type"] == "http":
            # Handle preflight OPTIONS requests
            if scope["method"] == "OPTIONS":
                response = Response(status_code=200)
                response.headers["Access-Control-Allow-Origin"] = ", ".join(self.allow_origins)
                response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
                await response(scope, receive, send)
                return
        
        await self.app(scope, receive, send)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size"""
    
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
        self.logger = structlog.get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        """Check request body size"""
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            if content_length > self.max_size:
                self.logger.warning(
                    "Request body too large",
                    content_length=content_length,
                    max_size=self.max_size,
                    path=request.url.path,
                    client_ip=request.client.host
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request entity too large",
                        "message": f"Request body size {content_length} exceeds limit of {self.max_size} bytes"
                    }
                )
        
        return await call_next(request) 