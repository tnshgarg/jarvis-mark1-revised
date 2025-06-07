"""
Authentication and Authorization

Session 20: API Layer & REST Endpoints
JWT-based authentication and role-based authorization system
"""

import jwt
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..utils.exceptions import AuthenticationException, AuthorizationException


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = "mark1-super-secret-key-change-in-production"  # TODO: Move to config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class User:
    """User model for authentication"""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[str] = None,
        permissions: Set[str] = None,
        is_active: bool = True
    ):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles or []
        self.permissions = permissions or set()
        self.is_active = is_active
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission"""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(perm in self.permissions for perm in permissions)
    
    def to_dict(self) -> Dict:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": list(self.permissions),
            "is_active": self.is_active
        }


class AuthenticationManager:
    """Manages authentication operations"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Mock user database - in production, this would be a real database
        self.users_db = {
            "admin": {
                "user_id": "user_1",
                "username": "admin",
                "email": "admin@mark1.ai",
                "hashed_password": pwd_context.hash("admin123"),
                "roles": ["admin", "user"],
                "permissions": {"read", "write", "delete", "admin"},
                "is_active": True
            },
            "user": {
                "user_id": "user_2", 
                "username": "user",
                "email": "user@mark1.ai",
                "hashed_password": pwd_context.hash("user123"),
                "roles": ["user"],
                "permissions": {"read", "write"},
                "is_active": True
            }
        }
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plaintext password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        user_data = self.users_db.get(username)
        if not user_data:
            return None
        
        if not self.verify_password(password, user_data["hashed_password"]):
            return None
        
        if not user_data["is_active"]:
            return None
        
        return User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data["roles"],
            permissions=user_data["permissions"],
            is_active=user_data["is_active"]
        )
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token has expired")
        except jwt.JWTError:
            raise AuthenticationException("Invalid token")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        for user_data in self.users_db.values():
            if user_data["user_id"] == user_id:
                return User(
                    user_id=user_data["user_id"],
                    username=user_data["username"],
                    email=user_data["email"],
                    roles=user_data["roles"],
                    permissions=user_data["permissions"],
                    is_active=user_data["is_active"]
                )
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_data = self.users_db.get(username)
        if not user_data:
            return None
        
        return User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data["roles"],
            permissions=user_data["permissions"],
            is_active=user_data["is_active"]
        )


class AuthorizationManager:
    """Manages authorization operations"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Permission definitions
        self.permissions = {
            # Agent permissions
            "agents:read": "Read agent information",
            "agents:write": "Create and update agents",
            "agents:delete": "Delete agents",
            "agents:test": "Test agent functionality",
            
            # Task permissions
            "tasks:read": "Read task information",
            "tasks:write": "Create and update tasks",
            "tasks:execute": "Execute tasks",
            "tasks:delete": "Delete tasks",
            
            # Context permissions
            "contexts:read": "Read context information",
            "contexts:write": "Create and update contexts",
            "contexts:delete": "Delete contexts",
            "contexts:share": "Share contexts between agents",
            
            # Orchestration permissions
            "orchestration:read": "Read orchestration status",
            "orchestration:write": "Create orchestration workflows",
            "orchestration:execute": "Execute orchestration workflows",
            
            # System permissions
            "system:read": "Read system information",
            "system:monitor": "Monitor system metrics",
            "system:admin": "System administration",
            
            # Admin permissions
            "admin:users": "User management",
            "admin:config": "System configuration",
            "admin:logs": "Access system logs"
        }
        
        # Role definitions
        self.roles = {
            "user": {
                "agents:read", "agents:test",
                "tasks:read", "tasks:write", "tasks:execute",
                "contexts:read", "contexts:write", "contexts:share",
                "orchestration:read", "orchestration:write", "orchestration:execute",
                "system:read"
            },
            "admin": {
                # Admin has all permissions
                *self.permissions.keys()
            },
            "readonly": {
                "agents:read",
                "tasks:read",
                "contexts:read",
                "orchestration:read",
                "system:read"
            }
        }
    
    def get_role_permissions(self, role: str) -> Set[str]:
        """Get permissions for a role"""
        return self.roles.get(role, set())
    
    def check_permission(self, user: User, required_permission: str) -> bool:
        """Check if user has required permission"""
        if not user.is_active:
            return False
        
        # Check direct permissions
        if user.has_permission(required_permission):
            return True
        
        # Check role-based permissions
        for role in user.roles:
            role_permissions = self.get_role_permissions(role)
            if required_permission in role_permissions:
                return True
        
        return False
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would be used as a dependency in FastAPI
                # The actual implementation would check the current user
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# FastAPI Security
security = HTTPBearer()
auth_manager = AuthenticationManager()
authz_manager = AuthorizationManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """FastAPI dependency to get current authenticated user"""
    try:
        token = credentials.credentials
        payload = auth_manager.verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        user = auth_manager.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return user
    
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


def require_permission(permission: str):
    """FastAPI dependency to require specific permission"""
    def check_permission_dependency(current_user: User = Depends(get_current_user)):
        if not authz_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return current_user
    
    return check_permission_dependency


def require_role(role: str):
    """FastAPI dependency to require specific role"""
    def check_role_dependency(current_user: User = Depends(get_current_user)):
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role}"
            )
        return current_user
    
    return check_role_dependency


# Utility functions for testing/development
def create_test_token(username: str = "admin") -> str:
    """Create a test token for development/testing"""
    user = auth_manager.get_user_by_username(username)
    if not user:
        raise ValueError(f"User {username} not found")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_manager.create_access_token(
        data={"sub": user.user_id, "username": user.username},
        expires_delta=access_token_expires
    )
    return access_token 