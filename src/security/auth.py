"""
JWT Authentication & OIDC Module

Provides token-based authentication for the FastAPI API layer.

Features:
- JWT access + refresh token creation and validation
- Password hashing with bcrypt (passlib)
- OIDC provider configuration (optional)
- OAuth2PasswordBearer scheme integration
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-production")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
REFRESH_TOKEN_EXPIRE_DAYS = int(
    os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
)

# OIDC (optional)
OIDC_ENABLED = os.environ.get("OIDC_ENABLED", "false").lower() == "true"
OIDC_ISSUER = os.environ.get("OIDC_ISSUER", "")
OIDC_CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "")
OIDC_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "")

# Password hashing context
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


# ---------------------------------------------------------------------------
#  Password Utilities
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return _pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
#  JWT Token Utilities
# ---------------------------------------------------------------------------

def create_access_token(
    subject: str,
    extra_claims: Optional[Dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: User identifier (user_id)
        extra_claims: Additional JWT claims (role, permissions, etc.)
        expires_delta: Custom expiry (default: ACCESS_TOKEN_EXPIRE_MINUTES)

    Returns:
        Encoded JWT string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_delta,
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        subject: User identifier
        expires_delta: Custom expiry (default: REFRESH_TOKEN_EXPIRE_DAYS)

    Returns:
        Encoded JWT string
    """
    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_delta,
        "type": "refresh",
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.

    Returns:
        Decoded token payload

    Raises:
        JWTError: If token is invalid or expired
    """
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


def decode_token_optional(token: str) -> Optional[Dict[str, Any]]:
    """Decode a JWT token, returning None on failure instead of raising."""
    try:
        return decode_token(token)
    except JWTError:
        return None


# ---------------------------------------------------------------------------
#  OIDC Configuration (placeholder for SSO integration)
# ---------------------------------------------------------------------------

class OIDCConfig:
    """OpenID Connect provider configuration."""

    def __init__(
        self,
        issuer: str = "",
        client_id: str = "",
        client_secret: str = "",
    ):
        self.issuer = issuer or OIDC_ISSUER
        self.client_id = client_id or OIDC_CLIENT_ID
        self.client_secret = client_secret or OIDC_CLIENT_SECRET
        self.enabled = bool(self.issuer and self.client_id)

    @property
    def discovery_url(self) -> str:
        """OIDC discovery endpoint URL."""
        return f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"

    @property
    def authorization_endpoint(self) -> str:
        """OIDC authorization endpoint."""
        return f"{self.issuer.rstrip('/')}/authorize"

    @property
    def token_endpoint(self) -> str:
        """OIDC token endpoint."""
        return f"{self.issuer.rstrip('/')}/token"

    @property
    def userinfo_endpoint(self) -> str:
        """OIDC userinfo endpoint."""
        return f"{self.issuer.rstrip('/')}/userinfo"


def get_oidc_config() -> OIDCConfig:
    """Get the global OIDC configuration."""
    return OIDCConfig()
