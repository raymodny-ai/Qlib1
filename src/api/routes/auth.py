"""
Authentication Routes

Provides login, token refresh, logout, and user info endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.security.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
    oauth2_scheme,
    get_oidc_config,
)
from src.api.main import get_current_user, get_rbac

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
#  Request / Response Models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Login credentials."""
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    """Login response with tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., min_length=1)


class RefreshResponse(BaseModel):
    """Token refresh response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInfoResponse(BaseModel):
    """Current user information."""
    user_id: str
    name: str
    role: str
    email: str
    active: bool


class LogoutResponse(BaseModel):
    """Logout confirmation."""
    message: str = "已登出"


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate with username and password.

    Returns JWT access token (short-lived) and refresh token (long-lived).
    The access token must be sent as `Authorization: Bearer <token>` header.
    """
    from src.security.auth import ACCESS_TOKEN_EXPIRE_MINUTES

    rbac = await get_rbac()

    # Look up user by name
    user = await rbac.get_user_by_name(request.username)
    if user is None:
        # Try by user_id as fallback
        user = await rbac.get_user(request.username)

    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not user.get("active", True):
        raise HTTPException(status_code=403, detail="账号已被禁用")

    # Verify password
    stored_hash = user.get("password_hash", "")
    if not stored_hash:
        # Development fallback: accept any password for users without hash
        # In production, all users should have password_hash set
        pass
    elif not verify_password(request.password, stored_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    user_id = user["user_id"]
    role = user.get("role", "unknown")

    # Create tokens
    access_token = create_access_token(
        subject=user_id,
        extra_claims={"role": role},
    )
    refresh_token = create_refresh_token(subject=user_id)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user_id,
        role=role,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(request: RefreshRequest):
    """
    Obtain a new access token using a refresh token.

    The old refresh token remains valid until expiry.
    """
    from src.security.auth import ACCESS_TOKEN_EXPIRE_MINUTES

    try:
        payload = decode_token(request.refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="无效或过期的刷新令牌")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="令牌类型错误，需要刷新令牌")

    user_id = payload.get("sub", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="令牌有效负载无效")

    # Verify user still exists and is active
    rbac = await get_rbac()
    user = await rbac.get_user(user_id)
    if user is None or not user.get("active", True):
        raise HTTPException(status_code=403, detail="用户不存在或已禁用")

    access_token = create_access_token(
        subject=user_id,
        extra_claims={"role": user.get("role", "unknown")},
    )

    return RefreshResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    current_user: str = Depends(get_current_user),
):
    """
    Logout endpoint (client-side token discard).

    In a stateless JWT setup, logout is handled client-side by
    discarding the tokens. For server-side invalidation, a token
    blacklist would be required.
    """
    return LogoutResponse(message="已登出")


@router.get("/me", response_model=UserInfoResponse)
async def get_me(
    current_user: str = Depends(get_current_user),
):
    """
    Get current authenticated user's information.

    Requires a valid JWT access token in Authorization header.
    """
    rbac = await get_rbac()
    user = await rbac.get_user(current_user)

    if user is None:
        raise HTTPException(status_code=404, detail="用户未找到")

    return UserInfoResponse(
        user_id=user["user_id"],
        name=user.get("name", ""),
        role=user.get("role", "unknown"),
        email=user.get("email", ""),
        active=user.get("active", True),
    )


@router.get("/oidc/config")
async def get_oidc_config_endpoint():
    """
    Get OIDC/SSO configuration status.

    Returns whether OIDC is configured and available.
    """
    config = get_oidc_config()
    return {
        "enabled": config.enabled,
        "issuer": config.issuer if config.enabled else None,
    }
