"""
FastAPI 安全中间件

对标 PRD 第5章: 全链路 TLS 1.2+ 传输加密强制 + HSTS 安全响应头。

核心功能:
- HTTPSRedirectMiddleware: HTTP → HTTPS 重定向
- HSTSMiddleware: Strict-Transport-Security 响应头
- SecurityHeadersMiddleware: CSP/X-Frame-Options 等安全头
- TLSValidationMiddleware: TLS 1.2+ 版本强制校验
- RateLimitMiddleware: 令牌桶 API 速率限制
- register_security_middleware: 一键注册所有安全中间件

使用方式 (在 main.py 中):
    from src.api.middleware import register_security_middleware
    register_security_middleware(app)
"""

import time
import threading
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import RedirectResponse, JSONResponse


# ========================================================================
#  HTTPS 重定向中间件
# ========================================================================

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """HTTP → HTTPS 强制重定向 (开发环境 localhost 自动跳过)"""

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if self.enabled:
            proto = request.headers.get("X-Forwarded-Proto", "")
            scheme = request.url.scheme
            is_local = request.client and request.client.host in (
                "127.0.0.1", "::1", "localhost",
            )
            if not is_local and proto != "https" and scheme != "https":
                return RedirectResponse(
                    str(request.url.replace(scheme="https")), status_code=301
                )
        return await call_next(request)


# ========================================================================
#  HSTS 响应头中间件
# ========================================================================

class HSTSMiddleware(BaseHTTPMiddleware):
    """Strict-Transport-Security (HSTS) — max-age=1年 + includeSubDomains"""

    DEFAULT_POLICY = "max-age=31536000; includeSubDomains; preload"

    def __init__(self, app, hsts_policy: Optional[str] = None):
        super().__init__(app)
        self.hsts_policy = hsts_policy or self.DEFAULT_POLICY

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
        if scheme == "https":
            response.headers["Strict-Transport-Security"] = self.hsts_policy
        return response


# ========================================================================
#  通用安全响应头中间件
# ========================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """CSP / X-Frame-Options / X-Content-Type-Options 等安全头"""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "camera=(), microphone=(), geolocation=(), payment=(), "
            "usb=(), magnetometer=(), accelerometer=(), gyroscope=()"
        ),
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        for name, value in self.SECURITY_HEADERS.items():
            if name not in response.headers:
                response.headers[name] = value
        return response


# ========================================================================
#  TLS 版本验证中间件
# ========================================================================

class TLSValidationMiddleware(BaseHTTPMiddleware):
    """TLS 1.2+ 强制 — 违规返回 426 Upgrade Required"""

    ALLOWED_VERSIONS = {"TLSv1.2", "TLSv1.3", "tls1.2", "tls1.3"}
    WEAK_CIPHERS = {"RC4-SHA", "DES-CBC3-SHA", "RC4-MD5", "NULL-SHA256", "NULL-MD5"}

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self._violation_count = 0
        self._lock = threading.Lock()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.enabled:
            return await call_next(request)

        # 本地开发跳过
        if request.client and request.client.host in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)

        tls_ver = request.headers.get("X-Forwarded-TLS-Version", "")
        cipher = request.headers.get("X-Forwarded-Cipher", "")

        if tls_ver and tls_ver not in self.ALLOWED_VERSIONS:
            with self._lock:
                self._violation_count += 1
            return JSONResponse(status_code=426, content={
                "error": "Upgrade Required",
                "message": f"TLS {tls_ver} 不符合安全策略 (要求 >= TLS 1.2)",
                "required_version": "TLSv1.2",
            })

        if cipher in self.WEAK_CIPHERS:
            with self._lock:
                self._violation_count += 1
            return JSONResponse(status_code=426, content={
                "error": "Upgrade Required",
                "message": f"不安全的密码套件: {cipher}",
            })

        return await call_next(request)

    @property
    def violation_count(self) -> int:
        with self._lock:
            return self._violation_count


# ========================================================================
#  速率限制中间件 (令牌桶算法)
# ========================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    基于令牌桶的 API 速率限制

    参数:
        rate: 每秒补充令牌数 (默认 100)
        burst: 最大突发容量 (默认 200)
    """

    def __init__(self, app, rate: float = 100.0, burst: int = 200):
        super().__init__(app)
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(float(self.burst), self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
            else:
                return JSONResponse(status_code=429, content={
                    "error": "Too Many Requests",
                    "message": "请求频率超限，请稍后重试",
                    "retry_after": int(1.0 / self.rate) + 1,
                })

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(int(self._tokens))
        return response


# ========================================================================
#  请求日志中间件
# ========================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个请求的方法、路径、状态码和耗时"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        # 跳过健康检查的详细日志
        if request.url.path != "/health":
            response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"

        return response


# ========================================================================
#  一键注册
# ========================================================================

def register_security_middleware(
    app: FastAPI,
    enable_https_redirect: bool = True,
    enable_hsts: bool = True,
    enable_tls_validation: bool = True,
    enable_rate_limit: bool = False,
    rate_limit_rate: float = 100.0,
    rate_limit_burst: int = 200,
) -> None:
    """
    一键注册所有安全中间件到 FastAPI 应用

    中间件顺序 (由外到内):
    1. RateLimit    → 最早拦截，防止资源耗尽
    2. TLSValidation → 协议安全校验
    3. HTTPSRedirect → HTTP → HTTPS
    4. HSTS          → 响应头注入
    5. SecurityHeaders → 通用安全头
    6. RequestLogging → 请求追踪

    Args:
        app: FastAPI 应用实例
        enable_https_redirect: 启用 HTTPS 重定向
        enable_hsts: 启用 HSTS
        enable_tls_validation: 启用 TLS 版本校验
        enable_rate_limit: 启用速率限制
        rate_limit_rate: 每秒令牌数
        rate_limit_burst: 突发容量
    """
    # 注册顺序: 后添加的先执行 (LIFO)
    app.add_middleware(RequestLoggingMiddleware)

    app.add_middleware(SecurityHeadersMiddleware)

    if enable_hsts:
        app.add_middleware(HSTSMiddleware)

    if enable_https_redirect:
        app.add_middleware(HTTPSRedirectMiddleware, enabled=enable_https_redirect)

    if enable_tls_validation:
        app.add_middleware(TLSValidationMiddleware, enabled=enable_tls_validation)

    if enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            rate=rate_limit_rate,
            burst=rate_limit_burst,
        )

    from src.utils.logger import get_logger
    logger = get_logger()
    logger.info(
        "安全中间件已注册",
        https_redirect=enable_https_redirect,
        hsts=enable_hsts,
        tls_validation=enable_tls_validation,
        rate_limit=enable_rate_limit,
    )
