"""
Database Engine & Session Factory

Provides async SQLAlchemy engine and session management.
Supports SQLite (dev) and PostgreSQL (production) via DATABASE_URL.
"""

import os
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./data/system.db",
)


def _ensure_data_dir() -> None:
    """Ensure the data directory exists (needed for SQLite)."""
    if DATABASE_URL.startswith("sqlite"):
        db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        if db_path.startswith("./"):
            db_path = db_path[2:]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


_ensure_data_dir()

# Async engine with connection pooling
_engine = create_async_engine(
    DATABASE_URL,
    echo=os.environ.get("ENVIRONMENT") == "development",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Async session factory
_AsyncSessionFactory = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


async def get_engine():
    """Return the global async engine."""
    return _engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (for FastAPI dependency injection)."""
    async with _AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all tables defined in models that inherit from Base."""
    from src.infrastructure.models import (  # noqa: F401 - register models
        UserModel,
        AuditLogModel,
        ExperimentModel,
        BacktestTaskModel,
        AccessGrantModel,
        AccessRequestModel,
    )
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session_factory():
    """Return the session factory for standalone use."""
    return _AsyncSessionFactory
