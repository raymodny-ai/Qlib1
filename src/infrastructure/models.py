"""
SQLAlchemy ORM Models

Database-backend persistence for users, roles, audit logs,
experiments, backtest tasks, access grants, and access requests.

All models use async SQLAlchemy with the Base from database.py.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
#  User & Role
# ---------------------------------------------------------------------------

class UserModel(Base):
    """Database-persisted system user."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="quant_researcher"
    )
    email: Mapped[str] = mapped_column(String(256), default="")
    password_hash: Mapped[str] = mapped_column(String(256), default="")
    api_key_hash: Mapped[str] = mapped_column(String(256), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    access_grants: Mapped[list["AccessGrantModel"]] = relationship(
        back_populates="user", lazy="selectin"
    )
    access_requests: Mapped[list["AccessRequestModel"]] = relationship(
        back_populates="requester", foreign_keys="AccessRequestModel.requester_id",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<User {self.user_id} role={self.role}>"


# ---------------------------------------------------------------------------
#  Audit Log
# ---------------------------------------------------------------------------

class AuditLogModel(Base):
    """Tamper-proof audit log entry persisted to database."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(24), unique=True, default=_new_id)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user: Mapped[str] = mapped_column(String(64), default="system", index=True)
    role: Mapped[str] = mapped_column(String(32), default="unknown")
    action: Mapped[str] = mapped_column(String(128), default="")
    resource: Mapped[str] = mapped_column(String(256), default="")
    detail: Mapped[str] = mapped_column(Text, default="{}")  # JSON string
    ip_address: Mapped[str] = mapped_column(String(45), default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    hash_chain: Mapped[str] = mapped_column(String(128), default="")

    def __repr__(self) -> str:
        return f"<AuditLog {self.event_id} {self.event_type}>"


# ---------------------------------------------------------------------------
#  Experiment
# ---------------------------------------------------------------------------

class ExperimentModel(Base):
    """Persisted experiment / backtest run record."""

    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=_new_id
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(64), default="topk_dropout")
    start_date: Mapped[str] = mapped_column(String(16), nullable=False)
    end_date: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True
    )  # pending | running | completed | failed
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    submitted_by: Mapped[str] = mapped_column(String(64), default="")


# ---------------------------------------------------------------------------
#  Backtest Task
# ---------------------------------------------------------------------------

class BacktestTaskModel(Base):
    """Persisted async backtest task."""

    __tablename__ = "backtest_tasks"

    task_id: Mapped[str] = mapped_column(String(24), primary_key=True, default=_new_id)
    experiment_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("experiments.experiment_id"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True
    )  # pending | running | completed | failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    submitted_by: Mapped[str] = mapped_column(String(64), default="")


# ---------------------------------------------------------------------------
#  Access Grant (temporary permissions)
# ---------------------------------------------------------------------------

class AccessGrantModel(Base):
    """Persisted temporary permission grant."""

    __tablename__ = "access_grants"

    grant_id: Mapped[str] = mapped_column(String(24), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False, index=True
    )
    permission: Mapped[str] = mapped_column(String(128), nullable=False)
    granted_by: Mapped[str] = mapped_column(String(64), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    max_uses: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped[UserModel] = relationship(back_populates="access_grants")


# ---------------------------------------------------------------------------
#  Access Request (approval workflow)
# ---------------------------------------------------------------------------

class AccessRequestModel(Base):
    """Persisted permission access request."""

    __tablename__ = "access_requests"

    request_id: Mapped[str] = mapped_column(
        String(24), primary_key=True, default=_new_id
    )
    requester_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False, index=True
    )
    requested_permission: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    approver_id: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(
        String(16), default="pending", index=True
    )  # pending | approved | rejected
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolution_note: Mapped[str] = mapped_column(Text, default="")
    duration_hours: Mapped[int] = mapped_column(Integer, default=24)

    requester: Mapped[UserModel] = relationship(
        back_populates="access_requests", foreign_keys=[requester_id]
    )
