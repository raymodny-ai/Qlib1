#!/usr/bin/env python3
"""
Database Seed Script

Initializes database tables and populates default users/roles.
Run with:
    python scripts/seed_db.py
"""

import asyncio
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main() -> None:
    from src.security.security import DBRBACManager, AuditLogger

    print("Initializing database...")
    audit = AuditLogger()
    rbac = DBRBACManager(audit_logger=audit)
    await rbac.initialize(seed_defaults=True)

    # Verify
    users = await rbac.list_users()
    print(f"Seeded {len(users)} users:")
    for u in users:
        print(f"  - {u['user_id']:15s} | {u['name']:25s} | {u['role']}")

    print("Database seed complete.")


if __name__ == "__main__":
    asyncio.run(main())
