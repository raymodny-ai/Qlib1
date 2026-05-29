"""
pytest 根级配置 — 注册自定义标记和共享 fixtures

标记:
- slow: 耗时超过1秒的测试（日常开发可跳过）
- integration: 需要外部服务的集成测试
- network: 需要真实网络访问的测试（仅 CI 全量运行）
"""

import pytest


def pytest_configure(config):
    """注册自定义 pytest 标记"""
    config.addinivalue_line("markers", "slow: 耗时超过1秒的测试")
    config.addinivalue_line("markers", "integration: 需要外部服务的集成测试")
    config.addinivalue_line("markers", "network: 需要真实网络访问的测试")
