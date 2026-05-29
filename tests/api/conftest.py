"""
API 测试共享 fixtures — 依赖注入隔离

通过 FastAPI dependency_override 替换所有外部依赖（DataServer / 模型 / RBAC），
使 API 测试无需真实数据服务和模型文件即可运行。
"""

from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# ===== Mock DataServer =====

@pytest.fixture
def mock_data_server():
    """模拟 DataServer，返回可控的测试数据"""
    ds = MagicMock()
    ds.is_initialized = True

    # list_instruments 返回50支模拟股票
    ds.registry = MagicMock()
    ds.registry.list_instruments.return_value = [
        f"STOCK_{i:03d}" for i in range(50)
    ]

    # load_features 返回模拟 DataFrame
    def mock_load_features(fields, instruments, start, end):
        dates = pd.date_range(start, end, freq="B")
        data = []
        for inst in instruments[:10]:
            for d in dates:
                data.append({
                    "instrument": inst,
                    "date": d.strftime("%Y-%m-%d"),
                    "close": 100.0 + (hash(inst + str(d)) % 100),
                    "volume": 1000000 + (hash(inst + str(d)) % 5000000),
                    "open": 99.0,
                    "high": 105.0,
                    "low": 98.0,
                })
        df = pd.DataFrame(data)
        if "instrument" in df.columns:
            df = df.set_index(["instrument", "date"])
        return df

    ds.load_features = mock_load_features
    return ds


# ===== Mock 模型 =====

@pytest.fixture
def mock_model():
    """模拟 ML 模型，返回可控预测值"""
    model = MagicMock()
    model.predict.return_value = np.array([0.05, 0.03, -0.01, 0.02, -0.04] * 10)[:50]
    return model


# ===== Mock RBAC / 用户 =====

@pytest.fixture
def mock_rbac():
    """模拟 RBAC 管理器，放行所有权限"""
    from src.security.security import Role, User

    rbac = MagicMock()
    test_user = User(user_id="test_user", name="Test User", role=Role.QUANT_RESEARCHER)
    rbac.get_user.return_value = test_user
    rbac.check_permission.return_value = True
    return rbac


# ===== TestClient with dependency overrides =====

@pytest.fixture
def api_client(mock_data_server, mock_model, mock_rbac):
    """
    FastAPI TestClient — 所有外部依赖已通过 dependency_override + 单例替换

    关键修补：
    - startup_event / require_permission 中的 get_rbac() 和 get_data_server()
      是直接调用（不走 Depends），因此 dependency_overrides 无法拦截。
      必须直接替换模块级单例 _rbac_manager / _data_server。

    使用方式:
        def test_my_endpoint(api_client):
            response = api_client.get("/api/v1/...")
            assert response.status_code == 200
    """
    import src.api.main as api_main

    # ── 保存原始单例 ──
    original_data_server = api_main._data_server
    original_rbac_manager = api_main._rbac_manager
    original_models = api_main._models.copy()

    # ── 直接替换模块级单例（绕过 Depends 机制） ──
    # startup_event 和 require_permission 内部都是直接调用 get_data_server() / get_rbac()
    api_main._data_server = mock_data_server
    api_main._rbac_manager = mock_rbac

    # ── 注入模型到全局 _models dict ──
    api_main._models["Test"] = mock_model
    api_main._models["LightGBM_v1"] = mock_model

    # ── 注册依赖覆盖（额外安全网，覆盖通过 Depends() 调用的场景） ──
    def override_get_data_server():
        return mock_data_server

    def override_get_rbac():
        return mock_rbac

    async def override_get_current_user():
        return "test_user"

    app.dependency_overrides[api_main.get_data_server] = override_get_data_server
    app.dependency_overrides[api_main.get_rbac] = override_get_rbac
    app.dependency_overrides[api_main.get_current_user] = override_get_current_user

    # ── 测试模式环境变量（避免 HTTPS 重定向） ──
    import os
    original_env = os.environ.get("QLIB_TEST_MODE")
    os.environ["QLIB_TEST_MODE"] = "1"

    with TestClient(app) as client:
        yield client

    # ── 清理 ──
    app.dependency_overrides.clear()
    api_main._data_server = original_data_server
    api_main._rbac_manager = original_rbac_manager
    api_main._models = original_models
    if original_env is not None:
        os.environ["QLIB_TEST_MODE"] = original_env
    else:
        os.environ.pop("QLIB_TEST_MODE", None)


# ===== 简化版 client（仅健康检查等无需 mock 的端点） =====

@pytest.fixture
def client():
    """基础 TestClient（无依赖覆盖）"""
    with TestClient(app) as c:
        yield c
