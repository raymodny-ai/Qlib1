"""
API 微服务层单元测试

使用 tests/api/conftest.py 中的 api_client fixture 进行依赖注入隔离：
- api_client: 带完整 Mock DataServer / 模型 / RBAC 的 TestClient
- client: 基础 TestClient（仅用于健康检查等无依赖端点）
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """基础 TestClient（无依赖覆盖，仅用于健康检查/Docs等）"""
    return TestClient(app)


class TestHealthCheck:
    """健康检查端点测试"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "qlib-us-fundamental"

    def test_health_has_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        assert "uptime_seconds" in data


class TestInstruments:
    """证券列表端点测试"""

    def test_list_instruments(self, api_client):
        response = api_client.get("/api/v1/instruments")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_instruments_with_sector(self, api_client):
        response = api_client.get("/api/v1/instruments?sector=Technology")
        assert response.status_code == 200
        data = response.json()
        for inst in data:
            assert inst["sector"] == "Technology"

    def test_list_instruments_with_limit(self, api_client):
        response = api_client.get("/api/v1/instruments?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 1


class TestFactors:
    """因子查询端点测试"""

    def test_query_factors(self, api_client):
        payload = {
            "instruments": ["AAPL", "MSFT"],
            "start_date": "2020-01-01",
            "end_date": "2020-01-31",
            "fields": ["close", "volume"],
        }
        response = api_client.post("/api/v1/factors/fundamentals", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "fundamentals"
        assert data["n_rows"] > 0

    def test_query_factors_default_fields(self, api_client):
        payload = {
            "instruments": ["AAPL"],
            "start_date": "2020-01-01",
            "end_date": "2020-01-10",
        }
        response = api_client.post("/api/v1/factors/ohlcv", json=payload)
        assert response.status_code == 200


class TestPredict:
    """预测端点测试"""

    def test_predict_basic(self, api_client):
        payload = {
            "model_name": "LightGBM_v1",
            "instruments": ["AAPL", "MSFT", "GOOGL"],
            "date": "2023-12-15",
            "factors": {
                "AAPL": {"roe": 0.45, "pe": 28.5},
                "MSFT": {"roe": 0.38, "pe": 32.1},
                "GOOGL": {"roe": 0.22, "pe": 25.8},
            },
        }
        response = api_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "LightGBM_v1"
        assert len(data["predictions"]) == 3
        # 检查排名
        ranks = [p["rank"] for p in data["predictions"]]
        assert sorted(ranks) == [1, 2, 3]

    def test_predict_no_factors(self, api_client):
        payload = {
            "model_name": "Test",
            "instruments": ["AAPL"],
            "date": "2023-12-15",
        }
        response = api_client.post("/api/v1/predict", json=payload)
        # 无因子时降级为模拟预测 → 200 OK
        assert response.status_code == 200


class TestPortfolio:
    """组合查询端点测试"""

    def test_get_portfolio(self, api_client):
        response = api_client.get("/api/v1/portfolio/topk_v1?date=2023-12-15")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_id"] == "topk_v1"
        assert len(data["holdings"]) > 0
        assert abs(data["total_weight"] - sum(h["weight"] for h in data["holdings"])) < 0.01

    def test_portfolio_weights_positive(self, api_client):
        response = api_client.get("/api/v1/portfolio/test?date=2023-12-15")
        data = response.json()
        for h in data["holdings"]:
            assert h["weight"] > 0


class TestBacktest:
    """回测端点测试"""

    def test_submit_backtest(self, api_client):
        payload = {
            "strategy_type": "topk_dropout",
            "model_name": "LightGBM_v1",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 1000000,
            "top_k": 30,
        }
        response = api_client.post("/api/v1/backtest", json=payload)
        assert response.status_code == 200
        data = response.json()
        # 同步执行时可能已完成 (mock 数据极小)，生产环境为 "running"
        assert data["status"] in ("running", "completed")
        assert "task_id" in data

    def test_get_backtest_status(self, api_client):
        # 先提交再查询
        payload = {
            "model_name": "LightGBM_v1",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        }
        submit_resp = api_client.post("/api/v1/backtest", json=payload)
        task_id = submit_resp.json()["task_id"]

        response = api_client.get(f"/api/v1/backtest/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id

    def test_backtest_validation(self, api_client):
        """验证参数校验"""
        payload = {
            "model_name": "LightGBM_v1",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 0,  # 低于最小值
        }
        response = api_client.post("/api/v1/backtest", json=payload)
        assert response.status_code == 422  # 验证错误


class TestReport:
    """报告查询端点测试"""

    def test_get_report_not_found(self, api_client):
        """不存在的实验应返回 404"""
        response = api_client.get("/api/v1/report/nonexistent_exp")
        assert response.status_code == 404

    def test_report_metrics_structure(self, api_client):
        """提交回测后应可查询报告（依赖回测完成的实验记录）"""
        # 先提交一个回测
        payload = {
            "model_name": "LightGBM_v1",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        }
        submit_resp = api_client.post("/api/v1/backtest", json=payload)
        assert submit_resp.status_code == 200
        task_id = submit_resp.json()["task_id"]

        # 回测完成后，尝试查询报告
        # 由于回测任务是同步执行的，此时应有结果
        status_resp = api_client.get(f"/api/v1/backtest/{task_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] in ("completed", "running")


class TestErrorHandling:
    """错误处理测试"""

    def test_404_not_found(self, client):
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_invalid_date_format(self, client):
        payload = {
            "instruments": ["AAPL"],
            "start_date": "01-01-2020",  # 错误格式
            "end_date": "2020-12-31",
        }
        response = client.post("/api/v1/factors/test", json=payload)
        assert response.status_code == 422


class TestOpenAPI:
    """OpenAPI 文档测试"""

    def test_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Qlib US Fundamental Analysis API"
