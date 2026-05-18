"""
API 微服务层

基于 FastAPI 的量化分析数据服务，将底层因子矩阵、
模型预测和风险指标以 JSON 格式暴露给外部消费端。
"""

from src.api.main import app

__all__ = ["app"]