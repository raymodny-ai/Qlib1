# ============================================================================
# Qlib美股基本面量化分析系统 — Makefile
# ============================================================================

.PHONY: help install install-dev data-ingest convert process train backtest report serve test lint format benchmark docker-build docker-up docker-down check clean

# 默认目标
help:
	@echo "Qlib美股基本面量化分析系统 — 开发命令"
	@echo ""
	@echo "  环境管理:"
	@echo "    make install         安装项目依赖"
	@echo "    make install-dev     安装开发依赖"
	@echo "    make check           检查开发环境"
	@echo ""
	@echo "  数据处理:"
	@echo "    make data-ingest     数据采集落盘"
	@echo "    make convert         数据格式转换 (.bin)"
	@echo "    make process         特征工程 pipeline"
	@echo ""
	@echo "  模型与策略:"
	@echo "    make train           模型训练"
	@echo "    make backtest        策略回测"
	@echo "    make report          报告生成"
	@echo ""
	@echo "  开发工具:"
	@echo "    make serve           启动 API 服务"
	@echo "    make test            运行测试"
	@echo "    make lint            代码检查"
	@echo "    make format          代码格式化"
	@echo "    make benchmark       运行性能基准测试"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build    构建 Docker 镜像"
	@echo "    make docker-up       启动开发容器"
	@echo "    make docker-down     停止容器"
	@echo ""
	@echo "  清理:"
	@echo "    make clean           清理临时文件"

# ===== 环境管理 =====
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio black isort ruff mypy

check:
	python task.py check

# ===== 数据处理 =====
data-ingest:
	python task.py data-ingest

convert:
	python task.py convert

process:
	python task.py process

# ===== 模型与策略 =====
train:
	python task.py train

backtest:
	python task.py backtest

report:
	python task.py report

# ===== API 服务 =====
serve:
	python task.py serve

# ===== 性能基准测试 (PRD 第4章) =====
benchmark:
	PYTHONPATH=. python scripts/benchmark.py

# ===== 开发工具 =====
test:
	python task.py test

lint:
	python task.py lint

format:
	python task.py format

# ===== Docker =====
docker-build:
	python task.py docker-build

docker-up:
	python task.py docker-up

docker-down:
	docker compose -f docker/docker-compose.yml down

# ===== 清理 =====
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/
	@echo "临时文件已清理"
