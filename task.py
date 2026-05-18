"""
Qlib美股基本面量化分析系统 — 统一开发命令入口

用法:
    python task.py data-ingest        # 数据采集落盘
    python task.py process            # 特征工程处理
    python task.py train              # 模型训练
    python task.py backtest           # 策略回测
    python task.py report             # 报告生成
    python task.py serve              # 启动 API 服务
    python task.py test               # 运行测试
    python task.py lint               # 代码检查
"""

import subprocess
import sys
from pathlib import Path

import click

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


@click.group()
@click.version_option(version="0.1.0", prog_name="qlib-us-fundamental")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Qlib美股基本面量化分析系统 CLI 工具集"""
    ctx.ensure_object(dict)
    ctx.obj["PROJECT_ROOT"] = PROJECT_ROOT


# ===== 数据采集 =====
@cli.command("data-ingest")
@click.option("--source", "-s", default="all",
              type=click.Choice(["all", "alpha_vantage", "eodhd", "intrinio", "sec_edgar"]),
              help="指定数据源")
@click.option("--start-date", default=None, help="采集起始日期 (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="采集截止日期 (YYYY-MM-DD)")
@click.option("--tickers", "-t", default=None, help="指定股票代码列表文件路径")
@click.pass_context
def data_ingest(ctx, source, start_date, end_date, tickers):
    """从外部数据源采集美股基本面与量价数据"""
    click.echo(f"🚀 启动数据采集 pipeline | 数据源: {source}")
    # TODO: 集成 src.workflow.data_ingestion_pipeline
    subprocess.run([
        sys.executable, "-m", "src.workflow.data_ingestion_pipeline",
        "--source", source,
        *(["--start-date", start_date] if start_date else []),
        *(["--end-date", end_date] if end_date else []),
        *(["--tickers", tickers] if tickers else []),
    ], check=False)


# ===== 数据转换 =====
@cli.command("convert")
@click.option("--input-dir", "-i", default="./data/raw", help="原始数据目录")
@click.option("--output-dir", "-o", default="./data/qlib_data/us_data", help="Qlib .bin 输出目录")
@click.option("--incremental/--full", default=True, help="增量转换 / 全量重建")
@click.pass_context
def convert_data(ctx, input_dir, output_dir, incremental):
    """将原始 CSV/Parquet 数据转换为 Qlib 二进制格式 (.bin)"""
    mode = "增量" if incremental else "全量"
    click.echo(f"🔄 数据格式转换 | 模式: {mode}")
    subprocess.run([
        sys.executable, "scripts/dump_bin.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        *(["--incremental"] if incremental else ["--full"]),
    ], check=False)


# ===== 特征工程 =====
@cli.command("process")
@click.option("--config", "-c", default="config/qlib_config.yaml", help="配置文件路径")
@click.pass_context
def process_features(ctx, config):
    """运行数据预处理与特征工程 pipeline"""
    click.echo("🔧 启动特征工程 pipeline")
    subprocess.run([
        sys.executable, "-m", "src.workflow.feature_pipeline",
        "--config", config,
    ], check=False)


# ===== 模型训练 =====
@cli.command("train")
@click.option("--model", "-m", default="lightgbm",
              type=click.Choice(["lightgbm", "xgboost", "double_ensemble", "adarnn", "tabnet", "all"]),
              help="选择模型类型")
@click.option("--config", "-c", default="config/qlib_config.yaml", help="配置文件路径")
@click.option("--gpu", "-g", default=0, help="GPU 设备 ID (-1 表示 CPU)")
@click.pass_context
def train_model(ctx, model, config, gpu):
    """训练 ML 预测模型"""
    click.echo(f"🧠 启动模型训练 | 模型: {model} | GPU: {gpu}")
    subprocess.run([
        sys.executable, "-m", "src.workflow.training_pipeline",
        "--model", model,
        "--config", config,
        "--gpu", str(gpu),
    ], check=False)


# ===== 策略回测 =====
@cli.command("backtest")
@click.option("--config", "-c", default="config/qlib_config.yaml", help="配置文件路径")
@click.option("--model-path", "-m", default=None, help="模型权重文件路径")
@click.option("--start-date", default=None, help="回测起始日期")
@click.option("--end-date", default=None, help="回测截止日期")
@click.pass_context
def run_backtest(ctx, config, model_path, start_date, end_date):
    """执行策略回测模拟"""
    click.echo("📊 启动策略回测引擎")
    cmd = [sys.executable, "-m", "src.workflow.backtest_pipeline", "--config", config]
    if model_path:
        cmd.extend(["--model-path", model_path])
    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])
    subprocess.run(cmd, check=False)


# ===== 报告生成 =====
@cli.command("report")
@click.option("--experiment-id", "-e", required=True, help="实验 ID")
@click.option("--output-dir", "-o", default="./reports", help="报告输出目录")
@click.option("--format", "-f", "fmt", default="html",
              type=click.Choice(["json", "html", "all"]),
              help="报告格式")
@click.pass_context
def generate_report(ctx, experiment_id, output_dir, fmt):
    """生成回测分析报告（IC曲线/收益曲线/风险图谱）"""
    click.echo(f"📄 生成回测报告 | 实验: {experiment_id} | 格式: {fmt}")
    subprocess.run([
        sys.executable, "-m", "src.analyzers.report_generator",
        "--experiment-id", experiment_id,
        "--output-dir", output_dir,
        "--format", fmt,
    ], check=False)


# ===== API 服务 =====
@cli.command("serve")
@click.option("--host", "-h", default="0.0.0.0", help="绑定地址")
@click.option("--port", "-p", default=8000, help="绑定端口")
@click.option("--reload/--no-reload", default=True, help="热重载（开发模式）")
@click.pass_context
def serve_api(ctx, host, port, reload):
    """启动 RESTful API 微服务"""
    click.echo(f"🌐 启动 API 服务 | http://{host}:{port}")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", str(port),
        *(["--reload"] if reload else []),
    ], check=False)


# ===== 测试 =====
@cli.command("test")
@click.option("--cov/--no-cov", default=True, help="生成覆盖率报告")
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
@click.pass_context
def run_tests(ctx, cov, verbose):
    """运行测试套件"""
    click.echo("🧪 运行测试套件")
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if cov:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    if verbose:
        cmd.append("-v")
    subprocess.run(cmd, check=False)


# ===== 代码检查 =====
@cli.command("lint")
@click.option("--fix/--no-fix", default=False, help="自动修复")
@click.pass_context
def lint_code(ctx, fix):
    """代码风格检查 (ruff + mypy)"""
    click.echo("🔍 代码质量检查")
    if fix:
        subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", "src/", "tests/"], check=False)
    else:
        subprocess.run([sys.executable, "-m", "ruff", "check", "src/", "tests/"], check=False)
        click.echo("---")
        subprocess.run([sys.executable, "-m", "mypy", "src/"], check=False)


# ===== 格式化 =====
@cli.command("format")
@click.pass_context
def format_code(ctx):
    """代码格式化 (black + isort)"""
    click.echo("✨ 代码格式化")
    subprocess.run([sys.executable, "-m", "isort", "src/", "tests/"], check=False)
    subprocess.run([sys.executable, "-m", "black", "src/", "tests/"], check=False)


# ===== Docker 相关 =====
@cli.command("docker-build")
@click.option("--target", "-t", default="development",
              type=click.Choice(["development", "production"]),
              help="构建目标阶段")
@click.pass_context
def docker_build(ctx, target):
    """构建 Docker 镜像"""
    click.echo(f"🐳 构建 Docker 镜像 | 目标: {target}")
    subprocess.run([
        "docker", "build",
        "-f", "docker/Dockerfile",
        "--target", target,
        "-t", f"qlib-us-fundamental:{target}",
        ".",
    ], check=False)


@cli.command("docker-up")
@click.option("--service", "-s", default="dev",
              type=click.Choice(["dev", "api", "data-ingestor", "all"]),
              help="启动的服务")
@click.pass_context
def docker_up(ctx, service):
    """启动 Docker Compose 服务"""
    click.echo(f"🐳 启动 Docker 服务 | 服务: {service}")
    compose_file = "docker/docker-compose.yml"
    if service == "all":
        subprocess.run(["docker", "compose", "-f", compose_file, "up", "-d"], check=False)
    else:
        subprocess.run(["docker", "compose", "-f", compose_file, "up", "-d", service], check=False)


# ===== 环境检查 =====
@cli.command("check")
@click.pass_context
def check_environment(ctx):
    """检查开发环境（Python版本、依赖、GPU）"""
    click.echo("🔎 环境检查")

    # Python 版本
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"  Python: {py_version} {'✅' if sys.version_info >= (3, 9) else '❌ (需要 ≥3.9)'}")

    # Qlib
    try:
        import qlib
        click.echo(f"  Qlib: {qlib.__version__} ✅")
    except ImportError:
        click.echo("  Qlib: 未安装 ❌")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            click.echo(f"  CUDA: 可用 | GPU: {torch.cuda.get_device_name(0)} ✅")
        else:
            click.echo("  CUDA: 不可用 ⚠️ (将使用 CPU 训练)")
    except ImportError:
        click.echo("  PyTorch: 未安装 ❌")

    # XGBoost
    try:
        import xgboost as xgb
        click.echo(f"  XGBoost: {xgb.__version__} ✅")
    except ImportError:
        click.echo("  XGBoost: 未安装 ❌")

    # LightGBM
    try:
        import lightgbm as lgb
        click.echo(f"  LightGBM: {lgb.__version__} ✅")
    except ImportError:
        click.echo("  LightGBM: 未安装 ❌")

    # 配置文件
    config_path = PROJECT_ROOT / "config" / "qlib_config.yaml"
    click.echo(f"  配置文件: {'存在 ✅' if config_path.exists() else '缺失 ❌'}")


if __name__ == "__main__":
    cli()
