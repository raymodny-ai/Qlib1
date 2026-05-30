"""
系统日志配置模块

基于 structlog 提供结构化日志输出，兼容标准 logging 接口。
配置来源于 qlib_config.yaml。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from pythonjsonlogger.json import JsonFormatter


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    use_json: bool = False,
) -> structlog.BoundLogger:
    """
    初始化结构化日志系统

    Args:
        log_level: 日志级别 (DEBUG | INFO | WARNING | ERROR)
        log_dir: 日志文件输出目录
        use_json: 是否输出 JSON 格式（生产环境推荐）

    Returns:
        配置好的 structlog logger
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置标准日志处理器
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if use_json:
        console_formatter = JsonFormatter(
            "%(timestamp)s %(name)s %(level)s %(message)s"
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    console_handler.setFormatter(console_formatter)

    # 文件输出
    file_handler = logging.FileHandler(log_path / "system.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = JsonFormatter(
        "%(timestamp)s %(name)s %(level)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 错误日志单独文件
    error_handler = logging.FileHandler(log_path / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # 减少第三方库日志噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return structlog.get_logger("qlib-us")


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """获取模块级别的 logger"""
    logger = structlog.get_logger(name or "qlib-us")
    return logger.bind(module=name) if name else logger
