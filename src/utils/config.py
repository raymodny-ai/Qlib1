"""
系统配置加载器

从 config/qlib_config.yaml 加载全局配置，并支持环境变量覆盖。
提供单例模式的配置访问接口。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class ConfigLoader:
    """
    单例配置加载器

    使用方式:
        from src.utils.config import get_config
        cfg = get_config()
        log_level = cfg.get("system.log_level")
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """加载 YAML 配置文件并合并环境变量"""
        config_paths = [
            Path("config/qlib_config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "qlib_config.yaml",
        ]

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            raise FileNotFoundError(
                "无法找到 qlib_config.yaml。请确保项目根目录的 config/ 文件夹中包含该文件。"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # 环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖 YAML 配置"""
        env_mappings = {
            "QLIB_DATA_DIR": "qlib.provider_uri",
            "QLIB_CACHE_DIR": "qlib.cache",
            "QLIB_LOG_LEVEL": "system.log_level",
            "ENVIRONMENT": "system.environment",
        }

        for env_key, config_path in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                self._set_nested(config_path, env_value)

    def _set_nested(self, path: str, value: Any) -> None:
        """设置嵌套字典值，使用点号分隔路径"""
        keys = path.split(".")
        d = self._config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            path: 点号分隔的配置路径，如 "qlib.provider_uri"
            default: 未找到时的默认值

        Returns:
            配置值
        """
        keys = path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_all(self) -> Dict[str, Any]:
        """返回完整配置字典（只读拷贝）"""
        import copy
        return copy.deepcopy(self._config)

    def reload(self) -> None:
        """重新加载配置文件"""
        self._load()


# 便捷的全局单例
_config_instance: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """获取 ConfigLoader 单例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance
