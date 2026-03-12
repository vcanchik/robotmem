"""配置管理 — 默认值 → ~/.robotmem/config.json

robotmem 不依赖 pyyaml，配置用 JSON。
三层合并：dataclass 默认值 → config.json → 环境变量 ROBOTMEM_HOME。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

# 全局配置目录（支持 ROBOTMEM_HOME 环境变量覆盖）
ROBOTMEM_HOME = (
    Path(os.environ["ROBOTMEM_HOME"])
    if os.environ.get("ROBOTMEM_HOME")
    else Path.home() / ".robotmem"
)
CONFIG_FILE = ROBOTMEM_HOME / "config.json"


@dataclass
class Config:
    """robotmem 核心配置"""

    # ── 数据库 ──
    db_path: str = ""  # 空 = ROBOTMEM_HOME / "memory.db"

    # ── Embedding ──
    embed_backend: str = "onnx"  # "onnx" | "ollama"
    onnx_model: str = "BAAI/bge-small-en-v1.5"
    onnx_dim: int = 384
    embedding_model: str = "nomic-embed-text"  # Ollama 模型名
    embedding_dim: int = 768  # Ollama 模型维度
    ollama_url: str = "http://localhost:11434"
    embed_api: str = "ollama"  # "ollama" | "openai_compat"
    fastembed_cache_dir: str = ""  # 空 = 系统默认

    # ── Web UI ──
    web_port: int = 7878

    # ── 搜索 ──
    top_k: int = 10
    rrf_k: int = 60

    # ── 记忆默认值 ──
    collection: str = "default"
    default_confidence: float = 0.9
    default_decay_rate: float = 0.01
    min_confidence: float = 0.3  # recall 默认过滤阈值

    def __post_init__(self):
        if not self.db_path:
            self.db_path = str(ROBOTMEM_HOME / "memory.db")
        valid_backends = ("onnx", "ollama")
        if self.embed_backend not in valid_backends:
            raise ValueError(
                f"embed_backend 必须是 {valid_backends} 之一，当前值: '{self.embed_backend}'"
            )
        valid_apis = ("ollama", "openai_compat")
        if self.embed_api not in valid_apis:
            raise ValueError(
                f"embed_api 必须是 {valid_apis} 之一，当前值: '{self.embed_api}'"
            )

    @property
    def db_path_resolved(self) -> Path:
        """数据库绝对路径（展开 ~）"""
        return Path(self.db_path).expanduser()

    @property
    def default_collection(self) -> str:
        """默认 collection 名"""
        return self.collection

    @property
    def effective_embedding_dim(self) -> int:
        """当前 embedding 后端的实际维度"""
        if self.embed_backend == "onnx":
            return self.onnx_dim
        return self.embedding_dim


def load_config() -> Config:
    """加载配置：默认值 → ~/.robotmem/config.json"""
    try:
        ROBOTMEM_HOME.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("无法创建配置目录 %s: %s（配置将不持久化）", ROBOTMEM_HOME, e)

    config = Config()

    if CONFIG_FILE.is_file():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                _merge_into_config(config, overrides)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("配置文件读取失败: %s", e)

    return config


def save_config(config: Config, path: Path | None = None) -> None:
    """保存非默认配置到 JSON 文件"""
    if path is None:
        path = CONFIG_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("无法创建目录 %s: %s", path.parent, e)

    data = {}
    default = Config()
    for f in fields(Config):
        value = getattr(config, f.name)
        if value != getattr(default, f.name):
            data[f.name] = value

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("配置文件保存失败: %s", e)


def _merge_into_config(config: Config, overrides: dict) -> None:
    """将 dict 中的有效键合并到 Config 实例"""
    valid_fields = {f.name for f in fields(Config)}
    for key, value in overrides.items():
        if key in valid_fields and value is not None:
            setattr(config, key, value)
