"""robotmem Web UI — Flask 应用工厂

宪法原则 2 落地：人能看懂 — 可见、可懂、可控。
技术栈：Flask + 原生 JS + 单 CSS 文件（零框架）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from ..config import Config, load_config
from ..db_cog import CogDatabase

logger = logging.getLogger(__name__)

# 模板和静态文件目录（相对于 src/robotmem/）
_PKG_DIR = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _PKG_DIR / "templates"
_STATIC_DIR = _PKG_DIR / "static"


def create_app(config: Config | None = None) -> Flask:
    """Flask 应用工厂

    Args:
        config: robotmem 配置，None 时自动加载
    """
    if config is None:
        config = load_config()

    app = Flask(
        __name__,
        template_folder=str(_TEMPLATE_DIR),
        static_folder=str(_STATIC_DIR),
    )

    # 存储到 app.config 供 API 使用
    db = CogDatabase(config)
    _ = db.conn  # 触发 lazy init
    app.config["ROBOTMEM_DB"] = db
    app.config["ROBOTMEM_CONFIG"] = config

    # 注册 API 蓝图
    from .api import api_bp
    app.register_blueprint(api_bp)

    # 注册首页路由
    from flask import render_template

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/outreach")
    def outreach_page():
        return render_template("outreach.html")

    # 关闭时清理
    @app.teardown_appcontext
    def _close_db(exc):
        pass  # DB 连接随进程生命周期，不在 request 级别关闭

    logger.info("robotmem Web UI 已创建: db=%s", config.db_path_resolved)
    return app
