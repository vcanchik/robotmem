"""Outreach — 外部发布链接追踪

链接配置 + HTTP 可达性检查。Web UI Outreach Tab 的数据源。
"""

from __future__ import annotations

import logging
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)


# ── 链接配置（更新时改此处） ──

OUTREACH_GROUPS: list[dict] = [
    {
        "key": "paper",
        "label": "Papers & Academic",
        "icon": "\U0001f4c4",
        "links": [
            {
                "name": "Zenodo",
                "url": "https://zenodo.org/records/18925678",
                "status": "published",
                "note": "DOI: 10.5281/zenodo.18925678",
            },
            {
                "name": "SSRN",
                "url": None,
                "status": "pending",
                "note": "\u5f85\u63d0\u4ea4",
            },
            {
                "name": "TechRxiv",
                "url": None,
                "status": "pending",
                "note": "\u5e73\u53f0\u8fc1\u79fb\u4e2d",
            },
            {
                "name": "arXiv",
                "url": None,
                "status": "blocked",
                "note": "\u9700\u8981 endorsement",
            },
        ],
    },
    {
        "key": "code",
        "label": "Code & Packages",
        "icon": "\U0001f4e6",
        "links": [
            {
                "name": "Official Website",
                "url": "https://robotmem.com",
                "status": "published",
                "note": "",
            },
            {
                "name": "GitHub",
                "url": "https://github.com/robotmem/robotmem",
                "status": "published",
                "note": "",
            },
            {
                "name": "PyPI",
                "url": "https://pypi.org/project/robotmem/",
                "status": "published",
                "note": "",
            },
        ],
    },
    {
        "key": "community",
        "label": "Social & Community",
        "icon": "\U0001f4ac",
        "links": [
            {
                "name": "Medium",
                "url": "https://medium.com/p/8f905e237064",
                "status": "published",
                "note": "",
            },
            {
                "name": "Hacker News",
                "url": None,
                "status": "published",
                "note": "Show HN",
            },
            {
                "name": "MuJoCo Discussions",
                "url": "https://github.com/google-deepmind/mujoco/discussions/3161",
                "status": "published",
                "note": "",
            },
            {
                "name": "Reddit r/reinforcementlearning",
                "url": None,
                "status": "published",
                "note": "",
            },
            {
                "name": "Reddit r/robotics",
                "url": None,
                "status": "planned",
                "note": "\u5f85\u53d1\u5e03",
            },
            {
                "name": "Product Hunt",
                "url": None,
                "status": "planned",
                "note": "\u8ba1\u5212\u4e2d",
            },
        ],
    },
    {
        "key": "integration",
        "label": "Framework Integration",
        "icon": "\U0001f517",
        "links": [
            {
                "name": "LeRobot Issue #3129",
                "url": "https://github.com/huggingface/lerobot/issues/3129",
                "status": "published",
                "note": "huggingface/lerobot Feature Request",
            },
        ],
    },
]


def get_outreach_data() -> list[dict]:
    """返回分组链接数据（纯配置读取，无 IO）"""
    return OUTREACH_GROUPS


def check_url_reachable(url: str, timeout: int = 5) -> bool:
    """HEAD 请求检查 URL 可达性，HEAD 失败 fallback GET"""
    for method in ("HEAD", "GET"):
        try:
            req = Request(url, method=method)
            req.add_header("User-Agent", "robotmem-outreach/0.1")
            with urlopen(req, timeout=timeout) as resp:
                return resp.status < 400
        except Exception:
            if method == "GET":
                return False
    return False


def check_all_urls(timeout: int = 5) -> list[dict]:
    """检查所有有 URL 的链接可达性"""
    results = []
    for group in OUTREACH_GROUPS:
        for link in group["links"]:
            if link.get("url"):
                reachable = check_url_reachable(link["url"], timeout)
                results.append({
                    "name": link["name"],
                    "url": link["url"],
                    "reachable": reachable,
                })
    return results
