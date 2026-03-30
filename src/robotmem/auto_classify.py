"""L0 自动分类引擎 — 从自由文本推断 category/confidence/scope

纯规则实现，不依赖 LLM。供 learn MCP 工具使用。
"""

from __future__ import annotations

import json
import logging
import re

from .tag_tree import VALID_TAGS

logger = logging.getLogger(__name__)

# --- 分类规则（顺序敏感，首命中返回） ---
# 优先级: constraint > preference > worldview > tradeoff > root_cause
#          > decision > revert > pattern > architecture > config
#          > postmortem > gotcha > self_defect > observation > code(默认)
# L3b (constraint/preference/worldview) 匹配最高优先级 — 不衰减
# L3a (pattern) 用于跨场景归纳

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # L3b: 世界观 — 不衰减
    (
        "constraint",
        re.compile(
            r"(?:must\s+(?:always|never|not)|^(?:always|never)\s+\w+|"
            r"(?:必须|禁止|不[允准许]许|强制|不可|绝不|一定要))",
            re.IGNORECASE,
        ),
    ),
    (
        "preference",
        re.compile(
            r"(?:"
            r"prefer(?:s|red)?\s+\w+.*?\s+(?:over|instead\s+of|rather\s+than)\s+\w+"
            r"|prefer(?:s|red)?\s+(?:using|to\s+use|to\s+\w+)\s+\w+"
            r"|preference\s+for\s+\w+"
            r"|preferred\s+(?:approach|method|way)\s+"
            r"|recommended\s+to\s+(?:use|keep|avoid)\s+\w+"
            r"|(?:优先|偏好)\s*(?:用|使用|选择|考虑)\s*\S+"
            r"|(?:推荐|建议)使用\s*\S+"
            r"|倾向于\s*(?:用|使用)\s*\S+"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "worldview",
        re.compile(
            r"(?:"
            r"\w+\s+is\s+(?:better|worse|superior|inferior)\s+(?:than|to)\s+\w+"
            r"|the\s+(?:right|best|proper|correct|ideal)\s+(?:way|approach|practice)"
            r"|I\s+(?:usually|always|generally|typically|normally)\s+\w+"
            r"|(?:from\s+now\s+on|going\s+forward|in\s+the\s+future)\b"
            r"|[\w\u4e00-\u9fff]+\s*(?:比|>|优于)\s*[\w\u4e00-\u9fff]+\s*(?:好|强|快|稳)"
            r"|(?:更好的|最[佳好]的?)\s*(?:做法|方式|实践|方案)"
            r"|(?:我一般|我习惯|以后都|今后都)\s*\S+"
            r")",
            re.IGNORECASE,
        ),
    ),
    # L2: 战术记忆
    (
        "tradeoff",
        re.compile(
            r"(?:tradeoff|trade-off|权衡|pros?\s+and\s+cons?|"
            r"advantage|disadvantage|优[缺]点|然而|不过|(?<![不])但(?:是|却)|"
            r"\bvs\.?\b|versus|对比|相比)",
            re.IGNORECASE,
        ),
    ),
    (
        "root_cause",
        re.compile(
            r"(?:root\s*cause|caused?\s*by|because|the\s+issue\s+was|"
            r"原因|根因|导致|问题出在|之所以|是因为|由于)",
            re.IGNORECASE,
        ),
    ),
    (
        "decision",
        re.compile(
            r"(?:chose|decided|选择|决定|决策|instead\s+of|rather\s+than|"
            r"we\s+should|采用|改[为用]|不[用再])",
            re.IGNORECASE,
        ),
    ),
    (
        "revert",
        re.compile(
            r"(?:revert(?:ed)?|回滚|rollback|undo|撤销)",
            re.IGNORECASE,
        ),
    ),
    # L3a: 战略模式 — 跨场景归纳
    (
        "pattern",
        re.compile(
            r"(?:pattern|规律|every\s+time|whenever|总是|"
            r"recurring|反复出现|common\s+(?:issue|problem|pattern))",
            re.IGNORECASE,
        ),
    ),
    (
        "architecture",
        re.compile(
            r"(?:architecture|架构|模块|module|pipeline|系统设计|system\s+design|"
            r"depends\s+on|依赖|层[级次]|分层)",
            re.IGNORECASE,
        ),
    ),
    (
        "config",
        re.compile(
            r"(?:config(?:uration)?|配置|environment|env\s*var|环境变量|"
            r"设置|setting|port\s*\d|url|版本|version\s*\d)",
            re.IGNORECASE,
        ),
    ),
    # experience: 教训/踩坑/AI缺陷
    (
        "postmortem",
        re.compile(
            r"(?:postmortem|lesson\s+learn|教训|复盘|事后分析|事后复盘|回顾总结)",
            re.IGNORECASE,
        ),
    ),
    (
        "gotcha",
        re.compile(
            r"(?:gotcha|pitfall|踩坑|陷阱|坑[：:]|掉坑|踩了|踩过)",
            re.IGNORECASE,
        ),
    ),
    (
        "self_defect",
        re.compile(
            r"(?:self.defect|AI.?缺陷|训练偏[好向]|幻觉倾向|注意力衰减|"
            r"讨好倾向|过度设计倾向|生成惯性|自回归偏[好向])",
            re.IGNORECASE,
        ),
    ),
    # observation 细分 — 更具体的子类别先匹配
    (
        "observation_debug",
        re.compile(
            r"(?:found\s+that|observed|noticed|discovered|发现|观察到|注意到)"
            r".*(?:error|bug|crash|fail|timeout|exception|报错|崩溃|异常|超时|泄漏|leak)",
            re.IGNORECASE,
        ),
    ),
    (
        "observation_code",
        re.compile(
            r"(?:found\s+that|observed|noticed|discovered|发现|观察到|注意到)"
            r".*(?:\.(?:py|rs|js|ts|go|md)\b|`\w+(?:\(\))?`|函数|模块|类|文件)",
            re.IGNORECASE,
        ),
    ),
    # 通用 observation
    (
        "observation",
        re.compile(
            r"(?:"
            r"found\s+that|observed\s+that|noticed\s+that"
            r"|discovered\s+that|reveals?\s+that|(?:it\s+)?turns?\s+out"
            r"|investigation\s+show"
            r"|发现[了到]?|观察到|注意到|实际上[是为]"
            r"|实测|数据显示|测试(?:表明|显示)|确认[了到]|验证[了到]"
            r")",
            re.IGNORECASE,
        ),
    ),
]

# --- scope 提取正则 ---

_FILE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(,])(/?\w[\w./-]*\.(?:py|rs|js|ts|tsx|go|md|toml|yaml|yml|json|sql|sh|css|html))"
    r"(?=[\s\"'`),.:;]|$)"
)

_BACKTICK_ENTITY_RE = re.compile(r"`(\w[\w.]*)(?:\(\))?`")

_PASCAL_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")

_CAUSAL_RE = re.compile(
    r"(?:because|caused?\s*by|原因|根因|导致|since|due\s+to|之所以)",
    re.IGNORECASE,
)


def classify_tags(text: str, context_json: str | None = None) -> list[str]:
    """多标签分类 — 返回所有命中的 category tag

    三层分类策略:
    - Layer 1 规则: 遍历 _CATEGORY_PATTERNS 正则，返回全部命中
    - Layer 1c scenario_tags: 从 context_json 解析 scenario_tags 映射
    - source='user' 的 tag 不在此处处理（由调用方保留）

    规则匹配 0 个 tag 时返回 ["code"]（兜底）。
    """
    tags: list[str] = []

    try:
        # 遍历全部正则，收集所有命中
        for category, pattern in _CATEGORY_PATTERNS:
            if pattern.search(text):
                if category not in tags:
                    tags.append(category)

        # 从 context_json 提取 scenario_tags（如有）
        if context_json:
            try:
                ctx = (
                    json.loads(context_json)
                    if isinstance(context_json, str)
                    else context_json
                )
                if isinstance(ctx, dict):
                    scenario = ctx.get("scenario_tags")
                    if isinstance(scenario, list):
                        for st in scenario:
                            if isinstance(st, str) and st not in tags:
                                if st in VALID_TAGS:
                                    tags.append(st)
                                else:
                                    logger.debug("scenario_tag 不在受控词表: %s", st)
            except (json.JSONDecodeError, ValueError):
                pass

        # 兜底
        if not tags:
            tags.append("code")

    except Exception as e:
        logger.error("classify_tags 内部错误，已降级到 ['code']: %s", e, exc_info=True)
        return ["code"]

    return tags


def classify_category(text: str) -> str:
    """L0 规则分类 — 正则首命中返回，默认 'code'

    比 classify_tags 更快：找到第一个匹配就返回，不做全量遍历。
    """
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return "code"


def estimate_confidence(text: str, context: str = "") -> float:
    """L0 置信度评估 — 基于内容丰富度信号

    base=0.8, 每个信号 +0.05, cap=0.95
    """
    conf = 0.80

    if _FILE_PATH_RE.search(text):
        conf += 0.05

    if "`" in text or re.search(r"\b\w+\(\)", text):
        conf += 0.05

    if _CAUSAL_RE.search(text):
        conf += 0.05

    if context and len(context.strip()) > 20:
        conf += 0.05

    return min(conf, 0.95)


def extract_scope(text: str) -> dict:
    """L0 scope 提取 — 从文本自动识别文件路径、实体名、模块名

    Returns:
        {"scope_files": [...], "scope_entities": [...], "scope_modules": [...]}
    """
    scope_files = list(dict.fromkeys(_FILE_PATH_RE.findall(text)))

    entities: list[str] = []
    for m in _BACKTICK_ENTITY_RE.finditer(text):
        name = m.group(1)
        if len(name) >= 2 and not any(
            name.endswith(ext) for ext in (".py", ".js", ".rs", ".ts")
        ):
            entities.append(name)
    for m in _PASCAL_CASE_RE.finditer(text):
        name = m.group(1)
        if name not in entities:
            entities.append(name)

    modules: list[str] = []
    for fp in scope_files:
        parts = fp.strip("/").split("/")
        if len(parts) >= 2:
            mod = parts[-2]
            if mod not in modules and mod not in ("src", "lib", "app", "tests"):
                modules.append(mod)

    return {
        "scope_files": scope_files,
        "scope_entities": entities,
        "scope_modules": modules,
    }


def normalize_scope_files(
    files: list[str],
    project_root: str | None = None,
) -> list[str]:
    """去重：绝对路径 → 项目相对路径，然后 set 去重"""
    if not files:
        return []
    normalized: set[str] = set()
    root = project_root.rstrip("/") + "/" if project_root else None
    for f in files:
        if not isinstance(f, str) or not f.strip():
            continue
        path = f.strip()
        if root and path.startswith(root):
            path = path[len(root) :]
        path = path.lstrip("/")
        if path:
            normalized.add(path)
    return sorted(normalized)


def build_context_json(insight: str, context: str) -> str:
    """构建 context_json — 合并用户提供的 context + 来源标记

    如果 context 是合法 JSON 则解析合并，否则作为 user_context 字段。
    """
    result: dict = {"source": "learn_tool"}

    if context and context.strip():
        try:
            parsed = json.loads(context)
            if isinstance(parsed, dict):
                result.update(parsed)
            else:
                result["user_context"] = context
        except (json.JSONDecodeError, ValueError):
            result["user_context"] = context

    return json.dumps(result, ensure_ascii=False)
