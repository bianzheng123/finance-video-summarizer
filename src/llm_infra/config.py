"""LLM 配置层（热拔插）：模型/阶段参数 + 管线批大小/阈值，两块合并在本模块。

两块分工清晰、同属「外部配置」这一职责：

1. **模型/阶段参数**（原 llm_profiles）：三层配置解析，后层字段级覆盖前层
   - profile  —— 模型级默认（model / base_url / api_key_env / thinking /
                reasoning_effort / temperature / max_tokens / usage_map / kv_user_id）
   - stages 表 —— 按阶段前缀覆盖。键形如 "*"（全局）、"2_*"（通配）、
                "3_1"（精确），匹配优先级 精确 > 通配 > 全局；
                阶段键从 step_name / log_name 的 "N_M" 前缀提取（零侵入）
   - call 显式参数 —— call()/call_conversation() 传的 model / temperature

   Provider 中立键（reasoning_effort / thinking / temperature / model /
   max_tokens）由本模块翻译成 DeepSeek 的真实参数形态（thinking:{type}）。

   `max_tokens` = 单次响应的输出预算，**思维链 token 也算在内**（DeepSeek 的
   completion_tokens 含 reasoning_tokens）。不设时各家按自己的默认值截断
   （DeepSeek 实测 131072），思考型阶段容易出现"整个预算被 reasoning 吃光、
   正文 0 token"→ 报 `length limit was reached` 或截断 JSON 解析失败。因此
   profile 层显式拉到模型上限（v4-flash 支持 393216），让思考再长也留得下正文。
   usage_map 把各家 usage 字段归一化成日志里的三个标准键
   （reasoning_tokens / prompt_cache_hit_tokens / prompt_cache_miss_tokens），
   llm_cost_stats.py 等下游工具不变。

   配置来源（后者字段级覆盖前者）：
   - config/llm_config.json（纳入版本控制）
   - 运行时 stage_overrides_scope()（benchmark 的 --stage-config 走这里）

   环境变量 LLM_PROFILE 选择默认 profile；profile_scope() 上下文管理器可临时切换
   （线程池经 log_config 的 submit 补丁自动继承，与 user_id_scope 同模式）。

2. **管线批大小/阈值**（原 parameter_config）：管「一次调用处理多少个工作单元
   （话题/块/条目/行）」与 embedding 相似度阈值等管线参数。

   配置来源（字段级覆盖，后者优先）：
   - config/llm_config.json（纳入版本控制）

   缺失/损坏/键缺失 → 回退内置默认值 + WARNING，绝不 raise（对齐 io
   「损坏视为不存在」哲学——配置是外部状态，不该打断管线）。
"""

import contextvars
import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 一、模型/阶段参数（profile → stages 表 → call 显式参数 三层解析）
# ═══════════════════════════════════════════════════════════════════════════

_LLM_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "llm_config.json"
_VIDEO_URL_PATH = Path(__file__).parent.parent.parent / "config" / "video_url.json"

# 配置文件缺失时的兜底 profile：与历史硬编码行为完全一致，保证无配置也能跑。
_DEFAULT_PROFILE: dict[str, Any] = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "thinking": True,
    "reasoning_effort": "max",
    "temperature": 0.0,
    "max_tokens": 393216,
    "usage_map": {
        "reasoning": ["completion_tokens_details", "reasoning_tokens"],
        "cache_hit": ["prompt_cache_hit_tokens"],
        "cache_miss": ["prompt_cache_miss_tokens"],
    },
    "kv_user_id": True,
}

_PROFILE_NAME: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_profile", default=None,
)
_STAGE_OVERRIDES: contextvars.ContextVar[dict[str, dict] | None] = contextvars.ContextVar(
    "llm_stage_overrides", default=None,
)

_STAGE_RE = re.compile(r"^(\d+_\d+(?:_\d+)?)")


def _merge_into(base: dict, patch: dict) -> dict:
    """字段级合并：patch 覆盖 base 的同名字段（不深挖嵌套，保持简单）。"""
    return {**base, **patch}


@lru_cache(maxsize=1)
def _load_config_file() -> dict[str, Any]:
    """读取 ``config/llm_config.json``；缺失/损坏/非对象时记 WARNING 并返回空 dict（绝不 raise）。"""
    if not _LLM_CONFIG_PATH.exists():
        return {}
    try:
        patch = json.loads(_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("配置文件 %s 损坏，跳过（使用内置默认值）: %s", _LLM_CONFIG_PATH, e)
        return {}
    if not isinstance(patch, dict):
        logger.warning("配置文件 %s 内容不是 JSON 对象，跳过", _LLM_CONFIG_PATH)
        return {}
    return patch


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """加载模型/阶段参数（default_profile / profiles / stages）。"""
    patch = _load_config_file()
    config: dict[str, Any] = {"default_profile": "deepseek-v4-flash", "profiles": {}, "stages": {}}
    config["default_profile"] = patch.get("default_profile", config["default_profile"])
    config["profiles"] = {**config["profiles"], **patch.get("profiles", {})}
    config["stages"] = {**config["stages"], **patch.get("stages", {})}
    return config


def clear_config_cache() -> None:
    """清空配置缓存（配置文件热更新后调用）。"""
    _load_config_file.cache_clear()
    load_config.cache_clear()
    load_parameter_config.cache_clear()


def stage_key_of(step_name: str | None, log_name: str | None) -> str | None:
    """从 step_name / log_name 提取阶段键 "N_M"；提取不到返回 None。"""
    for text in (step_name, log_name):
        if text:
            m = _STAGE_RE.match(text)
            if m:
                return m.group(1)
    return None


def resolve_profile(profile_name: str | None = None) -> dict[str, Any]:
    """按名字解析 profile（merge 兜底默认值）；None 时用 LLM_PROFILE 或 default_profile。"""
    config = load_config()
    name = profile_name or current_profile_name() or config["default_profile"]
    profile = config["profiles"].get(name)
    if profile is None:
        raise ValueError(f"llm_profiles 配置中没有 profile「{name}」（可用：{list(config['profiles'])}）")
    if not isinstance(profile, dict):
        raise ValueError(f"profile「{name}」不是 JSON 对象")
    return _merge_into(_DEFAULT_PROFILE, profile)


def _match(stage: str | None, key: str) -> bool:
    """阶段键匹配：key == "*"、key == stage（精确）、key == "N_*"（通配前缀）。"""
    return key == "*" or key == stage or (
        stage is not None and key.endswith("_*") and stage.startswith(key[:-1])
    )


def resolve_stage_overrides(stage: str | None) -> dict[str, Any]:
    """按阶段键解析覆盖参数：精确 > 通配 > 全局；运行时 overrides 最后叠加（优先级最高）。"""
    merged: dict[str, Any] = {}
    for key, value in load_config()["stages"].items():
        if isinstance(value, dict) and _match(stage, key):
            merged = _merge_into(merged, value)
    for key, value in (_STAGE_OVERRIDES.get() or {}).items():
        if isinstance(value, dict) and _match(stage, key):
            merged = _merge_into(merged, value)
    return merged


def effective_config(
    stage: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """三层解析后的生效配置：_build_llm 与日志审计的唯一数据来源。

    `max_tokens` 无 call 级显式参数——需要按阶段调小时写 stages 表（或
    benchmark 的 stage_overrides_scope），profile 层给的是模型上限兜底。
    """
    profile = resolve_profile()
    overrides = resolve_stage_overrides(stage)
    eff = _merge_into(profile, overrides)
    if model:
        eff["model"] = model
    if temperature is not None:
        eff["temperature"] = temperature
    eff["profile"] = current_profile_name() or load_config()["default_profile"]
    eff["stage"] = stage
    return eff


def build_extra_body(eff: dict[str, Any], user_id: str | None) -> dict[str, Any]:
    """按 provider 把 thinking 开关翻译进 extra_body；DeepSeek 追加 user_id KV 隔离。

    `max_tokens` 也走 extra_body（而不是 ChatOpenAI 的同名字段）：langchain-openai
    会把该字段改名成 OpenAI 新版的 `max_completion_tokens` 发出，而 DeepSeek
    **接受但完全忽略**这个键（实测限 16 仍生成 598 token、finish_reason=stop），
    只认经典的 `max_tokens`。extra_body 原样进 body，键名不被改写，且优先级高于
    顶层同名参数（实测 extra_body 的值生效），故它是唯一可靠的传法。
    """
    provider = eff.get("provider", "deepseek")
    thinking = eff.get("thinking")
    extra: dict[str, Any] = {}
    if eff.get("max_tokens") is not None:
        extra["max_tokens"] = eff["max_tokens"]
    if provider == "deepseek":
        if thinking is not None:
            extra["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif isinstance(thinking, dict):
        extra.update(thinking)
    if eff.get("kv_user_id") and user_id:
        extra["user_id"] = user_id
    return extra


@contextmanager
def profile_scope(profile_name: str) -> Iterator[dict[str, Any]]:
    """with 块内把当前上下文绑定到指定 profile（含线程池继承），退出后复位。"""
    token = _PROFILE_NAME.set(profile_name)
    try:
        yield resolve_profile(profile_name)
    finally:
        _PROFILE_NAME.reset(token)


@contextmanager
def stage_overrides_scope(overrides: dict[str, dict] | None) -> Iterator[None]:
    """with 块内叠加运行时阶段覆盖（benchmark 的 --stage-config 表走这里）。"""
    token = _STAGE_OVERRIDES.set(overrides)
    try:
        yield
    finally:
        _STAGE_OVERRIDES.reset(token)


def current_profile_name() -> str | None:
    """当前上下文绑定的 profile 名；未绑定时返回 None。"""
    return _PROFILE_NAME.get()


def resolve_api_key(eff: dict[str, Any]) -> str:
    """按 profile 的 api_key_env 取 API key；缺失直接抛错（回放模式不会走到这里）。"""
    env_name = eff.get("api_key_env") or "DEEPSEEK_API_KEY"
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(
            f"profile「{eff.get('profile')}」需要环境变量 {env_name}，未设置",
        )
    return api_key


def get_usage_map() -> dict[str, Any]:
    """当前 profile 的 usage 字段映射（model 模块在响应侧读取）。"""
    return resolve_profile().get("usage_map") or _DEFAULT_PROFILE["usage_map"]


def extract_usage(usage: dict, usage_map: dict[str, Any]) -> dict[str, int]:
    """按 usage_map 从 provider 返回的 usage 中提取归一化统计量。

    usage_map 每项可以是路径列表（逐层取值），或 {"op": "minus", "lhs": [...],
    "rhs": [...]}（lhs 减 rhs，如 prompt_tokens - cached_tokens 得未命中数）。
    取不到的值记 0，不抛异常——日志审计宁可缺项也不能中断调用。
    """
    def walk(path: list[str]) -> int | None:
        node: Any = usage
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node if isinstance(node, int) else None

    def read(spec: Any) -> int:
        if isinstance(spec, list):
            return walk(spec) or 0
        if isinstance(spec, dict) and spec.get("op") == "minus":
            lhs = walk(spec.get("lhs", []))
            rhs = walk(spec.get("rhs", []))
            if lhs is None or rhs is None:
                return 0
            return max(lhs - rhs, 0)
        return 0

    return {
        "reasoning_tokens": read(usage_map.get("reasoning")),
        "prompt_cache_hit_tokens": read(usage_map.get("cache_hit")),
        "prompt_cache_miss_tokens": read(usage_map.get("cache_miss")),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 二、管线批大小/分组/embedding 阈值（config/llm_config.json 的四个 section）
# ═══════════════════════════════════════════════════════════════════════════

# 默认批大小：每步每次 LLM 调用处理的工作单元数上限（默认值单点，与配置文件同构）。
# 多轮工具对话（2_3）默认 1 = 批路径代码在但不默认生效（搜索上下文互相干扰风险）。
_DEFAULT_BATCH_SIZES: dict[str, int] = {
    "7_1_主题分析": 3,
    # 引用校验是组件（步骤 5 与 7 内联调用），键不带步骤号
    "引用校验": 8,
    # 第 2 步词级三阶段：2_3 多轮对话默认 1
    "2_1_粗筛": 4,
    "2_2_词表判定": 4,
    "2_3_联网确认": 1,
    # 第 3 步：3_1 按话题批；3_2 的工作单元是「困惑段聚类」，各单元 [DATA] 切片
    # 不同，批内混切片会破坏共享前缀不变量，故默认 1（靠单元间并行拿并发）
    "3_1_关键词与剪枝": 3,
    "3_2_话题重置": 1,
    # 4_2 关键词合并：工作单元是「候选等价对」（一对主题名 + 相似度），单元极轻，批可较大
    "4_2_关键词合并": 128,
}

# embedding 相似度门限（4_1 向量主题去重：两主题余弦相似度 ≥ 此值才作为候选等价对）。
_DEFAULT_EMBEDDING_THRESHOLD = 0.6

# 4_4 候选父 embedding range 召回：相似度严格大于此阈值的候选父才召回。
_DEFAULT_EMBEDDING_RECALL_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_parameter_config() -> dict[str, Any]:
    """加载管线批大小 / 分组 / embedding 阈值（config/llm_config.json 的四个 section）。"""
    patch = _load_config_file()
    config: dict[str, Any] = {
        "llm_batch": {},
        "llm_group_count": {},
        "embedding_dedup": {},
        "embedding_recall": {},
    }
    for section in config:
        value = patch.get(section)
        if isinstance(value, dict):
            config[section].update(value)
    return config


def load_video_urls() -> list[str]:
    """读取 ``config/video_url.json``（纯 URL 数组），返回去空后的字符串列表。"""
    if not _VIDEO_URL_PATH.exists():
        return []
    try:
        urls = json.loads(_VIDEO_URL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("URL 配置文件 %s 损坏，忽略: %s", _VIDEO_URL_PATH, e)
        return []
    if not isinstance(urls, list):
        logger.warning("URL 配置文件 %s 内容不是数组，忽略", _VIDEO_URL_PATH)
        return []
    return [str(u).strip() for u in urls if isinstance(u, str) and u.strip()]


def _clamp_int(value: Any, step_name: str, default: int) -> int:
    """把配置值解析为正整数；非法值回退默认、非正数钳制为 1。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("步骤「%s」参数配置非法：%r，回退默认 %d", step_name, value, default)
        return default
    if parsed <= 0:
        logger.warning("步骤「%s」参数配置非正数：%r，钳制为 1", step_name, value)
        return 1
    return parsed


def get_batch_size(step_name: str) -> int:
    """读 llm_batch 节的批大小（每次调用处理的工作单元数上限）。

    键缺失回退 _DEFAULT_BATCH_SIZES；未知步骤回退 1（等价于不批量）。
    """
    default = _DEFAULT_BATCH_SIZES.get(step_name, 1)
    value = load_parameter_config()["llm_batch"].get(step_name, default)
    return _clamp_int(value, step_name, default)


def get_group_count(step_name: str, default: int) -> int:
    """读 llm_group_count 节的分组并行组数（4_x 用）。

    键缺失回退调用方传入的默认值（即聚类模块的历史常量）。
    """
    value = load_parameter_config()["llm_group_count"].get(step_name, default)
    return _clamp_int(value, step_name, default)


def get_embedding_threshold() -> float:
    """读 embedding_dedup 节的相似度门限（4_1 向量主题去重）。

    两主题余弦相似度 ≥ 此值才作为候选等价对。键缺失回退
    `_DEFAULT_EMBEDDING_THRESHOLD`；非法值（非数 / 越界 [0,1]）回退默认。
    """
    value = load_parameter_config()["embedding_dedup"].get(
        "threshold", _DEFAULT_EMBEDDING_THRESHOLD
    )
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "embedding 相似度门限配置非法：%r，回退默认 %s",
            value, _DEFAULT_EMBEDDING_THRESHOLD,
        )
        return _DEFAULT_EMBEDDING_THRESHOLD
    if not 0.0 <= parsed <= 1.0:
        logger.warning(
            "embedding 相似度门限越界 [0,1]：%r，回退默认 %s",
            value, _DEFAULT_EMBEDDING_THRESHOLD,
        )
        return _DEFAULT_EMBEDDING_THRESHOLD
    return parsed


def get_embedding_recall_threshold() -> float:
    """读 embedding_recall 节的 threshold（4_4 候选父 embedding range 召回的相似度阈值）。

    相似度严格大于此值的候选父才召回。键缺失回退 `_DEFAULT_EMBEDDING_RECALL_THRESHOLD`；
    非法值（非数 / 越界 [0,1]）回退默认。
    """
    value = load_parameter_config()["embedding_recall"].get(
        "threshold", _DEFAULT_EMBEDDING_RECALL_THRESHOLD
    )
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "embedding_recall 相似度阈值配置非法：%r，回退默认 %s",
            value, _DEFAULT_EMBEDDING_RECALL_THRESHOLD,
        )
        return _DEFAULT_EMBEDDING_RECALL_THRESHOLD
    if not 0.0 <= parsed <= 1.0:
        logger.warning(
            "embedding_recall 相似度阈值越界 [0,1]：%r，回退默认 %s",
            value, _DEFAULT_EMBEDDING_RECALL_THRESHOLD,
        )
        return _DEFAULT_EMBEDDING_RECALL_THRESHOLD
    return parsed
