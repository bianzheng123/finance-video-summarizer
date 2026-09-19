"""外部服务客户端：AKShare 市场热点 + 腾讯云 embedding + 腾讯云联网搜索（WSA）。

三个客户端都是进程内**单例**（`__new__` + `_initialized`，同 LLMClient 模式）：它们
分别持有腾讯云 SDK 连接 / HTTP headers / 按天缓存目录等全局唯一资源，重复构造只会
返回同一实例，避免各阶段模块各自初始化、浪费连接与配额。

三者同一取向：外部依赖（akshare / 腾讯云 SDK / API key）缺失或网络失败时，要么
惰性 import 降级为空结果，要么对瞬时服务端错误做有限指数退避重试——确定性错误
（4xx / 鉴权失败）快速失败不重试，绝不阻塞主管线。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.wsa.v20250508 import wsa_client, models

from .resource import RESOURCE_MANAGER

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 一、AKShare 财经热点数据客户端（按天缓存 + 失败降级）
# ═══════════════════════════════════════════════════════════════════════════

_DATABASE_DIR = Path(__file__).parent.parent.parent / "database"
_CACHE_DIR = _DATABASE_DIR / "cache" / "akshare"


def _pick(record: dict, *keys: str) -> str:
    """从 record 取第一个存在且非空的键值（akshare 列名随版本会变，做防御）。"""
    for k in keys:
        v = record.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _records(ak: Any, func_names: list[str]) -> list[dict] | None:
    """依次尝试多个 akshare 接口，返回首个成功的 records 列表；全失败返回 None。

    多源降级是为了对抗单接口的反爬 / 字段变动（如某些接口被服务端直接断连），
    此时自动切下一源。每个接口失败只 warning，不抛异常。
    """
    for name in func_names:
        func = getattr(ak, name, None)
        if func is None:
            logger.debug("akshare 无接口 %s，跳过", name)
            continue
        try:
            df = func()
            if hasattr(df, "to_dict"):
                records = df.to_dict("records")
                if isinstance(records, list):
                    return [r for r in records if isinstance(r, dict)]
        except Exception as e:  # noqa: BLE001 —— 单接口失败换下一源
            logger.warning("AKShare 接口 %s 拉取失败，尝试降级：%s", name, e)
            continue
    return None


class AkshareMarketClient:
    """AKShare 市场热点客户端（新闻 + 行业板块 + 热门关键词，按天缓存，单例）。"""

    _instance: "AkshareMarketClient | None" = None
    _initialized: bool = False

    def __new__(cls) -> "AkshareMarketClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._cache_dir = _CACHE_DIR
        self._initialized = True

    # ==================== 公共接口 ====================

    def get_market_snapshot(
        self,
        news_limit: int = 10,
        board_limit: int = 100,
    ) -> dict[str, Any]:
        """一次性返回三份市场快照，按天缓存。

        Returns:
            ``{"新闻": [{"标题","摘要","发布时间"}], "行业板块": [板块名],
            "热门关键词": [关键词]}``。失败时三份均为空列表。
        """
        today = date.today().isoformat()
        cache_path = self._cache_dir / f"market_snapshot_{today}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        snapshot = self._fetch_snapshot(news_limit, board_limit)
        self._write_cache(cache_path, snapshot)
        return snapshot

    def get_market_news(self, limit: int = 10) -> list[dict]:
        """仅取当日新闻列表（2_2 的 [DATABASE] 块用）。"""
        news = self.get_market_snapshot(news_limit=limit).get("新闻")
        return news if isinstance(news, list) else []

    # ==================== 拉取 ====================

    def _fetch_snapshot(self, news_limit: int, board_limit: int) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"新闻": [], "行业板块": [], "热门关键词": []}
        try:
            import akshare as ak  # noqa: PLC0415 —— 惰性导入，依赖缺失只降级
        except Exception as e:  # noqa: BLE001 —— 依赖缺失降级为空结构
            logger.warning("akshare 未安装，跳过市场快照拉取：%s", e)
            return snapshot

        snapshot["新闻"] = self._fetch_news(ak, news_limit)
        snapshot["行业板块"] = self._fetch_industry_boards(ak, board_limit)
        snapshot["热门关键词"] = self._fetch_hot_keywords(ak)
        return snapshot

    @staticmethod
    def _fetch_news(ak: Any, limit: int) -> list[dict]:
        records = _records(ak, ["stock_info_global_em"]) or []
        news: list[dict] = []
        for r in records:
            title = _pick(r, "标题")
            if not title:
                continue
            news.append({
                "标题": title,
                "摘要": _pick(r, "摘要"),
                "发布时间": _pick(r, "发布时间"),
            })
            if len(news) >= limit:
                break
        logger.info("AKShare 新闻快照拉取 %d 条", len(news))
        return news

    @staticmethod
    def _fetch_industry_boards(ak: Any, limit: int) -> list[str]:
        records = _records(
            ak,
            ["stock_board_industry_summary_ths"],
        ) or []

        def sort_key(r: dict) -> float:
            try:
                return float(r.get("涨跌幅", 0.0))
            except (TypeError, ValueError):
                return float("-inf")

        boards: list[str] = []
        for r in sorted(records, key=sort_key, reverse=True):
            name = _pick(r, "板块名称", "板块")
            if name and name not in boards:
                boards.append(name)
            if len(boards) >= limit:
                break
        logger.info("AKShare 行业板块快照拉取 %d 个", len(boards))
        return boards

    @staticmethod
    def _fetch_hot_keywords(ak: Any) -> list[str]:
        records = _records(
            ak,
            ["stock_board_concept_summary_ths", "stock_hot_keyword_em"],
        ) or []
        keywords: list[str] = []
        for r in records:
            kw = _pick(r, "概念名称", "关键词")
            if kw and kw not in keywords:
                keywords.append(kw)
        logger.info("AKShare 热门关键词快照拉取 %d 个", len(keywords))
        return keywords

    # ==================== 缓存 ====================

    @staticmethod
    def _read_cache(cache_path: Path) -> dict[str, Any] | None:
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("市场快照缓存损坏，忽略重拉：%s", cache_path)
            return None
        if not isinstance(data, dict):
            return None
        if all(k in data for k in ("新闻", "行业板块", "热门关键词")):
            logger.info("命中当日市场快照缓存：%s", cache_path.name)
            return data
        return None

    @staticmethod
    def _write_cache(cache_path: Path, snapshot: dict[str, Any]) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_name(cache_path.name + ".tmp")
            tmp.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(tmp, cache_path)
        except Exception as e:  # noqa: BLE001 —— 缓存写失败下次重拉，不阻塞
            logger.warning("市场快照缓存落盘失败：%s", e)


# ═══════════════════════════════════════════════════════════════════════════
# 二、腾讯云 TokenHub embedding 客户端（OpenAI 兼容）
# ═══════════════════════════════════════════════════════════════════════════

# 每次请求的文本条数上限（保守值；tokenhub 接口未公开上限，20 已实测稳定）。
_BATCH_SIZE = 20
_MAX_ATTEMPTS = 4
_RETRY_INITIAL = 1.0
_RETRY_MAX = 8.0
_RETRY_JITTER = 1.0


class EmbeddingClient:
    """腾讯云 TokenHub embedding 封装（kinfra-text-embedding-4b，单例）。"""

    _URL = "https://tokenhub.tencentmaas.com/v1/embeddings"
    _MODEL = "kinfra-text-embedding-4b"
    _API_KEY_ENV = "TENCENT_LLM_API_KEY"

    _instance: "EmbeddingClient | None" = None
    _initialized: bool = False

    def __new__(cls) -> "EmbeddingClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        api_key = os.getenv(self._API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"缺失 {self._API_KEY_ENV}，无法初始化 embedding 客户端"
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._initialized = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把文本列表编码成向量列表（按输入顺序对齐）。

        内部按 `_BATCH_SIZE` 分批调用；返回的向量维度以接口为准
        （kinfra-text-embedding-4b 为 2560）。
        """
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            embeddings.extend(self._embed_batch(texts[i : i + _BATCH_SIZE]))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单批调用，对瞬时错误做有限指数退避重试；4xx 直接抛出。"""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    self._URL,
                    headers=self._headers,
                    json={"model": self._MODEL, "input": texts},
                    timeout=120,
                )
                if resp.status_code >= 400:
                    resp.raise_for_status()
                data = resp.json().get("data") or []
                data = sorted(data, key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            except requests.HTTPError as e:
                # 4xx（鉴权/参数）是确定性错误，重试不会变好，直接抛
                status = e.response.status_code if e.response is not None else None
                if status is not None and 400 <= status < 500:
                    raise
                self._sleep_or_raise(attempt, e)
            except requests.RequestException as e:
                self._sleep_or_raise(attempt, e)
        # 不可达：_sleep_or_raise 在最后一次失败时必 raise
        raise RuntimeError("embedding 调用失败（不可达分支）")

    def _sleep_or_raise(self, attempt: int, exc: Exception) -> None:
        """非致命错误：退避后重试；最后一次失败直接抛出。"""
        if attempt >= _MAX_ATTEMPTS:
            raise exc
        delay = min(_RETRY_INITIAL * (2 ** (attempt - 1)), _RETRY_MAX) + random.uniform(
            0, _RETRY_JITTER
        )
        logger.warning(
            "embedding 第 %d/%d 次重试，%.1fs 后重试：%s",
            attempt, _MAX_ATTEMPTS, delay, exc,
        )
        time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════════
# 三、腾讯云联网搜索 API（WSA）客户端
# ═══════════════════════════════════════════════════════════════════════════

# 腾讯云 WSA 偶发服务端瞬时错误（"内部服务错误，请稍后重试" 等），属可自愈，
# 做有限次指数退避重试；耗尽后再抛出，交由上层 _wsa_round 降级为空结果。
# 与 retry 的取向一致：瞬时服务端错误应"熬过去"而非直接放弃整条后处理链路。
_WSA_MAX_ATTEMPTS = 4
_WSA_RETRY_INITIAL = 2.0   # 首次退避秒数
_WSA_RETRY_MAX = 30.0      # 退避封顶
_WSA_RETRY_JITTER = 2.0    # 附加 0~2s 抖动，避免并发查询同步重试
# 视为可重试的腾讯云错误码前缀（按 e.code 前缀匹配，覆盖带子码的形式如 InternalError.xxx）。
_WSA_RETRYABLE_CODE_PREFIXES = (
    "InternalError",          # 内部服务错误，请稍后重试（本次崩溃的错误码）
    "RequestLimitExceeded",   # 请求频率超限
    "ServiceUnavailable",     # 服务暂时不可用
    "ClientNetworkError",     # SDK 网络层异常（连接/超时等）
    "InternalServerError",    # 5xx 兜底
)


def _is_retryable_tencent_error(exc: TencentCloudSDKException) -> bool:
    """判断腾讯云 SDK 异常是否为可自愈的瞬时错误（按错误码前缀匹配）。

    鉴权 / 参数非法等确定性错误（如 AuthFailure / InvalidParameter）不命中，快速失败，
    不做无谓重试。
    """
    code = getattr(exc, "code", "") or ""
    return code.startswith(_WSA_RETRYABLE_CODE_PREFIXES)


class WSASearchClient:
    """SearchPro 联网搜索封装（单例）。"""

    _instance: "WSASearchClient | None" = None
    _initialized: bool = False

    def __new__(cls) -> "WSASearchClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, region: str = "") -> None:
        """初始化 WSA 客户端。

        Args:
            region: WSA 是全局服务，默认传空字符串即可（不要传 ap-* 区域，会报 UnsupportedRegion）。
        """
        if self._initialized:
            return
        secret_id = os.getenv("TENCENT_SECRET_ID")
        secret_key = os.getenv("TENCENT_SECRET_KEY")
        if not secret_id or not secret_key:
            raise RuntimeError(
                "缺失 TENCENT_SECRET_ID / TENCENT_SECRET_KEY，无法初始化 WSA 客户端"
            )

        cred = credential.Credential(secret_id, secret_key)
        cpf = ClientProfile()
        cpf.httpProfile.reqTimeout = 60
        self._client = wsa_client.WsaClient(cred, region, cpf)
        self._initialized = True

    def search(self, query: str, mode: int | None = None) -> list[dict]:
        """对单条查询调用 SearchPro。

        Args:
            query: 查询语
            mode: 返回结果类型。None=不传该参数（默认，轻量版唯一支持的方式）；
                0=自然检索；1=多模态VR；2=混合。
                注意：轻量版（lite）不接受 Mode 参数，传任何值都会报
                `InvalidParameter illegal Mode`，必须留空；0/1/2 仅标准版 / 尊享版可用。

        Returns:
            搜索结果列表，每条 dict 含 title / url / passage / site / score / date
            （字段以 SDK 返回为准；尊享版会多一个 content 字段）。
            空查询返回空列表。
        """
        # 不自打阶段标签：WSA 是 2_3 / 3_1 / 3_2 共用的工具执行器，
        # 沿用调用方的标签才能看出这次搜索是哪个阶段发起的
        if not query or not query.strip():
            return []

        req = models.SearchProRequest()
        req.Query = query
        if mode is not None:
            req.Mode = mode

        # 全局限流：WSA 并发统一由 ResourceManager 管控，跨视频跨主播共享
        # （限制总并发，避免触发 RequestLimitExceeded，该错误虽可重试但会拖慢整条管线）。
        with RESOURCE_MANAGER.wsa():
            resp = self._search_pro_with_retry(req, query)
        pages_raw = resp.Pages or []

        results: list[dict] = []
        for raw in pages_raw:
            try:
                results.append(json.loads(raw))
            except json.JSONDecodeError as e:
                logger.warning("Pages 元素 JSON 解析失败：%s（原文：%.80s...）", e, raw)
                continue
        logger.debug("query='%s' → %d 条结果", query, len(results))
        return results

    def _search_pro_with_retry(self, req: "models.SearchProRequest", query: str) -> "models.SearchProResponse":
        """调用 SearchPro，对腾讯云瞬时错误做有限指数退避重试。

        可重试错误（见 _is_retryable_tencent_error）耗尽 _WSA_MAX_ATTEMPTS 次后抛出；
        确定性错误（鉴权 / 参数非法等）立即抛出，不重试。异常最终由上层 _wsa_round
        捕获并把该条黑话降级为空搜索结果，从而不再拖垮整条后处理链路。
        """
        for attempt in range(1, _WSA_MAX_ATTEMPTS + 1):
            try:
                return self._client.SearchPro(req)
            except TencentCloudSDKException as e:
                if not _is_retryable_tencent_error(e) or attempt >= _WSA_MAX_ATTEMPTS:
                    if _is_retryable_tencent_error(e):
                        logger.error(
                            "query='%s' 瞬时错误已重试 %d 次仍失败: %s",
                            query, _WSA_MAX_ATTEMPTS, e,
                        )
                    raise
                delay = min(
                    _WSA_RETRY_INITIAL * (2 ** (attempt - 1)), _WSA_RETRY_MAX
                ) + random.uniform(0, _WSA_RETRY_JITTER)
                logger.warning(
                    "query='%s' 腾讯云瞬时错误，第 %d/%d 次重试，%.1fs 后重试: %s",
                    query, attempt, _WSA_MAX_ATTEMPTS, delay, e,
                )
                time.sleep(delay)
