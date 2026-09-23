"""不触网单测：词级三阶段（2_1-2_3）+ 3_1 剪枝终判的安全不变量与注释格式对齐。

覆盖本次重构最容易静默出错的几处：
  1. 词表三源合并 + hotspot 线索门控（「电梯」case：有 PCB 语境才命中）；
  2. `改字幕` 由代码从结论推导——模型无从干预这个高危决定；
  3. 坏输出降级链（越权字段忽略 / 不近音降级 / 编造正式名降级 / 漏判定补位）；
  4. 2_2 跳过条件的**词表兜底**（2_1 无词表可能漏标，词表扫描命中就必须进 2_2）；
  5. **注释格式逐字段对齐**——错了第 4 步的事件主题注入会静默失效；
  6. 3_1 增量改判重建剪枝 + 有金融实体关键词的行不被剪掉（原 2_4，已迁入第 3 步）。

用法：uv run pytest test/test_word_level_pipeline.py -q
"""

import pytest

from _common import setup_logging  # noqa: F401  —— 装配 sys.path 与 .env

from src.llm_summarize.subtitle_cleaning.common import (
    HotwordResources,
    build_row_annotations,
)
from src.llm_summarize.subtitle_cleaning.vocab_hints import (
    build_jargon_hints,
    build_jargon_map,
    gate_hotspot_alias,
    scan_jargon_in_text,
)
from src.llm_summarize.subtitle_cleaning.word_level_stages import (
    WordLevelStages,
    _hits_opinion_blacklist,
    _text_by_row,
    _validate_wrong_words,
)
from src.llm_summarize.subtitle_cleaning.word_level_step import (
    _build_vocab_suggestions,
    _confirmed_jargon,
)
from src.llm_infra.shared_state import SharedFindings

# 真实 case 的语境（钱博士 2026.8.6 行 1931）
PCB_CTX = "那个电梯PCB啊，市场上有小作文儿传他拿到大的订单了，覆铜板"
PLAIN_CTX = "我今天坐电梯上来的，电梯里人好多，等了半天"


# ==================== 1. 词表三源合并 + 门控 ====================


def test_jargon_map_includes_hotspot() -> None:
    """「电梯」必须在合并后的词表里——旧实现漏了 hotspot 源，这是本次修复的根因。"""
    jm = build_jargon_map()
    assert jm.get("电梯") == ["胜宏科技"], "hotspot 源未并入，「电梯」查不到"
    assert "电梯公司" in jm
    # stock / sector 两源不受影响
    assert jm.get("寒王") == ["寒武纪"]
    assert jm.get("达链") == ["英伟达供应链"]


def test_hotspot_clue_gating_both_directions() -> None:
    """门控双向：PCB 语境放行、日常语境拦截。

    只验证放行方向是不够的——不拦截等于用一个新错误（把真说电梯的行标成股票）
    换掉旧错误。
    """
    assert gate_hotspot_alias("电梯", PCB_CTX) is True
    assert gate_hotspot_alias("电梯", PLAIN_CTX) is False

    jm = build_jargon_map()
    assert "电梯" in [h["word"] for h in build_jargon_hints(jm, PCB_CTX)]
    assert "电梯" not in [h["word"] for h in build_jargon_hints(jm, PLAIN_CTX)]


def test_single_char_hotspot_aliases_excluded() -> None:
    """单字事件别名不参与子串匹配（防「天」命中「昨天」「等了半天」）。"""
    jm = build_jargon_map()
    for alias in ("天", "尔", "木", "兹", "伊"):
        assert alias not in jm, f"单字别名「{alias}」应被排除，否则子串匹配误伤"
    # 人工精选的 sector 单字别名保持原行为
    assert jm.get("光") == ["光通讯"]


# ==================== 2. 立场词黑名单 ====================

# 立场词黑名单：修正前后命中立场词集合不一致 → 必须丢弃（安全优先）。
# 这些词承载主讲人看多/看空、买卖、涨跌判断，改错一个字会翻转报告结论。
OPINION_FLIP_CASES: list[tuple[str, str]] = [
    # 多空
    ("看多", "看空"), ("做多", "做空"), ("多头", "空头"), ("翻多", "翻空"),
    # 买卖 / 仓位
    ("买入", "卖出"), ("加仓", "减仓"), ("增持", "减持"),
    # 涨跌
    ("看涨", "看跌"), ("上涨", "下跌"), ("涨停", "跌停"),
    # 加息降息 / 利好利空 / 突破跌破
    ("加息", "降息"), ("利好", "利空"), ("突破", "跌破"),
    # 立场翻转藏在更长的片段里
    ("看多做多", "看空做空"), ("利好出尽", "利空出尽"), ("全部买入", "全部卖出"),
    # 引入立场词（纠错却引入方向判断）与删除立场词（把方向判断改没了）
    ("看度", "看多"), ("看多", "看到"),
]

# 普通纠错：修正前后均未命中任何立场词 → 放行（不再用读音判据拦）。
NORMAL_FIXES: list[tuple[str, str]] = [
    # 本视频 workflow.log 里曾被音韵门禁误拦的真实 ASR 误识形态
    ("闪离", "闪迪"), ("国司", "国瓷"), ("双腿", "双底"), ("略入", "流入"),
    ("事件", "世界"), ("我叫", "我就"), ("陌陌", "某某"),
    # 声调差 / 同音近形
    ("通价", "铜价"), ("韩王", "寒王"), ("蓝旗", "澜起"),
    # 立场词本身不变、只纠别的字（不得误伤）
    ("新做空", "先做空"), ("新往下", "先往下"),
    # 数字 / 字母按口语读音计
    ("227年", "2027年"), ("2MB", "RMB"), ("CP", "CPI"),
    # 音节增删（吞音 / 漏字 / 复读折叠）
    ("人币", "人民币"), ("杀穿了年", "杀穿了年线"), ("对对", "对"),
    # 声母 / 韵母高频混淆
    ("回路", "回落"), ("半倒体", "半导体"), ("市盈利", "市盈率"),
    ("比录", "披露"), ("MMRCC", "MLCC"),
]


@pytest.mark.parametrize(("before", "after"), OPINION_FLIP_CASES)
def test_opinion_blacklist_blocks_flips(before: str, after: str) -> None:
    """修正前后命中立场词集合不一致 → 必须命中黑名单（降级给 2_3）。"""
    assert _hits_opinion_blacklist(before, after) is True


@pytest.mark.parametrize(("before", "after"), NORMAL_FIXES)
def test_opinion_blacklist_lets_through_normal_fixes(before: str, after: str) -> None:
    """普通纠错（修正前后均无立场词变化）→ 黑名单不拦，放行。"""
    assert _hits_opinion_blacklist(before, after) is False


# ==================== 3. 坏输出降级链 ====================


def _stages() -> WordLevelStages:
    return WordLevelStages()


# 渲染后的 [DATA] 形态：`rowID\t文本` 两列（说话人**不进 [DATA]**，见
# `subtitle_dict_to_llm_text`；步骤 4 起会多一列行号标识，两者都由第二列取文本）
SUB = "1\t那个电梯PCB啊\n2\t通价又涨"
SUSPECTS = [
    {"rowID": 1, "原先的片段": "电梯", "初判类型": "疑似黑话", "上下文线索": ["PCB"]},
    {"rowID": 2, "原先的片段": "通价", "初判类型": "疑似错别字", "上下文线索": []},
]
A2_BASES = {"数据库", "拼音候选", "上下文"}


def test_filter_wrong_word_opinion_flip_dropped() -> None:
    """2_2 错词涉及立场词翻转（通价→看空）→ 立场门禁丢弃，疑似点保守补位。"""
    out = _stages()._validate_filter(
        {"疑似点": [], "指代列表": [], "错词列表": [
            {"rowID": 2, "原先的词": "通价", "修改的词": "看空", "依据": "上下文"},
        ]},
        SUSPECTS[1:], SUB,
        wrong_bases={"数据库", "拼音候选", "上下文"},
        referent_bases={"数据库", "上下文"},
        stage="2_2",
    )
    assert out["错词列表"] == [], "立场翻转的错词应被丢弃"
    assert [s["原先的片段"] for s in out["疑似点"]] == ["通价"], "未解决的疑似点应补位保留"


def test_filter_missing_suspect_backfilled() -> None:
    """输入疑似点未获归宿 → 保守补位到疑似点，不静默丢弃。"""
    out = _stages()._validate_filter(
        {"疑似点": [], "指代列表": [], "错词列表": []},
        SUSPECTS, SUB,
        wrong_bases={"数据库", "拼音候选", "上下文"},
        referent_bases={"数据库", "上下文"},
        stage="2_2",
    )
    assert len(out["疑似点"]) == 2


def test_filter_new_suspect_allowed() -> None:
    """2_2 新增的疑似点（不在输入清单，但在 [DATA] 内）允许保留。"""
    out = _stages()._validate_filter(
        {"疑似点": [{"rowID": 1, "原先的片段": "电梯", "初判类型": "疑似黑话",
                     "违和原因": "x", "上下文线索": ["PCB"]}],
         "指代列表": [], "错词列表": []},
        [], SUB,
        wrong_bases={"数据库", "拼音候选", "上下文"},
        referent_bases={"数据库", "上下文"},
        stage="2_2",
    )
    assert len(out["疑似点"]) == 1


def test_filter_referent_anchors_after_wrong_word() -> None:
    """黑话错别字拆两列表：错词改字面（韩王→寒王），指代锚定改后词（寒王→寒武纪）。"""
    sub = "1\t韩王这个票"
    out = _stages()._validate_filter(
        {"疑似点": [], "指代列表": [
            {"rowID": 1, "原先的词": "寒王", "指代": ["寒武纪"], "依据": "数据库"},
        ], "错词列表": [
            {"rowID": 1, "原先的词": "韩王", "修改的词": "寒王", "依据": "数据库"},
        ]},
        [{"rowID": 1, "原先的片段": "韩王", "初判类型": "疑似黑话错别字", "上下文线索": []}],
        sub,
        wrong_bases={"数据库", "拼音候选", "上下文"},
        referent_bases={"数据库", "上下文"},
        stage="2_2",
    )
    assert len(out["错词列表"]) == 1
    assert len(out["指代列表"]) == 1, "指代「寒王」应能锚定改后字幕"
    assert out["指代列表"][0]["指代"] == ["寒武纪"]


def test_a1_wrong_words_validated_and_gated() -> None:
    """2_1 错词列表校验：越界/非子串/无效丢弃，立场词翻转被门禁拦下，合法错词透传。"""
    text_by_id = _text_by_row(SUB)
    raw = [
        {"rowID": 2, "原先的词": "通价", "修改的词": "铜价", "依据": "上下文"},  # 合法
        {"rowID": 99, "原先的词": "通价", "修改的词": "铜价"},                 # 越界
        {"rowID": 2, "原先的词": "不是子串", "修改的词": "铜价"},               # 非子串
        {"rowID": 2, "原先的词": "通价", "修改的词": "通价"},                   # 修改无效
    ]
    kept, blocked = _validate_wrong_words(raw, {1, 2}, text_by_id, set())
    assert len(kept) == 1
    assert kept[0]["原先的词"] == "通价"
    assert kept[0]["修改的词"] == "铜价"
    assert blocked == []

    # 立场词翻转（看多→看空）→ 立场门禁拦下：不改字面，降级为「不确定+候选」
    sub = "1\t我看多这个票"
    kept2, blocked2 = _validate_wrong_words(
        [{"rowID": 1, "原先的词": "看多", "修改的词": "看空"}],
        {1}, _text_by_row(sub), set(),
    )
    assert kept2 == []
    assert len(blocked2) == 1
    assert blocked2[0]["原先的词"] == "看多"
    assert blocked2[0]["候选"] == "看空"


# ==================== 4. 2_1 校验 ====================


def test_a1_drops_out_of_scope_and_fabricated() -> None:
    """2_1 校验：越界行号丢弃、片段非子串丢弃、脑补线索词剔除、依据不得为数据库。"""
    result = {
        "无关行号": [1, 999],          # 999 越界
        "噪音行号": [2],
        "疑似点": [
            {"rowID": 1, "原先的片段": "电梯", "初判类型": "疑似黑话",
             "违和原因": "x", "上下文线索": ["PCB", "字幕里没有的词"]},
            {"rowID": 1, "原先的片段": "不存在的片段", "初判类型": "疑似错别字",
             "违和原因": "x", "上下文线索": []},
        ],
        "指代列表": [],
    }
    out = _stages()._validate_2_1(result, [1, 2], SUB)
    assert 999 not in out["无关行号"]
    assert len(out["疑似点"]) == 1, "非子串片段应被丢弃"
    assert out["疑似点"][0]["上下文线索"] == ["PCB"], "脑补线索词应被剔除"


def test_a1_conflicting_prune_resolves_to_irrelevant() -> None:
    """同一行同时标无关与噪音 → 判为无关（不重复剪枝）。"""
    result = {"无关行号": [1], "噪音行号": [1], "疑似点": [], "指代列表": []}
    out = _stages()._validate_2_1(result, [1, 2], SUB)
    assert out["无关行号"] == [1]
    assert out["噪音行号"] == []


# ==================== 5. 注释格式对齐（第 4 步依赖）====================


def test_annotation_format_matches_downstream_contract() -> None:
    """注释字段必须与 3_1 输出逐字段对齐，否则第 4 步会静默失效。"""
    ann = build_row_annotations(
        {1: {"text": "那个电梯PCB啊"}, 2: {"text": "噪音行"}},
        [{"rowID": 1, "修正前词组": "通价", "修正后词组": "铜价"}],
        {2: "噪音"},
        [{"rowID": 1, "原先的词": "它", "指代": "胜宏科技", "依据": "上下文"}],
        [{"rowID": 1, "词": "电梯", "正式名": "胜宏科技", "类型": "股票"}],
    )
    assert set(ann[1]) == {"类型", "纠错", "指代", "关键词"}
    kw = ann[1]["关键词"][0]
    assert set(kw) == {"词", "正式名", "类型"}
    assert ann[1]["纠错"] == [{"修改前": "通价", "修改后": "铜价"}]
    # 剪枝行三列表清空
    assert ann[2]["类型"] == "噪音"
    assert ann[2]["纠错"] == [] and ann[2]["指代"] == [] and ann[2]["关键词"] == []


def test_confirmed_jargon_entry_shape() -> None:
    """已确证黑话条目形态必须能直接当注释关键词用（只收词表/联网确证的黑话映射）。"""
    referents = [
        {"rowID": 1, "原先的词": "电梯", "指代": "胜宏科技", "依据": "数据库"},
        {"rowID": 2, "原先的词": "寒王", "指代": "寒武纪", "依据": "数据库"},
        {"rowID": 3, "原先的词": "某新词", "指代": "某公司", "依据": "联网"},
        # 代词消解（依据=上下文）不是黑话，不进已确证黑话表
        {"rowID": 4, "原先的词": "它", "指代": "胜宏科技", "依据": "上下文"},
        # 词 = 指代（正式名本身），不是黑话映射
        {"rowID": 5, "原先的词": "宁德时代", "指代": "宁德时代", "依据": "数据库"},
    ]
    out = _confirmed_jargon(referents)
    by_row = {c["rowID"]: c for c in out}
    assert 4 not in by_row, "代词消解（上下文依据）不是黑话"
    assert 5 not in by_row, "词=指代（正式名本身）不是黑话映射"
    assert by_row[1]["词"] == "电梯"
    assert by_row[2]["词"] == "寒王"
    assert by_row[3]["词"] == "某新词"
    # 依据归一到「数据库」，困惑恒 false（新架构无未解析黑话）
    assert all(c["依据"] == "数据库" for c in out)
    assert all(c["困惑"] is False for c in out)


# ==================== 6. 3_1 校验（原 2_4，已迁入第 3 步）====================


def _refine_stages():
    """3_1 执行器（词表资源复用 subtitle_cleaning 的 HotwordResources）。"""
    from src.llm_summarize.step3_refine.refine_stages import RefineStages

    return RefineStages()


def test_s3_1_applies_pruning_deltas() -> None:
    """剪枝增量重建：第 2 步初判为基线，改判说明增量得出最终无关/噪音集合。"""
    result = {
        "关键词列表": [],
        "改判说明": [
            {"rowID": 1, "原判": "无关", "终判": "重点", "原因": "纠错后语义完整"},
            {"rowID": 2, "原判": "重点", "终判": "噪音", "原因": "纯语气词"},
        ],
    }
    pruned = {"无关": {1}, "噪音": set()}
    out = _refine_stages()._validate_3_1(result, [1, 2], SUB, pruned)
    assert out["无关行号"] == [], "row1 被改判为重点"
    assert out["噪音行号"] == [2], "row2 被改判为噪音"


def test_s3_1_protects_rows_with_financial_entities() -> None:
    """有金融实体关键词的行被判剪枝 → 保留为重点（宁可漏标不误标）。"""
    result = {
        "关键词列表": [{"rowID": 1, "词": "电梯", "正式名": "胜宏科技", "类型": "股票"}],
        "改判说明": [],
    }
    pruned = {"无关": {1}, "噪音": set()}
    out = _refine_stages()._validate_3_1(result, [1, 2], SUB, pruned)
    assert out["无关行号"] == [], "含金融实体的行不该被剪掉"


def test_s3_1_drops_non_substring_keywords() -> None:
    """关键词必须是该行原文子串（它是下游引用的锚点）。"""
    result = {
        "关键词列表": [
            {"rowID": 1, "词": "电梯", "正式名": "胜宏科技", "类型": "股票"},
            {"rowID": 1, "词": "字幕没有这个词", "正式名": "某公司", "类型": "股票"},
        ],
        "改判说明": [],
    }
    out = _refine_stages()._validate_3_1(
        result, [1, 2], SUB, {"无关": set(), "噪音": set()},
    )
    assert len(out["关键词列表"]) == 1
    assert out["关键词列表"][0]["词"] == "电梯"


# ==================== 7. 跳过条件的词表兜底 ====================


def test_vocab_scan_forces_a2_when_a1_silent() -> None:
    """2_1 没有词表、可能自信地把「电梯」当普通名词漏标——词表扫描必须兜住。

    这正是本次 case 的漏洞：若只凭 2_1 自报「没问题」就跳过 2_2，这类词直接漏过。
    """
    jm = build_jargon_map()
    assert scan_jargon_in_text(jm, PCB_CTX), "PCB 语境命中词表 → 必须强制进 2_2"
    assert not scan_jargon_in_text(jm, PLAIN_CTX), "日常语境不该被强制进 2_2"


# ==================== 8. 其它纯逻辑 ====================


def test_stage_checkpoints_carry_only_own_outputs(tmp_path) -> None:
    """2_2 的检查点「结果」必须是 2_2 自己的三字段产出，不含 2_3 的。

    回归：`_run_chain` 曾把 2_3 的判定覆盖回同一个 `verdicts` 变量，`_persist_chain`
    又拿它当 2_2 的「结果」落盘——同一文件「输出」与「结果」自相矛盾。新架构按
    `_merge_stage_outputs` 分别并集各阶段三字段，互不污染。
    """
    import json

    from src.llm_summarize.subtitle_cleaning.word_level_step import WordLevelStep

    merged = {
        "outputs": {
            "2_1_粗筛_b0": {"无关行号": [], "噪音行号": []},
            "2_2_词表判定_b0": {"疑似点": [], "指代列表": [], "错词列表": [
                {"rowID": 1273, "原先的词": "延伸", "修改的词": "延伸", "依据": "上下文"},
            ]},
            "2_3_联网确认_b0": {"疑似点": [], "指代列表": [
                {"rowID": 9, "原先的词": "大摩", "指代": "摩根士丹利", "依据": "联网"},
            ], "错词列表": []},
        },
        "wrong_words": [],
        "referents": [],
        "pruned": {"无关": set(), "噪音": set()},
        "suggestions": [],
    }
    WordLevelStep()._persist_chain(tmp_path, merged)

    d_2_2 = json.loads((tmp_path / "2_2_vocab_decision.json").read_text("utf-8"))
    assert len(d_2_2["结果"]["错词列表"]) == 1, "2_2 的「结果」应是 2_2 自己的错词"
    assert d_2_2["结果"]["指代列表"] == [], "2_2 的「结果」不该混入 2_3 的指代"

    d_2_3 = json.loads((tmp_path / "2_3_web_confirm.json").read_text("utf-8"))
    assert len(d_2_3["结果"]["指代列表"]) == 1, "2_3 的「结果」应是 2_3 自己的指代"
    assert d_2_3["结果"]["错词列表"] == [], "2_3 的「结果」不该混入 2_2 的错词"


def test_description_formal_name_never_becomes_jargon() -> None:
    """事件描述/修辞释义式的正式名不得进已确证黑话（否则会变成假主题）。

    回归（真实 case：钱博士 2026.8.6 行 1273「延伸管制」、行 995「鬼故事」）：
    2_3 没有「不是黑话」的出口，把 50 字事件描述填进 `正式名` 判成
    `疑似黑话未收录`；经 `confirmed_jargon` 强制注入 3_1 关键词后，第 4 步聚合出
    一个「名字是整句描述、类型是个股黑话」的主题。
    """
    from src.llm_summarize.subtitle_cleaning.word_level_step import (
        _looks_like_entity_name,
    )

    desc = "中方2025年10月9日公布的'域外延伸'出口管制措施（对境外稀土物项实施管制）"
    gloss = "吓人的利空传闻/恐慌叙事（多指不会兑现的利空故事）"
    referents = [
        {"rowID": 1273, "原先的词": "延伸管制", "指代": desc, "依据": "联网"},
        {"rowID": 995, "原先的词": "鬼故事", "指代": gloss, "依据": "联网"},
        # 正常代称必须照旧通过——门禁只拦描述，不拦真黑话
        {"rowID": 631, "原先的词": "大波波", "指代": "波兰", "依据": "联网"},
    ]
    out = _confirmed_jargon(referents)
    assert [c["rowID"] for c in out] == [631], "描述式正式名不该被收录为黑话"
    # 入库建议同一道门禁：不该拿事件描述去浪费人工审核
    sugg = _build_vocab_suggestions(referents, [], {})
    assert [s["别名"] for s in sugg] == ["大波波"]

    assert _looks_like_entity_name("特斯拉Optimus人形机器人") is True
    assert _looks_like_entity_name("航天/商业航天板块") is True, "并列实体名不该被拦"
    assert _looks_like_entity_name("韩国存储厂商（三星电子、SK海力士）") is False


def test_jargon_type_falls_back_to_name_shape() -> None:
    """词表外的板块名靠词尾判 `板块黑话`，不该一律兜底成 `个股黑话`。

    回归（真实 case：钱博士 2026.8.6「棒子→韩国半导体板块」「渣男→航天/商业航天板块」）：
    `_guess_jargon_type` 原先只查板块表，而 `疑似黑话未收录` 的正式名是模型新造的、
    **必然不在任何词表里**，于是所有新黑话无论词形一律判成 `个股黑话`。
    """
    from src.llm_summarize.subtitle_cleaning.word_level_step import _guess_jargon_type

    assert _guess_jargon_type("韩国半导体板块") == "板块黑话"
    assert _guess_jargon_type("航天/商业航天板块") == "板块黑话"
    assert _guess_jargon_type("低空经济概念") == "板块黑话"
    # 个股形态与判不出的都仍按个股黑话兜底（「绝不丢弃」优先）
    assert _guess_jargon_type("特斯拉Optimus人形机器人") == "个股黑话"
    assert _guess_jargon_type("波兰") == "个股黑话"


def test_vocab_suggestions_use_real_queries() -> None:
    """入库建议的证据搜索词取自 trace 里模型实际用过的 query。

    trace 必须用 `call_conversation` **真实返回**的形状：每个工具调用一项、
    英文键 `{"round","name","args"}`。回归：本测试原先喂的是落盘日志的中文键
    形状（`{"工具调用":[{"名称","参数"}]}`），于是测试通过而线上恒为空——
    真实视频跑完 26 条入库建议的证据搜索词全是 `[]`。
    """
    referents = [{
        "rowID": 1, "原先的词": "某新词", "指代": "某公司", "依据": "联网",
    }]
    trace = [
        {"round": 1, "name": "web_search", "args": {"query": "某新词 A股"}, "result_chars": 12},
        {"round": 2, "name": "web_search", "args": {"query": "某新词 简称"}, "result_chars": 8},
        # 重复 query 去重
        {"round": 2, "name": "web_search", "args": {"query": "某新词 A股"}, "result_chars": 8},
    ]
    out = _build_vocab_suggestions(referents, trace, build_jargon_map())
    assert len(out) == 1
    assert out[0]["证据搜索词"] == ["某新词 A股", "某新词 简称"]
    assert out[0]["别名"] == "某新词"


def test_vocab_suggestions_dedup_by_alias() -> None:
    """同一别名跨行重复只保留一条——26 条建议里 12 条是同一个「渣男」，人工审核要看去重后的。"""
    referents = [
        {"rowID": r, "原先的词": "渣男", "指代": "商业航天", "依据": "联网"}
        for r in (1841, 1842, 1843)
    ] + [
        {"rowID": 631, "原先的词": "大波波", "指代": "波兰", "依据": "联网"},
    ]
    out = _build_vocab_suggestions(referents, [], build_jargon_map())
    aliases = [o["别名"] for o in out]
    assert aliases == ["渣男", "大波波"], f"应按别名去重并保序，实际 {aliases}"
    assert out[0]["出现行号"] == [1841, 1842, 1843], "去重后要保留全部出现行号供人工定位"


def test_persist_chain_prefixes_match_producers(tmp_path) -> None:
    """落盘的前缀过滤必须与生产侧 log_name 前缀一致。

    回归：A1-A3 改名成 2_1-2_3 后，`_persist_chain` 仍按 `A1_`/`A2_`/`A3_` 过滤，
    三个检查点的「输出」节全部静默写成空 dict——不报错、不告警，
    只有去翻落盘文件才发现 LLM 输出没留痕（违反「LLM 调用必落盘」）。
    """
    import json

    from src.llm_summarize.subtitle_cleaning.word_level_step import WordLevelStep

    step = WordLevelStep()
    merged = {
        # 键名与生产侧 outputs[f"2_1_粗筛_b{i}"] 等严格同源
        "outputs": {
            "2_1_粗筛_b0": {"x": 1},
            "2_2_词表判定_b0": {"y": 2},
            "2_3_联网确认_b0": {"z": 3},
        },
        # 每个检查点的「结果」只放该阶段自己的三字段产出
        "wrong_words": [],
        "referents": [],
        "pruned": {"无关": {7}, "噪音": set()},
        "suggestions": [],
    }
    step._persist_chain(tmp_path, merged)

    for fname, key in (
        ("2_1_coarse_screen.json", "2_1_粗筛_b0"),
        ("2_2_vocab_decision.json", "2_2_词表判定_b0"),
        ("2_3_web_confirm.json", "2_3_联网确认_b0"),
    ):
        data = json.loads((tmp_path / fname).read_text(encoding="utf-8"))
        assert data["输出"], f"{fname} 的「输出」节为空——前缀过滤与生产侧不一致"
        assert key in data["输出"], f"{fname} 缺少 {key}"


def test_shared_findings_scope_and_replay(monkeypatch) -> None:
    """共享发现：confirmed_map 按范围过滤；回放模式读冻结集保确定性。"""
    sf = SharedFindings()
    sf.add_many([
        {"rowID": 1, "词": "电梯", "正式名": "胜宏科技", "类型": "个股黑话"},
        {"rowID": 99, "词": "达链", "正式名": "英伟达供应链", "类型": "板块黑话"},
    ])
    scoped = sf.confirmed_map({1})
    assert scoped == {"电梯": ["胜宏科技"]}

    frozen = sf.freeze()
    sf2 = SharedFindings()
    sf2.load_frozen(frozen)
    monkeypatch.setenv("LLM_REPLAY", "1")
    sf2.add_many([{"rowID": 5, "词": "新加的", "正式名": "X", "类型": "股票"}])
    assert 5 not in sf2.snapshot(), "回放模式应读冻结集，忽略运行时新增"


def test_data_slice_carries_row_type_column() -> None:
    """步骤 4 起：[DATA] 只含范围内行，且**每行自带行号标识**（第三列）。

    剪枝事实随行走取代了 `[ANNOTATION]` 的剪枝行号数组——旧实现两处都写，靠
    `scope_prunes` 补丁把数组收敛到切片范围内（否则会标出切片里不存在的行号，
    模型去找找不到）。行号标识列让这个问题从根上消失：行在切片里，它的标识就在；
    行不在，标识也不在，不可能不一致。故 `render_annotations` 不再输出剪枝数组。
    """
    import json

    from src.llm_summarize.schemas import SubtitleRow
    from src.llm_summarize.utils.llm_text_utils import (
        render_annotation_dict,
        render_annotations,
        render_scoped_subtitle,
    )
    from src.llm_summarize.utils.rowid_scope import scope_id_set

    subs = []
    for rid in range(1, 11):
        ann = {"类型": "重点", "指代": [], "关键词": []}
        if rid == 3:
            ann = {"类型": "无关", "指代": [], "关键词": []}   # 范围外剪枝行
        if rid == 7:
            ann = {"类型": "噪音", "指代": [], "关键词": []}   # 范围内剪枝行
        if rid == 5:
            ann = {"类型": "重点", "指代": [], "关键词": [
                {"词": "电梯", "正式名": "胜宏科技", "类型": "个股黑话",
                 "困惑": False, "困惑原因": ""}]}
        subs.append(SubtitleRow(rowID=rid, text=f"第{rid}行", annotation=ann))

    scope = scope_id_set([4, 5, 6, 7, 8])

    sliced = render_scoped_subtitle(subs, row_scope=scope, with_type=True)
    rows = [line.split("\t") for line in sliced.splitlines()]
    assert [int(r[0]) for r in rows] == [4, 5, 6, 7, 8]
    # 第三列即行号标识；范围内的噪音行（7）自带标识，无需注释块再说一遍
    assert {int(r[0]): r[2] for r in rows} == {
        4: "重点", 5: "重点", 6: "重点", 7: "噪音", 8: "重点",
    }
    # 两列模式（步骤 1-3）不带标识列
    assert all(
        len(line.split("\t")) == 2
        for line in render_scoped_subtitle(subs, row_scope=scope).splitlines()
    )

    ann_txt = render_annotations(subs, include_keywords=True, row_scope=scope)
    parsed = json.loads(ann_txt)
    assert "无关行号" not in parsed and "噪音行号" not in parsed, (
        "步骤 4 起剪枝行号只由 [DATA] 第三列携带，注释块不得重复给（双写必然打架）"
    )
    assert "胜宏科技" in ann_txt, "范围内的黑话映射须保留（靠注释逐行传递）"

    # 第 2 步内部仍需靠注释块传剪枝初判（那时 [DATA] 只有两列）
    full_ann = json.loads(render_annotation_dict(
        {s.rowID: s.annotation for s in subs},
    ))
    assert full_ann["无关行号"] == [3] and full_ann["噪音行号"] == [7]


def test_text_by_row_parsing() -> None:
    """[DATA] 行文本反解：**第二列**为文本，首列为 rowID。

    旧版本取 `parts[-1]`，并拿一条五列的字幕文件行（start/end/speaker/text）当输入
    ——那不是 `[DATA]` 的形态，`render_full_subtitle` 从来只产出两列。取末列在
    三列 `[DATA]`（步骤 4 起带行号标识）下会解出 `重点`/`无关`，使所有"片段是否为
    原文子串"的校验全数失败，故改为固定取第二列，并按两种真实形态断言。
    """
    two_col = "1\t那个电梯PCB啊\n2\t通价又涨"
    assert _text_by_row(two_col) == {1: "那个电梯PCB啊", 2: "通价又涨"}

    three_col = "1\t那个电梯PCB啊\t重点\n2\t谢谢老铁的礼物\t无关"
    assert _text_by_row(three_col) == {1: "那个电梯PCB啊", 2: "谢谢老铁的礼物"}
