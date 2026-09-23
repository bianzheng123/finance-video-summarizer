"""拼音匹配工具：用于字幕近音误识纠错。

提供拼音转换、距离计算、候选查找等功能。
LLM 纠错前用 _find_suspect_candidates 扫出可疑词清单注入 prompt。

本模块的音韵判据只服务于**候选提示**，别混用：

- `_is_exact_homophone` / `_find_suspect_candidates`（`max_score=0`）：严格同音。
- `_weighted_pinyin_distance`（`scan_row_candidates` 的放宽档）：容许一个字的混淆音，
  只用于 2_1 产出 `[ANNOTATION]` 拼音候选**提示**（有 prompt 反锚定规则复核）。

2_2/2_3 的改字面门禁已改为**立场词黑名单**（`word_level_stages._hits_opinion_blacklist`），
不再使用读音判据。
"""

from pypinyin import pinyin, Style

# 声母列表，按长度倒序便于最长前缀匹配
_INITIALS: tuple[str, ...] = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "r", "z", "c", "s", "y", "w",
)


def _split_syllable(syllable_with_tone: str) -> tuple[str, str, int]:
    """把 pypinyin TONE3 风格音节（如 'lan2'、'an1'）拆成 (声母, 韵母, 声调)。"""
    if syllable_with_tone and syllable_with_tone[-1].isdigit():
        tone = int(syllable_with_tone[-1])
        body = syllable_with_tone[:-1]
    else:
        tone = 0
        body = syllable_with_tone
    for ini in _INITIALS:
        if body.startswith(ini):
            return ini, body[len(ini):], tone
    return "", body, tone


def _to_pinyin_tuple(word: str) -> tuple[tuple[str, str, int], ...]:
    """词 → 每字 (声母, 韵母, 声调) 的元组。"""
    syllables = pinyin(word, style=Style.TONE3, heteronym=False)
    result: list[tuple[str, str, int]] = []
    for syll_list in syllables:
        if not syll_list:
            result.append(("", "", 0))
            continue
        result.append(_split_syllable(syll_list[0]))
    return tuple(result)


def _pinyin_display(word: str) -> str:
    """词 → 带声调标记的拼音字符串，例 '澜起' → 'lán qǐ'。"""
    syllables = pinyin(word, style=Style.TONE, heteronym=False)
    return " ".join(s[0] for s in syllables if s)


def row_reading(text: str) -> str:
    """整行文本的带调拼音（供 [ANNOTATION] 的 `行读音`）。

    给模型整行而非单个片段，是为了让它能**重新切分词边界**——上一阶段圈出的
    片段边界本身可能就是错的（「小某书」被切成「小卯叔」的一部分）。
    汉字段走 `heteronym=False` 取上下文音，与 `_readings_in_context` 同口径。
    """
    return _pinyin_display(text)


def _pinyin_distance(
    pa: tuple[tuple[str, str, int], ...],
    pb: tuple[tuple[str, str, int], ...],
    tone_sensitive: bool = False,
) -> int | None:
    """两个拼音元组的编辑距离（仅同长度）。

    每字声母不同 +1，韵母不同 +1，声调不同 +1（仅当 tone_sensitive=True）。
    长度不同返回 None（不跨长度匹配）。
    """
    if len(pa) != len(pb):
        return None
    score = 0
    for (ia, fa, ta), (ib, fb, tb) in zip(pa, pb):
        if ia != ib:
            score += 1
        if fa != fb:
            score += 1
        if tone_sensitive and ta != tb:
            score += 1
    return score


def _is_exact_homophone(
    pa: tuple[tuple[str, str, int], ...],
    pb: tuple[tuple[str, str, int], ...],
) -> bool:
    """完全同音（同长度、声母韵母全同，忽略声调）。"""
    if len(pa) != len(pb):
        return False
    for (ia, fa, _), (ib, fb, _) in zip(pa, pb):
        if ia != ib or fa != fb:
            return False
    return True


# ==================== 混淆读音判据（供候选放宽档 _weighted_pinyin_distance 复用）====================
#
# 与上面的严格同音判据分开：这里问的不是"是否同音"，而是"ASR 可能这么错吗"。
# 三个缺陷驱动了这套实现（旧的等长+逐字全等判据实测全踩）：
# 1. pypinyin 把数字/字母整块原样返回（'227' → ('','227')），永远配不上汉字读音
# 2. 等长硬约束让音节增删一律不通过（人币→人民币、杀穿了年→杀穿了年线）
# 3. 无混淆容差（xin/xian、lu/luo、zi/zhe 都被拒）

# 数字 → 口语读音（含变读：1 可念 yāo、2 可念 liǎng、0 可念 ō）
_DIGIT_READINGS: dict[str, tuple[str, ...]] = {
    "0": ("ling", "o"), "1": ("yi", "yao"), "2": ("er", "liang"),
    "3": ("san",), "4": ("si",), "5": ("wu",), "6": ("liu",),
    "7": ("qi",), "8": ("ba",), "9": ("jiu",),
}
# 拉丁字母 → 中文使用者念法（近似，按单伪音节处理）。
# ASR 把字母听成汉字/别的字母极常见：2MB→RMB 靠 2=èr ↔ R=ar 成立。
_LETTER_READINGS: dict[str, tuple[str, ...]] = {
    "a": ("ei", "a"), "b": ("bi",), "c": ("xi", "sei"), "d": ("di",),
    "e": ("yi",), "f": ("ef",), "g": ("ji",), "h": ("ei", "eichi"),
    "i": ("ai",), "j": ("jei",), "k": ("kei",), "l": ("el",),
    "m": ("em",), "n": ("en",), "o": ("ou", "ling"), "p": ("pi",),
    "q": ("kiu",), "r": ("er", "ar"), "s": ("es",), "t": ("ti",),
    "u": ("you",), "v": ("wei",), "w": ("dabliu",), "x": ("eks",),
    "y": ("wai",), "z": ("zi", "zei"),
}
# 声母混淆对：平翘舌、鼻边音、送气与否、零声母等 ASR 高频混淆。
# 含送气对 b/p、d/t、g/k：实测「比录」→「披露」（bǐ lù → pī lù）是真实误识形态，
# 排除它们会把这类正确判定误拦。担心的「突破」→「跌破」并不靠声母放行——
# 它被**韵母**拦住（突 tū 的 u 与 跌 diē 的 ie 不是混淆对），前提是汉字读音走
# 上下文音（见 `_readings_in_context`）。
#
# 塞擦音 ↔ 塞音（ch/t、zh/d、sh/t、q/t、j/d）：实测「生成」→「昇腾」
# （shēng chéng → shēng téng）是真实误识形态。注意本组**天然对称**——
# frozenset 无序，放行 `ch → t` 必然同时放行 `t → ch`（「电梯」→「电池」）。
# 这个副作用由**词表门控**消化，不在音韵层解决：`scan_row_candidates` 对已命中
# jargon_map 的片段不产候选，故 PCB 语境下的「电梯」（＝胜宏科技的词表别名）
# 不会被建议改成「电池」。音韵判据只回答"读音上像不像"，不该承担"它是不是黑话"。
_FUZZY_INITIALS: frozenset[frozenset[str]] = frozenset({
    frozenset({"z", "zh"}), frozenset({"c", "ch"}), frozenset({"s", "sh"}),
    frozenset({"n", "l"}), frozenset({"f", "h"}), frozenset({"r", "l"}),
    frozenset({"j", "zh"}), frozenset({"q", "ch"}), frozenset({"x", "s"}),
    frozenset({"r", "y"}), frozenset({"", "y"}), frozenset({"", "w"}),
    frozenset({"b", "p"}), frozenset({"d", "t"}), frozenset({"g", "k"}),
    frozenset({"ch", "t"}), frozenset({"zh", "d"}), frozenset({"sh", "t"}),
    frozenset({"q", "t"}), frozenset({"j", "d"}),
})
# 韵母混淆对：前后鼻音、介音脱落、复元音单化等
_FUZZY_FINALS: frozenset[frozenset[str]] = frozenset({
    frozenset({"an", "ang"}), frozenset({"en", "eng"}), frozenset({"in", "ing"}),
    frozenset({"ian", "iang"}), frozenset({"uan", "uang"}), frozenset({"ong", "eng"}),
    frozenset({"iong", "ong"}), frozenset({"u", "uo"}), frozenset({"u", "ou"}),
    frozenset({"o", "uo"}), frozenset({"iu", "ou"}), frozenset({"ui", "uei"}),
    frozenset({"i", "ie"}), frozenset({"ian", "iao"}), frozenset({"an", "ai"}),
    frozenset({"in", "ian"}), frozenset({"ing", "iang"}), frozenset({"er", "ar"}),
    frozenset({"e", "er"}), frozenset({"ei", "e"}), frozenset({"ai", "ei"}),
    frozenset({"ao", "ou"}), frozenset({"v", "u"}), frozenset({"ve", "ue"}),
    frozenset({"i", "v"}), frozenset({"ia", "a"}),
    # 字母名之间的混淆（M/L/N/R 的呼读音只差韵尾，ASR 极易听混：MMRCC → MLCC）
    frozenset({"em", "el"}), frozenset({"em", "en"}), frozenset({"el", "en"}),
    frozenset({"er", "el"}), frozenset({"ar", "el"}),
})
# 舌尖元音 [ɨ] 与 e 的混淆只在这些声母下成立（自 zì ↔ 这 zhè）。
# 不设声母条件会让 -i 与 -e 在任意声母下互通，容差过宽。
_APICAL_INITIALS: frozenset[str] = frozenset({"z", "c", "s", "zh", "ch", "sh", "r"})

_FUZZY_COST = 0.25      # 混淆音对的单维代价
_UNRELATED_COST = 2.0   # 任一维完全不同即视为无关（语义改写），禁止该转移


def _is_phonetic_token(ch: str) -> bool:
    """该字符是否参与读音比较（汉字 / 数字 / 拉丁字母；标点空格不参与）。"""
    return ch.isdigit() or (ch.isascii() and ch.isalpha()) or "一" <= ch <= "鿿"


def _phonetic_tokens(text: str) -> list[str]:
    return [c for c in text if _is_phonetic_token(c)]


def _readings_in_context(text: str) -> list[set[tuple[str, str]]]:
    """逐 token 出可能读音集合 `{(声母, 韵母)}`，声调一律忽略。

    汉字段整体走 `heteronym=False` 取**上下文读音**——这一点是必须的：
    `heteronym=True` 会带出生僻异读（「跌」有 tú），使「突破」→「跌破」
    这种语义改写被判成同音而漏过。
    """
    tokens = _phonetic_tokens(text)
    han_chars = [c for c in tokens if not (c.isdigit() or c.isascii())]
    han_readings: list[str] = []
    if han_chars:
        han_readings = [
            s[0] for s in pinyin("".join(han_chars), style=Style.TONE3, heteronym=False)
        ]

    result: list[set[tuple[str, str]]] = []
    han_idx = 0
    for ch in tokens:
        if ch.isdigit():
            result.append({_split_syllable(r)[:2] for r in _DIGIT_READINGS[ch]})
        elif ch.isascii():
            result.append({_split_syllable(r)[:2] for r in _LETTER_READINGS[ch.lower()]})
        else:
            syll = han_readings[han_idx] if han_idx < len(han_readings) else ""
            han_idx += 1
            result.append({_split_syllable(syll)[:2]})
    return result


def _token_sub_cost(
    ra: set[tuple[str, str]], rb: set[tuple[str, str]],
) -> float:
    """两 token 读音集合的最小替换代价：0 同音 / 0.5 双维混淆 / ≥2.0 无关。"""
    best = _UNRELATED_COST
    for ia, fa in ra:
        for ib, fb in rb:
            if ia == ib and fa == fb:
                return 0.0
            if ia == ib:
                ini_cost = 0.0
            elif frozenset({ia, ib}) in _FUZZY_INITIALS:
                ini_cost = _FUZZY_COST
            else:
                ini_cost = 1.0
            if fa == fb:
                fin_cost = 0.0
            elif frozenset({fa, fb}) in _FUZZY_FINALS:
                fin_cost = _FUZZY_COST
            elif (frozenset({fa, fb}) == frozenset({"i", "e"})
                    and ia in _APICAL_INITIALS and ib in _APICAL_INITIALS):
                fin_cost = _FUZZY_COST
            else:
                fin_cost = 1.0
            # 任一维完全不同 → 无关：不是听错，是换了个词
            cost = (_UNRELATED_COST if 1.0 in (ini_cost, fin_cost)
                    else ini_cost + fin_cost)
            best = min(best, cost)
    return best


def _build_hotword_pinyin_index(
    hotwords_by_weight: dict,
    jargon_map: dict,
    min_weight: int = 100,
    min_word_len: int = 2,
    max_word_len: int = 4,
) -> dict[int, list[tuple[str, tuple, str, int]]]:
    """预构建热词拼音索引：{word_len: [(word, pinyin_tuple, formal_name, weight)]}。

    Args:
        hotwords_by_weight: hotword_stock_enhance.json 的内容，{"100": [...], "11": [...]}。
        jargon_map: 黑话→正式名映射，{"澜起": ["澜起科技"], ...}。formal 取列表第一项。
        min_weight: 只索引权重 >= min_weight 的热词。
        min_word_len/max_word_len: 词长度过滤。
    """
    index: dict[int, list[tuple[str, tuple, str, int]]] = {}
    for weight_str, words in hotwords_by_weight.items():
        try:
            weight = int(weight_str)
        except (ValueError, TypeError):
            continue
        if weight < min_weight or not isinstance(words, list):
            continue
        for w in words:
            if not isinstance(w, str) or not w:
                continue
            wl = len(w)
            if wl < min_word_len or wl > max_word_len:
                continue
            formal_list = jargon_map.get(w)
            formal = formal_list[0] if formal_list else w
            index.setdefault(wl, []).append((w, _to_pinyin_tuple(w), formal, weight))
    return index


def _is_all_han(token: str) -> bool:
    return all("一" <= c <= "鿿" for c in token)


# 候选放宽档的代价上限：等长片段逐字比，允许总代价 ≤ 该值。
# 0.5 = 恰好容纳**一个**字的双维混淆（声母 0.25 + 韵母 0.25），
# 或两个字各一维混淆。任一字出现"无关"维（代价 1.0）即整体超限被拒。
_CANDIDATE_FUZZY_BUDGET = 0.5


def _weighted_pinyin_distance(a: str, b: str) -> float | None:
    """等长片段的**分维度加权**读音距离（同音 0 / 混淆 0.25 每维 / 无关 1.0 每维）。

    与 `_pinyin_distance` 的关键差别：后者**无差别计数**，只要有一维不同就 +1，
    于是 `生成→昇腾`（ch/t 是真混淆对）与 `深宏→神华`（ong/ua 毫无关系）距离都是 1，
    抬 `max_score` 会把两者一并放进来。本函数复用 `_token_sub_cost`，
    让"真混淆"（0.25/维）与"无关"（1.0/维）拉开量级，
    放宽覆盖面的同时不放进垃圾。

    只处理等长片段（候选滑窗本就等长）；长度不等返回 None。
    """
    ra, rb = _readings_in_context(a), _readings_in_context(b)
    if len(ra) != len(rb) or not ra:
        return None
    total = 0.0
    for x, y in zip(ra, rb):
        cost = _token_sub_cost(x, y)
        if cost >= _UNRELATED_COST:
            return None  # 任一字读音无关 → 不是候选
        total += cost
    return total


def _find_suspect_candidates(
    line_text: str,
    hotword_index: dict[int, list[tuple[str, tuple, str, int]]],
    max_per_line: int = 3,
    max_score: int = 0,
    fuzzy: bool = False,
) -> list[dict]:
    """一行字幕里用滑窗找同音 / 音近的热词候选。

    两档判据（**不要混用**）：
    - `fuzzy=False`（默认，严格）：`max_score=0` 即声母+韵母全同、仅容许声调不同，
      判据必须严格。
    - `fuzzy=True`（放宽）：走 `_weighted_pinyin_distance` 的分维度加权判据，
      容许**一个字的混淆音**（如「生成」→「昇腾」的 ch/t）。只给
      **有 LLM 复核**的路径用（`scan_row_candidates` → `[ANNOTATION]` 提示）。
      注意此时 `max_score` 参数被忽略——加权判据自带预算 `_CANDIDATE_FUZZY_BUDGET`。

    跳过本身就是热词的滑窗（已经匹配的不再当作可疑）。每行最多 max_per_line 条。
    """
    if not line_text or not hotword_index:
        return []

    hotword_set: set[str] = set()
    for entries in hotword_index.values():
        for w, _, _, _ in entries:
            hotword_set.add(w)

    candidates: list[dict] = []
    seen_originals: set[str] = set()

    for win_len in sorted(hotword_index.keys()):
        entries = hotword_index[win_len]
        if not entries:
            continue
        text_len = len(line_text)
        for i in range(text_len - win_len + 1):
            token = line_text[i:i + win_len]
            if token in seen_originals or token in hotword_set:
                continue
            if not _is_all_han(token):
                continue
            token_py = _to_pinyin_tuple(token)
            best: tuple | None = None  # (代价, tone_diff, suspect, formal)
            for hot_word, hot_py, formal, _ in entries:
                if fuzzy:
                    wd = _weighted_pinyin_distance(token, hot_word)
                    if wd is None or wd > _CANDIDATE_FUZZY_BUDGET:
                        continue
                    # 同音者代价 0，优先于混淆者；同代价再按声调、字面排序保确定性
                    cand = (wd, 0 if wd == 0 else 1, hot_word, formal)
                    if best is None or cand < best:
                        best = cand
                    continue
                d = _pinyin_distance(token_py, hot_py, tone_sensitive=False)
                if d is None or d > max_score:
                    continue
                tone_d = _pinyin_distance(token_py, hot_py, tone_sensitive=True)
                tone_diff = 0 if tone_d == d else 1
                cand = (0.0, tone_diff, hot_word, formal)
                if best is None or cand < best:
                    best = cand
            if best is not None:
                cost, tone_diff, suspect, formal = best
                candidates.append({
                    "original": token,
                    "suspect": suspect,
                    "formal": formal,
                    "score": cost,
                    "tone_match": tone_diff == 0,
                })
                seen_originals.add(token)
                if len(candidates) >= max_per_line:
                    return candidates
    return candidates


def scan_row_candidates(
    text_by_id: dict[int, str],
    hotword_index: dict[int, list[tuple[str, tuple, str, int]]],
    truncation_index: dict[str, tuple[str, str]] | None = None,
    max_per_line: int = 3,
    max_score: int = 0,
    include_formal: bool = False,
    fuzzy: bool = True,
    jargon_aliases: frozenset[str] | set[str] | None = None,
) -> dict[int, list[dict]]:
    """批量扫多行的拼音候选，产出 `[ANNOTATION]` 的 `拼音候选` 结构。

    与 `_find_suspect_candidates` 的关系：本函数是它的多行封装 + 注释形态整形。

    **判据是放宽档**（`fuzzy=True`，走 `_weighted_pinyin_distance`）：容许一个字的
    混淆音，故「生成」→「昇腾」（ch/t）能进候选。这一档只在此处使用——它的输出是
    给 LLM 的**提示**，有 prompt 反锚定规则 + 2_2 词表判定 + 立场词黑名单写盘
    门禁三重复核。无 LLM 复核的自动替换路径（预清洗、兜底替换）仍走严格同音档。

    **`jargon_aliases` 门控（放宽档的必要配套）**：片段本身已命中黑话别名表时
    **不为它产候选**。原因是 ch↔t 这类混淆对天然对称，放行「生成」→「昇腾」必然
    同时放行「电梯」→「电池」，而 PCB 语境下的「电梯」是词表确认的胜宏科技别名，
    建议它改成「电池」是直接对抗线索门控。这个冲突在音韵层无解（两者是同一变换），
    只能靠词表消化：**词表说它对，就不要拿读音去质疑它**。

    Args:
        text_by_id: `{rowID: 行文本}`
        include_formal: 是否带上正式名（词表知识）。**2_1 必须传 False**——
            它的定位就是"不给词表"，带正式名会废掉 2_1/2_2 的分工。
        fuzzy: 是否走放宽判据。置 False 退回严格同音（仅供对照/回归用）。
        jargon_aliases: 黑话别名集合（通常传 `HotwordResources.jargon_map` 的键）。
            命中者不产候选。为 None 时不做门控——**放宽档下不建议**。

    Returns:
        `{rowID: [{"片段", "读音", "音近": [...], "正式名"?}]}`。同一片段在同一行
        内只出一条（`_find_suspect_candidates` 已按 original 去重）。
    """
    aliases = jargon_aliases or frozenset()
    out: dict[int, list[dict]] = {}
    for row_id, text in text_by_id.items():
        if not text:
            continue
        raw = _find_suspect_candidates(
            text, hotword_index, max_per_line=max_per_line, max_score=max_score,
            fuzzy=fuzzy,
        )
        if truncation_index:
            raw = raw + _find_truncation_candidates(text, truncation_index)
        entries: list[dict] = []
        seen: set[str] = set()
        for c in raw:
            frag = c["original"]
            if frag in seen:
                continue
            # 词表说这个字面是对的 → 不拿读音去质疑它（见 docstring 的门控说明）
            if frag in aliases:
                continue
            seen.add(frag)
            entry: dict = {
                "片段": frag,
                "读音": _pinyin_display(frag),
                "音近": [c["suspect"]],
            }
            if include_formal and c.get("formal") and c["formal"] != c["suspect"]:
                entry["正式名"] = c["formal"]
            entries.append(entry)
        if entries:
            out[row_id] = entries
    return out


def _build_truncation_index(
    hotwords_by_weight: dict,
    jargon_map: dict,
    min_weight: int = 100,
    min_word_len: int = 3,
) -> dict[str, tuple[str, str]]:
    """构建吞首字索引：{去掉第一个字的形式: (完整热词, 正式名)}。

    只对字数 >= min_word_len 的热词建索引，避免 2 字热词去掉一字变单字误伤。
    """
    index: dict[str, tuple[str, str]] = {}
    for weight_str, words in hotwords_by_weight.items():
        try:
            weight = int(weight_str)
        except (ValueError, TypeError):
            continue
        if weight < min_weight or not isinstance(words, list):
            continue
        for w in words:
            if not isinstance(w, str) or len(w) < min_word_len:
                continue
            truncated = w[1:]
            if truncated in index:
                continue
            formal_list = jargon_map.get(w)
            index[truncated] = (w, formal_list[0] if formal_list else w)
    return index


def _find_truncation_candidates(
    line_text: str,
    truncation_index: dict[str, tuple[str, str]],
) -> list[dict]:
    """在一行字幕里直接字符串匹配，找出可能是吞首字音节的片段。

    Returns:
        候选列表，每条 {"original": ..., "suspect": ..., "formal": ..., "type": "truncation"}
    """
    if not line_text or not truncation_index:
        return []

    candidates: list[dict] = []
    seen: set[str] = set()
    for truncated, (full_word, formal) in truncation_index.items():
        if truncated in line_text and truncated not in seen:
            candidates.append({
                "original": truncated,
                "suspect": full_word,
                "formal": formal,
                "type": "truncation",
            })
            seen.add(truncated)
    return candidates
