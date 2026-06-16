"""
Stock metadata utilities: fetch basic info via AkShare, assign classification tags via LLM.

Tags schema (exactly 3, one per dimension):
  - 规模: 大盘 / 中盘 / 小盘
  - 风格: 成长 / 价值 / 红利
  - 行业: 金融 / 消费 / 科技 / 制造 / 周期 / 基础设施
"""
from __future__ import annotations
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SIZE_OPTIONS    = {"大盘", "中盘", "小盘"}
_STYLE_OPTIONS   = {"成长", "价值", "红利"}
_SECTOR_OPTIONS  = {"金融", "消费", "科技", "制造", "周期", "基础设施"}

_TAG_DIMENSIONS = [_SIZE_OPTIONS, _STYLE_OPTIONS, _SECTOR_OPTIONS]
_TAG_DIM_NAMES  = ["规模", "风格", "行业"]

# Index single-tag options: per user spec, indexes use the SAME 12-tag vocabulary as
# stocks (规模/风格/行业 three dimensions, 12 specific tags total), but pick ONLY ONE
# tag per index based on which dimension best characterizes it.
# Broad-market indexes (沪深300 / 上证综指 / 中证500 / etc.) get a 规模 tag like 大盘.
_INDEX_TAG_OPTIONS = _SIZE_OPTIONS | _STYLE_OPTIONS | _SECTOR_OPTIONS


def _board_from_symbol(symbol: str) -> str:
    code   = symbol.split(".")[0]
    suffix = symbol.split(".")[-1].upper()
    if suffix in ("SS", "SH"):
        return "科创板" if code.startswith("688") else "上交所主板"
    if suffix == "SZ":
        return "创业板" if code.startswith("300") else "深交所主板"
    return "未知"


def fetch_stock_basic_info(symbol: str) -> dict:
    """
    Fetch stock name, SW-industry, and market cap from AkShare.
    Returns dict with keys: name, industry, total_mv_yi (亿), board.
    """
    import akshare as ak

    code = symbol.split(".")[0]
    info = ak.stock_individual_info_em(symbol=code)
    d = dict(zip(info["item"], info["value"]))

    try:
        mv_yi = float(d.get("总市值", 0)) / 1e8
    except (TypeError, ValueError):
        mv_yi = 0.0

    return {
        "name":        d.get("股票简称", code),
        "industry":    d.get("行业", ""),
        "total_mv_yi": mv_yi,
        "board":       _board_from_symbol(symbol),
    }


def fetch_index_basic_info(symbol: str) -> dict:
    """Fetch index name from Wind aindexdescription, returns {symbol, name}."""
    try:
        from finagent.data.fetcher import _get_wind_connection, _wind_code
        conn = _get_wind_connection()
        cur = conn.cursor()
        wind_code = _wind_code(symbol)
        cur.execute(
            "SELECT S_INFO_NAME FROM winddb.aindexdescription WHERE S_INFO_WINDCODE = :1",
            [wind_code],
        )
        row = cur.fetchone()
        cur.close()
        name = row[0] if row else ""
    except Exception as e:
        logger.warning(f"fetch_index_basic_info: Wind lookup failed for {symbol}: {e}")
        name = ""
    return {"symbol": symbol, "name": name}


def assign_tags_via_llm(stock_info: dict, model: Optional[str] = None) -> list[str]:
    """
    Call LLM to assign exactly 3 tags — one per dimension.
    Returns [规模, 风格, 行业].
    Raises ValueError if response is malformed after retries.
    """
    from finagent.config import (
        DEFAULT_MODEL, LLM_PROVIDER,
        ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
        OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL,
    )

    model = model or DEFAULT_MODEL
    mv_str = f"{stock_info['total_mv_yi']:.0f}亿" if stock_info["total_mv_yi"] else "未知"

    prompt = (
        f"为以下A股股票选择分类标签，每个维度必须且只能选一个：\n"
        f"- 规模：大盘 / 中盘 / 小盘\n"
        f"- 风格：成长 / 价值 / 红利\n"
        f"- 行业：金融 / 消费 / 科技 / 制造 / 周期 / 基础设施\n\n"
        f"股票：{stock_info['name']}\n"
        f"板块：{stock_info['board']}\n"
        f"行业：{stock_info['industry']}\n"
        f"总市值：{mv_str}\n\n"
        f"只返回JSON数组，共3个标签，顺序为[规模, 风格, 行业]，例如：[\"大盘\", \"价值\", \"金融\"]"
    )

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_COMPAT_API_KEY, base_url=OPENAI_COMPAT_BASE_URL)
        msg = client.chat.completions.create(
            model=model,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.choices[0].message.content.strip()
    else:
        import anthropic
        client_kwargs: dict = {"api_key": ANTHROPIC_API_KEY}
        if ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = ANTHROPIC_BASE_URL
        client = anthropic.Anthropic(**client_kwargs)
        msg = client.messages.create(
            model=model,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

    # Strip possible markdown code fence
    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    tags = json.loads(raw)

    if not isinstance(tags, list) or len(tags) != 3:
        raise ValueError(f"Expected list of 3 tags, got: {tags}")

    for tag, dimension in zip(tags, _TAG_DIMENSIONS):
        if tag not in dimension:
            raise ValueError(f"Tag '{tag}' not in {sorted(dimension)}")

    return tags


def assign_index_tag_via_llm(symbol: str, model: Optional[str] = None) -> list[str]:
    """
    For index profiles: pick ONE tag from {宽基, 风格大类, 行业大类}.
    - 宽基 covers broad-market indexes (沪深300, 上证综指, 中证500, 创业板指 etc.)
    - 风格大类: 成长 / 价值 / 红利
    - 行业大类: 金融 / 消费 / 科技 / 制造 / 周期 / 基础设施
    Returns a 1-element list, e.g. ["宽基"] / ["成长"] / ["科技"].
    """
    from finagent.config import (
        DEFAULT_MODEL, LLM_PROVIDER,
        ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
        OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL,
    )

    model = model or DEFAULT_MODEL
    info = fetch_index_basic_info(symbol)
    name = info.get("name") or symbol

    prompt = (
        f"为以下A股指数分配一个分类标签。\n"
        f"先判断该指数最能代表哪个维度，再从该维度内选一个具体标签：\n"
        f"- 规模维度（适合宽基/全市场综合指数：沪深300、上证综指、中证500、创业板指、科创50 等）："
        f"大盘 / 中盘 / 小盘\n"
        f"- 风格维度（适合成长/价值/红利等风格指数）：成长 / 价值 / 红利\n"
        f"- 行业维度（适合行业/赛道指数）：金融 / 消费 / 科技 / 制造 / 周期 / 基础设施\n\n"
        f"指数代码：{symbol}\n"
        f"指数名称：{name}\n\n"
        f'只返回 JSON 数组，恰好一个标签，例如 ["大盘"] / ["成长"] / ["科技"]。'
    )

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_COMPAT_API_KEY, base_url=OPENAI_COMPAT_BASE_URL)
        msg = client.chat.completions.create(
            model=model, max_tokens=40,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.choices[0].message.content.strip()
    else:
        import anthropic
        client_kwargs: dict = {"api_key": ANTHROPIC_API_KEY}
        if ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = ANTHROPIC_BASE_URL
        client = anthropic.Anthropic(**client_kwargs)
        msg = client.messages.create(
            model=model, max_tokens=40,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    tags = json.loads(raw)
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list) or len(tags) != 1:
        raise ValueError(f"Expected list of 1 index tag, got: {tags!r}")
    if tags[0] not in _INDEX_TAG_OPTIONS:
        raise ValueError(f"Tag '{tags[0]}' not in {sorted(_INDEX_TAG_OPTIONS)}")
    return tags


def ensure_symbol_tags(
    symbol: str,
    strategy: dict,
    model: Optional[str] = None,
) -> tuple[list[str], bool]:
    """
    Return (tags, was_updated).
    If symbol already has valid 3-tag entry in strategy["symbol_tags"], return it as-is.
    Otherwise fetch from AkShare + LLM, update strategy dict in-place.
    Caller is responsible for persisting strategy if was_updated is True.
    """
    is_index = strategy.get("data_source_type") == "index"
    tags_map: dict = strategy.setdefault("symbol_tags", {})
    existing = tags_map.get(symbol, [])

    if is_index:
        # Index path: single tag from _INDEX_TAG_OPTIONS
        if (
            isinstance(existing, list)
            and len(existing) == 1
            and existing[0] in _INDEX_TAG_OPTIONS
        ):
            return existing, False
        logger.info(f"Assigning index tag for {symbol}...")
        try:
            tag = assign_index_tag_via_llm(symbol, model=model)
            tags_map[symbol] = tag
            logger.info(f"{symbol} tagged: {tag}")
            return tag, True
        except Exception as e:
            logger.warning(f"Failed to auto-tag index {symbol}: {e}. Using empty tags.")
            return [], False

    # Stock path: exactly 3 tags (size / style / sector)
    if (
        isinstance(existing, list)
        and len(existing) == 3
        and existing[0] in _SIZE_OPTIONS
        and existing[1] in _STYLE_OPTIONS
        and existing[2] in _SECTOR_OPTIONS
    ):
        return existing, False

    logger.info(f"Fetching stock info and tags for {symbol}...")
    try:
        info = fetch_stock_basic_info(symbol)
        tags = assign_tags_via_llm(info, model=model)
        tags_map[symbol] = tags
        logger.info(f"{symbol} ({info['name']}) tagged: {tags}")
        return tags, True
    except Exception as e:
        logger.warning(f"Failed to auto-tag {symbol}: {e}. Using empty tags.")
        return [], False
