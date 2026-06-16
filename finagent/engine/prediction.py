# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mode 2: Single prediction using the active profile.
"""
from __future__ import annotations
import logging
from typing import Optional

from rich.console import Console

from finagent.config import DEFAULT_MODEL
from finagent.storage.profile_store import ProfileStore
from finagent.storage.memory_store import MemoryManager
from finagent.wyckoff_bridge import get_wyckoff_snapshot
from finagent.agents.predictor import PredictorAgent, PredictionParseError
from finagent.prompts.templates import build_predictor_user_prompt

logger = logging.getLogger(__name__)
console = Console()


async def run_prediction(
    symbol: str,
    as_of_date: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    profile_name: Optional[str] = None,
) -> dict:
    """
    Mode 2: Load active profile, run Wyckoff analysis, predict next 20 days.
    Returns structured forecast dict.
    """
    store = ProfileStore()
    if profile_name:
        strategy = store.load(profile_name)
        active_name = profile_name
    else:
        active_name, strategy = store.load_active()

    version = strategy.get("profile_version", 0)
    data_source_type = strategy.get("data_source_type", "stock")

    # First-run: build the seqstats reference table from CSV if needed.
    from finagent.stats.seqstats import ensure_seqstats_built
    ensure_seqstats_built()

    console.print(
        f"[bold cyan]finagent predict[/] {symbol}  "
        f"profile=[magenta]{active_name}[/] v{version}  model={model}  data={data_source_type}"
    )

    # Run Wyckoff analysis
    console.print("运行威科夫分析...")
    try:
        snapshot = get_wyckoff_snapshot(symbol, end_date=as_of_date, days=500, data_source_type=data_source_type)
    except Exception as e:
        console.print(f"[red]威科夫分析失败: {e}[/]")
        raise

    actual_date = as_of_date or snapshot["raw"].get("date", "")
    prob = snapshot.get("probability", {})
    phase = snapshot.get("market_phase", {})
    console.print(f"  截止日期: {actual_date}")
    console.print(
        f"  当前阶段: {phase.get('phase_name', '?')}  "
        f"上涨概率: {prob.get('up', 0):.1%}  "
        f"下跌概率: {prob.get('down', 0):.1%}"
    )

    # Predict
    from finagent.storage.embedder import get_default_embedder
    symbol_tags: list = strategy.get("symbol_tags", {}).get(symbol, [])
    mem = MemoryManager(active_name, embedder=get_default_embedder())
    await mem.ensure_embeddings_built()
    user_msg = build_predictor_user_prompt(snapshot["text"], strategy, symbol_tags=symbol_tags)
    predictor = PredictorAgent(strategy, model=model, memory_manager=mem, symbol_tags=symbol_tags)

    console.print("LLM 预测中...")
    try:
        prediction = await predictor.predict(user_msg, snapshot=snapshot)
    except PredictionParseError as e:
        console.print(f"[red]预测解析失败: {e}[/]")
        raise

    result = {
        "symbol": symbol,
        "as_of_date": actual_date,
        "profile_name": active_name,
        "profile_version": version,
        "symbol_tags": symbol_tags,
        "current_price": snapshot.get("current_price"),
        "wyckoff_phase": phase.get("phase_name", ""),
        "wyckoff_probability": prob,
        "wyckoff_events": snapshot["raw"].get("recent_events", []),
        **prediction,
    }

    _print_prediction(result)
    return result


def _format_rationale(text: str) -> list[str]:
    import re
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Step 1: LLM puts "N. \n" at end of line — join number with following content
    text = re.sub(r'(\d+)\. *\n+', r'\1. ', text)
    # Step 2: insert blank line before each numbered item after a sentence-ending punctuation
    text = re.sub(r'([。．\.，,、])\s*(\d+)\.\s+', r'\1\n\n\2. ', text)
    # Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.splitlines()


def _print_prediction(result: dict) -> None:
    direction_color = {
        "bullish": "green",
        "bearish": "red",
        "neutral": "yellow",
    }.get(result.get("direction", "neutral"), "white")
    direction_cn = {
        "bullish": "看涨",
        "bearish": "看跌",
        "neutral": "中性",
    }.get(result.get("direction", "neutral"), "?")

    tags = result.get("symbol_tags", [])
    if tags:
        console.print(f"\n[cyan]标签: {' / '.join(tags)}[/]")

    events = result.get("wyckoff_events", [])
    if events:
        console.print(f"\n[bold white]形态序列[/]")
        pol_icon = {1: "[red]▲[/]", -1: "[green]▼[/]", 0: "[dim]—[/]"}
        status_icon = {"confirmed": "[green]✓[/]", "pending": "[yellow]?[/]", "failed": "[dim]✗[/]"}
        # Group same-day events into one line
        from collections import OrderedDict
        day_groups: OrderedDict = OrderedDict()
        for ev in events[-20:]:
            day = ev.get("date", "")
            day_groups.setdefault(day, []).append(ev)
        for day, evs in day_groups.items():
            close = evs[0].get("close_at_event")
            close_s = f"  {close:.4f}" if close is not None else ""
            if len(evs) == 1:
                ev = evs[0]
                arrow = pol_icon.get(ev.get("polarity", 0), "[dim]—[/]")
                st = status_icon.get(ev.get("status", ""), "[dim]?[/]")
                name = ev.get("event", "")
                name_cn = ev.get("name_cn", "")
                label = f"{name}({name_cn})" if name_cn else name
                console.print(f"  {day}  {st}{arrow}  {label}{close_s}")
            else:
                # Multiple events same day — show combined
                labels = []
                for ev in evs:
                    name = ev.get("event", "")
                    name_cn = ev.get("name_cn", "")
                    st_char = {"confirmed": "✓", "pending": "?", "failed": "✗"}.get(ev.get("status", ""), "?")
                    pol_char = {1: "▲", -1: "▼", 0: "—"}.get(ev.get("polarity", 0), "")
                    labels.append(f"{st_char}{pol_char}{name}({name_cn})" if name_cn else f"{st_char}{pol_char}{name}")
                console.print(f"  {day}  [dim]同日多形态:[/] {' / '.join(labels)}{close_s}")
        console.print()

    # Prediction result — between 形态序列 and 分析理由
    console.print(f"[bold white]预测结果[/]")
    console.print(
        f"  方向: [{direction_color}]{direction_cn}[/]  "
        f"置信度: {result.get('confidence', 0):.1%}"
    )
    console.print(f"  当前价格: {result.get('current_price', '?')}")
    if result.get("target_price"):
        console.print(f"  目标价: {result['target_price']:.4f}")
    if result.get("key_support"):
        console.print(f"  关键支撑: {result['key_support']:.4f}")
    if result.get("key_resistance"):
        console.print(f"  关键阻力: {result['key_resistance']:.4f}")
    console.print(f"  预测周期: {result.get('time_horizon_days', 20)} 个交易日")
    console.print()

    console.print(f"[bold white]分析理由[/]")
    for line in _format_rationale(result.get("rationale", "")):
        if line.strip():
            console.print(f"  {line.strip()}")
        else:
            console.print()
    console.print()

    signals = result.get("key_signals", [])
    if signals:
        console.print(f"[bold white]关键信号[/]")
        for s in signals:
            console.print(f"  • {s}")
        console.print()

    risks = result.get("risk_factors", [])
    if risks:
        console.print(f"[bold white]风险因素[/]")
        for r in risks:
            console.print(f"  ⚠ {r}")
        console.print()

    memory = result.get("triggered_memory", [])
    if memory:
        console.print(f"[bold white]【触发的记忆】[/]")
        for m in memory:
            label = m.replace("_", " ").replace(".md", "")
            console.print(f"  [dim]◈ {label}[/]")
        console.print()
