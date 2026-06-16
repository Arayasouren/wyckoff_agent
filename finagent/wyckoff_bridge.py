# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bridge layer between finagent and the Wyckoff computation service.

The Wyckoff engine AND the market-data fetching both run server-side (the
service may be backed by Wind WDS). This client just asks the service for an
analysis snapshot by symbol/date and formats the result for the LLM. The client
holds no data backend; snapshot text formatting (incl. seqstats) stays local.

Requires WYCKOFF_API_URL and WYCKOFF_API_KEY (your authorization code) in env.
"""
from __future__ import annotations
import logging
from typing import Optional

from finagent.service import service_post, WyckoffServiceError  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


def get_wyckoff_snapshot(
    symbol: str,
    end_date: Optional[str] = None,
    days: int = 500,
    data_source_type: str = "stock",
) -> dict:
    """
    Run Wyckoff analysis for a symbol up to end_date via the remote service.

    The service fetches the market data itself and runs the Wyckoff engine.
    Returns a dict with the same shape as before:
      - raw: analysis result dict from the service
      - text: LLM-friendly formatted string (built client-side)
      - current_price / market_phase / probability

    data_source_type: "stock" (default) or "index" — passed through to the service.
    """
    result = service_post("/v1/snapshot", {
        "symbol": symbol,
        "end_date": end_date,
        "days": days,
        "data_source_type": data_source_type,
    })

    # Format client-side (includes the optional seqstats block).
    text = _format_for_llm(result)
    return {
        "raw": result,
        "text": text,
        "current_price": result.get("current_price"),
        "market_phase": result.get("market_phase", {}),
        "probability": result.get("probability", {}),
    }


def _format_for_llm(result: dict) -> str:
    """Format Wyckoff analysis result into structured text for LLM prompts."""
    lines = []

    # Header
    code = result.get("code", "")
    date = result.get("date", "")
    price = result.get("current_price", 0)
    data_range = result.get("data_range", {})
    lines.append(f"=== 威科夫分析快照: {code} @ {date} ===")
    lines.append(f"当前价格: {price:.4f}")
    lines.append(
        f"数据范围: {data_range.get('start', '')} ~ {data_range.get('end', '')} "
        f"({data_range.get('bars', 0)} 根K线)"
    )
    lines.append("")

    # Market phase
    phase = result.get("market_phase", {})
    lines.append("【市场阶段】")
    lines.append(f"  阶段: {phase.get('phase_name', '未知')} (ID={phase.get('phase_id', '?')})")
    sup_lines = phase.get("support_lines") or []
    res_lines = phase.get("resistance_lines") or []
    ice = phase.get("ice_line")
    if sup_lines:
        if len(sup_lines) == 1:
            lines.append(f"  支撑位: {sup_lines[0]:.4f}")
        else:
            formatted = " / ".join(f"{x:.4f}" for x in sup_lines)
            lines.append(f"  支撑位(由近及远): {formatted}")
    if res_lines:
        if len(res_lines) == 1:
            lines.append(f"  阻力位: {res_lines[0]:.4f}")
        else:
            formatted = " / ".join(f"{x:.4f}" for x in res_lines)
            lines.append(f"  阻力位(由近及远): {formatted}")
    if ice:
        lines.append(f"  冰线(分销底部): {ice:.4f}")
    lines.append("")

    # Probability
    prob = result.get("probability", {})
    lines.append("【阶段概率（规则引擎）】")
    lines.append("  说明: 由规则引擎按威科夫微观阶段(accum_A..D/dist_A..D)映射计算，非序列历史匹配。")
    lines.append("        叠加宏观阶段过滤、信号强度与P&F目标调整得出；仅作阶段倾向参考，")
    lines.append("        不等同于历史序列统计概率，不可直接作结论。")
    lines.append(f"  上涨: {prob.get('up', 0):.1%}  下跌: {prob.get('down', 0):.1%}  横盘: {prob.get('flat', 0):.1%}")
    lines.append(f"  主导阶段: {prob.get('phase_label', '未知')}")
    lines.append(f"  阶段描述: {prob.get('phase_description', '')}")
    lines.append("")

    # P&F targets
    pnf = result.get("pnf", {})
    lines.append("【点数图目标】")
    latest_target = pnf.get("latest_target")
    latest_dir = pnf.get("latest_target_dir", "")
    if latest_target:
        dir_cn = "上行" if latest_dir == "up" else "下行"
        lines.append(f"  最新目标: {latest_target:.4f} ({dir_cn})")
    all_targets = pnf.get("all_targets", [])
    if all_targets:
        for t in all_targets[-3:]:
            dir_cn = "上行" if t.get("direction") == "up" else "下行"
            lines.append(
                f"  目标 {t.get('target', 0):.4f} ({dir_cn}) "
                f"— 格宽{t.get('box_size', 0)}, 列宽{t.get('width', 0)}"
            )
    lines.append("")

    # Recent waves
    waves = result.get("waves", [])
    lines.append("【最近波浪】(最多6波，由远及近)")
    for w in waves[-6:]:
        dir_cn = "上涨" if w.get("direction") == "up" else "下跌"
        lines.append(
            f"  {dir_cn} {w.get('amplitude_pct', 0):.2f}%  "
            f"持续{w.get('length_bars', 0)}根K线"
        )
    lines.append("")

    # Recent events
    events = result.get("recent_events", [])
    lines.append(f"【最近威科夫事件】(共{len(events)}个，由远及近)")
    for ev in events[-20:]:
        status_map = {"confirmed": "✓", "pending": "?", "failed": "✗"}
        status_icon = status_map.get(ev.get("status", ""), "?")
        polarity_icon = "▲" if ev.get("polarity", 0) > 0 else ("▼" if ev.get("polarity", 0) < 0 else "—")
        bfe = ev.get("bars_from_end")
        cae = ev.get("close_at_event")
        bfe_s = f"距今{bfe}根K线" if bfe is not None else ""
        cae_s = f"收盘={cae:.4f}" if cae is not None else ""
        tail = "  ".join(s for s in (bfe_s, cae_s) if s)
        lines.append(
            f"  [{ev.get('date', '')}] {status_icon}{polarity_icon} "
            f"{ev.get('event', '')}({ev.get('name_cn', '')})  "
            f"{tail}  "
            f"{ev.get('description', '')[:60]}"
        )
    lines.append("")

    # Statistical prior from historical sequence stats (optional, requires seqstats table).
    try:
        from finagent.stats.seqstats import format_seqstats_block
        symbol = result.get("code", "")
        stat_lines = format_seqstats_block(symbol, events)
        if stat_lines:
            lines.extend(stat_lines)
    except Exception:
        pass

    lines.append("=== 快照结束 ===")

    return "\n".join(lines)


def format_actual_outcome(
    entry_price: float,
    horizon_prices: list[float],  # list of closing prices over horizon
    prediction_horizon: int = 20,
) -> dict:
    """
    Compute actual outcome stats from realized price data.
    Used by CriticAgent to evaluate predictions.
    """
    if not horizon_prices:
        return {}

    exit_price = horizon_prices[-1]
    actual_return_pct = (exit_price - entry_price) / entry_price * 100

    # Max drawdown from entry
    peak = entry_price
    max_drawdown = 0.0
    for p in horizon_prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # Max gain
    max_gain_pct = (max(horizon_prices) - entry_price) / entry_price * 100

    # Direction
    if actual_return_pct > 2.0:
        actual_direction = "bullish"
    elif actual_return_pct < -2.0:
        actual_direction = "bearish"
    else:
        actual_direction = "neutral"

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "actual_return_pct": round(actual_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "max_gain_pct": round(max_gain_pct, 4),
        "actual_direction": actual_direction,
        "horizon_days": len(horizon_prices),
    }
