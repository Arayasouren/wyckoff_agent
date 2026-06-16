# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Post-evolution K-line chart with prediction arrows.
Saved to data/figure/ after each evolve run.
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def save_evolution_chart(
    symbol: str,
    profile_name: str,
    version: int,
    stats: dict,
    holdout_scores: dict,
    best_key: str,
    baseline_score: float,
    train_records: list[dict],
    holdout_records: list[dict],
    fig_dir: Path,
    data_source_type: str = "stock",
) -> Optional[Path]:
    """
    Draw and save a candlestick chart with prediction arrows.

    train_records / holdout_records: [{window_end_date, direction, confidence, ...}]
    Returns saved Path or None if matplotlib is unavailable.
    """
    try:
        import sys, os as _os
        _old_stderr = sys.stderr
        sys.stderr = open(_os.devnull, "w")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib.gridspec as gridspec
        finally:
            sys.stderr.close()
            sys.stderr = _old_stderr
    except ImportError as e:
        logger.warning(f"Chart skipped (matplotlib unavailable): {e}")
        return None

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
        "SimHei", "Microsoft YaHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        from finagent.data.fetcher import fetch_ohlcv_df
        df = fetch_ohlcv_df(symbol, days=2000, data_source_type=data_source_type)
    except Exception as e:
        logger.warning(f"Chart: OHLCV fetch failed for {symbol}: {e}")
        return None

    if df.empty:
        return None

    # Restrict display range to window coverage + small margin
    all_pred_dates = {
        r["window_end_date"] for r in train_records + holdout_records
        if r.get("window_end_date")
    }
    df_dates = [d.strftime("%Y-%m-%d") for d in df.index]

    if all_pred_dates:
        min_pd = min(all_pred_dates)
        max_pd = max(all_pred_dates)
        lo = max(0, _nearest_idx(df_dates, min_pd) - 20)
        hi = min(len(df) - 1, _nearest_idx(df_dates, max_pd) + 40)
        df = df.iloc[lo : hi + 1]
        df_dates = [d.strftime("%Y-%m-%d") for d in df.index]

    # Cap at 800 bars for readability
    if len(df) > 800:
        df = df.iloc[-800:]
        df_dates = [d.strftime("%Y-%m-%d") for d in df.index]

    n = len(df_dates)
    if n == 0:
        return None

    date_to_x = {d: i for i, d in enumerate(df_dates)}
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    vols   = df["volume"].values

    price_range = max(highs) - min(lows) if max(highs) > min(lows) else 1.0

    # ── Layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 12), facecolor="white")
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.7, 4.5, 1.3], hspace=0.04)
    ax_info = fig.add_subplot(gs[0])
    ax_k    = fig.add_subplot(gs[1])
    ax_vol  = fig.add_subplot(gs[2], sharex=ax_k)

    # ── Info panel ─────────────────────────────────────────────
    ax_info.axis("off")
    scores_str = "  ".join(
        f"{k}={v:.3f}" for k, v in sorted(holdout_scores.items())
    )
    best_score = holdout_scores.get(best_key, 0.0)
    line1 = (
        f"Symbol: {symbol}    Profile: {profile_name}  v{version}    "
        f"Train windows: {stats.get('total_windows', 0)}    "
        f"Direction acc: {stats.get('direction_accuracy', 0):.1%}    "
        f"Avg score: {stats.get('avg_score', 0):.3f}    "
        f"Avg return: {stats.get('avg_return_pct', 0):+.2f}%"
    )
    line2 = (
        f"Holdout → baseline: {baseline_score:.3f}    {scores_str}    "
        f"Selected: {best_key} ({best_score:.3f})"
    )
    ax_info.text(0.01, 0.72, line1, transform=ax_info.transAxes,
                 fontsize=9, va="center", fontfamily="monospace", color="#1a1a1a")
    ax_info.text(0.01, 0.28, line2, transform=ax_info.transAxes,
                 fontsize=9, va="center", fontfamily="monospace", color="#1a1a1a")

    # ── Candlesticks ───────────────────────────────────────────
    bar_w = 0.6
    up_c   = "#e03030"  # red  = close >= open
    down_c = "#1fa640"  # green = close < open

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        col = up_c if c >= o else down_c
        body_lo = min(o, c)
        body_hi = max(o, c)
        # Ensure minimum body height (e.g. doji)
        if body_hi - body_lo < price_range * 0.001:
            body_hi = body_lo + price_range * 0.001
        ax_k.add_patch(mpatches.Rectangle(
            (i - bar_w / 2, body_lo), bar_w, body_hi - body_lo,
            color=col, linewidth=0, zorder=2,
        ))
        ax_k.plot([i, i], [l, h], color=col, linewidth=0.7, zorder=1)

    ax_k.set_xlim(-1, n + 1)
    ax_k.set_ylim(min(lows) * 0.993, max(highs) * 1.007)
    ax_k.set_ylabel("Price", fontsize=9)
    ax_k.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax_k.tick_params(axis="x", labelbottom=False)

    # ── Prediction arrows ──────────────────────────────────────
    offset = price_range * 0.018   # vertical offset from close

    def _plot_arrow(x: int, close: float, direction: str, is_holdout: bool) -> None:
        if direction == "bullish":
            y      = close - offset
            marker = "^"
            color  = "#d42020"
        elif direction == "bearish":
            y      = close + offset
            marker = "v"
            color  = "#17963a"
        else:
            y      = close
            marker = "D"
            color  = "#888888"

        mfc = "none" if is_holdout else color
        mew = 1.8 if is_holdout else 0
        ms  = 9   if is_holdout else 8
        ax_k.plot(
            x, y, marker=marker, color=color,
            markersize=ms, markerfacecolor=mfc,
            markeredgecolor=color, markeredgewidth=mew,
            alpha=0.85, zorder=6, linestyle="none",
        )

    for r in train_records:
        x = date_to_x.get(r.get("window_end_date", ""))
        if x is None:
            continue
        _plot_arrow(x, closes[x], r.get("direction", ""), is_holdout=False)

    for r in holdout_records:
        x = date_to_x.get(r.get("window_end_date", ""))
        if x is None:
            continue
        _plot_arrow(x, closes[x], r.get("direction", ""), is_holdout=True)

    # ── Legend ─────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#d42020",
                   markersize=9, label="多头-训练"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#17963a",
                   markersize=9, label="空头-训练"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="none",
                   markeredgecolor="#d42020", markeredgewidth=1.8,
                   markersize=9, label=f"多头-{best_key}验证"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="none",
                   markeredgecolor="#17963a", markeredgewidth=1.8,
                   markersize=9, label=f"空头-{best_key}验证"),
    ]
    ax_k.legend(handles=legend_handles, loc="upper left",
                fontsize=8, framealpha=0.85, ncol=2)

    # ── Volume bars ────────────────────────────────────────────
    for i in range(n):
        col = up_c if closes[i] >= opens[i] else down_c
        ax_vol.bar(i, vols[i], color=col, width=bar_w, alpha=0.75, linewidth=0)

    ax_vol.set_ylabel("Volume", fontsize=8)
    _fmt_vol = plt.FuncFormatter(
        lambda x, _: f"{x/1e8:.1f}亿" if x >= 1e8
        else (f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )
    ax_vol.yaxis.set_major_formatter(_fmt_vol)
    ax_vol.grid(axis="y", linestyle="--", alpha=0.2)

    # ── X-axis date ticks ──────────────────────────────────────
    tick_step = max(1, n // 20)
    ticks = list(range(0, n, tick_step))
    ax_vol.set_xticks(ticks)
    ax_vol.set_xticklabels(
        [df_dates[i][:7] for i in ticks],
        rotation=45, ha="right", fontsize=7,
    )
    ax_vol.set_xlim(-1, n + 1)

    # ── Save ──────────────────────────────────────────────────
    fig_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol}_{profile_name}_v{version}_{ts}.png"
    out_path = fig_dir / fname
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Evolution chart saved: {out_path}")
    return out_path


def _nearest_idx(dates: list[str], target: str) -> int:
    """Return index of nearest date >= target, or last index if all are earlier."""
    for i, d in enumerate(dates):
        if d >= target:
            return i
    return len(dates) - 1
