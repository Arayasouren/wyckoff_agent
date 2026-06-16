# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Visualization module: generates multi-panel PNG charts for Wyckoff analysis.
Charts are then assembled into PDF by main.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd

# ─── CJK font setup ───────────────────────────────────────────────────────────
# Candidate font paths (macOS). Tried in order; first one found wins.
_CJK_FONT_CANDIDATES = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/Library/Fonts/AdobeHeitiStd-Regular.otf',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Songti.ttc',
]

_CJK_FONT_PATH: str = None
_CJK_FONT_NAME: str = None

def _find_cjk_font() -> tuple:
    """Return (path, name) of first available CJK font, or (None, None)."""
    for path in _CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                fp = fm.FontProperties(fname=path)
                return path, fp.get_name()
            except Exception:
                continue
    return None, None

def setup_cjk_font():
    """Register CJK font with matplotlib and set as default family.
    Call this after any matplotlib.use() call to ensure font takes effect.
    """
    global _CJK_FONT_PATH, _CJK_FONT_NAME
    if _CJK_FONT_PATH is None:
        _CJK_FONT_PATH, _CJK_FONT_NAME = _find_cjk_font()
    if _CJK_FONT_PATH:
        fm.fontManager.addfont(_CJK_FONT_PATH)
        rcParams['font.family'] = _CJK_FONT_NAME
    else:
        # Fallback: try common names without path
        rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

# Apply immediately for PNG generation
setup_cjk_font()

# ─── Color palette ────────────────────────────────────────────────────────────
BULL_COLOR   = '#26a69a'
BEAR_COLOR   = '#ef5350'
NEUTRAL_COLOR = '#90a4ae'
BG_COLOR     = '#1a1a2e'
PANEL_COLOR  = '#16213e'
TEXT_COLOR   = '#e0e0e0'
GRID_COLOR   = '#2a2a4a'
SUP_COLOR    = '#00e676'
RES_COLOR    = '#ff1744'
TARGET_UP_COLOR   = '#64ffda'
TARGET_DOWN_COLOR = '#ff6e40'
ACCENT_COLOR = '#ffd54f'

# Event marker styles
EVENT_MARKERS = {
    'SC':       dict(marker='v', color='#ff1744', size=14, zorder=10),
    'BC':       dict(marker='^', color='#ff6e40', size=14, zorder=10),
    'PS':       dict(marker='o', color='#80cbc4', size=9,  zorder=9),
    'PSY':      dict(marker='o', color='#ffab91', size=9,  zorder=9),
    'SPRING':   dict(marker='*', color='#00e676', size=18, zorder=11),
    'TSO':      dict(marker='*', color='#69f0ae', size=16, zorder=11),
    'UT':       dict(marker='D', color='#ff5252', size=12, zorder=10),
    'LPS':      dict(marker='P', color='#40c4ff', size=12, zorder=10),
    'LPSY':     dict(marker='P', color='#ff4081', size=12, zorder=10),
    'JOC':      dict(marker='^', color='#00e676', size=16, zorder=11),
    'BOI':      dict(marker='v', color='#ff1744', size=16, zorder=11),
    'AR_down':  dict(marker='>', color='#80deea', size=9,  zorder=9),
    'AR_up':    dict(marker='<', color='#ffcc80', size=9,  zorder=9),
    'ST_down':  dict(marker='s', color='#b2ebf2', size=9,  zorder=9),
    'ST_up':    dict(marker='s', color='#ffe082', size=9,  zorder=9),
    'SOT_down': dict(marker='_', color='#a5d6a7', size=12, zorder=8),
    'SOT_up':   dict(marker='_', color='#ef9a9a', size=12, zorder=8),
    'ASY':      dict(marker='^', color='#b9f6ca', size=10, zorder=9),
    'AS':       dict(marker='v', color='#ff8a80', size=10, zorder=9),
    'OKR_up':   dict(marker='H', color='#00e676', size=14, zorder=10),
    'OKR_down': dict(marker='H', color='#ff1744', size=14, zorder=10),
    '3H':       dict(marker='1', color='#ff8a80', size=12, zorder=8),
    '3L':       dict(marker='2', color='#b9f6ca', size=12, zorder=8),
    'VDB':      dict(marker='|', color='#64ffda', size=14, zorder=9),
    'VSB':      dict(marker='|', color='#ff6e40', size=14, zorder=9),
    'OB_up':    dict(marker='x', color='#ff8a80', size=10, zorder=8),
    'OS_down':  dict(marker='x', color='#b9f6ca', size=10, zorder=8),
    # v2 new events
    'SOS':      dict(marker='^', color='#00e676', size=15, zorder=11),
    'SOW':      dict(marker='v', color='#ff1744', size=15, zorder=11),
    'ABS':      dict(marker='h', color='#64ffda', size=12, zorder=10),
    'COB':      dict(marker='8', color='#ffd54f', size=12, zorder=10),
}
DEFAULT_MARKER = dict(marker='o', color='#ffd54f', size=8, zorder=7)

ICE_COLOR     = '#80d8ff'   # ice line — distribution range bottom
HALF_COLOR    = '#ce93d8'   # 50% retracement level


def _draw_candles(ax, df: pd.DataFrame, width: float = 0.6):
    """Draw candlestick chart."""
    opens  = df['open'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n = len(df)

    for i in range(n):
        color = BULL_COLOR if closes[i] >= opens[i] else BEAR_COLOR
        # Wick
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8, zorder=3)
        # Body
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        rect = plt.Rectangle((i - width/2, body_lo), width, body_hi - body_lo,
                              color=color, zorder=4)
        ax.add_patch(rect)


def _draw_aux_lines(ax, aux: dict, n: int):
    """Draw support/resistance, channel lines, ICE line, and 50% retracement."""
    # v2.1: prefer multi-line lists; fallback to single scalar if absent
    supports = aux.get('support_lines') or []
    resistances = aux.get('resistance_lines') or []
    if not supports and aux.get('support_level') is not None:
        supports = [aux['support_level']]
    if not resistances and aux.get('resistance_level') is not None:
        resistances = [aux['resistance_level']]

    # Supports: first entry (closest to close) solid+darker, others dashed+lighter
    for idx, s in enumerate(supports):
        primary = (idx == 0)
        ax.axhline(s, color=SUP_COLOR,
                   linestyle='--' if primary else ':',
                   linewidth=1.2 if primary else 0.9,
                   alpha=0.75 if primary else 0.45, zorder=5)
        ax.text(n - 1, s, f'  支撑 {s:.2f}', color=SUP_COLOR,
                va='center', fontsize=7, alpha=0.85 if primary else 0.55)

    for idx, r in enumerate(resistances):
        primary = (idx == 0)
        ax.axhline(r, color=RES_COLOR,
                   linestyle='--' if primary else ':',
                   linewidth=1.2 if primary else 0.9,
                   alpha=0.75 if primary else 0.45, zorder=5)
        ax.text(n - 1, r, f'  阻力 {r:.2f}', color=RES_COLOR,
                va='center', fontsize=7, alpha=0.85 if primary else 0.55)

    # ICE line: distribution range bottom (only present for consolidation phases)
    ice = aux.get('ice_line')
    if ice is not None:
        too_close = any(abs(ice - s) <= max(s, 1e-9) * 0.005 for s in supports)
        if not too_close:
            ax.axhline(ice, color=ICE_COLOR, linestyle=(0, (3, 1, 1, 1)), linewidth=1.0,
                       alpha=0.75, zorder=5)
            ax.text(n - 1, ice, f'  冰线', color=ICE_COLOR, va='center', fontsize=7, alpha=0.9)

    # Trend lines from wave points
    sup_pts = aux.get('support', [])
    res_pts = aux.get('resistance', [])

    if len(sup_pts) >= 2:
        xs = [p[0] for p in sup_pts]
        ys = [p[1] for p in sup_pts]
        ax.plot(xs, ys, color=SUP_COLOR, linewidth=1.0, alpha=0.5, linestyle='-.', zorder=4)

    if len(res_pts) >= 2:
        xs = [p[0] for p in res_pts]
        ys = [p[1] for p in res_pts]
        ax.plot(xs, ys, color=RES_COLOR, linewidth=1.0, alpha=0.5, linestyle='-.', zorder=4)


def _draw_events(ax, events: list, df: pd.DataFrame):
    """Draw event markers on price chart."""
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values

    for e in events:
        i = e['bar_idx']
        if i >= len(closes):
            continue
        polarity = e.get('polarity', 0)
        style = EVENT_MARKERS.get(e['event'], DEFAULT_MARKER)

        # Place marker above/below candle
        if polarity > 0:
            y = lows[i] * 0.992
        elif polarity < 0:
            y = highs[i] * 1.008
        else:
            y = closes[i]

        # Pending events are dimmed; failed events are shown as faint outline only
        status = e.get('status', 'confirmed')
        alpha = 0.4 if status == 'pending' else (0.25 if status == 'failed' else 0.9)
        ax.scatter(i, y, marker=style['marker'], c=style['color'],
                   s=style['size']**2, zorder=style['zorder'], alpha=alpha)

        # Label for key events
        key_events = {'SC', 'BC', 'SPRING', 'TSO', 'UT', 'LPS', 'LPSY', 'JOC', 'BOI', 'OKR_up', 'OKR_down'}
        if e['event'] in key_events:
            name = e.get('name_cn', e['event'])
            y_text = y * 0.985 if polarity > 0 else y * 1.015
            ax.annotate(name, (i, y),
                        xytext=(0, -14 if polarity > 0 else 14),
                        textcoords='offset points',
                        fontsize=6.5, color=style['color'],
                        ha='center', va='top' if polarity > 0 else 'bottom',
                        arrowprops=dict(arrowstyle='-', color=style['color'], alpha=0.4),
                        zorder=12)


def _draw_50pct_retracement(ax, waves: list, n: int):
    """Draw 50% retracement level of the most recent significant wave."""
    if not waves or len(waves) < 2:
        return
    # Find the most recent wave with significant amplitude (>2%)
    for wave in reversed(waves):
        start_p, end_p = wave[1], wave[3]
        amplitude = abs(end_p - start_p) / start_p
        if amplitude >= 0.02:
            half = (start_p + end_p) / 2.0
            ax.axhline(half, color=HALF_COLOR, linestyle=':', linewidth=0.9,
                       alpha=0.65, zorder=5)
            ax.text(n - 1, half, f'  50%回调 {half:.2f}', color=HALF_COLOR,
                    va='center', fontsize=6.5, alpha=0.85)
            return


def _draw_pnf_targets(ax, pnf_result: dict, n: int):
    """Draw P&F target price lines."""
    targets = pnf_result.get('targets', [])
    current_bar = n - 1

    for t in targets[-3:]:  # Show last 3 targets only
        color = TARGET_UP_COLOR if t['primary_dir'] == 'up' else TARGET_DOWN_COLOR
        label = f"P&F目标 {'↑' if t['primary_dir'] == 'up' else '↓'} {t['primary_target']:.2f}"
        ax.axhline(t['primary_target'], color=color, linestyle=':', linewidth=1.0,
                   alpha=0.65, zorder=5)
        ax.text(current_bar, t['primary_target'], f"  {label}",
                color=color, va='center', fontsize=7, alpha=0.85)


def plot_candlestick_panel(ax, df: pd.DataFrame, events: list,
                           aux: dict, pnf_result: dict, code: str):
    """Main candlestick panel with all overlays."""
    ax.set_facecolor(PANEL_COLOR)
    n = len(df)

    _draw_candles(ax, df)
    _draw_aux_lines(ax, aux, n)
    _draw_events(ax, events, df)
    _draw_pnf_targets(ax, pnf_result, n)

    # Wave classification markers (subtle vertical lines at wave ends)
    from .wave_classifier import classify_waves
    closes = df['close'].values
    waves = classify_waves(closes)
    for w in waves:
        ax.axvline(w[2], color='#4a4a6a', linewidth=0.4, alpha=0.5, zorder=2)

    # 50% retracement of most recent significant wave
    _draw_50pct_retracement(ax, waves, n)

    # X axis: show dates every ~20 bars
    dates = df.index
    step = max(1, n // 10)
    ticks = range(0, n, step)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([str(dates[i])[:10] for i in ticks],
                       rotation=30, ha='right', fontsize=7, color=TEXT_COLOR)
    ax.set_xlim(-1, n + 5)

    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelsize=8, colors=TEXT_COLOR)
    ax.set_title(f'{code} — 威科夫K线分析', color=ACCENT_COLOR, fontsize=11, pad=6)
    ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.7)

    # Event legend
    legend_elements = []
    shown_events = {e['event'] for e in events}
    for ev in sorted(shown_events):
        style = EVENT_MARKERS.get(ev, DEFAULT_MARKER)
        name_cn = events[[e['event'] for e in events].index(ev)]['name_cn'] if events else ev
        legend_elements.append(
            mpatches.Patch(facecolor=style['color'], label=f'{ev}({name_cn})')
        )
    if legend_elements:
        ax.legend(handles=legend_elements[:10], loc='upper left', fontsize=6,
                  facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  ncol=2, framealpha=0.8)


def plot_volume_panel(ax, df: pd.DataFrame, events: list):
    """Volume bar chart with event highlights."""
    ax.set_facecolor(PANEL_COLOR)
    volumes = df['volume'].values
    closes  = df['close'].values
    opens   = df['open'].values
    n = len(df)

    # Determine highlight events
    event_bars = {e['bar_idx']: e['polarity'] for e in events}

    colors = []
    for i in range(n):
        if i in event_bars:
            colors.append(BULL_COLOR if event_bars[i] > 0 else BEAR_COLOR)
        else:
            colors.append(BULL_COLOR if closes[i] >= opens[i] else BEAR_COLOR)

    ax.bar(range(n), volumes, color=colors, alpha=0.7, width=0.8)

    # 20-bar moving average volume
    ma_vol = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    ax.plot(range(n), ma_vol, color=ACCENT_COLOR, linewidth=0.8, alpha=0.8)

    ax.set_xlim(-1, n + 5)
    ax.set_xticks([])
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelsize=7, colors=TEXT_COLOR)
    ax.set_ylabel('成交量', color=TEXT_COLOR, fontsize=8)
    ax.grid(color=GRID_COLOR, linewidth=0.3, alpha=0.5)


def plot_macd_panel(ax, df: pd.DataFrame):
    """MACD histogram panel."""
    ax.set_facecolor(PANEL_COLOR)
    from .wave_classifier import compute_macd
    closes = df['close'].values
    n = len(df)
    dif, dea, hist = compute_macd(closes)

    colors = [BULL_COLOR if h >= 0 else BEAR_COLOR for h in hist]
    ax.bar(range(n), hist, color=colors, alpha=0.7, width=0.8)
    ax.plot(range(n), dif, color='#ffd54f', linewidth=0.9, label='DIF')
    ax.plot(range(n), dea, color='#80cbc4', linewidth=0.9, label='DEA')
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)

    ax.set_xlim(-1, n + 5)
    ax.set_xticks([])
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelsize=7, colors=TEXT_COLOR)
    ax.set_ylabel('MACD', color=TEXT_COLOR, fontsize=8)
    ax.legend(fontsize=6, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.8)
    ax.grid(color=GRID_COLOR, linewidth=0.3, alpha=0.5)


def plot_pnf_chart(ax, pnf_result: dict):
    """Point & Figure chart."""
    ax.set_facecolor(PANEL_COLOR)
    columns = pnf_result.get('columns', [])
    base_price = pnf_result.get('base_price', 0)
    box_size = pnf_result.get('box_size', 1)
    trading_ranges = pnf_result.get('trading_ranges', [])

    if not columns:
        ax.text(0.5, 0.5, 'P&F数据不足', ha='center', va='center',
                color=TEXT_COLOR, fontsize=10, transform=ax.transAxes)
        return

    # Warn when column count is too low for reliable targets
    if len(columns) < 5:
        ax.text(0.5, 0.92, '⚠ 列数不足（<5列），目标价不可信', ha='center', va='top',
                color='#ffab40', fontsize=8, fontweight='bold', transform=ax.transAxes)

    # Draw columns
    for col_idx, col in enumerate(columns):
        lo = min(col['start_box'], col['end_box'])
        hi = max(col['start_box'], col['end_box'])
        color = BULL_COLOR if col['direction'] == 1 else BEAR_COLOR
        char  = 'X' if col['direction'] == 1 else 'O'
        for box in range(lo, hi + 1):
            price = base_price + box * box_size
            ax.text(col_idx, price, char, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    # Highlight trading ranges
    for r in trading_ranges:
        sc, ec = r['start_col'], r['end_col']
        lo_p = base_price + r['min_box'] * box_size
        hi_p = base_price + r['max_box'] * box_size
        color = TARGET_UP_COLOR if r['direction'] >= 0 else TARGET_DOWN_COLOR
        rect = plt.Rectangle((sc - 0.5, lo_p), ec - sc + 1, hi_p - lo_p,
                              linewidth=1.2, edgecolor=color, facecolor='none',
                              alpha=0.6, zorder=5, linestyle='--')
        ax.add_patch(rect)

    # Compute y limits from actual column prices
    all_col_prices = []
    for col in columns[-60:]:  # limit to last 60 columns for display
        lo = min(col['start_box'], col['end_box'])
        hi = max(col['start_box'], col['end_box'])
        all_col_prices.extend([base_price + lo * box_size, base_price + hi * box_size])

    if all_col_prices:
        y_lo = min(all_col_prices) * 0.97
        y_hi = max(all_col_prices) * 1.03
    else:
        y_lo, y_hi = 0, 1

    # Target lines (only draw if within ±50% of y range)
    for t in pnf_result.get('targets', [])[-3:]:
        pt = t['primary_target']
        if y_lo * 0.5 <= pt <= y_hi * 1.5:
            color = TARGET_UP_COLOR if t['primary_dir'] == 'up' else TARGET_DOWN_COLOR
            ax.plot([-1, len(columns) + 1], [pt, pt], color=color,
                    linestyle=':', linewidth=1.0, alpha=0.7,
                    label=f"目标 {pt:.2f}")

    # Only show last 60 columns to keep chart readable
    display_start = max(0, len(columns) - 60)
    ax.set_xlim(display_start - 1, len(columns) + 1)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelsize=7, colors=TEXT_COLOR)
    ax.set_xlabel('列序', color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel('价格', color=TEXT_COLOR, fontsize=8)
    ax.set_title(f'P&F点数图  (格值={box_size:.4g}, 反转=3格)',
                 color=ACCENT_COLOR, fontsize=9, pad=4)
    ax.grid(color=GRID_COLOR, linewidth=0.3, alpha=0.4)
    if pnf_result.get('targets'):
        ax.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, framealpha=0.8)


def plot_probability_panel(ax, scores: dict, code: str, current_price: float):
    """Probability bar chart + phase label."""
    ax.set_facecolor(PANEL_COLOR)
    ax.axis('off')

    probs = [scores['prob_up'], scores['prob_flat'], scores['prob_down']]
    labels = ['上涨', '盘整', '下跌']
    colors = [BULL_COLOR, NEUTRAL_COLOR, BEAR_COLOR]

    # Horizontal bars
    bar_y = [0.72, 0.58, 0.44]
    for y, p, lbl, c in zip(bar_y, probs, labels, colors):
        ax.barh(y, p, height=0.10, color=c, alpha=0.85, left=0.12)
        ax.text(0.10, y, lbl, ha='right', va='center', color=TEXT_COLOR, fontsize=9, fontweight='bold')
        ax.text(0.13 + p, y, f'{p*100:.1f}%', ha='left', va='center', color=c, fontsize=9)

    ax.set_xlim(0, 0.9)
    ax.text(0.5, 0.90, f'当前阶段: {scores.get("phase_label", "")}',
            ha='center', va='center', color=ACCENT_COLOR, fontsize=9, fontweight='bold',
            transform=ax.transAxes)

    pnf_note = ''
    if scores.get('pnf_adjusted'):
        pnf_dir = '↑' if scores['pnf_dir'] == 'up' else '↓'
        pnf_note = f'  P&F目标 {pnf_dir} {scores["pnf_target"]:.2f}'

    ax.text(0.5, 0.28, f'{scores.get("phase_description", "")}{pnf_note}',
            ha='center', va='center', color=TEXT_COLOR, fontsize=7.5,
            wrap=True, transform=ax.transAxes, style='italic')

    # Signal strength indicator
    strength = scores.get('signal_strength', 0)
    strength_color = BULL_COLOR if strength > 0.5 else (NEUTRAL_COLOR if strength > 0.3 else BEAR_COLOR)
    ax.text(0.5, 0.12, f'信号强度: {strength*100:.0f}%',
            ha='center', va='center', color=strength_color, fontsize=8,
            fontweight='bold', transform=ax.transAxes)


def generate_charts_png(df: pd.DataFrame, events: list, aux: dict,
                        pnf_result: dict, scores: dict, code: str,
                        output_path: str):
    """
    Generate multi-panel chart PNG.
    Layout: candlestick | P&F (top row) + volume | MACD | probability (bottom row)
    """
    fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[5, 1.5, 1.5],
        width_ratios=[2.2, 1],
        hspace=0.08,
        wspace=0.06
    )

    ax_candle = fig.add_subplot(gs[0, 0])
    ax_pnf    = fig.add_subplot(gs[0, 1])
    ax_vol    = fig.add_subplot(gs[1, 0])
    ax_macd   = fig.add_subplot(gs[2, 0])
    ax_prob   = fig.add_subplot(gs[1:3, 1])

    current_price = df['close'].iloc[-1]
    recent_events = events[-40:] if len(events) > 40 else events

    plot_candlestick_panel(ax_candle, df, recent_events, aux, pnf_result, code)
    plot_pnf_chart(ax_pnf, pnf_result)
    plot_volume_panel(ax_vol, df, recent_events)
    plot_macd_panel(ax_macd, df)
    plot_probability_panel(ax_prob, scores, code, current_price)

    # Title
    from datetime import date
    today = str(date.today())
    phase_label = scores.get('phase_label', '')
    fig.suptitle(
        f'威科夫技术分析报告  |  {code}  |  {today}  |  当前价: {current_price:.2f}',
        color=ACCENT_COLOR, fontsize=13, fontweight='bold', y=0.995
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    return output_path
