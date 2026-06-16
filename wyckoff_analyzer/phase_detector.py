# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Trend vs. Consolidation detector and auxiliary line builder.
Translates MATLAB trendorconsolid.m and auxline.m logic to Python.
"""
import numpy as np
import pandas as pd
from .wave_classifier import classify_waves, get_wave_df


PHASE_UPTREND = 1
PHASE_DOWNTREND = 2
PHASE_SHRINKING = 3   # 收缩盘整: higher lows, lower highs
PHASE_EXPANDING = 4   # 扩张盘整: lower lows, higher highs

PHASE_NAMES = {
    PHASE_UPTREND: '上升趋势',
    PHASE_DOWNTREND: '下降趋势',
    PHASE_SHRINKING: '收缩盘整',
    PHASE_EXPANDING: '扩张盘整',
}


def detect_phase(waves: list) -> int:
    """
    Determine market phase from last 4 swing waves.
    Returns PHASE_* constant.
    """
    wdf = get_wave_df(waves)
    if len(wdf) < 4:
        return PHASE_SHRINKING  # default

    # Last 4 completed waves (exclude current incomplete)
    last4 = wdf.iloc[-5:-1] if len(wdf) > 4 else wdf.iloc[-4:]

    # Collect tops and bottoms from last 4 waves
    tops = []
    bottoms = []
    for _, row in last4.iterrows():
        if row['end_price'] > row['start_price']:  # up wave
            tops.append(row['end_price'])
        else:  # down wave
            bottoms.append(row['end_price'])

    if len(tops) < 2 or len(bottoms) < 2:
        return PHASE_SHRINKING

    top_rising = tops[-1] > tops[-2]
    bottom_rising = bottoms[-1] > bottoms[-2]

    if top_rising and bottom_rising:
        return PHASE_UPTREND
    elif not top_rising and not bottom_rising:
        return PHASE_DOWNTREND
    elif not top_rising and bottom_rising:
        return PHASE_SHRINKING
    else:  # top_rising and not bottom_rising
        return PHASE_EXPANDING


def build_aux_lines(close: np.ndarray, waves: list, phase: int) -> dict:
    """
    Build support/resistance/channel auxiliary lines.

    Returns dict with:
      'support': list of (idx, price) points for support line
      'resistance': list of (idx, price) points for resistance line
      'support_level': float (most recent support price)
      'resistance_level': float (most recent resistance price)
      'channel_high': float or None
      'channel_low': float or None
      'consolidation_confidence': float [0,1]
    """
    wdf = get_wave_df(waves)
    n = len(close)
    result = {
        'support': [],
        'resistance': [],
        'support_level': None,
        'resistance_level': None,
        'support_lines': [],       # close 下方参考价（降序，最近 close 的在前）
        'resistance_lines': [],    # close 上方参考价（升序，最近 close 的在前）
        'channel_high': None,
        'channel_low': None,
        'consolidation_confidence': 0.0,
        'ice_line': None,      # Distribution range bottom — reference for BOI detection
    }

    if len(wdf) < 4:
        return result

    last4 = wdf.iloc[-5:-1] if len(wdf) > 4 else wdf.iloc[-4:]
    up_waves = last4[last4['direction'] == 1]
    dn_waves = last4[last4['direction'] == -1]

    if phase == PHASE_UPTREND:
        # Bottom support line connecting prior lows (start of up waves)
        bottoms = [(int(r['start_idx']), r['start_price'])
                   for _, r in last4[last4['direction'] == 1].iterrows()]
        if len(bottoms) >= 2:
            result['support'] = bottoms
            result['support_level'] = bottoms[-1][1]
            # Slope from last two bottoms
            if len(bottoms) >= 2:
                x1, y1 = bottoms[-2]
                x2, y2 = bottoms[-1]
                slope = (y2 - y1) / max(x2 - x1, 1)
                proj = y2 + slope * (n - 1 - x2)
                result['channel_low'] = proj
        tops = [(int(r['end_idx']), r['end_price'])
                for _, r in last4[last4['direction'] == 1].iterrows()]
        if tops:
            result['resistance_level'] = tops[-1][1]
            result['resistance'] = tops

    elif phase == PHASE_DOWNTREND:
        # Top resistance line connecting prior highs (end of up waves / start of down waves)
        tops = [(int(r['start_idx']), r['start_price'])
                for _, r in last4[last4['direction'] == -1].iterrows()]
        if len(tops) >= 2:
            result['resistance'] = tops
            result['resistance_level'] = tops[-1][1]
        bottoms = [(int(r['end_idx']), r['end_price'])
                   for _, r in last4[last4['direction'] == -1].iterrows()]
        if bottoms:
            result['support_level'] = bottoms[-1][1]
            result['support'] = bottoms

    else:  # Consolidation
        # Horizontal support/resistance at key swing levels
        all_highs = sorted(
            [r['end_price'] for _, r in last4[last4['direction'] == 1].iterrows()],
            reverse=True
        )
        all_lows = sorted(
            [r['end_price'] for _, r in last4[last4['direction'] == -1].iterrows()]
        )

        resistance_lvl = all_highs[0] if all_highs else close[-1] * 1.02
        support_lvl = all_lows[0] if all_lows else close[-1] * 0.98

        result['resistance_level'] = resistance_lvl
        result['support_level'] = support_lvl
        result['channel_high'] = resistance_lvl
        result['channel_low'] = support_lvl
        # ICE line: bottom support of the consolidation range
        # In distribution phases this is the critical level whose break confirms markdown
        result['ice_line'] = support_lvl

        # Confidence: how close current price is to key levels
        current = close[-1]
        mid = (resistance_lvl + support_lvl) / 2
        range_size = resistance_lvl - support_lvl
        if range_size > 0:
            dist_ratio = abs(current - mid) / (range_size / 2)
            result['consolidation_confidence'] = max(0.0, 1.0 - dist_ratio)

    # --- Multi-line support/resistance output (v2.1) ---
    # Does not affect event detection (which still uses support_level/resistance_level scalars).
    close_now = float(close[-1])
    wdf_full = wdf  # includes last4 + current unconfirmed wave

    if phase in (PHASE_SHRINKING, PHASE_EXPANDING):
        # Candidate price pool: all endpoints from last 4 confirmed waves + current unconfirmed wave's extreme
        candidates = []
        for _, r in last4.iterrows():
            candidates.append(float(r['start_price']))
            candidates.append(float(r['end_price']))
        # Current unconfirmed wave's extreme
        if len(wdf_full) > 4:
            candidates.append(float(wdf_full.iloc[-1]['end_price']))

        # Deduplicate (treat prices within 0.01% as identical)
        dedup = []
        for p in candidates:
            if not any(abs(p - q) / max(q, 1e-9) < 1e-4 for q in dedup):
                dedup.append(p)

        supports = sorted([p for p in dedup if p < close_now], reverse=True)
        resistances = sorted([p for p in dedup if p > close_now])

        result['support_lines'] = supports
        result['resistance_lines'] = resistances
        result['ice_line'] = min(supports) if supports else None

    elif phase == PHASE_UPTREND:
        # Two most recent bottoms (start of up waves in last4)
        bots = [(int(r['start_idx']), float(r['start_price']))
                for _, r in last4[last4['direction'] == 1].iterrows()]
        if len(bots) >= 2:
            (x1, y1), (x2, y2) = bots[-2], bots[-1]
            k = (y2 - y1) / max(x2 - x1, 1)
            supp_today = y2 + k * (n - 1 - x2)
            # Top between x1 and x2 (up-wave end within that range)
            mids = [(int(r['end_idx']), float(r['end_price']))
                    for _, r in last4[last4['direction'] == 1].iterrows()
                    if x1 < int(r['end_idx']) < x2]
            if mids:
                x_top, y_top = max(mids, key=lambda t: t[1])
                resi_today = y_top + k * (n - 1 - x_top)
                result['support_lines'] = [float(supp_today)]
                result['resistance_lines'] = [float(resi_today)]

    elif phase == PHASE_DOWNTREND:
        # Two most recent tops (start of down waves in last4)
        tops = [(int(r['start_idx']), float(r['start_price']))
                for _, r in last4[last4['direction'] == -1].iterrows()]
        if len(tops) >= 2:
            (x1, y1), (x2, y2) = tops[-2], tops[-1]
            k = (y2 - y1) / max(x2 - x1, 1)
            resi_today = y2 + k * (n - 1 - x2)
            # Bottom between x1 and x2 (down-wave end within that range)
            mids = [(int(r['end_idx']), float(r['end_price']))
                    for _, r in last4[last4['direction'] == -1].iterrows()
                    if x1 < int(r['end_idx']) < x2]
            if mids:
                x_bot, y_bot = min(mids, key=lambda t: t[1])
                supp_today = y_bot + k * (n - 1 - x_bot)
                result['support_lines'] = [float(supp_today)]
                result['resistance_lines'] = [float(resi_today)]

    return result
