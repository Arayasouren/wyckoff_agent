# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Wyckoff phase probability scorer v2.

Key improvements from v1:
- Nested stage model: phase_detector macro context constrains micro-stage labels
- Uptrend context: accum_A/B/C events are treated as intra-trend consolidation noise
- Downtrend context: dist_A/B/C events are treated as dead-cat bounce noise
- Added SOS, SOW, ABS, COB to event→phase mapping
- Signal strength: weak event signals blend toward neutral/background
- Chain completeness: bonus weight when key event chains are present
"""
import numpy as np
from typing import Dict

# Macro phase IDs from phase_detector
PHASE_UPTREND   = 1
PHASE_DOWNTREND = 2
PHASE_CONSOL_CONTRACTING = 3
PHASE_CONSOL_EXPANDING   = 4

# Phase probability table (up%, down%, flat%)
_PHASE_PROBS = {
    # Accumulation stages
    'accum_A':  (0.40, 0.35, 0.25),  # PS/SC/AR — early bottom, direction unclear
    'accum_B':  (0.45, 0.25, 0.30),  # Multiple STs — base building
    'accum_C':  (0.65, 0.15, 0.20),  # SPRING/LPS — key test passed
    'accum_D':  (0.78, 0.08, 0.14),  # JOC/SOS — markup underway
    # Distribution stages
    'dist_A':   (0.35, 0.40, 0.25),  # PSY/BC/AR — early top
    'dist_B':   (0.25, 0.45, 0.30),  # Multiple STs — topping
    'dist_C':   (0.15, 0.65, 0.20),  # UT — key test failed
    'dist_D':   (0.08, 0.78, 0.14),  # LPSY/SOW/BOI — markdown underway
    # Trend continuation (no reversal signal)
    'uptrend':   (0.62, 0.18, 0.20),  # in uptrend, no warning signs
    'downtrend': (0.18, 0.62, 0.20),  # in downtrend, no reversal signs
    'neutral':   (0.33, 0.33, 0.34),
}

# Map events to Wyckoff micro-phases
_EVENT_PHASE = {
    # Accumulation signals
    'SC':       'accum_A',
    'PS':       'accum_A',
    'AR_down':  'accum_A',
    'OS_down':  'accum_A',
    'OS_line':  'accum_A',
    'ST_down':  'accum_B',
    'SOT_down': 'accum_B',
    '3L':       'accum_B',
    'TSO':      'accum_C',
    'SPRING':   'accum_C',
    'LPS':      'accum_C',
    'OKR_up':   'accum_C',
    'VDB':      'accum_C',
    'SOS':      'accum_D',
    'JOC':      'accum_D',
    'ASY':      'accum_D',
    'ABS':      'accum_D',  # absorption at resistance = demand overwhelming supply
    # Distribution signals
    'BC':       'dist_A',
    'PSY':      'dist_A',
    'AR_up':    'dist_A',
    'OB_up':    'dist_A',
    'OB_line':  'dist_A',
    'ST_up':    'dist_B',
    'SOT_up':   'dist_B',
    '3H':       'dist_B',
    'UT':       'dist_C',
    'LPSY':     'dist_C',
    'OKR_down': 'dist_C',
    'VSB':      'dist_C',
    'SOW':      'dist_D',
    'BOI':      'dist_D',
    'AS':       'dist_D',
    # COB is a phase-change signal — handled separately
}

# Background probability by macro phase
_MACRO_BASE_PROBS = {
    PHASE_UPTREND:             (0.62, 0.18, 0.20),
    PHASE_DOWNTREND:           (0.18, 0.62, 0.20),
    PHASE_CONSOL_CONTRACTING:  (0.33, 0.33, 0.34),
    PHASE_CONSOL_EXPANDING:    (0.30, 0.30, 0.40),
}

# In uptrend, accum_A/B/C events are intra-trend pullbacks — suppress them
# In downtrend, dist_A/B/C events are dead-cat bounces — suppress them
_PHASE_SUPPRESSION = {
    PHASE_UPTREND:   {'accum_A': 0.15, 'accum_B': 0.20, 'accum_C': 0.30},
    PHASE_DOWNTREND: {'dist_A':  0.15, 'dist_B':  0.20, 'dist_C':  0.30},
}

# Key accumulation chains: list of event sets that form complete chains
_ACCUM_CHAINS = [
    {'PS', 'SC', 'AR_down'},        # A-stage chain
    {'SC', 'AR_down', 'ST_down'},   # A→B transition
    {'ST_down', 'SPRING'},          # B→C
    {'ST_down', 'LPS'},             # B→C (alternative)
    {'SPRING', 'SOS'},              # C→D
    {'SOS', 'JOC'},                 # D confirmation
    {'LPS', 'JOC'},                 # D confirmation (alternative)
]

_DIST_CHAINS = [
    {'PSY', 'BC', 'AR_up'},         # A-stage chain
    {'BC', 'AR_up', 'ST_up'},       # A→B transition
    {'ST_up', 'UT'},                # B→C
    {'UT', 'LPSY'},                 # C→D
    {'SOW', 'BOI'},                 # D confirmation
    {'LPSY', 'BOI'},                # D confirmation (alternative)
]


def _chain_completeness(present_events: set, chains: list) -> float:
    """
    Returns a strength multiplier [1.0, 1.5] based on how many complete
    event chains are present. Full chain = 1.5× weight boost.
    """
    complete = sum(1 for chain in chains if chain.issubset(present_events))
    if complete == 0:
        return 1.0
    return min(1.5, 1.0 + complete * 0.2)


def score_events(events: list, market_phase: int = None, half_life: int = 10) -> Dict:
    """
    Compute probability scores from recent event list.

    Args:
      events: list of event dicts from get_recent_events()
      market_phase: integer phase ID from phase_detector.detect_phase()
                    (1=uptrend, 2=downtrend, 3=consol_contracting, 4=consol_expanding)
      half_life: exponential decay half-life in bars

    Returns dict with prob_up, prob_down, prob_flat, dominant_phase, phase_label.
    """
    # Default to consolidation if not provided
    macro = market_phase if market_phase in _MACRO_BASE_PROBS else PHASE_CONSOL_CONTRACTING

    if not events:
        base_up, base_dn, base_fl = _MACRO_BASE_PROBS[macro]
        macro_key = _macro_phase_label(macro)
        return {
            'prob_up':    round(base_up, 3),
            'prob_down':  round(base_dn, 3),
            'prob_flat':  round(base_fl, 3),
            'dominant_phase':    macro_key,
            'phase_label':       _phase_display_label(macro_key, macro),
            'phase_description': _phase_desc(macro_key, macro),
            'phase_weights':     {},
            'signal_strength':   0.0,
        }

    max_bar = max(e['bar_idx'] for e in events)
    decay = np.log(2) / half_life

    phase_weights: Dict[str, float] = {}
    present_events: set = set()
    cob_present = False

    for e in events:
        ev = e['event']
        age = max_bar - e['bar_idx']
        weight = np.exp(-decay * age) * e['confidence']

        if ev == 'COB':
            cob_present = True
            continue  # COB handled separately

        micro_phase = _EVENT_PHASE.get(ev, 'neutral')
        phase_weights[micro_phase] = phase_weights.get(micro_phase, 0.0) + weight
        present_events.add(ev)

    # Apply phase-context suppression (uptrend suppresses accum_A/B/C, etc.)
    suppression = _PHASE_SUPPRESSION.get(macro, {})
    for phase, factor in suppression.items():
        if phase in phase_weights:
            phase_weights[phase] *= factor

    # Add chain completeness bonus for the dominant direction
    accum_weight = sum(phase_weights.get(p, 0) for p in ['accum_A', 'accum_B', 'accum_C', 'accum_D'])
    dist_weight  = sum(phase_weights.get(p, 0) for p in ['dist_A',  'dist_B',  'dist_C',  'dist_D'])

    if accum_weight > dist_weight:
        chain_multiplier = _chain_completeness(present_events, _ACCUM_CHAINS)
        for p in ['accum_A', 'accum_B', 'accum_C', 'accum_D']:
            if p in phase_weights:
                phase_weights[p] *= chain_multiplier
    elif dist_weight > accum_weight:
        chain_multiplier = _chain_completeness(present_events, _DIST_CHAINS)
        for p in ['dist_A', 'dist_B', 'dist_C', 'dist_D']:
            if p in phase_weights:
                phase_weights[p] *= chain_multiplier

    # Add macro background as a baseline weight (anchors interpretation)
    macro_baseline_weight = max(sum(phase_weights.values()) * 0.3, 0.5)
    macro_phase_key = _macro_phase_label(macro)
    phase_weights[macro_phase_key] = phase_weights.get(macro_phase_key, 0) + macro_baseline_weight

    # COB: if present, add a weight toward transition phase (away from current trend)
    if cob_present:
        if macro == PHASE_UPTREND:
            phase_weights['dist_A'] = phase_weights.get('dist_A', 0) + macro_baseline_weight * 0.8
        elif macro == PHASE_DOWNTREND:
            phase_weights['accum_A'] = phase_weights.get('accum_A', 0) + macro_baseline_weight * 0.8

    if not phase_weights:
        phase_weights['neutral'] = 1.0

    # Determine dominant phase
    dominant_phase = max(phase_weights, key=phase_weights.get)
    total_weight = sum(phase_weights.values())

    # Blend probabilities weighted by phase occurrence
    prob_up = prob_down = prob_flat = 0.0
    for phase, w in phase_weights.items():
        pu, pd, pf = _PHASE_PROBS.get(phase, _PHASE_PROBS['neutral'])
        frac = w / total_weight
        prob_up   += pu * frac
        prob_down += pd * frac
        prob_flat += pf * frac

    # Signal strength: if event signal is weak, blend toward background
    event_total = total_weight - phase_weights.get(macro_phase_key, 0)
    if event_total > 0:
        strength = min(1.0, event_total / (event_total + macro_baseline_weight))
    else:
        strength = 0.0

    if strength < 0.5:
        # Blend toward background
        bg_up, bg_dn, bg_fl = _MACRO_BASE_PROBS[macro]
        blend = (1.0 - strength * 2)  # 1.0 at strength=0, 0.0 at strength=0.5
        prob_up   = prob_up   * (1 - blend) + bg_up * blend
        prob_down = prob_down * (1 - blend) + bg_dn * blend
        prob_flat = prob_flat * (1 - blend) + bg_fl * blend

    # Normalize
    total = prob_up + prob_down + prob_flat
    if total > 0:
        prob_up   /= total
        prob_down /= total
        prob_flat /= total

    return {
        'prob_up':    round(prob_up, 3),
        'prob_down':  round(prob_down, 3),
        'prob_flat':  round(prob_flat, 3),
        'dominant_phase':    dominant_phase,
        'phase_label':       _phase_display_label(dominant_phase, macro),
        'phase_description': _phase_desc(dominant_phase, macro),
        'phase_weights':     {k: round(v, 3) for k, v in phase_weights.items()},
        'signal_strength':   round(strength, 3),
    }


def _macro_phase_label(macro: int) -> str:
    return {
        PHASE_UPTREND:   'uptrend',
        PHASE_DOWNTREND: 'downtrend',
        PHASE_CONSOL_CONTRACTING: 'neutral',
        PHASE_CONSOL_EXPANDING:   'neutral',
    }.get(macro, 'neutral')


def _phase_display_label(phase: str, macro: int = None) -> str:
    # Context-aware overrides when dominant event phase contradicts macro trend
    if macro == PHASE_UPTREND and phase in ('dist_A', 'dist_B'):
        return '上升趋势警示 — 分布信号出现，注意见顶风险'
    if macro == PHASE_UPTREND and phase in ('dist_C', 'dist_D'):
        return '上升趋势转弱 — 强分布信号，谨慎减仓或离场'
    if macro == PHASE_UPTREND and phase == 'accum_D':
        return '上升趋势加速 — 吸筹突破完成，趋势延续'
    if macro == PHASE_DOWNTREND and phase in ('accum_A', 'accum_B'):
        return '下降趋势中出现吸筹信号 — 谨慎，可能是死猫弹'
    if macro == PHASE_DOWNTREND and phase in ('accum_C', 'accum_D'):
        return '下降趋势可能见底 — 吸筹信号明确，关注止跌'
    return {
        'accum_A':   '吸筹A阶段 — 下跌止步，底部初探',
        'accum_B':   '吸筹B阶段 — 区间震荡，筑底蓄势',
        'accum_C':   '吸筹C阶段 — 弹簧/最后支撑，看涨临界',
        'accum_D':   '吸筹D阶段 — 突破启动，上升趋势确立',
        'dist_A':    '派发A阶段 — 上涨止步，顶部初探',
        'dist_B':    '派发B阶段 — 区间震荡，顶部蓄势',
        'dist_C':    '派发C阶段 — 上冲回落/最后供应，看跌临界',
        'dist_D':    '派发D阶段 — 破位下行，下降趋势确立',
        'uptrend':   '上升趋势持仓 — 趋势延续，回调是机会',
        'downtrend': '下降趋势持仓 — 趋势延续，反弹是陷阱',
        'neutral':   '盘整无明确信号 — 等待方向突破',
    }.get(phase, '未知阶段')


def _phase_desc(phase: str, macro: int = None) -> str:
    if macro == PHASE_UPTREND and phase in ('dist_A', 'dist_B'):
        return '上升趋势中出现分布迹象（三高、关键反包下等），需关注量价关系是否真正转弱，尚未确认见顶'
    if macro == PHASE_UPTREND and phase in ('dist_C', 'dist_D'):
        return '上升趋势中出现明确派发信号，主力出货特征显现，建议减仓或止损保护利润'
    if macro == PHASE_DOWNTREND and phase in ('accum_A', 'accum_B'):
        return '下降趋势中出现初步吸筹迹象，但需警惕反弹后继续下跌，宜等待更多确认后再行动'
    if macro == PHASE_DOWNTREND and phase in ('accum_C', 'accum_D'):
        return '下降趋势中出现明确吸筹信号（弹簧/SOS），止跌可能性较高，可轻仓关注做多机会'
    return {
        'accum_A':   '供给初步耗尽，但反弹力度尚弱，需等待更多确认信号',
        'accum_B':   '主力资金在区间内吸筹，浮筹逐步减少，方向尚未明朗',
        'accum_C':   '关键测试通过（弹簧/最后支撑），主力意图较为明显，可关注做多机会',
        'accum_D':   '放量突破阻力，吸筹完成，趋势性上涨开始，回调是买入良机',
        'dist_A':    '需求初步耗尽，但下跌力度尚弱，需等待更多确认信号',
        'dist_B':    '主力在区间内出货，浮筹逐步增加，方向尚未明朗',
        'dist_C':    '关键测试失败（上冲回落），主力出货意图明显，可关注做空机会',
        'dist_D':    '破位下跌，派发完成，趋势性下跌开始，反弹是卖出良机',
        'uptrend':   '上升趋势延续，当前无明显派发信号，持仓或逢低买入',
        'downtrend': '下降趋势延续，当前无明显吸筹信号，空仓或逢高做空',
        'neutral':   '暂无明显威科夫阶段特征，建议观望等待方向突破',
    }.get(phase, '暂无明显威科夫阶段特征')


def adjust_with_pnf(scores: Dict, pnf_result: Dict, _current_price: float = 0) -> Dict:
    """
    Adjust probability scores using P&F target direction.
    P&F shifts probabilities by up to 5%.
    """
    latest_target = pnf_result.get('latest_target')
    latest_dir    = pnf_result.get('latest_target_dir')

    if latest_target is None or latest_dir is None:
        return scores

    adjusted = dict(scores)
    shift = 0.05

    if latest_dir == 'up':
        adjusted['prob_up']   = min(0.95, scores['prob_up']   + shift)
        adjusted['prob_down'] = max(0.02, scores['prob_down'] - shift * 0.6)
        adjusted['prob_flat'] = max(0.02, scores['prob_flat'] - shift * 0.4)
    else:
        adjusted['prob_down'] = min(0.95, scores['prob_down'] + shift)
        adjusted['prob_up']   = max(0.02, scores['prob_up']   - shift * 0.6)
        adjusted['prob_flat'] = max(0.02, scores['prob_flat'] - shift * 0.4)

    total = adjusted['prob_up'] + adjusted['prob_down'] + adjusted['prob_flat']
    adjusted['prob_up']   = round(adjusted['prob_up']   / total, 3)
    adjusted['prob_down'] = round(adjusted['prob_down'] / total, 3)
    adjusted['prob_flat'] = round(adjusted['prob_flat'] / total, 3)
    adjusted['pnf_adjusted'] = True
    adjusted['pnf_target']   = latest_target
    adjusted['pnf_dir']      = latest_dir

    return adjusted
