"""
Wyckoff event detection engine v2.

Changes from v1:
- MIN_GAP dedup: each event type has a minimum bar gap to prevent repetitive triggers
- Z-score SC/BC: uses 60-bar rolling Z-score instead of absolute multiples
- State machine: events can be pending -> confirmed / failed
- New events: SOS, SOW, ABS, COB (+ FT as internal confirmer)
- OKR position check: must be near support/resistance
- Two-phase scan: phase 1 detects candidates, phase 2 confirms/fails them
"""
import numpy as np
import pandas as pd
from .wave_classifier import classify_waves, get_wave_df, compute_macd
from .phase_detector import detect_phase, build_aux_lines


# ─── helper ───────────────────────────────────────────────────────────────────

def _sig(x: float, k: float = 1.0) -> float:
    """Sigmoid mapping: maps any real -> (0,1). Clamps to avoid overflow."""
    z = -k * x
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1.0 / (1.0 + np.exp(z))


def _msig(x: float) -> float:
    """MATLAB-style sigmoid: sig.m from wyckoff reference.
    Shifts input by 0.5 and scales by 10: y = 1/(1+exp(-(x-0.5)*10)).
    So x=0.5 -> y=0.5, x=1 -> y≈0.993, x=0 -> y≈0.007.
    """
    z = -(x - 0.5) * 10.0
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1.0 / (1.0 + np.exp(z))


def _vol_ratio(vol: np.ndarray, idx: int, window: int = 20) -> float:
    start = max(0, idx - window)
    avg = np.mean(vol[start:idx]) if idx > start else vol[idx]
    return vol[idx] / avg if avg > 0 else 1.0


def _body_ratio(op: float, cl: float, hi: float, lo: float) -> float:
    rng = hi - lo
    return abs(cl - op) / rng if rng > 0 else 0.5


def _rolling_avg_body(opens, closes, idx, window=20):
    start = max(0, idx - window)
    bodies = np.abs(closes[start:idx] - opens[start:idx])
    return np.mean(bodies) if len(bodies) > 0 else abs(closes[idx] - opens[idx])


def _z_score(arr: np.ndarray, idx: int, window: int = 60) -> float:
    """Rolling Z-score of arr[idx] relative to arr[idx-window:idx]."""
    start = max(0, idx - window)
    segment = arr[start:idx]
    if len(segment) < 10:
        return 0.0
    mean = np.mean(segment)
    std = np.std(segment)
    if std < 1e-9:
        return 0.0
    return (arr[idx] - mean) / std


def _macd_negative_run(hist: np.ndarray, idx: int, lookback: int = 20) -> bool:
    """Check if MACD histogram has been predominantly negative for lookback bars."""
    start = max(0, idx - lookback)
    segment = hist[start:idx]
    if len(segment) < 5:
        return False
    return np.mean(segment < 0) >= 0.7  # at least 70% of bars negative


def _macd_positive_run(hist: np.ndarray, idx: int, lookback: int = 20) -> bool:
    """Check if MACD histogram has been predominantly positive for lookback bars."""
    start = max(0, idx - lookback)
    segment = hist[start:idx]
    if len(segment) < 5:
        return False
    return np.mean(segment > 0) >= 0.7


def _near_level(price: float, level: float, tolerance: float = 0.02) -> bool:
    """Check if price is within tolerance% of level."""
    if level is None or level <= 0:
        return False
    return abs(price - level) / level <= tolerance


# ─── event definitions ────────────────────────────────────────────────────────

EVENT_INFO = {
    # Accumulation / downtrend signals
    'PS':       ('初次支撑',   '下跌趋势中首次出现的支撑，放量止跌阳线，预示抛压减弱'),
    'SC':       ('恐慌抛售',   '极量大阴线，供应耗尽的高潮，常为盘整底部的起点'),
    'AR_down':  ('自动反弹',   'SC后的首次反弹，需求暂时超过供给，确立区间上沿'),
    'ST_down':  ('二次测试',   '重测SC低点，量须小于SC，确认底部供应已耗尽'),
    'TSO':      ('终极震仓',   '强力假跌破支撑后的急速拉升，清洗弱手，常为强力看涨信号'),
    'SPRING':   ('弹簧形态',   '微幅假跌破支撑后快速收复，量缩为佳，经典底部反转信号'),
    'LPS':      ('最后支撑点', '标记阶段开始前的最后低量回调，是良好的买入机会'),
    'JOC':      ('跳越溪流',   '放量突破盘整区间上沿（阻力），标志吸筹完成、上升启动'),
    'SOT_down': ('推进缩短',   '下跌趋势中每次下跌波幅递减，下跌动能减弱，预示底部'),
    'ASY':      ('突破阻力',   '有效突破阻力线，看多信号，确认趋势由跌转涨'),
    'OS_down':  ('超卖信号',   '下跌连续+高量累积，下跌动能过度释放，暗示接近极限反弹'),
    'OS_line':  ('超卖破线',   '下跌趋势中收盘跌破下通道线，趋势加速但伴随超卖风险'),

    # Distribution / uptrend signals
    'PSY':      ('初次供应',   '上涨趋势中首次出现供给，放量上影，预示买盘减弱'),
    'BC':       ('抢购高潮',   '极量大阳线，需求耗尽的高潮，常为盘整顶部的起点'),
    'AR_up':    ('自动回落',   'BC后的首次下跌，供给暂时超过需求，确立区间下沿'),
    'ST_up':    ('二次测试',   '重测BC高点，量须小于BC，确认顶部需求已耗尽'),
    'UT':       ('上冲回落',   '假突破阻力后快速跌回，量能不足，经典顶部反转信号'),
    'LPSY':     ('最后供应点', '下跌阶段前的最后低量反弹，是良好的卖出机会'),
    'BOI':      ('破冰',       '跌回之前盘整区间（破冰线），确认派发完成、下跌启动'),
    'SOT_up':   ('推进缩短',   '上涨趋势中每次上涨波幅递减，上涨动能减弱，预示顶部'),
    'AS':       ('突破支撑',   '有效跌破支撑线，看空信号，确认趋势由涨转跌'),
    'OB_up':    ('超买信号',   '上涨连续+高量累积，上涨动能过度释放，暗示接近极限回调'),
    'OB_line':  ('超买破线',   '上升趋势中收盘突破上通道线，趋势加速但伴随超买风险'),

    # New behavior events
    'SOS':      ('强势信号',   '震荡区右侧放量长阳，需求控制市场，上涨秩序形成信号'),
    'SOW':      ('弱势信号',   '震荡区右侧放量长阴，供应控制市场，下跌秩序形成信号'),
    'ABS':      ('吸收行为',   '阻力位附近高量不跌，需求吸收供应，后续大概率向上突破'),
    'COB':      ('背景改变',   '市场控制力量由需求转为供应（或反之），阶段转换信号'),

    # Universal patterns
    '3H':       ('三高形态',   '连续三根K线顶部创新高，盘整区间顶部耗尽信号'),
    '3L':       ('三低形态',   '连续三根K线底部创新低，盘整区间底部耗尽信号'),
    'VDB':      ('垂直需求柱', '大实体小上影线放量阳线，强劲需求的体现'),
    'VSB':      ('垂直供应柱', '大实体小下影线放量阴线，强劲供给的体现'),
    'OKR_up':   ('关键反包上', '大阳线实体完整覆盖前根大阴线，看涨吞没，底部反转信号'),
    'OKR_down': ('关键反包下', '大阴线实体完整覆盖前根大阳线，看跌吞没，顶部反转信号'),
}

# Polarity: +1 = bullish, -1 = bearish
EVENT_POLARITY = {
    'PS': +1, 'SC': +1, 'AR_down': +1, 'ST_down': +1, 'TSO': +1,
    'SPRING': +1, 'LPS': +1, 'JOC': +1, 'SOT_down': +1, 'ASY': +1,
    'OS_down': +1, 'OS_line': +1,
    'PSY': -1, 'BC': -1, 'AR_up': -1, 'ST_up': -1, 'UT': -1,
    'LPSY': -1, 'BOI': -1, 'SOT_up': -1, 'AS': -1,
    'OB_up': -1, 'OB_line': -1,
    'SOS': +1, 'SOW': -1, 'ABS': +1, 'COB': 0,
    '3H': +1, '3L': -1, 'VDB': +1, 'VSB': -1,
    'OKR_up': +1, 'OKR_down': -1,
}

# ─── MIN_GAP: minimum bars between same event type ──────────────────────────

MIN_GAP = {
    'PS': 30, 'PSY': 30,
    'SC': 60, 'BC': 60,
    'AR_down': 20, 'AR_up': 20,
    'ST_down': 5, 'ST_up': 5,      # allow multiple secondary tests
    'SPRING': 30, 'TSO': 60, 'UT': 30,
    'LPS': 20, 'LPSY': 20,
    'JOC': 30, 'BOI': 30,
    'SOS': 20, 'SOW': 20,
    'ASY': 30, 'AS': 30,
    'SOT_down': 10, 'SOT_up': 10,
    'OS_down': 0, 'OB_up': 0,      # state, can be continuous
    'OS_line': 30, 'OB_line': 30,  # channel break, one-shot event
    'ABS': 30, 'COB': 60,
    '3H': 3, '3L': 3,
    'OKR_up': 10, 'OKR_down': 10,
    'VDB': 10, 'VSB': 10,
}


# ─── detection logic per event ────────────────────────────────────────────────

def _detect_SC(i, opens, highs, lows, closes, volumes, ranges, macd_hist):
    """Selling Climax: Z-score based extreme volume + range, in downtrend."""
    if closes[i] >= opens[i]:
        return 0.0
    if not _macd_negative_run(macd_hist, i):
        return 0.0
    vol_z = _z_score(volumes, i)
    range_z = _z_score(ranges, i)
    # Need at least one z > 1.8 AND both z > 1.2
    if not ((vol_z > 1.8 or range_z > 1.8) and vol_z > 1.2 and range_z > 1.2):
        return 0.0
    br = _body_ratio(opens[i], closes[i], highs[i], lows[i])
    score = _sig((max(vol_z, range_z) - 1.5) * 1.5) * _sig((br - 0.4) * 5)
    return float(score)


def _detect_BC(i, opens, highs, lows, closes, volumes, ranges, macd_hist):
    """Buying Climax: Z-score based extreme volume + range, in uptrend."""
    if closes[i] <= opens[i]:
        return 0.0
    if not _macd_positive_run(macd_hist, i):
        return 0.0
    vol_z = _z_score(volumes, i)
    range_z = _z_score(ranges, i)
    if not ((vol_z > 1.8 or range_z > 1.8) and vol_z > 1.2 and range_z > 1.2):
        return 0.0
    br = _body_ratio(opens[i], closes[i], highs[i], lows[i])
    score = _sig((max(vol_z, range_z) - 1.5) * 1.5) * _sig((br - 0.4) * 5)
    return float(score)


def _detect_PS(i, opens, highs, lows, closes, volumes, waves):
    """Preliminary Support: first support after sustained downtrend."""
    if closes[i] <= opens[i]:
        return 0.0
    wdf = get_wave_df(waves)
    if len(wdf) < 2:
        return 0.0
    last_wave = wdf.iloc[-1]
    if last_wave['direction'] != -1:
        return 0.0
    vr = _vol_ratio(volumes, i)
    score = _sig((vr - 0.8) * 2) * _sig((closes[i] - opens[i]) / max(opens[i], 1e-9) * 20)
    return float(score * 0.7)


def _detect_PSY(i, opens, highs, lows, closes, volumes, waves):
    """Preliminary Supply: first supply after sustained uptrend."""
    if closes[i] >= opens[i]:
        return 0.0
    wdf = get_wave_df(waves)
    if len(wdf) < 2:
        return 0.0
    last_wave = wdf.iloc[-1]
    if last_wave['direction'] != 1:
        return 0.0
    vr = _vol_ratio(volumes, i)
    score = _sig((vr - 0.8) * 2) * _sig((opens[i] - closes[i]) / max(opens[i], 1e-9) * 20)
    return float(score * 0.7)


def _detect_AR(i, opens, highs, lows, closes, volumes, event_history, kind='down'):
    """Automatic Rally (after SC) or Automatic Reaction (after BC)."""
    if kind == 'down':
        if closes[i] <= opens[i]:
            return 0.0
        recent = [e for e in event_history if e[0] == 'SC' and i - e[1] <= 10]
        if not recent:
            return 0.0
        vr = _vol_ratio(volumes, i)
        score = _sig((closes[i] - opens[i]) / max(opens[i], 1e-9) * 30) * _sig((vr - 0.5) * 2)
    else:
        if closes[i] >= opens[i]:
            return 0.0
        recent = [e for e in event_history if e[0] == 'BC' and i - e[1] <= 10]
        if not recent:
            return 0.0
        vr = _vol_ratio(volumes, i)
        score = _sig((opens[i] - closes[i]) / max(opens[i], 1e-9) * 30) * _sig((vr - 0.5) * 2)
    return float(score)


def _detect_ST(i, opens, highs, lows, closes, volumes, event_history, kind='down'):
    """Secondary Test: retest of SC low or BC high with lower volume."""
    if kind == 'down':
        sc_events = [e for e in event_history if e[0] == 'SC']
        if not sc_events:
            return 0.0
        sc_bar = sc_events[-1][1]
        sc_price = lows[sc_bar]
        dist = abs(lows[i] - sc_price) / max(sc_price, 1e-9)
        vol_vs_sc = volumes[i] / max(volumes[sc_bar], 1) if volumes[sc_bar] > 0 else 1.0
        score = _sig((0.05 - dist) * 30) * _sig((0.7 - vol_vs_sc) * 5)
    else:
        bc_events = [e for e in event_history if e[0] == 'BC']
        if not bc_events:
            return 0.0
        bc_bar = bc_events[-1][1]
        bc_price = highs[bc_bar]
        dist = abs(highs[i] - bc_price) / max(bc_price, 1e-9)
        vol_vs_bc = volumes[i] / max(volumes[bc_bar], 1) if volumes[bc_bar] > 0 else 1.0
        score = _sig((0.05 - dist) * 30) * _sig((0.7 - vol_vs_bc) * 5)
    return float(score)


def _detect_SPRING(i, opens, highs, lows, closes, volumes, support_level):
    """Spring: false break below support, quick recovery."""
    if support_level is None:
        return 0.0
    if lows[i] >= support_level:
        return 0.0
    if closes[i] <= support_level:
        return 0.0
    penetration = (support_level - lows[i]) / support_level
    recovery = (closes[i] - lows[i]) / max(highs[i] - lows[i], 1e-9)
    vr = _vol_ratio(volumes, i)
    score = _sig((0.03 - penetration) * 80) * _sig((recovery - 0.5) * 6) * _sig((1.5 - vr) * 2)
    return float(score)


def _detect_TSO(i, opens, highs, lows, closes, volumes, support_level):
    """Terminal Shakeout: violent false break below support with strong recovery."""
    if support_level is None:
        return 0.0
    if lows[i] >= support_level:
        return 0.0
    if closes[i] <= support_level:
        return 0.0
    penetration = (support_level - lows[i]) / support_level
    recovery = (closes[i] - lows[i]) / max(highs[i] - lows[i], 1e-9)
    vr = _vol_ratio(volumes, i)
    score = _sig((penetration - 0.01) * 60) * _sig((recovery - 0.6) * 8) * _sig((vr - 1.2) * 3)
    return float(score)


def _detect_UT(i, opens, highs, lows, closes, volumes, resistance_level):
    """Upthrust: false break above resistance, quick reversal."""
    if resistance_level is None:
        return 0.0
    if highs[i] <= resistance_level:
        return 0.0
    if closes[i] >= resistance_level:
        return 0.0
    penetration = (highs[i] - resistance_level) / resistance_level
    rejection = (highs[i] - closes[i]) / max(highs[i] - lows[i], 1e-9)
    vr = _vol_ratio(volumes, i)
    score = _sig((0.03 - penetration) * 80) * _sig((rejection - 0.5) * 6) * _sig((1.5 - vr) * 2)
    return float(score)


def _detect_LPS(i, opens, highs, lows, closes, volumes, support_level, event_history):
    """Last Point of Support: low-volume pullback after SOS/JOC confirmed."""
    if support_level is None:
        return 0.0
    # Must have prior confirmed SOS or confirmed JOC
    prior = [e for e in event_history
             if e[0] in ('SOS', 'JOC') and i - e[1] <= 30]
    if not prior:
        return 0.0
    dist = abs(closes[i] - support_level) / support_level
    vr = _vol_ratio(volumes, i)
    score = _sig((0.04 - dist) * 40) * _sig((0.9 - vr) * 5)
    return float(score)


def _detect_LPSY(i, opens, highs, lows, closes, volumes, resistance_level, event_history):
    """Last Point of Supply: low-volume rally after SOW confirmed."""
    if resistance_level is None:
        return 0.0
    prior = [e for e in event_history
             if e[0] in ('SOW', 'BOI', 'AS') and i - e[1] <= 30]
    if not prior:
        return 0.0
    dist = abs(closes[i] - resistance_level) / resistance_level
    vr = _vol_ratio(volumes, i)
    score = _sig((0.04 - dist) * 40) * _sig((0.9 - vr) * 5)
    return float(score)


def _detect_JOC(i, opens, highs, lows, closes, volumes, resistance_level):
    """Jump over Creek / Sign of Strength: volume breakout above resistance."""
    if resistance_level is None:
        return 0.0
    if closes[i] <= resistance_level:
        return 0.0
    vr = _vol_ratio(volumes, i)
    breakout = (closes[i] - resistance_level) / resistance_level
    score = _sig((vr - 1.2) * 3) * _sig((breakout - 0.005) * 50)
    return float(score)


def _detect_BOI(i, opens, highs, lows, closes, volumes, support_level):
    """Break of Ice: fall back into prior trading range."""
    if support_level is None:
        return 0.0
    if closes[i] >= support_level:
        return 0.0
    vr = _vol_ratio(volumes, i)
    breakdown = (support_level - closes[i]) / support_level
    score = _sig((breakdown - 0.005) * 50) * _sig((vr - 0.8) * 2)
    return float(score)


def _detect_SOT(i, opens, highs, lows, closes, volumes, waves, kind='down'):
    """Shortening of Thrust: wave amplitude shrinking in trend direction."""
    wdf = get_wave_df(waves)
    target_dir = -1 if kind == 'down' else 1
    trend_waves = wdf[wdf['direction'] == target_dir].tail(3)
    if len(trend_waves) < 3:
        return 0.0
    amps = trend_waves['amplitude'].values
    if amps[0] <= 0:
        return 0.0
    shrink = (amps[0] - amps[-1]) / amps[0]
    score = _sig((shrink - 0.2) * 10)
    return float(score)


def _detect_ASY(i, closes, resistance_level):
    """Breaking resistance (ASY): close above resistance."""
    if resistance_level is None:
        return 0.0
    if closes[i] <= resistance_level:
        return 0.0
    breakout = (closes[i] - resistance_level) / resistance_level
    return float(_sig((breakout - 0.01) * 60))


def _detect_AS(i, closes, support_level):
    """Breaking support (AS): close below support."""
    if support_level is None:
        return 0.0
    if closes[i] >= support_level:
        return 0.0
    breakdown = (support_level - closes[i]) / support_level
    return float(_sig((breakdown - 0.01) * 60))


def _detect_OS_down(i, closes, volumes, waves):
    """Oversold (MATLAB wyckoffevent.m L384-393):
    Triggered when current close < previous close (down bar).
    var1 = sig((downcount + 0.5) / 10)   — consecutive down bars since last up bar
    var2 = sig(rank of vol[i] within current wave vols)
    score = var1 * var2.
    """
    if i < 1 or closes[i] >= closes[i-1]:
        return 0.0
    rets = closes[1:i+1] / closes[:i] - 1.0  # rets[k] is return from bar k to k+1
    # lastup = most recent bar index (in 1..i) where return is > 0; 0 if none
    up_idxs = np.where(rets > 0)[0]
    lastup = int(up_idxs[-1]) + 1 if len(up_idxs) > 0 else 0
    downcount = i - lastup  # bars from lastup+1 through i, inclusive → i - lastup
    var1 = _msig((downcount + 0.5) / 10.0)
    # Current wave's start bar (waves[-1] is the unconfirmed wave)
    if not waves:
        return 0.0
    wstart = int(waves[-1][0])
    wave_vols = volumes[wstart:i+1]
    if len(wave_vols) == 0:
        return 0.0
    rank = float(np.sum(volumes[i] >= wave_vols)) / len(wave_vols)
    var2 = _msig(rank)
    return float(var1 * var2)


def _detect_OB_up(i, closes, volumes, waves):
    """Overbought (MATLAB wyckoffevent.m L363-372):
    Triggered when current close > previous close (up bar).
    var1 = sig((upcount + 0.5) / 10)  — consecutive up bars since last down bar
    var2 = sig(rank of vol[i] within current wave vols)
    score = var1 * var2.
    """
    if i < 1 or closes[i] <= closes[i-1]:
        return 0.0
    rets = closes[1:i+1] / closes[:i] - 1.0
    down_idxs = np.where(rets < 0)[0]
    lastdown = int(down_idxs[-1]) + 1 if len(down_idxs) > 0 else 0
    upcount = i - lastdown
    var1 = _msig((upcount + 0.5) / 10.0)
    if not waves:
        return 0.0
    wstart = int(waves[-1][0])
    wave_vols = volumes[wstart:i+1]
    if len(wave_vols) == 0:
        return 0.0
    rank = float(np.sum(volumes[i] >= wave_vols)) / len(wave_vols)
    var2 = _msig(rank)
    return float(var1 * var2)


# ─── Channel-break events (OB_line / OS_line per MATLAB) ─────────────────────

def _trend_channel_line_prices(waves, phase, bar_i):
    """Return (line_at_i_minus_1, line_at_i) for the parallel channel line
    (MATLAB aline{2}) at current phase. None if not applicable.

    For uptrend:  line = parallel to support, offset through the top between the
                  two most recent bottoms (the upper channel line).
    For downtrend: line = parallel to resistance, offset through the bottom
                  between the two most recent tops (the lower channel line).
    Returns None if fewer than required swing points or phase is consolidation.
    """
    if not waves or len(waves) < 4:
        return None
    wdf = get_wave_df(waves)
    # Exclude current unconfirmed wave when selecting anchors, matching auxline.m
    last4 = wdf.iloc[-5:-1] if len(wdf) > 4 else wdf.iloc[-4:]

    if phase == 1:  # uptrend
        bots = [(int(r['start_idx']), float(r['start_price']))
                for _, r in last4[last4['direction'] == 1].iterrows()]
        if len(bots) < 2:
            return None
        (x1, y1), (x2, y2) = bots[-2], bots[-1]
        mids = [(int(r['end_idx']), float(r['end_price']))
                for _, r in last4[last4['direction'] == 1].iterrows()
                if x1 < int(r['end_idx']) < x2]
        if not mids:
            return None
        x_top, y_top = max(mids, key=lambda t: t[1])
        k = (y2 - y1) / max(x2 - x1, 1)
        return (y_top + k * ((bar_i - 1) - x_top),
                y_top + k * (bar_i - x_top))

    if phase == 2:  # downtrend
        tops = [(int(r['start_idx']), float(r['start_price']))
                for _, r in last4[last4['direction'] == -1].iterrows()]
        if len(tops) < 2:
            return None
        (x1, y1), (x2, y2) = tops[-2], tops[-1]
        mids = [(int(r['end_idx']), float(r['end_price']))
                for _, r in last4[last4['direction'] == -1].iterrows()
                if x1 < int(r['end_idx']) < x2]
        if not mids:
            return None
        x_bot, y_bot = min(mids, key=lambda t: t[1])
        k = (y2 - y1) / max(x2 - x1, 1)
        return (y_bot + k * ((bar_i - 1) - x_bot),
                y_bot + k * (bar_i - x_bot))

    return None


def _detect_OB_line(i, closes, waves, phase):
    """OB_line (MATLAB L354-361): uptrend + 2 aux lines; close crosses
    the upper channel line from below to above between i-1 and i.
    Confidence is fixed at 1.0 in MATLAB.
    """
    if phase != 1 or i < 1:
        return 0.0
    pair = _trend_channel_line_prices(waves, phase, i)
    if pair is None:
        return 0.0
    line_prev, line_now = pair
    if closes[i-1] < line_prev and closes[i] > line_now:
        return 1.0
    return 0.0


def _detect_OS_line(i, closes, waves, phase):
    """OS_line (MATLAB L375-382): downtrend + 2 aux lines; close crosses
    the lower channel line from above to below between i-1 and i.
    """
    if phase != 2 or i < 1:
        return 0.0
    pair = _trend_channel_line_prices(waves, phase, i)
    if pair is None:
        return 0.0
    line_prev, line_now = pair
    if closes[i-1] > line_prev and closes[i] < line_now:
        return 1.0
    return 0.0


def _detect_3H(i, highs):
    """Three Higher Highs."""
    if i < 2:
        return 0.0
    if highs[i] > highs[i-1] > highs[i-2]:
        margin = (highs[i] - highs[i-2]) / highs[i-2]
        return float(_sig((margin - 0.005) * 100))
    return 0.0


def _detect_3L(i, lows):
    """Three Lower Lows."""
    if i < 2:
        return 0.0
    if lows[i] < lows[i-1] < lows[i-2]:
        margin = (lows[i-2] - lows[i]) / lows[i-2]
        return float(_sig((margin - 0.005) * 100))
    return 0.0


def _detect_VDB(i, opens, highs, lows, closes, volumes):
    """Vertical Demand Bar."""
    if closes[i] <= opens[i]:
        return 0.0
    body = closes[i] - opens[i]
    upper_shadow = highs[i] - closes[i]
    total_range = highs[i] - lows[i]
    if total_range <= 0:
        return 0.0
    vr = _vol_ratio(volumes, i)
    br = body / total_range
    shadow_ratio = upper_shadow / total_range
    score = _sig((br - 0.6) * 8) * _sig((0.2 - shadow_ratio) * 20) * _sig((vr - 1.5) * 2)
    return float(score)


def _detect_VSB(i, opens, highs, lows, closes, volumes):
    """Vertical Supply Bar."""
    if closes[i] >= opens[i]:
        return 0.0
    body = opens[i] - closes[i]
    lower_shadow = closes[i] - lows[i]
    total_range = highs[i] - lows[i]
    if total_range <= 0:
        return 0.0
    vr = _vol_ratio(volumes, i)
    br = body / total_range
    shadow_ratio = lower_shadow / total_range
    score = _sig((br - 0.6) * 8) * _sig((0.2 - shadow_ratio) * 20) * _sig((vr - 1.5) * 2)
    return float(score)


def _detect_OKR_up(i, opens, highs, lows, closes, volumes, support_level):
    """Outside Key Reversal Up: bullish engulfing near support."""
    if i < 1:
        return 0.0
    prev_open, prev_close = opens[i-1], closes[i-1]
    curr_open, curr_close = opens[i], closes[i]
    if prev_close >= prev_open:
        return 0.0
    if curr_close <= curr_open:
        return 0.0
    prev_body_lo = min(prev_open, prev_close)
    prev_body_hi = max(prev_open, prev_close)
    curr_body_lo = min(curr_open, curr_close)
    curr_body_hi = max(curr_open, curr_close)
    if curr_body_lo > prev_body_lo or curr_body_hi < prev_body_hi:
        return 0.0
    engulf_ratio = (prev_body_hi - prev_body_lo) / max(curr_body_hi - curr_body_lo, 1e-9)
    vr = _vol_ratio(volumes, i)
    # Position check: must be near support level
    position_bonus = 1.0
    if support_level and _near_level(lows[i], support_level, 0.02):
        position_bonus = 1.3
    elif support_level and not _near_level(lows[i], support_level, 0.05):
        position_bonus = 0.5  # far from support, weaker signal
    score = _sig((engulf_ratio - 0.4) * 5) * _sig((vr - 0.8) * 3) * position_bonus
    return float(min(score, 1.0))


def _detect_OKR_down(i, opens, highs, lows, closes, volumes, resistance_level):
    """Outside Key Reversal Down: bearish engulfing near resistance."""
    if i < 1:
        return 0.0
    prev_open, prev_close = opens[i-1], closes[i-1]
    curr_open, curr_close = opens[i], closes[i]
    if prev_close <= prev_open:
        return 0.0
    if curr_close >= curr_open:
        return 0.0
    prev_body_lo = min(prev_open, prev_close)
    prev_body_hi = max(prev_open, prev_close)
    curr_body_lo = min(curr_open, curr_close)
    curr_body_hi = max(curr_open, curr_close)
    if curr_body_lo > prev_body_lo or curr_body_hi < prev_body_hi:
        return 0.0
    engulf_ratio = (prev_body_hi - prev_body_lo) / max(curr_body_hi - curr_body_lo, 1e-9)
    vr = _vol_ratio(volumes, i)
    # Position check: must be near resistance level
    position_bonus = 1.0
    if resistance_level and _near_level(highs[i], resistance_level, 0.02):
        position_bonus = 1.3
    elif resistance_level and not _near_level(highs[i], resistance_level, 0.05):
        position_bonus = 0.5
    score = _sig((engulf_ratio - 0.4) * 5) * _sig((vr - 0.8) * 3) * position_bonus
    return float(min(score, 1.0))


# ─── new event detectors ─────────────────────────────────────────────────────

def _detect_SOS(i, opens, highs, lows, closes, volumes, resistance_level):
    """Sign of Strength: demand controls market (long bullish bar + volume)."""
    if closes[i] <= opens[i]:
        return 0.0
    body = closes[i] - opens[i]
    avg_body = _rolling_avg_body(opens, closes, i, 60)
    if avg_body <= 0:
        return 0.0
    body_ratio = body / avg_body
    vr = _vol_ratio(volumes, i)
    # Close near high (demand in control)
    rng = highs[i] - lows[i]
    close_near_high = (closes[i] - lows[i]) / rng if rng > 0 else 0.5
    score = (_sig((body_ratio - 1.2) * 3) *
             _sig((vr - 1.2) * 2) *
             _sig((close_near_high - 0.6) * 5))
    # Bonus if near resistance (about to break out)
    if resistance_level and _near_level(closes[i], resistance_level, 0.03):
        score = min(score * 1.2, 1.0)
    return float(score)


def _detect_SOW(i, opens, highs, lows, closes, volumes, support_level):
    """Sign of Weakness: supply controls market (long bearish bar + volume)."""
    if closes[i] >= opens[i]:
        return 0.0
    body = opens[i] - closes[i]
    avg_body = _rolling_avg_body(opens, closes, i, 60)
    if avg_body <= 0:
        return 0.0
    body_ratio = body / avg_body
    vr = _vol_ratio(volumes, i)
    # Close near low (supply in control)
    rng = highs[i] - lows[i]
    close_near_low = (highs[i] - closes[i]) / rng if rng > 0 else 0.5
    score = (_sig((body_ratio - 1.2) * 3) *
             _sig((vr - 1.2) * 2) *
             _sig((close_near_low - 0.6) * 5))
    if support_level and _near_level(closes[i], support_level, 0.03):
        score = min(score * 1.2, 1.0)
    return float(score)


def _detect_ABS(i, opens, highs, lows, closes, volumes, resistance_level):
    """Absorption: high volume near resistance but price holds — demand absorbing supply."""
    if resistance_level is None:
        return 0.0
    if i < 10:
        return 0.0
    # Check last 8 bars near resistance
    window = min(8, i)
    near_count = 0
    bull_body_sum = 0.0
    bear_body_sum = 0.0
    high_vol_count = 0
    max_drawdown = 0.0
    ref_price = closes[i]

    for j in range(i - window, i + 1):
        if _near_level(highs[j], resistance_level, 0.03):
            near_count += 1
        body = abs(closes[j] - opens[j])
        if closes[j] > opens[j]:
            bull_body_sum += body
        else:
            bear_body_sum += body
        if _vol_ratio(volumes, j) > 1.0:
            high_vol_count += 1
        dd = (ref_price - lows[j]) / ref_price
        max_drawdown = max(max_drawdown, dd)

    if near_count < 3:
        return 0.0
    if max_drawdown > 0.02:
        return 0.0  # price dropped too much, not absorption
    if bear_body_sum >= bull_body_sum:
        return 0.0  # down days stronger than up days

    vol_score = _sig((high_vol_count / window - 0.3) * 8)
    body_score = _sig((bull_body_sum / max(bear_body_sum, 1e-9) - 1.2) * 3)
    near_score = _sig((near_count / window - 0.3) * 8)
    return float(vol_score * body_score * near_score)


def _detect_COB(i, event_history, phase, prev_phase):
    """Change of Behavior: background shift between bull/bear control."""
    if phase == prev_phase:
        return 0.0
    # Look for evidence of shift in last 30 bars
    recent = [e for e in event_history if i - e[1] <= 30]
    recent_names = set(e[0] for e in recent)

    # Bull -> Bear COB
    if prev_phase in (1,):  # was uptrend
        bear_evidence = recent_names & {'BC', 'PSY', 'AR_up', 'SOT_up', 'UT', 'SOW', '3L'}
        if len(bear_evidence) >= 2:
            return float(_sig((len(bear_evidence) - 1.5) * 3))

    # Bear -> Bull COB
    if prev_phase in (2,):  # was downtrend
        bull_evidence = recent_names & {'SC', 'PS', 'AR_down', 'SPRING', 'SOS', '3H'}
        if len(bull_evidence) >= 2:
            return float(_sig((len(bull_evidence) - 1.5) * 3))

    return 0.0


# ─── FT (Follow-Through) internal confirmer ─────────────────────────────────

def _check_bullish_ft(i, highs, lows, closes, volumes, lookback=5):
    """Check for bullish follow-through: >=2 bars of three-high with volume in lookback."""
    count = 0
    for j in range(max(2, i - lookback + 1), i + 1):
        if (highs[j] > highs[j-1] and lows[j] > lows[j-1] and
                closes[j] > closes[j-1] and _vol_ratio(volumes, j) >= 1.0):
            count += 1
    return count >= 2


def _check_bearish_ft(i, highs, lows, closes, volumes, lookback=5):
    """Check for bearish follow-through: >=2 bars of three-low with volume."""
    count = 0
    for j in range(max(2, i - lookback + 1), i + 1):
        if (highs[j] < highs[j-1] and lows[j] < lows[j-1] and
                closes[j] < closes[j-1] and _vol_ratio(volumes, j) >= 1.0):
            count += 1
    return count >= 2


# ─── state machine: confirm / fail pending events ───────────────────────────

def _confirm_events(detected, opens, highs, lows, closes, volumes, n_bars):
    """
    Phase 2: scan forward from each pending event to confirm or fail it.
    Modifies detected list in place, adding 'status' field.
    """
    for rec in detected:
        ev = rec['event']
        bar = rec['bar_idx']
        rec['status'] = 'confirmed'  # default: most events are self-confirming

        # PS: confirmed if SC follows within 20 bars
        if ev == 'PS':
            rec['status'] = 'pending'
            for other in detected:
                if other['event'] == 'SC' and 0 < other['bar_idx'] - bar <= 20:
                    rec['status'] = 'confirmed'
                    break

        # SC: confirmed if AR_down follows within 10 bars
        elif ev == 'SC':
            rec['status'] = 'pending'
            for other in detected:
                if other['event'] == 'AR_down' and 0 < other['bar_idx'] - bar <= 10:
                    rec['status'] = 'confirmed'
                    break
            # Also confirm if no AR but price rebounds significantly
            if rec['status'] == 'pending' and bar + 5 < n_bars:
                max_close_after = max(closes[bar+1:min(bar+11, n_bars)])
                if (max_close_after - closes[bar]) / max(abs(closes[bar]), 1e-9) > 0.02:
                    rec['status'] = 'confirmed'

        # BC: confirmed if AR_up follows within 10 bars
        elif ev == 'BC':
            rec['status'] = 'pending'
            for other in detected:
                if other['event'] == 'AR_up' and 0 < other['bar_idx'] - bar <= 10:
                    rec['status'] = 'confirmed'
                    break
            if rec['status'] == 'pending' and bar + 5 < n_bars:
                min_close_after = min(closes[bar+1:min(bar+11, n_bars)])
                if (closes[bar] - min_close_after) / max(abs(closes[bar]), 1e-9) > 0.02:
                    rec['status'] = 'confirmed'

        # PSY: confirmed if BC follows within 20 bars
        elif ev == 'PSY':
            rec['status'] = 'pending'
            for other in detected:
                if other['event'] == 'BC' and 0 < other['bar_idx'] - bar <= 20:
                    rec['status'] = 'confirmed'
                    break

        # SPRING: confirmed if low-volume ST follows within 5-15 bars
        elif ev == 'SPRING':
            rec['status'] = 'pending'
            spring_low = lows[bar]
            spring_vol = volumes[bar]
            for j in range(bar + 3, min(bar + 16, n_bars)):
                near_spring = abs(lows[j] - spring_low) / max(spring_low, 1e-9) <= 0.02
                vol_shrink = volumes[j] < spring_vol * 0.7
                small_body = abs(closes[j] - opens[j]) < abs(closes[bar] - opens[bar]) * 0.8
                above_spring = closes[j] > spring_low
                if near_spring and vol_shrink and small_body and above_spring:
                    rec['status'] = 'confirmed'
                    break

        # JOC: confirmed if low-volume pullback follows; failed if heavy selling
        elif ev == 'JOC':
            rec['status'] = 'pending'
            joc_vol = volumes[bar]
            for j in range(bar + 2, min(bar + 11, n_bars)):
                vr_j = _vol_ratio(volumes, j)
                if vr_j < 0.8 and abs(closes[j] - opens[j]) < abs(closes[bar] - opens[bar]) * 0.5:
                    rec['status'] = 'confirmed'
                    break
                if closes[j] < opens[j] and volumes[j] > joc_vol * 0.9:
                    rec['status'] = 'failed'
                    break

        # UT: confirmed if price drops back quickly
        elif ev == 'UT':
            rec['status'] = 'pending'
            for j in range(bar + 1, min(bar + 6, n_bars)):
                if closes[j] < opens[j] and _vol_ratio(volumes, j) < 1.0:
                    rec['status'] = 'confirmed'
                    break

        # SOS: confirmed if bullish FT follows within 5 bars
        elif ev == 'SOS':
            rec['status'] = 'pending'
            end = min(bar + 6, n_bars)
            if end > bar + 2 and _check_bullish_ft(end - 1, highs, lows, closes, volumes, lookback=5):
                rec['status'] = 'confirmed'

        # SOW: confirmed if bearish FT or no-demand rally follows
        elif ev == 'SOW':
            rec['status'] = 'pending'
            end = min(bar + 6, n_bars)
            if end > bar + 2 and _check_bearish_ft(end - 1, highs, lows, closes, volumes, lookback=5):
                rec['status'] = 'confirmed'
            # Also confirm if followed by low-volume rally (LPSY pattern)
            for j in range(bar + 2, min(bar + 11, n_bars)):
                if closes[j] > opens[j] and _vol_ratio(volumes, j) < 0.7:
                    rec['status'] = 'confirmed'
                    break

    # Apply confidence penalty for pending events
    for rec in detected:
        if rec['status'] == 'pending':
            rec['confidence'] *= 0.6
        elif rec['status'] == 'failed':
            rec['confidence'] *= 0.1  # keep in list but nearly zero weight

    return detected


# ─── main detection loop ──────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.6


def detect_all_events(df: pd.DataFrame) -> list:
    """
    Run event detection over the full OHLCV DataFrame.

    Two-phase approach:
    1. Scan all bars, detect candidates with MIN_GAP dedup
    2. Confirm/fail pending events based on subsequent price action

    Returns list of dicts with fields:
      event, bar_idx, date, confidence, name_cn, description, polarity, status
    """
    opens   = df['open'].values
    highs   = df['high'].values
    lows    = df['low'].values
    closes  = df['close'].values
    volumes = df['volume'].values.astype(float)
    dates   = df.index
    n_bars  = len(closes)

    # Pre-compute ranges and MACD histogram
    ranges = highs - lows
    _, _, macd_hist = compute_macd(closes)

    detected = []
    event_history = []  # (event_name, bar_idx)
    last_event_bar = {}  # {event_name: last_bar_idx} for MIN_GAP
    prev_phase = None

    start_bar = 35

    for i in range(start_bar, n_bars):
        close_slice = closes[:i+1]
        waves = classify_waves(close_slice)
        if len(waves) < 4:
            continue
        phase = detect_phase(waves)
        aux = build_aux_lines(close_slice, waves, phase)
        support = aux['support_level']
        resistance = aux['resistance_level']

        candidates = {}

        # --- Downtrend / Accumulation events ---
        candidates['SC']       = _detect_SC(i, opens, highs, lows, closes, volumes, ranges, macd_hist)
        candidates['PS']       = _detect_PS(i, opens, highs, lows, closes, volumes, waves)
        candidates['AR_down']  = _detect_AR(i, opens, highs, lows, closes, volumes, event_history, 'down')
        candidates['ST_down']  = _detect_ST(i, opens, highs, lows, closes, volumes, event_history, 'down')
        candidates['TSO']      = _detect_TSO(i, opens, highs, lows, closes, volumes, support)
        candidates['SPRING']   = _detect_SPRING(i, opens, highs, lows, closes, volumes, support)
        candidates['LPS']      = _detect_LPS(i, opens, highs, lows, closes, volumes, support, event_history)
        candidates['JOC']      = _detect_JOC(i, opens, highs, lows, closes, volumes, resistance)
        candidates['SOT_down'] = _detect_SOT(i, opens, highs, lows, closes, volumes, waves, 'down')
        candidates['ASY']      = _detect_ASY(i, closes, resistance)
        candidates['OS_down']  = _detect_OS_down(i, closes, volumes, waves)
        candidates['OS_line']  = _detect_OS_line(i, closes, waves, phase)

        # --- Uptrend / Distribution events ---
        candidates['BC']       = _detect_BC(i, opens, highs, lows, closes, volumes, ranges, macd_hist)
        candidates['PSY']      = _detect_PSY(i, opens, highs, lows, closes, volumes, waves)
        candidates['AR_up']    = _detect_AR(i, opens, highs, lows, closes, volumes, event_history, 'up')
        candidates['ST_up']    = _detect_ST(i, opens, highs, lows, closes, volumes, event_history, 'up')
        candidates['UT']       = _detect_UT(i, opens, highs, lows, closes, volumes, resistance)
        candidates['LPSY']     = _detect_LPSY(i, opens, highs, lows, closes, volumes, resistance, event_history)
        candidates['BOI']      = _detect_BOI(i, opens, highs, lows, closes, volumes, support)
        candidates['SOT_up']   = _detect_SOT(i, opens, highs, lows, closes, volumes, waves, 'up')
        candidates['AS']       = _detect_AS(i, closes, support)
        candidates['OB_up']    = _detect_OB_up(i, closes, volumes, waves)
        candidates['OB_line']  = _detect_OB_line(i, closes, waves, phase)

        # --- New behavior events ---
        candidates['SOS']      = _detect_SOS(i, opens, highs, lows, closes, volumes, resistance)
        candidates['SOW']      = _detect_SOW(i, opens, highs, lows, closes, volumes, support)
        candidates['ABS']      = _detect_ABS(i, opens, highs, lows, closes, volumes, resistance)
        candidates['COB']      = _detect_COB(i, event_history, phase, prev_phase)

        # --- Universal patterns ---
        candidates['3H']       = _detect_3H(i, highs)
        candidates['3L']       = _detect_3L(i, lows)
        candidates['VDB']      = _detect_VDB(i, opens, highs, lows, closes, volumes)
        candidates['VSB']      = _detect_VSB(i, opens, highs, lows, closes, volumes)
        candidates['OKR_up']   = _detect_OKR_up(i, opens, highs, lows, closes, volumes, support)
        candidates['OKR_down'] = _detect_OKR_down(i, opens, highs, lows, closes, volumes, resistance)

        # Apply MIN_GAP filter + threshold
        valid = []
        for ev, sc in candidates.items():
            if sc < CONFIDENCE_THRESHOLD:
                continue
            gap = MIN_GAP.get(ev, 0)
            if gap > 0 and i - last_event_bar.get(ev, -9999) < gap:
                continue
            valid.append((ev, sc))

        valid.sort(key=lambda x: x[1], reverse=True)
        top = valid[:1]  # single highest-confidence event per bar

        for ev, sc in top:
            name_cn, desc = EVENT_INFO[ev]
            record = {
                'event': ev,
                'bar_idx': i,
                'date': dates[i],
                'confidence': sc,
                'name_cn': name_cn,
                'description': desc,
                'polarity': EVENT_POLARITY[ev],
                'status': 'confirmed',  # will be updated in phase 2
            }
            detected.append(record)
            event_history.append((ev, i))
            last_event_bar[ev] = i

        # Phase reset: clear last_event_bar when phase changes
        # but preserve events from current bar so they still enforce MIN_GAP
        if prev_phase is not None and phase != prev_phase:
            current_bar_events = {ev: idx for ev, idx in last_event_bar.items() if idx == i}
            last_event_bar.clear()
            last_event_bar.update(current_bar_events)
        prev_phase = phase

    # Phase 2: confirm/fail pending events
    detected = _confirm_events(detected, opens, highs, lows, closes, volumes, n_bars)

    return detected


def get_recent_events(events: list, n_bars: int = 60) -> list:
    """Return events from the last n_bars."""
    if not events:
        return []
    max_bar = max(e['bar_idx'] for e in events)
    return [e for e in events if max_bar - e['bar_idx'] <= n_bars]
