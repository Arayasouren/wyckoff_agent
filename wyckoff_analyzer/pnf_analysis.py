"""
Point & Figure (P&F) Chart Analysis.
Translates MATLAB plot_pnf.m / pnfcount.m / pnftarget.m to Python.

Core methodology (研报 Vol.2):
  - Box size = 2% of price (dynamic)
  - Reversal = 3 boxes
  - X = up column, O = down column
  - Trading range identified by RHS/LHS overlap rules
  - Target = column_count × box_size × reversal ± base_price
"""
import numpy as np
import pandas as pd
from typing import Optional


def compute_box_size(price: float) -> float:
    """Dynamic box size: 2% of reference price, rounded to sensible units."""
    raw = price * 0.02
    if raw >= 100:
        return round(raw / 10) * 10
    elif raw >= 10:
        return round(raw)
    elif raw >= 1:
        return round(raw * 2) / 2
    elif raw >= 0.1:
        return round(raw * 20) / 20
    else:
        return round(raw * 200) / 200


def build_pnf_matrix(highs: np.ndarray, lows: np.ndarray,
                     box_size: float, reversal: int = 3) -> tuple:
    """
    Build P&F chart from price data.

    Returns:
      columns: list of dicts {direction: +1/-1, start_box: int, end_box: int, bar: int}
      price_levels: array of price levels (descending)
      base_price: reference price
    """
    if len(highs) == 0:
        return [], np.array([]), 0.0

    base_price = lows[0]
    rev_amount = box_size * reversal

    columns = []
    direction = None  # +1=X (up), -1=O (down)
    col_start_box = 0
    col_end_box = 0

    def price_to_box(p):
        return int((p - base_price) / box_size)

    for i in range(len(highs)):
        hi_box = price_to_box(highs[i])
        lo_box = price_to_box(lows[i])

        if direction is None:
            # Initialize
            if hi_box > lo_box:
                direction = 1
                col_start_box = lo_box
                col_end_box = hi_box
            else:
                direction = -1
                col_start_box = hi_box
                col_end_box = lo_box
            continue

        if direction == 1:  # X column
            if hi_box > col_end_box:
                col_end_box = hi_box
            elif lo_box <= col_end_box - reversal:
                # Reversal: save current X column, start O column
                columns.append({
                    'direction': 1,
                    'start_box': col_start_box,
                    'end_box': col_end_box,
                    'bar': i - 1
                })
                col_start_box = col_end_box - 1
                col_end_box = col_end_box - reversal
                if lo_box < col_end_box:
                    col_end_box = lo_box
                direction = -1
        else:  # O column
            if lo_box < col_end_box:
                col_end_box = lo_box
            elif hi_box >= col_end_box + reversal:
                # Reversal
                columns.append({
                    'direction': -1,
                    'start_box': col_start_box,
                    'end_box': col_end_box,
                    'bar': i - 1
                })
                col_start_box = col_end_box + 1
                col_end_box = col_end_box + reversal
                if hi_box > col_end_box:
                    col_end_box = hi_box
                direction = 1

    # Append last (current) column
    if direction is not None:
        columns.append({
            'direction': direction,
            'start_box': col_start_box,
            'end_box': col_end_box,
            'bar': len(highs) - 1
        })

    # Determine price range
    if not columns:
        return [], np.array([]), base_price

    all_boxes = [c['start_box'] for c in columns] + [c['end_box'] for c in columns]
    min_box = min(all_boxes)
    max_box = max(all_boxes) + 1
    price_levels = np.array([base_price + b * box_size for b in range(max_box, min_box - 1, -1)])

    return columns, price_levels, base_price


def _col_box_range(col: dict) -> set:
    lo = min(col['start_box'], col['end_box'])
    hi = max(col['start_box'], col['end_box'])
    return set(range(lo, hi + 1))


def _overlap_fraction(col_a: dict, col_b: dict) -> float:
    """Fraction of col_a's range that overlaps with col_b."""
    range_a = _col_box_range(col_a)
    range_b = _col_box_range(col_b)
    if not range_a:
        return 0.0
    overlap = range_a & range_b
    return len(overlap) / len(range_a)


def find_trading_ranges(columns: list) -> list:
    """
    Identify P&F trading ranges using RHS/LHS overlap rules (研报方法).

    RHS condition: current col overlaps ≥20% with next 3 cols, AND 4th col overlaps ≥50%
    LHS condition: prev col overlaps <20% with next 4 cols (no leap)
    Extension: continue range while new col overlaps ≥50% with prior 3

    Returns list of dicts:
      {start_col, end_col, width, direction, base_box, target_box, target_price_up, target_price_down}
    """
    n = len(columns)
    if n < 6:
        return []

    ranges = []
    i = 0
    while i < n - 5:
        # Check LHS: previous column has <20% overlap with next 4
        lhs_ok = True
        if i > 0:
            for j in range(i, min(i + 4, n)):
                if _overlap_fraction(columns[i-1], columns[j]) >= 0.2:
                    lhs_ok = False
                    break

        if not lhs_ok:
            i += 1
            continue

        # Check RHS: current col overlaps ≥20% with next 3, AND 4th ≥50%
        if i + 4 >= n:
            i += 1
            continue

        rhs_3 = all(_overlap_fraction(columns[i], columns[i+k]) >= 0.2 for k in range(1, 4))
        rhs_4 = _overlap_fraction(columns[i], columns[i+4]) >= 0.5

        if not (rhs_3 and rhs_4):
            i += 1
            continue

        # Found a range start at column i
        range_start = i
        range_end = i + 4

        # Extend range rightward
        while range_end + 1 < n:
            prior_3 = columns[max(0, range_end-2):range_end+1]
            new_col = columns[range_end + 1]
            overlap_ok = all(_overlap_fraction(new_col, pc) >= 0.5 for pc in prior_3)
            if overlap_ok:
                range_end += 1
            else:
                break

        # Quality filter: height/width ratio ≤ 1.5
        cols_in_range = columns[range_start:range_end+1]
        all_boxes = [b for c in cols_in_range for b in _col_box_range(c)]
        if not all_boxes:
            i = range_end + 1
            continue
        height = max(all_boxes) - min(all_boxes)
        width = range_end - range_start + 1
        if height > 0 and (height / width) > 1.5:
            i = range_end + 1
            continue

        # Determine range direction (price before vs after range)
        mid_price_before = (columns[range_start]['start_box'] + columns[range_start]['end_box']) / 2
        mid_price_after  = (columns[range_end]['start_box'] + columns[range_end]['end_box']) / 2
        if mid_price_after > mid_price_before:
            direction = 1   # upside breakout
        elif mid_price_after < mid_price_before:
            direction = -1  # downside breakdown
        else:
            direction = 0

        ranges.append({
            'start_col': range_start,
            'end_col': range_end,
            'width': width,
            'height': height,
            'direction': direction,
            'min_box': min(all_boxes),
            'max_box': max(all_boxes),
        })

        i = range_end + 1

    return ranges


def compute_targets(trading_ranges: list, columns: list,
                    box_size: float, reversal: int,
                    base_price: float) -> list:
    """
    Calculate P&F price targets for each trading range.

    Formula: target = width × box_size × reversal ± base_price_of_range
    """
    results = []
    for r in trading_ranges:
        width = r['width']
        count_projection = width * box_size * reversal

        # Upside target: from top of range
        top_price = base_price + r['max_box'] * box_size
        # Downside target: from bottom of range
        bot_price = base_price + r['min_box'] * box_size

        target_up   = top_price + count_projection
        target_down = bot_price - count_projection

        # Determine primary target based on direction
        if r['direction'] >= 0:
            primary_target = target_up
            primary_dir = 'up'
        else:
            primary_target = target_down
            primary_dir = 'down'

        results.append({
            **r,
            'target_up': target_up,
            'target_down': target_down,
            'primary_target': primary_target,
            'primary_dir': primary_dir,
            'box_size': box_size,
        })

    return results


def run_pnf_analysis(df: pd.DataFrame) -> dict:
    """
    Full P&F analysis on price DataFrame.

    Returns dict with:
      columns, price_levels, base_price, box_size, trading_ranges,
      targets, latest_target, latest_target_dir
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values

    # Reference price for box size (use median close)
    ref_price = np.median(closes)
    box_size = compute_box_size(ref_price)

    columns, price_levels, base_price = build_pnf_matrix(highs, lows, box_size)

    if not columns:
        return {
            'columns': [], 'price_levels': np.array([]), 'base_price': 0,
            'box_size': box_size, 'trading_ranges': [], 'targets': [],
            'latest_target': None, 'latest_target_dir': None,
        }

    trading_ranges = find_trading_ranges(columns)
    targets = compute_targets(trading_ranges, columns, box_size, reversal=3, base_price=base_price)

    # Latest valid target (most recent range that hasn't been penetrated in opposite direction)
    latest_target = None
    latest_target_dir = None
    current_price = closes[-1]

    for t in reversed(targets):
        if t['primary_dir'] == 'up' and current_price < t['primary_target']:
            latest_target = t['primary_target']
            latest_target_dir = 'up'
            break
        elif t['primary_dir'] == 'down' and current_price > t['primary_target']:
            latest_target = t['primary_target']
            latest_target_dir = 'down'
            break

    return {
        'columns': columns,
        'price_levels': price_levels,
        'base_price': base_price,
        'box_size': box_size,
        'trading_ranges': trading_ranges,
        'targets': targets,
        'latest_target': latest_target,
        'latest_target_dir': latest_target_dir,
    }
