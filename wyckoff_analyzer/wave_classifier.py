# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
MACD-based swing wave classifier.
Translates MATLAB waveclassify.m logic to Python.

Returns alternating swing waves: [(start_idx, start_price, end_idx, end_price), ...]
"""
import numpy as np
import pandas as pd


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.zeros_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def compute_macd(close: np.ndarray, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    macd_hist = dif - dea
    return dif, dea, macd_hist


def classify_waves(close: np.ndarray) -> list:
    """
    Segment price into swing waves using MACD histogram sign changes.

    Returns list of tuples: (start_idx, start_price, end_idx, end_price)
    Each entry represents one completed swing.
    """
    if len(close) < 35:
        return []

    _, _, hist = compute_macd(close)

    waves = []
    direction = 1 if hist[-1] >= 0 else -1  # +1 up, -1 down

    # Find initial direction from first non-zero bar
    for i in range(len(hist)):
        if hist[i] != 0:
            direction = 1 if hist[i] > 0 else -1
            break

    wave_start_idx = 0
    wave_start_price = close[0]
    wave_extreme_idx = 0
    wave_extreme_price = close[0]

    for i in range(1, len(hist)):
        new_dir = 1 if hist[i] >= 0 else -1

        if new_dir == direction:
            # Continue current wave: track extreme
            if direction == 1 and close[i] > wave_extreme_price:
                wave_extreme_idx = i
                wave_extreme_price = close[i]
            elif direction == -1 and close[i] < wave_extreme_price:
                wave_extreme_idx = i
                wave_extreme_price = close[i]
        else:
            # Direction changed: save completed wave
            waves.append((wave_start_idx, wave_start_price,
                          wave_extreme_idx, wave_extreme_price))
            # Start new wave
            wave_start_idx = wave_extreme_idx
            wave_start_price = wave_extreme_price
            wave_extreme_idx = i
            wave_extreme_price = close[i]
            direction = new_dir

    # Add the current (incomplete) wave
    waves.append((wave_start_idx, wave_start_price,
                  wave_extreme_idx, wave_extreme_price))

    return waves


def get_wave_df(waves: list) -> pd.DataFrame:
    """Convert wave list to DataFrame for easier access."""
    if not waves:
        return pd.DataFrame(columns=['start_idx', 'start_price', 'end_idx', 'end_price'])
    df = pd.DataFrame(waves, columns=['start_idx', 'start_price', 'end_idx', 'end_price'])
    df['direction'] = np.where(df['end_price'] > df['start_price'], 1, -1)
    df['amplitude'] = (df['end_price'] - df['start_price']).abs() / df['start_price']
    df['length'] = df['end_idx'] - df['start_idx']
    return df
