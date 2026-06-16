# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Wyckoff Analyzer — main entry point.

Usage:
  python -m wyckoff_analyzer AAPL
  python -m wyckoff_analyzer 600519.SS --days 400
  python -m wyckoff_analyzer ^GSPC --output-json
  python -m wyckoff_analyzer TSLA --generate-pdf --ai-analysis "AI analysis text..."
"""
import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .data_fetcher import fetch_ohlcv, _normalize_code
from .wave_classifier import classify_waves, get_wave_df
from .phase_detector import detect_phase, build_aux_lines, PHASE_NAMES
from .wyckoff_events import detect_all_events, get_recent_events
from .pnf_analysis import run_pnf_analysis
from .probability_scorer import score_events, adjust_with_pnf


def _safe_json(obj):
    """Make objects JSON serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def run_analysis(code: str, days: int = 500, end_date: str = None) -> dict:
    """
    Full Wyckoff analysis pipeline.

    Returns structured results dict.
    """
    suffix = f" ~ {end_date}" if end_date else ""
    print(f"[Wyckoff] Fetching data for {code} ({days} days{suffix})...", flush=True)
    df = fetch_ohlcv(code, days=days, end_date=end_date)
    print(f"[Wyckoff] Got {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})", flush=True)

    closes = df['close'].values
    current_price = closes[-1]

    # Wave classification
    print("[Wyckoff] Classifying waves...", flush=True)
    waves = classify_waves(closes)
    wdf = get_wave_df(waves)

    # Phase detection
    phase = detect_phase(waves)
    phase_name = PHASE_NAMES.get(phase, '未知')

    # Auxiliary lines
    aux = build_aux_lines(closes, waves, phase)

    # Wyckoff event detection
    print("[Wyckoff] Detecting events (this may take a moment)...", flush=True)
    all_events = detect_all_events(df)
    recent_events = get_recent_events(all_events, n_bars=90)

    # P&F analysis
    print("[Wyckoff] Running P&F analysis...", flush=True)
    pnf = run_pnf_analysis(df)

    # Probability scoring — pass macro phase for nested stage model
    scores = score_events(recent_events, market_phase=phase)
    scores = adjust_with_pnf(scores, pnf, current_price)

    # Wave summary (last 4 waves)
    wave_summary = []
    for _, row in wdf.tail(6).iterrows():
        wave_summary.append({
            'direction': 'up' if row['direction'] == 1 else 'down',
            'start_idx': int(row['start_idx']),
            'end_idx':   int(row['end_idx']),
            'amplitude_pct': round(float(row['amplitude']) * 100, 2),
            'length_bars':   int(row['length']),
        })

    # Recent events summary
    event_summary = []
    _last_idx = len(df) - 1
    for e in recent_events[-20:]:
        _bar_idx = int(e['bar_idx'])
        _bars_from_end = _last_idx - _bar_idx
        try:
            _close_at_event = round(float(df['close'].iloc[_bar_idx]), 4)
        except Exception:
            _close_at_event = None
        event_summary.append({
            'event': e['event'],
            'name_cn': e['name_cn'],
            'date': str(e['date'])[:10],
            'bar_idx': _bar_idx,
            'bars_from_end': _bars_from_end,
            'close_at_event': _close_at_event,
            'polarity': int(e['polarity']),
            'status': e.get('status', 'confirmed'),
            'description': e['description'],
        })

    # P&F summary
    pnf_summary = {
        'box_size': float(pnf['box_size']),
        'num_columns': len(pnf['columns']),
        'trading_ranges': len(pnf['trading_ranges']),
        'latest_target': float(pnf['latest_target']) if pnf['latest_target'] else None,
        'latest_target_dir': pnf['latest_target_dir'],
    }
    if pnf['targets']:
        pnf_summary['all_targets'] = [
            {
                'direction': t['primary_dir'],
                'target': round(float(t['primary_target']), 4),
                'width': int(t['width']),
                'box_size': float(t['box_size']),
            }
            for t in pnf['targets'][-5:]
        ]

    result = {
        'code': code,
        'normalized_code': _normalize_code(code),
        'date': end_date if end_date else str(date.today()),
        'current_price': round(float(current_price), 4),
        'data_range': {
            'start': str(df.index[0])[:10],
            'end':   str(df.index[-1])[:10],
            'bars':  len(df),
        },
        'market_phase': {
            'phase_id': int(phase),
            'phase_name': phase_name,
            'support_lines':    [round(float(x), 4) for x in aux.get('support_lines', [])],
            'resistance_lines': [round(float(x), 4) for x in aux.get('resistance_lines', [])],
            'ice_line': round(float(aux['ice_line']), 4) if aux.get('ice_line') else None,
        },
        'waves': wave_summary,
        'recent_events': event_summary,
        'pnf': pnf_summary,
        'probability': {
            'up':   scores['prob_up'],
            'down': scores['prob_down'],
            'flat': scores['prob_flat'],
            'dominant_phase': scores['dominant_phase'],
            'phase_label': scores['phase_label'],
            'phase_description': scores['phase_description'],
            'pnf_adjusted': scores.get('pnf_adjusted', False),
        },
        # Raw objects for visualization (not JSON serialized)
        '_df': df,
        '_waves': waves,
        '_all_events': all_events,
        '_recent_events': recent_events,
        '_aux': aux,
        '_pnf': pnf,
        '_scores': scores,
    }

    return result


def generate_png(result: dict, output_path: str) -> str:
    """Generate chart PNG from analysis result."""
    from .visualizer import generate_charts_png
    df          = result['_df']
    events      = result['_recent_events']
    aux         = result['_aux']
    pnf_result  = result['_pnf']
    scores      = result['_scores']
    code        = result['code']
    return generate_charts_png(df, events, aux, pnf_result, scores, code, output_path)


def generate_pdf(result: dict, output_path: str, ai_analysis: str = '',
                 existing_png_path: str = None) -> str:
    """
    Assemble PDF report with charts + AI analysis text.

    Args:
      existing_png_path: if provided, use this PNG instead of regenerating it.
                         Used with --reuse-cache to skip re-running analysis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    from matplotlib.backends.backend_pdf import PdfPages
    from .visualizer import _CJK_FONT_CANDIDATES
    import textwrap

    # Build a FontProperties that directly points to the font file.
    # This bypasses the font cache and works reliably in PDF backend.
    _fp_path = next((p for p in _CJK_FONT_CANDIDATES if os.path.exists(p)), None)
    if _fp_path:
        def _fp(size=10, bold=False):
            return _fm.FontProperties(fname=_fp_path, size=size,
                                      weight='bold' if bold else 'normal')
    else:
        def _fp(size=10, bold=False):
            return _fm.FontProperties(size=size, weight='bold' if bold else 'normal')

    BG   = '#1a1a2e'
    TXT  = '#e0e0e0'
    ACC  = '#ffd54f'
    BULL = '#26a69a'
    BEAR = '#ef5350'
    NEUT = '#90a4ae'

    code  = result['code']
    today = result['date']
    price = result['current_price']
    phase = result['market_phase']['phase_name']
    prob  = result['probability']

    if existing_png_path and os.path.exists(existing_png_path):
        tmp_png = existing_png_path
        _delete_tmp = False
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_png = tmp.name
        _delete_tmp = True
        generate_png(result, tmp_png)

    with PdfPages(output_path) as pdf:

        # ── Page 1: Cover ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
        ax.set_facecolor(BG)
        ax.axis('off')

        ax.text(0.5, 0.85, '威科夫技术分析报告', ha='center', va='center',
                color=ACC, fontproperties=_fp(26, bold=True), transform=ax.transAxes)
        ax.text(0.5, 0.75, f'标的: {code}    日期: {today}    当前价: {price:.4g}',
                ha='center', va='center', color=TXT, fontproperties=_fp(14), transform=ax.transAxes)
        ax.text(0.5, 0.67, f'市场状态: {phase}',
                ha='center', va='center', color=ACC, fontproperties=_fp(13), transform=ax.transAxes)

        # Probability summary
        bar_data = [prob['up'], prob['flat'], prob['down']]
        bar_labels = ['上涨概率', '盘整概率', '下跌概率']
        bar_colors = [BULL, NEUT, BEAR]
        bar_y = [0.54, 0.46, 0.38]
        for y, p, lbl, c in zip(bar_y, bar_data, bar_labels, bar_colors):
            ax.barh(y, p * 0.6, height=0.055, color=c, alpha=0.85,
                    left=0.15, transform=ax.transAxes)
            ax.text(0.13, y, lbl, ha='right', va='center', color=TXT,
                    fontproperties=_fp(11), transform=ax.transAxes)
            ax.text(0.16 + p * 0.6, y, f'{p*100:.1f}%', ha='left', va='center',
                    color=c, fontproperties=_fp(11), transform=ax.transAxes)

        phase_lbl = prob.get('phase_label', '')
        ax.text(0.5, 0.28, f'威科夫阶段: {phase_lbl}',
                ha='center', va='center', color=ACC, fontproperties=_fp(11), transform=ax.transAxes)
        ax.text(0.5, 0.21, prob.get('phase_description', ''),
                ha='center', va='center', color=TXT, fontproperties=_fp(10),
                transform=ax.transAxes)
        ax.text(0.5, 0.08,
                '本报告基于威科夫技术分析理论（形态学 + 点数图），结合AI主观判断，仅供参考，不构成投资建议。',
                ha='center', va='center', color='#607d8b', fontproperties=_fp(8),
                transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight', facecolor=BG)
        plt.close(fig)

        # ── Page 2: Multi-panel chart ──────────────────────────────────────
        chart_img = plt.imread(tmp_png)
        fig2, ax2 = plt.subplots(figsize=(16, 11), facecolor=BG)
        ax2.set_facecolor(BG)
        ax2.axis('off')
        ax2.imshow(chart_img, aspect='auto')
        pdf.savefig(fig2, bbox_inches='tight', facecolor=BG)
        plt.close(fig2)

        # ── Page 3: Events table ───────────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(12, 9), facecolor=BG)
        ax3.set_facecolor(BG)
        ax3.axis('off')
        ax3.text(0.5, 0.97, '近期威科夫事件列表', ha='center', va='top',
                 color=ACC, fontproperties=_fp(14, bold=True), transform=ax3.transAxes)

        events = result['recent_events'][-18:]
        if events:
            col_labels = ['日期', '事件代码', '中文名称', '距今(K)', '当日收盘', '信号方向', '含义']
            col_widths = [0.11, 0.09, 0.10, 0.07, 0.09, 0.08, 0.44]
            header_y = 0.91
            for j, (lbl, xw) in enumerate(zip(col_labels, col_widths)):
                x = sum(col_widths[:j]) + 0.01
                ax3.text(x, header_y, lbl, ha='left', va='top', color=ACC,
                         fontproperties=_fp(8.5, bold=True), transform=ax3.transAxes)

            ax3.axhline(header_y - 0.025, color='#4a4a6a', linewidth=0.5, xmin=0.01, xmax=0.99)

            for row_idx, e in enumerate(reversed(events)):
                y = header_y - 0.035 - row_idx * 0.044
                if y < 0.02:
                    break
                color = BULL if e['polarity'] > 0 else (BEAR if e['polarity'] < 0 else NEUT)
                _bfe = e.get('bars_from_end', '')
                _cae = e.get('close_at_event')
                row_data = [
                    e['date'],
                    e['event'],
                    e['name_cn'],
                    f"{_bfe}",
                    f"{_cae:.4g}" if _cae is not None else '-',
                    '看涨' if e['polarity'] > 0 else ('看跌' if e['polarity'] < 0 else '中性'),
                    e['description'][:48],
                ]
                for j, (val, xw) in enumerate(zip(row_data, col_widths)):
                    x = sum(col_widths[:j]) + 0.01
                    ax3.text(x, y, val, ha='left', va='top',
                             color=color if j in (1, 5) else TXT,
                             fontproperties=_fp(7.5), transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, '近期无显著威科夫事件', ha='center', va='center',
                     color=TXT, fontproperties=_fp(12), transform=ax3.transAxes)

        pdf.savefig(fig3, bbox_inches='tight', facecolor=BG)
        plt.close(fig3)

        # ── Page 4: AI analysis ────────────────────────────────────────────
        if ai_analysis:
            fig4, ax4 = plt.subplots(figsize=(12, 9), facecolor=BG)
            ax4.set_facecolor(BG)
            ax4.axis('off')
            ax4.text(0.5, 0.97, 'AI主观分析与判断', ha='center', va='top',
                     color=ACC, fontproperties=_fp(14, bold=True), transform=ax4.transAxes)

            wrapped = textwrap.wrap(ai_analysis, width=95)
            y_pos = 0.91
            for line in wrapped[:50]:
                ax4.text(0.03, y_pos, line, ha='left', va='top', color=TXT,
                         fontproperties=_fp(8.5), transform=ax4.transAxes)
                y_pos -= 0.018
                if y_pos < 0.02:
                    break

            pdf.savefig(fig4, bbox_inches='tight', facecolor=BG)
            plt.close(fig4)

        # ── PDF metadata ──────────────────────────────────────────────────
        d = pdf.infodict()
        d['Title'] = f'威科夫分析报告 - {code} - {today}'
        d['Author'] = 'Wyckoff Analyzer (Claude Skill)'
        d['Subject'] = 'Wyckoff Technical Analysis'

    # Cleanup temp PNG (only if we created it, not if it was passed in)
    if _delete_tmp:
        try:
            os.unlink(tmp_png)
        except Exception:
            pass

    return output_path


def _print_summary(result: dict) -> None:
    code  = result['code']
    today = result['date']
    prob  = result['probability']
    pnf   = result['pnf']
    print('\n' + '=' * 60)
    print(f'  威科夫分析: {code}  ({today})')
    print('=' * 60)
    print(f'  当前价格: {result["current_price"]:.4g}')
    print(f'  市场状态: {result["market_phase"]["phase_name"]}')
    print(f'  支撑线: {result["market_phase"]["support_lines"]}')
    print(f'  阻力线: {result["market_phase"]["resistance_lines"]}')
    print(f'\n  概率评估:')
    print(f'    上涨: {prob["up"]*100:.1f}%')
    print(f'    盘整: {prob["flat"]*100:.1f}%')
    print(f'    下跌: {prob["down"]*100:.1f}%')
    print(f'\n  威科夫阶段: {prob["phase_label"]}')
    print(f'  {prob["phase_description"]}')
    if pnf['latest_target']:
        dir_str = '上涨' if pnf['latest_target_dir'] == 'up' else '下跌'
        print(f'\n  P&F目标价 ({dir_str}): {pnf["latest_target"]:.4g}')
    print(f'\n  近期关键事件:')
    for e in result['recent_events'][-8:]:
        pol = '↑' if e['polarity'] > 0 else ('↓' if e['polarity'] < 0 else '→')
        _bfe = e.get('bars_from_end', '?')
        _cae = e.get('close_at_event')
        _cae_s = f'{_cae:.4g}' if _cae is not None else '-'
        print(f'    {e["date"]}  {pol} {e["event"]}({e["name_cn"]}) 距今{_bfe}K 收盘={_cae_s}')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Wyckoff Technical Analysis')
    parser.add_argument('code', help='Stock/index code (e.g. AAPL, 600519.SS, ^GSPC)')
    parser.add_argument('--days', type=int, default=500, help='Lookback days (default 500)')
    parser.add_argument('--end-date', type=str, default=None, help='Analysis end date YYYY-MM-DD (default today)')
    parser.add_argument('--output-json', action='store_true', help='Output JSON analysis data')
    parser.add_argument('--output-png', action='store_true', help='Save chart PNG for AI viewing')
    parser.add_argument('--generate-pdf', action='store_true', help='Generate PDF report')
    parser.add_argument('--ai-analysis', type=str, default='', help='AI analysis text for PDF')
    parser.add_argument('--reuse-cache', action='store_true',
                        help='Load existing JSON+PNG from output-dir instead of re-running analysis')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    code = args.code
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_code = code.replace('^', '').replace('.', '_')

    # --reuse-cache: load from existing JSON, skip re-running analysis
    if args.reuse_cache and args.generate_pdf:
        # Find latest JSON for this code in output-dir
        candidates = sorted(out_dir.glob(f'wyckoff_{safe_code}_*.json'))
        if not candidates:
            print(f'[Wyckoff] ERROR: No cached JSON found in {out_dir} for {code}', flush=True)
            return
        json_path = candidates[-1]
        with open(json_path, encoding='utf-8') as f:
            result = json.load(f)
        today = result['date']
        # Look for existing PNG
        png_candidates = sorted(out_dir.glob(f'wyckoff_{safe_code}_*.png'))
        existing_png = str(png_candidates[-1]) if png_candidates else None
        pdf_path = out_dir / f'wyckoff_{safe_code}_{today}.pdf'
        generate_pdf(result, str(pdf_path), ai_analysis=args.ai_analysis,
                     existing_png_path=existing_png)
        print(f'[Wyckoff] PDF saved: {pdf_path}')
        _print_summary(result)
        return

    result = run_analysis(code, days=args.days, end_date=args.end_date)
    today = result['date']

    if args.output_json:
        json_path = out_dir / f'wyckoff_{safe_code}_{today}.json'
        json_data = {k: v for k, v in result.items() if not k.startswith('_')}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=_safe_json)
        print(f'[Wyckoff] JSON saved: {json_path}')

    if args.output_png or args.output_json:
        png_path = out_dir / f'wyckoff_{safe_code}_{today}.png'
        generate_png(result, str(png_path))
        print(f'[Wyckoff] Chart PNG saved: {png_path}')

    if args.generate_pdf:
        # Reuse PNG already saved above if available
        existing_png = str(out_dir / f'wyckoff_{safe_code}_{today}.png')
        existing_png = existing_png if os.path.exists(existing_png) else None
        pdf_path = out_dir / f'wyckoff_{safe_code}_{today}.pdf'
        generate_pdf(result, str(pdf_path), ai_analysis=args.ai_analysis,
                     existing_png_path=existing_png)
        print(f'[Wyckoff] PDF saved: {pdf_path}')
    elif not args.output_json and not args.output_png:
        # Default: generate PNG + print summary
        png_path = out_dir / f'wyckoff_{safe_code}_{today}.png'
        generate_png(result, str(png_path))
        print(f'[Wyckoff] Chart saved: {png_path}')

    _print_summary(result)


if __name__ == '__main__':
    main()
