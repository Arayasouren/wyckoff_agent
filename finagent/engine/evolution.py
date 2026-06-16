"""
Mode 1: Self-learning evolution pipeline.
Rolls through historical windows, predicts, critiques, then evolves the active profile.
"""
from __future__ import annotations
import asyncio
import itertools
import logging
import math
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console

from finagent.config import (
    PREDICTION_HORIZON, STEP_SIZE, HOLDOUT_RATIO,
    INDEX_BATCH_MONTHS, INDEX_TRAIN_MONTHS, INDEX_HOLDOUT_MONTHS,
    WORST_N_FOR_REFLECTOR, WORST_SCORE_CAP_FOR_REFLECTOR,
    DEFAULT_MODEL,
    TRAIN_CONCURRENCY,
    MAX_TOKENS_EVOLVER,
    PROFILE_HISTORY_DIR,
    PROFILES_DIR,
)
from finagent.storage.database import Database, PredictionRecord, CritiqueRecord
from finagent.storage.profile_store import ProfileStore
from finagent.wyckoff_bridge import get_wyckoff_snapshot, format_actual_outcome
from finagent.engine.rolling_window import (
    generate_date_windows, generate_monthly_windows,
    fetch_trading_dates_and_prices,
    get_horizon_prices, InsufficientDataError,
)
from finagent.agents.predictor import PredictorAgent, PredictionParseError
from finagent.agents.base import LLMCallError, get_api_error_summary
from finagent.agents.critic import CriticAgent
from finagent.agents.reflector import ReflectorAgent
from finagent.agents.evolver import EvolverAgent
from finagent.prompts.templates import build_predictor_user_prompt
from finagent.storage.memory_store import MemoryManager
from finagent.utils.stock_info import ensure_symbol_tags

logger = logging.getLogger(__name__)
console = Console()

_EVOLVABLE_PER_SYMBOL = frozenset({
    "symbol_specific_notes",
})


def _apply_symbol_overrides(strategy: dict, symbol: str) -> dict:
    """Merge per-symbol overrides into a copy of strategy before use."""
    overrides = strategy.get("per_symbol_overrides", {}).get(symbol, {})
    if not overrides:
        return strategy
    merged = dict(strategy)
    for key in _EVOLVABLE_PER_SYMBOL:
        if key in overrides:
            merged[key] = overrides[key]
    return merged


async def run_evolution_index_batches(
    symbol: str,
    model: str = DEFAULT_MODEL,
    auto_apply: bool = False,
    profile_name: Optional[str] = None,
    concurrency: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Index-mode driver: train through ALL non-overlapping 60-month batches sequentially.
    Each batch = 40 train + 20 holdout. Profile evolves between batches.
    Leftover months at the end (< 60) are skipped — only floor(N/60) batches run.

    max_batches: if set, only run the first N batches (for debug / smoke test).

    Returns {symbol, profile, n_batches_planned, n_batches_completed, batch_results: [...]}.
    """
    store = ProfileStore()
    if profile_name:
        strategy = store.load(profile_name)
        active_name = profile_name
    else:
        active_name, strategy = store.load_active()

    data_source_type = strategy.get("data_source_type", "stock")
    if data_source_type != "index":
        console.print(f"[red]ERROR: profile '{active_name}' 不是 index 类型，请用 run_evolution 直接调用[/]")
        return []

    # Fetch data once + generate all monthly windows
    try:
        trading_dates, _ = fetch_trading_dates_and_prices(
            symbol, data_source_type="index",
        )
    except InsufficientDataError as e:
        console.print(f"[red]ERROR 数据不足: {e}[/]")
        return {
            "symbol": symbol, "profile": active_name,
            "n_batches_planned": 0, "n_batches_completed": 0,
            "batch_results": [{"error": str(e)}],
        }

    all_windows = generate_monthly_windows(trading_dates)
    n_total = len(all_windows)
    n_batches_full = n_total // INDEX_BATCH_MONTHS
    n_batches = n_batches_full if max_batches is None else min(n_batches_full, max_batches)

    _cap_note = f"（用户限制 max_batches={max_batches}）" if max_batches and max_batches < n_batches_full else ""
    console.print(
        f"\n[bold cyan]━━━ 指数批量训练 {symbol} ━━━[/]\n"
        f"  数据范围: {trading_dates[0]} → {trading_dates[-1]}\n"
        f"  月度窗口: {n_total}  完整批次: floor({n_total}/{INDEX_BATCH_MONTHS}) = {n_batches_full}  本次跑 {n_batches} 批{_cap_note}\n"
        f"  尾部余 {n_total - n_batches_full * INDEX_BATCH_MONTHS} 个月度窗口跳过"
    )

    if n_batches == 0:
        console.print(f"[yellow]月度窗口不足 {INDEX_BATCH_MONTHS} 个，单次按比例切分跑[/]")
        result = await run_evolution(
            symbol, model=model, auto_apply=auto_apply,
            profile_name=profile_name, concurrency=concurrency,
            force_evolution=True,
        )
        return {
            "symbol": symbol, "profile": active_name,
            "n_batches_planned": 0, "n_batches_completed": 1 if "error" not in result else 0,
            "batch_results": [result],
        }

    # Critical: between-batch handoff requires that each batch's evolved profile be
    # written to the ACTIVE file (not the candidate), so the next batch's store.load()
    # picks up the new prompts/memory. Force auto_apply=True regardless of caller flag.
    if not auto_apply:
        console.print(
            "[dim]  注: 批次模式强制 auto_apply=True（否则批次间无法传递进化结果）[/]"
        )

    batch_results: list[dict] = []
    for batch_idx in range(n_batches):
        start = batch_idx * INDEX_BATCH_MONTHS
        end = start + INDEX_BATCH_MONTHS
        batch_windows = all_windows[start:end]
        label = (
            f"[批次 {batch_idx + 1}/{n_batches}  "
            f"{batch_windows[0].window_end_date} → {batch_windows[-1].window_end_date}]"
        )
        console.print(f"\n[bold magenta]══════ {label} ═════[/]")

        try:
            result = await run_evolution(
                symbol, model=model,
                auto_apply=True,            # forced — inter-batch handoff requires it
                profile_name=active_name, concurrency=concurrency,
                force_evolution=True,
                evolve_every_n_new_windows=0,  # batch driver controls cadence externally
                windows_override=batch_windows,
                batch_label=label,
            )
            batch_results.append({"batch": batch_idx + 1, **result})
            if "error" in result:
                console.print(f"[red]批次 {batch_idx + 1} 返回 error，终止后续批次[/]")
                break
        except Exception as e:
            console.print(f"[red]批次 {batch_idx + 1} 抛异常: {type(e).__name__}: {e}[/]")
            batch_results.append({"batch": batch_idx + 1, "error": f"{type(e).__name__}: {e}"})
            break  # don't continue — next batch would start from corrupted state

    _completed = [r for r in batch_results if "error" not in r]
    console.print(f"\n[bold green]━━━ 完成 {len(_completed)}/{n_batches} 个批次 ━━━[/]")
    return {
        "symbol": symbol, "profile": active_name,
        "n_batches_planned": n_batches, "n_batches_completed": len(_completed),
        "batch_results": batch_results,
    }


async def run_evolution(
    symbol: str,
    model: str = DEFAULT_MODEL,
    step_size: int = STEP_SIZE,
    max_windows: Optional[int] = None,
    auto_apply: bool = False,
    skip_evolution: bool = False,
    force_evolution: bool = False,
    profile_name: Optional[str] = None,
    evolve_every_n_new_windows: Optional[int] = None,  # None = use profile setting
    concurrency: Optional[int] = None,  # None = use config.TRAIN_CONCURRENCY
    windows_override: Optional[list] = None,  # Skip auto-generation, use these windows
    batch_label: str = "",                    # Shown in console for batch logging
) -> dict:
    """
    Full Mode 1 pipeline.
    Trains against the active profile (or the specified profile_name).
    Returns summary dict.
    """
    # First-run: build the seqstats reference table from CSV if needed.
    from finagent.stats.seqstats import ensure_seqstats_built
    ensure_seqstats_built()

    db = Database()
    await db.init_schema()
    store = ProfileStore()

    if profile_name:
        strategy = store.load(profile_name)
        active_name = profile_name
    else:
        active_name, strategy = store.load_active()

    # Apply per-symbol overrides (symbol_specific_notes)
    strategy = _apply_symbol_overrides(strategy, symbol)

    version = strategy.get("profile_version", 0)
    data_source_type = strategy.get("data_source_type", "stock")
    _batch_tag = f"  {batch_label}" if batch_label else ""
    console.print(
        f"[bold cyan]evolve[/] {symbol}  profile=[magenta]{active_name}[/] v{version}"
        f"  model={model}  data={data_source_type}{_batch_tag}"
    )

    # Save profile snapshot before any changes
    import shutil as _shutil
    from datetime import datetime as _dt
    PROFILE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _snap_name = f"{active_name}_v{version}_{_dt.now().strftime('%Y%m%d_%H%M%S')}.json"
    _shutil.copy2(store.dir / f"{active_name}.json", PROFILE_HISTORY_DIR / _snap_name)

    # Fetch full trading history (routed by data_source_type)
    try:
        trading_dates, price_series = fetch_trading_dates_and_prices(
            symbol, data_source_type=data_source_type,
        )
    except InsufficientDataError as e:
        console.print(f"[red]ERROR 数据不足: {e}[/]")
        return {"error": str(e)}

    console.print(
        f"  交易日数: {len(trading_dates)}  ({trading_dates[0]} ~ {trading_dates[-1]})"
    )

    # Windows: use override if caller already sliced a batch, else generate from data
    if windows_override is not None:
        windows = list(windows_override)
    elif data_source_type == "index":
        windows = generate_monthly_windows(trading_dates)
    else:
        windows = list(generate_date_windows(
            trading_dates,
            horizon_days=PREDICTION_HORIZON,
            step_days=step_size,
        ))
    if max_windows:
        windows = windows[:max_windows]

    total = len(windows)

    # Train/holdout split
    if data_source_type == "index" and total >= INDEX_BATCH_MONTHS:
        # 60 monthly windows: 40 train + 20 holdout
        batch_windows = windows[-INDEX_BATCH_MONTHS:] if total > INDEX_BATCH_MONTHS else windows
        train_windows = batch_windows[:INDEX_TRAIN_MONTHS]
        holdout_windows = batch_windows[INDEX_TRAIN_MONTHS:]
        console.print(
            f"  本批次窗口: {len(batch_windows)} (训练 {len(train_windows)} + 验证 {len(holdout_windows)})  "
            f"{batch_windows[0].window_end_date} → {batch_windows[-1].window_end_date}"
        )
    else:
        # Stock path (or index with insufficient months): existing HOLDOUT_RATIO logic
        holdout_count = max(1, math.ceil(total * HOLDOUT_RATIO))
        train_windows = windows[:-holdout_count]
        holdout_windows = windows[-holdout_count:]
        if data_source_type == "index":
            console.print(
                f"  [yellow]月度窗口仅 {total} 个 < {INDEX_BATCH_MONTHS}，按比例 {HOLDOUT_RATIO:.0%} 切分[/]"
            )
        console.print(f"  总窗口: {total}  训练: {len(train_windows)}  验证: {len(holdout_windows)}")

    n_train = len(train_windows)

    # Phase 1: Predict + Critique (training windows) — concurrent via semaphore
    from finagent.storage.embedder import get_default_embedder
    _embedder = get_default_embedder()
    mem = MemoryManager(active_name, embedder=_embedder)
    # Auto-build embedding index if missing or stale (new notes added since last run)
    await mem.ensure_embeddings_built()

    symbol_tags, tags_updated = ensure_symbol_tags(symbol, strategy, model=model)
    if tags_updated:
        store.patch(active_name, {"symbol_tags": strategy["symbol_tags"]})
    tags_label = "已更新" if tags_updated else "已有"
    console.print(f"  标签({tags_label}): {symbol_tags}")
    predictor = PredictorAgent(strategy, model=model, memory_manager=mem, symbol_tags=symbol_tags)
    critic = CriticAgent(strategy, model=model)

    processed = 0
    skipped = 0
    in_flight = 0
    loop = asyncio.get_running_loop()

    _DIR_ICON = {"bullish": "▲", "bearish": "▼", "neutral": "—"}
    _concurrency = concurrency if concurrency and concurrency > 0 else TRAIN_CONCURRENCY
    semaphore = asyncio.Semaphore(_concurrency)

    # Spinner runs in a real OS thread — completely independent of the asyncio event loop.
    # A threading.Lock coordinates with result prints so they don't interleave.
    _SPIN_CHARS = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    _spin_stop = threading.Event()
    _last_spin_len = [0]
    # Buffered output: (window_index, line) — printed in order after training completes
    _result_lines: list[tuple[int, str]] = []

    def _spin_thread_fn() -> None:
        while not _spin_stop.is_set():
            done = processed + skipped
            pct = int(done / n_train * 100) if n_train else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            line = (
                f"  {next(_SPIN_CHARS)} 训练  "
                f"|{bar}| {done}/{n_train} {pct}%  "
                f"新={processed} 跳={skipped} 并={in_flight}"
            )
            sys.stderr.write(f"\r{line}")
            sys.stderr.flush()
            _last_spin_len[0] = len(line)
            time.sleep(0.12)
        sys.stderr.write("\r" + " " * (_last_spin_len[0] + 2) + "\r")
        sys.stderr.flush()

    spin_thread = threading.Thread(target=_spin_thread_fn, daemon=True)
    spin_thread.start()

    async def _process_window(i: int, window) -> None:
        nonlocal processed, skipped, in_flight
        async with semaphore:
            in_flight += 1
            date = window.window_end_date
            try:
                # Skip already processed
                if await db.prediction_exists(active_name, symbol, date):
                    skipped += 1
                    return

                # Wyckoff snapshot (sync → executor), retry once with longer timeout
                snapshot = None
                for _snap_attempt, (_timeout, _label) in enumerate([(90.0, ""), (180.0, " retry")]):
                    try:
                        snapshot = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda d=date: get_wyckoff_snapshot(symbol, end_date=d, days=500, data_source_type=data_source_type),
                            ),
                            timeout=_timeout,
                        )
                        break
                    except asyncio.TimeoutError:
                        if _snap_attempt == 0:
                            continue
                        _result_lines.append((i, f"[yellow]TIMEOUT {symbol} {date} snapshot[/]"))
                        return
                    except Exception as e:
                        _result_lines.append((i, f"[red]ERROR {symbol} {date} snapshot: {e}[/]"))
                        return

                # Predict
                user_msg = build_predictor_user_prompt(snapshot["text"], strategy, symbol_tags=symbol_tags)
                try:
                    prediction = await predictor.predict(user_msg, snapshot=snapshot)
                except (PredictionParseError, LLMCallError) as e:
                    _result_lines.append((i, f"[red]ERROR {symbol} {date} predict: {e}[/]"))
                    return

                # Save prediction
                pred_rec = PredictionRecord(
                    profile_name=active_name,
                    symbol=symbol,
                    strategy_version=version,
                    window_end_date=date,
                    horizon_start_date=window.horizon_start_date,
                    horizon_end_date=window.horizon_end_date,
                    direction=prediction["direction"],
                    confidence=prediction["confidence"],
                    key_support=prediction.get("key_support"),
                    key_resistance=prediction.get("key_resistance"),
                    target_price=prediction.get("target_price"),
                    rationale=prediction.get("rationale", ""),
                    key_signals=prediction.get("key_signals", []),
                    risk_factors=prediction.get("risk_factors", []),
                    raw_llm_response=prediction.get("raw_llm_response", ""),
                )
                pred_id = await db.save_prediction(pred_rec)

                # Actual horizon prices
                horizon_prices = get_horizon_prices(
                    price_series, window.horizon_start_date,
                    window.horizon_end_date, trading_dates,
                )
                if not horizon_prices:
                    _result_lines.append((i, f"  [dim]{symbol} {date} 无收益数据，跳过批判[/]"))
                    return

                entry_price = price_series.get(date, horizon_prices[0])
                actual = format_actual_outcome(entry_price, horizon_prices)

                # Critique
                try:
                    critique = await critic.evaluate(prediction, actual)
                except Exception as e:
                    _result_lines.append((i, f"[red]ERROR {symbol} {date} critique: {e}[/]"))
                    return

                crit_rec = CritiqueRecord(
                    prediction_id=pred_id,
                    symbol=symbol,
                    actual_direction=critique["actual_direction"],
                    actual_return_pct=critique["actual_return_pct"],
                    max_drawdown_pct=critique["max_drawdown_pct"],
                    max_gain_pct=critique["max_gain_pct"],
                    direction_correct=critique["direction_correct"],
                    support_hit=critique.get("support_hit"),
                    resistance_hit=critique.get("resistance_hit"),
                    target_hit=critique.get("target_hit"),
                    score=critique["score"],
                    critique_text=critique.get("critique_text", ""),
                    what_worked=critique.get("what_worked", ""),
                    what_failed=critique.get("what_failed", ""),
                    improvement_hints=critique.get("improvement_hints", []),
                    raw_llm_response=critique.get("raw_llm_response", ""),
                )
                await db.save_critique(crit_rec)
                processed += 1

                dir_icon = _DIR_ICON.get(prediction["direction"], "?")
                correct_sym = "✓" if critique["direction_correct"] else "✗"
                color = "green" if critique["direction_correct"] else "red"
                _result_lines.append((i,
                    f"  [{i}/{n_train}]  {symbol}  {date}  "
                    f"{dir_icon} {prediction['direction'][:4]}  "
                    f"conf={prediction['confidence']:.2f}  score={critique['score']:.2f}  "
                    f"[{color}]{correct_sym}[/]"
                ))

                # B1/B3: Update cross-stock validation scores for ALL triggered notes
                triggered = prediction.get("triggered_memory", [])
                if triggered:
                    window_date = window.window_end_date
                    was_correct = critique["direction_correct"]
                    for fn in triggered:
                        await loop.run_in_executor(
                            None,
                            lambda f=fn: mem.update_note_outcome(f, symbol, was_correct, symbol_tags=symbol_tags)
                        )
                        await db.save_memory_outcome(
                            note_filename=fn,
                            symbol=symbol,
                            window_end_date=window_date,
                            direction_correct=was_correct,
                            score=critique["score"],
                        )

                # L2 Deepening: if memory fired but we were still wrong, refine the note
                if triggered and not critique["direction_correct"] and critique["score"] < 0.30:
                    async def _refine(fn=None):
                        refined = await mem.refine_note_if_needed(
                            fn, snapshot, prediction, critique,
                            llm_fn=lambda p: predictor.call(p),
                        )
                        if refined:
                            _result_lines.append((i, f"  [dim cyan]  ↻ L2 refined: {fn}[/]"))
                    await asyncio.gather(*[_refine(fn) for fn in triggered])

            finally:
                in_flight -= 1

    await asyncio.gather(*[_process_window(i, w) for i, w in enumerate(train_windows, 1)])
    _spin_stop.set()
    spin_thread.join()

    stats = await db.get_summary_stats(active_name)
    sym_stats = await db.get_summary_stats(active_name, symbol)
    api_errors = get_api_error_summary()
    console.print(
        f"\n[green]预测+批判完成[/] 新处理={processed} 跳过={skipped}  "
        f"总记录={stats['total_windows']}  "
        f"avg_score={sym_stats['avg_score']:.3f}  "
        f"方向准确率={sym_stats['direction_accuracy']:.1%}"
    )
    if api_errors:
        parts = ", ".join(f"{reason}: {cnt}次" for reason, cnt in api_errors.items())
        console.print(f"[yellow]  API失败（已重试仍失败）: {parts}[/]")

    # Print all results in window order (after summary so the summary line stays visible)
    if _result_lines:
        console.print(f"[dim]── 训练明细（共 {len(_result_lines)} 条，含新处理+错误）──[/]")
        for _, line in sorted(_result_lines, key=lambda x: x[0]):
            console.print(line)

    # Print memory hit report
    _print_memory_hit_report(mem)

    # Memory compression: merge similar notes if MEMORY.md exceeds threshold
    try:
        merged_count = await mem.compress_if_needed(
            llm_fn=lambda p: predictor.call(p),
        )
        if merged_count:
            console.print(f"  [dim cyan]记忆压缩: 合并了 {merged_count} 条相似笔记[/]")
    except Exception as e:
        logger.warning(f"Memory compression failed: {e}")

    # Check skip conditions for evolution phase
    evolve_every = evolve_every_n_new_windows  # CLI override
    if evolve_every is None:
        evolve_every = strategy.get("evolution_params", {}).get("evolve_every_n_new_windows", 0)

    if skip_evolution:
        console.print("[yellow]跳过进化阶段（--skip-evolution）[/]")
        return {**sym_stats, "new_windows": processed}
    if sym_stats["total_windows"] < 10:
        console.print("[yellow]跳过进化阶段（总窗口数不足10）[/]")
        return {**sym_stats, "new_windows": processed}
    if processed == 0 and not force_evolution:
        console.print("[yellow]跳过进化阶段（无新窗口，数据未变化）[/]")
        return {**sym_stats, "new_windows": processed}
    if evolve_every > 0 and processed < evolve_every and not force_evolution:
        console.print(
            f"[yellow]跳过进化阶段（新处理窗口 {processed} < 阈值 {evolve_every}）[/]"
        )
        return {**sym_stats, "new_windows": processed}

    # Phase 2: Reflect + Evolve
    console.print("\n开始进化阶段...")

    # In batch mode (windows_override set), restrict Reflector inputs to the CURRENT
    # batch's window date range — avoids the "old grudges" problem where ancient bad
    # predictions keep dominating worst-N selection across batches.
    _scope_since = None
    _scope_until = None
    _scope_label = ""
    if windows_override is not None and train_windows:
        _scope_since = train_windows[0].window_end_date
        _scope_until = train_windows[-1].window_end_date
        _scope_label = f"（范围 {_scope_since} → {_scope_until}）"

    worst = await db.get_worst_predictions(
        active_name, symbol=symbol,
        n=WORST_N_FOR_REFLECTOR, max_score=WORST_SCORE_CAP_FOR_REFLECTOR,
        since_date=_scope_since, until_date=_scope_until,
    )
    best = await db.get_best_predictions(
        active_name, symbol=symbol, n=WORST_N_FOR_REFLECTOR,
        since_date=_scope_since, until_date=_scope_until,
    )
    recent = await db.get_recent_predictions(
        active_name, n=20,
        symbol=symbol if windows_override is not None else None,
        since_date=_scope_since, until_date=_scope_until,
    )
    direction_stats = await db.get_direction_stats(active_name, symbol=symbol)

    if not worst:
        console.print(
            f"[yellow]跳过进化阶段：无评分 < {WORST_SCORE_CAP_FOR_REFLECTOR} 的失败记录，"
            f"当前表现已达标无需反思[/]"
        )
        return stats

    console.print(
        f"  Reflector 分析 {len(worst)} 条最差（score<{WORST_SCORE_CAP_FOR_REFLECTOR}）"
        f" + {len(best)} 条最优 + {len(recent)} 条近期预测"
        f"{_scope_label}"
        f"（当前股票方向准确率 {direction_stats['overall_win_pct']:.1%}，"
        f"neutral率 {direction_stats['neutral_pct']:.1%}）..."
    )
    reflector = ReflectorAgent(strategy, model=model)
    memory_index = mem.load_index()
    # Build compact retrieval_text list so reflector can see existing notes and avoid duplicates
    _all_notes = mem.load_all_notes()
    _rt_lines = []
    for _n in _all_notes:
        _rt = _n.get("retrieval_text", "").strip()
        _s = _n.get("situation", "").strip()
        if _rt:
            _rt_lines.append(f"- {_s}: {_rt[:120]}")
        elif _s:
            _rt_lines.append(f"- {_s}")
    memory_triggers = "\n".join(_rt_lines) if _rt_lines else ""
    diagnosis, memory_suggestions = await reflector.diagnose(
        worst, memory_index=memory_index, recent_records=recent,
        direction_stats=direction_stats,
        symbol=symbol, symbol_tags=symbol_tags,
        best_records=best,
        predictor_system_prompt=strategy.get("predictor_system_prompt", ""),
        memory_triggers=memory_triggers,
    )
    import re as _re
    _diag = _re.sub(r'^#{1,6}\s*', '', diagnosis, flags=_re.MULTILINE)
    _diag = _re.sub(r'\*\*(.+?)\*\*', r'\1', _diag)
    _diag = _re.sub(r'\*(.+?)\*', r'\1', _diag)
    _diag = _re.sub(r'`(.+?)`', r'\1', _diag)
    console.print("\n---\n[bold]诊断报告[/]")
    console.print(_diag.strip())
    console.print("\n---\n")

    console.print("  Evolver 生成3个候选策略...")
    evolver = EvolverAgent(strategy, model=model)
    candidates = await evolver.generate_candidates(
        diagnosis, stats, best, worst, symbol=symbol, symbol_tags=symbol_tags,
    )

    # Phase 3: holdout evaluation with statistical-significance gate.
    # Require ≥12 valid windows for either baseline or the best candidate; otherwise retry once,
    # then ask the user whether to try again or abort.
    _holdout_cap = min(len(holdout_windows), 20)
    MIN_VALID_HOLDOUT = 12

    async def _run_holdout_eval():
        console.print(f"  评估当前策略基准（{_holdout_cap} 个验证窗口）...")
        _baseline_results, _ = await _evaluate_candidates_on_holdout(
            {"baseline": strategy}, holdout_windows[:_holdout_cap],
            symbol, price_series, trading_dates, model,
            concurrency=_concurrency, mem=mem, symbol_tags=symbol_tags,
            data_source_type=data_source_type,
        )
        _baseline_detail = _baseline_results.get("baseline", {"score": 0.0, "valid_count": 0, "total_count": 0})
        console.print(
            f"  当前策略基准验证分: {_baseline_detail['score']:.4f}  "
            f"dir_acc={_baseline_detail.get('dir_acc', 0):.1%}  "
            f"({_baseline_detail['valid_count']}/{_baseline_detail.get('total_count', 0)} 有效)"
        )
        console.print(f"  在 {_holdout_cap} 个验证窗口上评估候选策略...")
        _holdout_results, _holdout_preds = await _evaluate_candidates_on_holdout(
            candidates, holdout_windows[:_holdout_cap],
            symbol, price_series, trading_dates, model,
            concurrency=_concurrency, mem=mem, symbol_tags=symbol_tags,
            data_source_type=data_source_type,
        )
        return _baseline_detail, _holdout_results, _holdout_preds

    attempt = 0
    _last_holdout_preds: dict = {}
    while True:
        attempt += 1
        console.print(f"[dim]  Holdout 评估 第 {attempt} 次尝试[/]")
        baseline_detail, holdout_results, _last_holdout_preds = await _run_holdout_eval()
        baseline_valid = baseline_detail["valid_count"]
        max_cand_valid = max((v["valid_count"] for v in holdout_results.values()), default=0)

        if baseline_valid >= MIN_VALID_HOLDOUT and max_cand_valid >= MIN_VALID_HOLDOUT:
            break

        console.print(
            f"[yellow]⚠ 有效验证窗口不足 {MIN_VALID_HOLDOUT}（baseline={baseline_valid}，"
            f"候选最大={max_cand_valid}），样本量不具备统计意义。[/]"
        )

        if attempt == 1:
            console.print("[yellow]自动重试一次...[/]")
            continue

        console.print(
            "[red]第二次评估仍然有效窗口不足。请检查数据源/网络/LLM 调用是否异常。[/]"
        )
        if sys.stdin.isatty():
            ans = console.input("[bold]是否再试一次？(Y/N): [/]").strip().upper()
            if ans == "Y":
                continue
        else:
            console.print("[yellow]非交互模式，自动放弃重试。[/]")
        console.print("[red]用户取消本次进化,不保存任何修改。[/]")
        return {
            **stats,
            "profile_name": active_name,
            "aborted": "holdout_insufficient",
            "attempts": attempt,
        }

    baseline_score = baseline_detail["score"]
    holdout_scores = {k: v["score"] for k, v in holdout_results.items()}
    best_holdout_preds = _last_holdout_preds  # per-key prediction lists for chart

    best_key, best_strategy = evolver.select_best_candidate(candidates, holdout_scores)
    best_strategy["notes"] = best_strategy.get("notes", "")
    best_strategy["evolution_notes"] = candidates.get("evolution_notes", "")

    # Regression protection against baseline floor.
    best_holdout = max(holdout_scores.values(), default=0.0)
    floor = max(baseline_score, 0.20)
    if best_holdout <= floor:
        console.print(
            f"[yellow]最优候选验证分 {best_holdout:.3f} 未超过基准 {floor:.3f}"
            f"（当前策略基准 {baseline_score:.3f}）。"
            f"全局策略保持不变,仅保留 per-symbol 的具体教训。[/]"
        )
        # 不让 evolver 的全局调整生效,但 per-symbol symbol_specific_notes 仍写入
        _preserved_per_symbol = {
            f: best_strategy[f] for f in _EVOLVABLE_PER_SYMBOL if f in best_strategy
        }
        best_strategy = dict(strategy)
        best_strategy.update(_preserved_per_symbol)
        best_strategy["evolution_notes"] = candidates.get("evolution_notes", "")

    # Snapshot win-rate for this version period and append to log
    period_stats = await db.get_stats_for_version(active_name, version)
    win_rate_log = list(best_strategy.get("win_rate_log", []))
    if period_stats["total"] > 0:
        log_entry = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "profile_version": version,
            "windows": period_stats["total"],
            "win_rate": period_stats["win_rate"],
            "avg_score": period_stats["avg_score"],
            "avg_return_pct": period_stats["avg_return_pct"],
            "date_from": period_stats["date_from"],
            "date_to": period_stats["date_to"],
            "symbols": period_stats["symbols"],
        }
        win_rate_log.append(log_entry)
        console.print(
            f"\n[bold]v{version} 期间胜率[/]  窗口={period_stats['total']}  "
            f"胜率=[{'green' if period_stats['win_rate'] >= 0.5 else 'red'}]"
            f"{period_stats['win_rate']:.1%}[/]  "
            f"avg_score={period_stats['avg_score']:.3f}  "
            f"品种={','.join(period_stats['symbols']) or '—'}"
        )
    else:
        console.print(
            f"\n[dim]v{version} 期间胜率  窗口=0 （本版本尚无训练记录，不计入日志）[/]"
        )
    best_strategy["win_rate_log"] = win_rate_log

    # C1: Merge evolved predictor_system_prompt only if not frozen by evolvable_fields
    evolvable = strategy.get("evolvable_fields", {})
    prompt_frozen = evolvable.get("predictor_system_prompt", True) is False
    if not prompt_frozen:
        old_prompt = strategy.get("predictor_system_prompt", "")
        new_prompt = best_strategy.get("predictor_system_prompt", "")
        if old_prompt and new_prompt and old_prompt != new_prompt:
            # Build fallback callable directly from predictor's pre-configured
            # fallback client — primary 400 errors bypass BaseLLMAgent's retry chain.
            _fb_client = getattr(predictor, "_fallback_client", None)
            _fb_model = getattr(predictor, "_fallback_model", "") or ""

            async def _merge_primary(p: str) -> str:
                return await predictor.call(p, max_tokens_override=MAX_TOKENS_EVOLVER)

            async def _fallback_llm(p: str) -> str:
                resp = await _fb_client.chat.completions.create(
                    model=_fb_model,
                    max_tokens=MAX_TOKENS_EVOLVER,
                    temperature=predictor.temperature,
                    messages=[
                        {"role": "system", "content": predictor.system_prompt},
                        {"role": "user", "content": p},
                    ],
                )
                return resp.choices[0].message.content

            fallback_fn = _fallback_llm if (_fb_client and _fb_model) else None

            merged = await _merge_prompts(
                old_prompt, new_prompt,
                llm_fn=_merge_primary,
                fallback_llm_fn=fallback_fn,
            )
            if merged:
                best_strategy["predictor_system_prompt"] = merged
                console.print("[dim cyan]  ✦ Prompt已合并（保留跨股通用规则）[/]")
                # Validate merged prompt on the same holdout windows
                console.print("  验证融合prompt在holdout窗口上的表现...")
                _merged_cand = dict(best_strategy)
                _merged_results, _ = await _evaluate_candidates_on_holdout(
                    {"merged": _merged_cand},
                    holdout_windows[:_holdout_cap],
                    symbol, price_series, trading_dates, model,
                    concurrency=_concurrency, mem=mem, symbol_tags=symbol_tags,
                    data_source_type=data_source_type,
                )
                _merged_dir_acc = _merged_results.get("merged", {}).get("dir_acc", 0.0)
                _best_dir_acc = holdout_results.get(best_key, {}).get("dir_acc", 0.0)
                if _merged_dir_acc < _best_dir_acc - 0.05:
                    best_strategy["predictor_system_prompt"] = new_prompt
                    console.print(
                        f"[yellow]  ⚠ 融合prompt方向准确率({_merged_dir_acc:.1%})低于最佳候选"
                        f"({_best_dir_acc:.1%})超过5%，回退使用候选版本[/]"
                    )
                else:
                    console.print(
                        f"[dim cyan]  ✦ 融合prompt验证通过 dir_acc={_merged_dir_acc:.1%}"
                        f"（最佳候选={_best_dir_acc:.1%}）[/]"
                    )
            else:
                console.print("[yellow]  ⚠ Prompt合并全部失败，直接采用候选版本[/]")
    else:
        # Frozen: discard any prompt changes from evolver
        best_strategy["predictor_system_prompt"] = strategy.get("predictor_system_prompt", "")

    # Save adopted prompt version to log dir
    _adopted_prompt = best_strategy.get("predictor_system_prompt", "")
    if _adopted_prompt:
        _next_version = best_strategy.get("profile_version", version + 1)
        _log_dir = PROFILES_DIR / f"{active_name}_log"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _ts_prompt = datetime.now().strftime("%Y%m%d_%H%M%S")
        _prompt_file = _log_dir / f"prompt_v{_next_version}_{_ts_prompt}.txt"
        try:
            _prompt_file.write_text(_adopted_prompt, encoding="utf-8")
            console.print(f"[dim]  ✦ Prompt版本已保存: {_prompt_file.name}[/]")
        except Exception as _pe:
            logger.warning(f"Failed to save prompt version file: {_pe}")

    # Print what changed
    _print_diff(strategy, best_strategy)

    # Execute memory operations from reflector suggestions
    if strategy.get("evolvable_fields", {}).get("memory", True):
        await _apply_memory_suggestions(
            mem, memory_suggestions,
            llm_fn=lambda p: predictor.call(p),
        )

    # Absorb memory notes covered by the new predictor_system_prompt.
    # Two sources: (1) Evolver's declared list (blind — Evolver never saw note full text);
    # (2) per-note semantic check (reads full content, LLM-verified).
    # Only fires when the prompt actually changed.
    evolver_absorbed = best_strategy.get("absorbed_memory_notes") or []
    new_prompt_text = best_strategy.get("predictor_system_prompt", "")
    prompt_changed = new_prompt_text != strategy.get("predictor_system_prompt", "")

    if prompt_changed:
        console.print("  扫描记忆笔记，逐条核查是否应被新 Prompt 吸收...")
        auto_absorbed = await _check_note_absorptions(new_prompt_text, mem, model)
        new_finds = set(auto_absorbed) - set(evolver_absorbed)
        if new_finds:
            console.print(f"[dim cyan]  ✦ 语义核查额外发现 {len(new_finds)} 条可吸收笔记[/]")
        all_absorbed = list(set(evolver_absorbed) | set(auto_absorbed))
    else:
        all_absorbed = []

    if all_absorbed:
        for fn in all_absorbed:
            try:
                mem.deprecate_memory(fn)
                console.print(f"[dim cyan]  ✦ 记忆已吸收并删除: {fn}[/]")
            except Exception as e:
                logger.warning(f"Failed to deprecate absorbed memory {fn}: {e}")
    # Don't persist this field into the saved profile
    best_strategy.pop("absorbed_memory_notes", None)

    # Extract evolved per-symbol fields → go to per_symbol_overrides[symbol]
    per_symbol_update = {}
    for field in _EVOLVABLE_PER_SYMBOL:
        if field in best_strategy:
            per_symbol_update[field] = best_strategy[field]

    # Global profile keeps the ORIGINAL values for per-symbol fields (cross-stock isolation).
    # Only non-per-symbol changes (win_rate_log, evolution_notes, etc.) are saved globally.
    global_strategy = dict(best_strategy)
    for field in _EVOLVABLE_PER_SYMBOL:
        if field in strategy:
            global_strategy[field] = strategy[field]  # restore original (pre-evolution) value
        else:
            global_strategy.pop(field, None)

    # Save global profile FIRST (without per-symbol fields)
    # Must be done before patch_symbol_override, otherwise store.save() overwrites the file
    # and loses any per-symbol data just written.
    if auto_apply:
        path = store.save(active_name, global_strategy, as_candidate=False)
        console.print(f"[green]档案已自动更新 → {path}[/]")
    else:
        path = store.save(active_name, global_strategy, as_candidate=True)
        console.print(f"\n[bold yellow]候选档案已保存至:[/] {path}")
        console.print(f"运行 [cyan]finagent apply[/] 来正式部署此档案")

    # Write per-symbol overrides AFTER saving the profile.
    # IMPORTANT: patch the SAME file (active or candidate) that store.save() just wrote.
    # Otherwise `finagent apply` (promote_candidate) would overwrite the active file
    # and lose any per_symbol_overrides that were written there.
    if per_symbol_update:
        store.patch_symbol_override(active_name, symbol, per_symbol_update, as_candidate=not auto_apply)
        console.print(f"[green]  ✦ {symbol} per-symbol overrides已写入[/]")

    # Cleanup ghost entries in MEMORY.md (index entries whose files no longer exist)
    ghost_count = mem.cleanup_ghost_entries()
    if ghost_count:
        console.print(f"[dim]  ✦ 清理幽灵记忆索引: {ghost_count} 条[/]")

    new_version = best_strategy.get("profile_version", version + 1)
    await db.save_evolution_run(
        profile_name=active_name,
        symbol=symbol,
        from_version=version,
        to_version=new_version,
        windows_processed=stats["total_windows"],
        avg_score=stats["avg_score"],
        direction_accuracy=stats["direction_accuracy"],
        candidate_chosen=best_key,
        evolution_notes=candidates.get("evolution_notes", ""),
    )

    # Draw and save K-line chart with prediction arrows.
    # Disabled by default — enable with FINAGENT_ENABLE_CHART=1 (needs matplotlib).
    from finagent.config import ENABLE_CHART
    chart_path = None
    if ENABLE_CHART:
      try:
        from finagent.utils.chart import save_evolution_chart
        from finagent.config import FIGURE_DIR
        train_records = await db.get_all_predictions(active_name, symbol)
        chart_path = save_evolution_chart(
            symbol=symbol,
            profile_name=active_name,
            version=version,
            stats=sym_stats,
            holdout_scores=holdout_scores,
            best_key=best_key,
            baseline_score=baseline_score,
            train_records=train_records,
            holdout_records=best_holdout_preds.get(best_key, []),
            fig_dir=FIGURE_DIR,
            data_source_type=data_source_type,
        )
        if chart_path:
            console.print(f"[dim cyan]  ✦ 图表已保存: {chart_path.name}[/]")
      except Exception as _chart_err:
        logger.warning(f"Chart generation failed: {_chart_err}")
        chart_path = None

    return {
        **sym_stats,
        "new_windows": processed,
        "profile_name": active_name,
        "chart_path": str(chart_path) if chart_path else None,
        "best_candidate": best_key,
        "holdout_scores": holdout_scores,
        "new_profile_version": new_version,
        "profile_path": str(path),
    }


def _print_memory_hit_report(mem) -> None:
    """Print how many times each memory note was triggered across the training windows."""
    attempts = mem.match_attempts
    counts = mem.match_counts
    if not attempts:
        return

    console.print(f"\n[bold]情境记忆命中报告[/]  检查窗口={attempts}")
    if not counts:
        console.print("  [yellow]无任何记忆被触发（所有触发器均未匹配）[/]")
        return

    # Load note titles for display
    notes_meta = {n["filename"]: n["situation"] for n in mem.load_all_notes()}
    all_notes = {n["filename"] for n in mem.load_all_notes()}
    zero_hit = all_notes - set(counts.keys())

    # Sort by count desc, show top 10
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for fn, cnt in ranked[:10]:
        pct = cnt / attempts * 100
        bar = "█" * min(int(pct / 5), 20)
        title = notes_meta.get(fn, fn)[:40]
        console.print(f"  {pct:5.1f}% {bar:<20} {cnt:3d}次  {title}")
    if len(ranked) > 10:
        console.print(f"  [dim]...还有 {len(ranked) - 10} 条[/]")

    if zero_hit:
        zero_list = sorted(zero_hit)
        console.print(f"\n  [dim]从未触发（共 {len(zero_hit)} 条，显示前10）:[/]")
        for fn in zero_list[:10]:
            title = notes_meta.get(fn, fn)[:50]
            console.print(f"  [dim]  0次  {title}[/]")
        if len(zero_list) > 10:
            console.print(f"  [dim]  ...还有 {len(zero_list) - 10} 条[/]")


def _print_diff(old: dict, new: dict) -> None:
    """Print a human-readable summary of what changed between two strategy dicts."""
    lines = []

    # Long text fields — show char count change
    for field in ("predictor_system_prompt", "critic_system_prompt",
                  "reflector_system_prompt", "evolver_system_prompt"):
        ov, nv = old.get(field, ""), new.get(field, "")
        if ov != nv:
            lines.append(f"  {field}: 已修改（旧 {len(ov)} 字 → 新 {len(nv)} 字）")

    # Short text / notes fields — show first 80 chars of new value
    for field in ("symbol_specific_notes", "evolution_notes"):
        ov, nv = old.get(field, ""), new.get(field, "")
        if ov != nv:
            preview = (nv or "")[:80].replace("\n", " ")
            lines.append(f"  {field}: 已修改 → {preview}{'…' if len(nv or '') > 80 else ''}")

    if lines:
        console.print("\n[bold]进化变更摘要[/]")
        for line in lines:
            console.print(line)
    else:
        console.print("\n[yellow]进化完成，无字段变更（候选与当前档相同）[/]")
        console.print("[dim]  提示: 如果Evolver未在候选中返回完整prompt，视为无变更，可查看LLM日志确认[/]")


_ABSORPTION_CHECK_SYSTEM = (
    "你是策略审核员，判断一条情境记忆笔记是否已被新版 predictor_system_prompt 完整体系化覆盖。"
    "只回答 JSON，不输出其他内容。"
)


def _build_absorption_check_prompt(new_prompt: str, note: dict) -> str:
    return "\n".join([
        "【新版 predictor_system_prompt】",
        new_prompt,
        "",
        "【待核查笔记】",
        f"情境: {note['situation']}",
        f"检索签名: {note.get('retrieval_text', '')}",
        f"经验内容: {note.get('content', '')[:600]}",
        "",
        "判断此笔记是否应被吸收（必须全部满足才能 absorb=true）：",
        "1. 笔记描述的具体情境已被 prompt 中明确可识别的条件覆盖（非隐含）",
        "2. 笔记的经验要点已以更通用的措辞写入 prompt，LLM 仅凭 prompt 即可在该情境下正确判断",
        "3. 部分覆盖 / 疑似 / 措辞含糊 → absorb=false",
        "",
        '返回 JSON：{"absorb": true|false, "reason": "一句话"}',
    ])


async def _check_note_absorptions(
    new_prompt: str,
    mem: MemoryManager,
    model: str,
    sim_threshold: float = 0.62,
    concurrency: int = 3,
) -> list[str]:
    """
    Embed new_prompt, pre-filter notes by cosine similarity to retrieval_text embeddings,
    then LLM-verify each candidate. Returns filenames confirmed for absorption.
    """
    import json as _json
    import numpy as np
    from finagent.agents.base import BaseLLMAgent

    mem._load_embeddings()
    if not mem._embeddings:
        return []

    embedder = mem._get_embedder()
    if embedder is None or not getattr(embedder, "configured", False):
        logger.debug("_check_note_absorptions: embedder not configured, skipping")
        return []

    try:
        prompt_vec = await embedder.aembed(new_prompt[:3000])
    except Exception as e:
        logger.warning(f"_check_note_absorptions: failed to embed prompt: {e}")
        return []

    prompt_norm = float(np.linalg.norm(prompt_vec)) + 1e-9
    candidates: list[str] = []
    for fn, note_vec in mem._embeddings.items():
        sim = float(np.dot(prompt_vec, note_vec) / (prompt_norm * (np.linalg.norm(note_vec) + 1e-9)))
        if sim >= sim_threshold:
            candidates.append(fn)

    if not candidates:
        return []

    notes = mem.load_notes(candidates)
    if not notes:
        return []

    checker = BaseLLMAgent(
        system_prompt=_ABSORPTION_CHECK_SYSTEM,
        max_tokens=256,
        model=model,
        temperature=0.1,
    )

    sem = asyncio.Semaphore(concurrency)
    absorbed: list[str] = []

    async def _check_one(note: dict) -> None:
        async with sem:
            prompt = _build_absorption_check_prompt(new_prompt, note)
            try:
                raw = await checker.call(prompt)
                data = checker.extract_json(raw)
                if data.get("absorb"):
                    absorbed.append(note["filename"])
                    logger.info(
                        f"_check_note_absorptions: '{note['filename']}' confirmed for absorption"
                        f" — {data.get('reason', '')}"
                    )
            except Exception as e:
                logger.debug(f"_check_note_absorptions: check failed for {note['filename']}: {e}")

    await asyncio.gather(*[_check_one(n) for n in notes])
    return absorbed


async def _merge_prompts(
    old_prompt: str, new_prompt: str, llm_fn, fallback_llm_fn=None,
) -> str:
    """
    C1: Merge evolved predictor_system_prompt with current one.
    Preserves rules validated across multiple stocks; replaces single-stock-specific rules.

    Escalation: primary llm_fn → retry once → fallback_llm_fn (if provided) → "" (caller
    keeps new_prompt). This guards against transient 400/502 from the primary provider
    that the in-agent retry chain doesn't cover (BadRequestError is raised immediately).
    """
    prompt = f"""你是威科夫策略专家，负责整合两个版本的预测规则 Prompt。

## 当前 Prompt（旧版，包含多只股票积累的规则）
{old_prompt}

## 新候选 Prompt（刚由进化生成，针对最新股票优化）
{new_prompt}

## 合并要求
1. 保留两个版本中**通用性强、跨股票成立**的规则（如"phase 7-8 避免做多"、"neutral 条件"等）
2. 若新旧版本描述的是同一规则但表述不同，取**更精确清晰**的那个
3. 若新版本新增了针对特定股票类型的规则（如"红利股横盘允许 neutral"），保留并标注
4. 若旧版本有某规则在新版本被删除，判断：如果是针对之前某只股票的特殊规则则删除；如果是普遍规律则保留
5. 输出格式与新候选 Prompt 结构一致，保持 markdown 格式
6. 不超过 2000 字

直接输出合并后的 Prompt 正文，不加任何解释。"""

    failures: list[str] = []

    async def _try(fn, label: str) -> str:
        try:
            merged = await fn(prompt)
            merged = (merged or "").strip()
            if len(merged) > 200:
                return merged
            failures.append(f"{label}: output too short ({len(merged)} chars)")
        except Exception as e:
            failures.append(f"{label} failed: {e}")
        return ""

    for attempt in range(2):
        result = await _try(llm_fn, f"primary attempt {attempt + 1}/2")
        if result:
            return result

    if fallback_llm_fn is not None:
        result = await _try(fallback_llm_fn, "fallback")
        if result:
            return result

    if failures:
        logger.warning(f"_merge_prompts all attempts failed: {'; '.join(failures)}")
    return ""


async def _apply_memory_suggestions(
    mem: MemoryManager, suggestions: dict, llm_fn=None,
) -> None:
    """Execute memory CRUD operations from reflector suggestions.

    If ``llm_fn`` is provided, each proposed new memory is first checked against
    all existing memories for contradictions via ``resolve_new_memory_conflicts``;
    the LLM decides whether to add / skip / replace / branch.
    """
    new_memories = suggestions.get("new_memories", [])
    update_memories = suggestions.get("update_memories", [])
    deprecate_memories = suggestions.get("deprecate_memories", [])

    if not new_memories and not update_memories and not deprecate_memories:
        return

    console.print("\n[bold]记忆更新[/]")

    for item in new_memories:
        situation = item.get("situation", "")
        if not situation:
            continue
        retrieval_text = item.get("retrieval_text", "")
        insight = item.get("insight", "")
        adjustments = item.get("suggested_adjustments", "")
        content = f"## 经验总结\n{insight}"
        if adjustments:
            content += f"\n\n## 建议调整\n{adjustments}"
        summary = insight[:80].replace("\n", " ") if insight else ""
        sector_scope = item.get("sector_scope", None)
        sector_excluded = item.get("sector_excluded", None)

        proposed = {
            "situation": situation,
            "retrieval_text": retrieval_text,
            "content": content,
            "summary": summary,
        }

        decision = {"action": "add"}
        if llm_fn is not None:
            try:
                decision = await mem.resolve_new_memory_conflicts(proposed, llm_fn)
            except Exception as e:
                logger.warning(f"conflict check failed for '{situation}': {e}")
                decision = {"action": "add"}

        action = decision.get("action", "add")
        try:
            if action == "skip":
                existing = decision.get("existing_file", "")
                console.print(
                    f"  [dim]= 跳过新增(与现有冲突/冗余): {situation} ← {existing}[/]"
                )
            elif action == "replace":
                replace_files = decision.get("replace_files", []) or []
                for fn in replace_files:
                    try:
                        mem.deprecate_memory(fn)
                    except Exception as e:
                        logger.warning(f"deprecate {fn} failed: {e}")
                path = await mem.add_memory(
                    situation=decision.get("situation", situation),
                    content=decision.get("content", content),
                    retrieval_text=decision.get("retrieval_text", retrieval_text),
                    summary=decision.get("summary", summary),
                    confidence=float(decision.get("confidence", 0.5)),
                    sector_scope=decision.get("sector_scope", sector_scope),
                    sector_excluded=decision.get("sector_excluded", sector_excluded),
                )
                console.print(
                    f"  [cyan]↻[/] 合并替换 {replace_files} → {path.name}"
                )
            elif action == "branch":
                sibling = decision.get("sibling_file", "")
                path = await mem.add_memory(
                    situation=decision.get("situation", situation),
                    content=decision.get("content", content),
                    retrieval_text=decision.get("retrieval_text", retrieval_text),
                    summary=decision.get("summary", summary),
                    confidence=float(decision.get("confidence", 0.5)),
                    sector_scope=decision.get("sector_scope", sector_scope),
                    sector_excluded=decision.get("sector_excluded", sector_excluded),
                )
                console.print(
                    f"  [magenta]⎇[/] 新增分支记忆(与 {sibling} 并存): {path.name}"
                )
            else:
                path = await mem.add_memory(
                    situation=situation,
                    content=content,
                    retrieval_text=retrieval_text,
                    summary=summary,
                    sector_scope=sector_scope,
                    sector_excluded=sector_excluded,
                )
                console.print(f"  [green]+[/] 新增记忆: {situation} → {path.name}")
        except Exception as e:
            logger.warning(f"Failed to apply memory decision '{situation}': {e}")

    for item in update_memories:
        filename = item.get("file", "")
        revision = item.get("revision", "")
        if not filename or not revision:
            continue
        try:
            mem.update_memory(filename, revision)
            console.print(f"  [yellow]~[/] 更新记忆: {filename}")
        except FileNotFoundError:
            logger.warning(f"Memory file not found for update: {filename}")

    for filename in deprecate_memories:
        if not filename:
            continue
        mem.deprecate_memory(filename)
        console.print(f"  [red]-[/] 废弃记忆: {filename}")


async def _evaluate_candidates_on_holdout(
    candidates: dict,
    holdout_windows: list,
    symbol: str,
    price_series: dict,
    trading_dates: list,
    model: str,
    concurrency: int = TRAIN_CONCURRENCY,
    mem: Optional[MemoryManager] = None,
    symbol_tags: Optional[list] = None,
    data_source_type: str = "stock",
) -> tuple[dict, dict]:
    """
    Evaluate candidates sequentially; within each candidate windows run concurrently.
    Returns (results_dict, preds_by_key) where:
      results_dict: {key: {score, valid_count, failed_count, total_count}}
      preds_by_key: {key: [{window_end_date, direction, confidence}]}
    """
    loop = asyncio.get_running_loop()
    results = {}
    preds_by_key: dict[str, list[dict]] = {}
    total_windows = len(holdout_windows)

    async def _eval_one(key: str) -> None:
        cand = candidates.get(key)
        if not isinstance(cand, dict):
            results[key] = {"score": 0.0, "valid_count": 0, "failed_count": 0, "total_count": total_windows}
            preds_by_key[key] = []
            return

        predictor = PredictorAgent(cand, model=model, memory_manager=mem, symbol_tags=symbol_tags or [])
        critic = CriticAgent(cand, model=model)
        window_scores: list[float] = []
        window_preds: list[dict] = []
        failed_count = [0]
        done_count = [0]
        dir_correct_count = [0]
        sem = asyncio.Semaphore(concurrency)

        # Independent spinner for this candidate
        spin_chars = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        spin_stop = threading.Event()
        spin_len = [0]

        def _spin_fn() -> None:
            while not spin_stop.is_set():
                pct = int(done_count[0] / total_windows * 100) if total_windows else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                line = f"  {next(spin_chars)} {key}  |{bar}| {done_count[0]}/{total_windows} {pct}%"
                sys.stderr.write(f"\r{line}")
                sys.stderr.flush()
                spin_len[0] = len(line)
                time.sleep(0.12)
            sys.stderr.write("\r" + " " * (spin_len[0] + 2) + "\r")
            sys.stderr.flush()

        spin_thread = threading.Thread(target=_spin_fn, daemon=True)
        spin_thread.start()

        async def _eval_window(window) -> None:
            async with sem:
                try:
                    snapshot = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda d=window.window_end_date: get_wyckoff_snapshot(symbol, end_date=d, days=500, data_source_type=data_source_type),
                        ),
                        timeout=180.0,
                    )
                    user_msg = build_predictor_user_prompt(snapshot["text"], cand, symbol_tags=symbol_tags or [])
                    prediction = await predictor.predict(user_msg, snapshot=snapshot)

                    horizon_prices = get_horizon_prices(
                        price_series, window.horizon_start_date,
                        window.horizon_end_date, trading_dates,
                    )
                    if not horizon_prices:
                        failed_count[0] += 1
                        return

                    entry_price = price_series.get(window.window_end_date, horizon_prices[0])
                    actual = format_actual_outcome(entry_price, horizon_prices)
                    critique = await critic.evaluate(prediction, actual)
                    window_scores.append(critique["score"])
                    if critique.get("direction_correct"):
                        dir_correct_count[0] += 1
                    window_preds.append({
                        "window_end_date": window.window_end_date,
                        "direction": prediction.get("direction", ""),
                        "confidence": prediction.get("confidence", 0.0),
                    })
                except asyncio.TimeoutError:
                    logger.warning(f"Holdout snapshot timeout for {key} at {window.window_end_date}")
                    failed_count[0] += 1
                except Exception as e:
                    logger.warning(f"Holdout eval error for {key} at {window.window_end_date}: {e}")
                    failed_count[0] += 1
                finally:
                    done_count[0] += 1

        await asyncio.gather(*[_eval_window(w) for w in holdout_windows])
        spin_stop.set()
        spin_thread.join()

        score = round(sum(window_scores) / len(window_scores), 4) if window_scores else 0.0
        valid = len(window_scores)
        dir_acc = dir_correct_count[0] / valid if valid else 0.0
        console.print(
            f"  {key}: score={score:.4f}  dir_acc={dir_acc:.1%}  ({valid}/{total_windows} 有效, {failed_count[0]} 失败)"
        )
        if total_windows > 0 and valid < total_windows * 0.5:
            console.print(f"  [yellow]⚠ {key}: 有效窗口 <50%,评分不可靠[/]")
        results[key] = {
            "score": score,
            "dir_acc": dir_acc,
            "valid_count": valid,
            "failed_count": failed_count[0],
            "total_count": total_windows,
        }
        preds_by_key[key] = window_preds

    eval_keys = [k for k in candidates if isinstance(candidates.get(k), dict)]
    for key in eval_keys:
        await _eval_one(key)

    return results, preds_by_key
