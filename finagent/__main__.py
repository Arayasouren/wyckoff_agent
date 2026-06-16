# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
CLI entry point: python -m finagent  or  finagent (after pip install -e .)

档案管理：
  finagent new-profile [name]      新建档案（不指定名则自动用时间戳命名），并设为当前档
  finagent use-profile <name>      切换当前档
  finagent list-profiles           列出所有档案
  finagent apply                   将当前档的候选版正式部署

训练与预测（均使用当前档）：
  finagent evolve <symbol>         对某支股票做历史滚动训练并进化当前档
  finagent predict <symbol>        对某支股票给出当前预测
  finagent status [symbol]         查看当前档状态（可选按股票过滤统计）
"""
import argparse
import asyncio
import json
import sys
import logging
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.table import Table
from finagent.config import STEP_SIZE

console = Console()


@contextmanager
def _log_console(log_path: Path):
    """Tee all Rich console.print() calls to a plain-text log file."""
    import finagent.engine.evolution as _evo
    import finagent.engine.prediction as _pred

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    file_con = Console(file=log_fh, no_color=True, highlight=False, width=120)

    _mods = [sys.modules[__name__], _evo, _pred]
    _originals = {m: m.console for m in _mods if hasattr(m, "console")}

    class _Tee:
        def __init__(self, orig):
            self._o = orig
        def print(self, *a, **kw):
            self._o.print(*a, **kw)
            file_con.print(*a, **kw)
        def print_json(self, *a, **kw):
            self._o.print_json(*a, **kw)
            file_con.print_json(*a, **kw)
        def __getattr__(self, n):
            return getattr(self._o, n)

    for m, orig in _originals.items():
        m.console = _Tee(orig)
    try:
        yield log_path
    finally:
        for m, orig in _originals.items():
            m.console = orig
        log_fh.close()


def _check_api_key() -> None:
    from finagent.config import ANTHROPIC_API_KEY, LLM_PROVIDER, OPENAI_COMPAT_API_KEY
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        console.print("[red]错误: ANTHROPIC_API_KEY 未设置[/]")
        console.print("请在 .env 文件或环境变量中设置 ANTHROPIC_API_KEY")
        sys.exit(1)
    if LLM_PROVIDER == "openai" and not OPENAI_COMPAT_API_KEY:
        console.print("[red]错误: OPENAI_COMPAT_API_KEY 未设置[/]")
        sys.exit(1)


def _apply_fallback_override() -> None:
    """Rewire LLM config so all calls go through the fallback OpenAI-compat endpoint.

    Runs *before* any finagent.agents module is imported — those modules bind config
    constants at import time, so we mutate os.environ and reload finagent.config first.
    """
    import os
    import importlib
    fb_key = os.getenv("OPENAI_COMPAT_FALLBACK_API_KEY", "")
    fb_url = os.getenv("OPENAI_COMPAT_FALLBACK_BASE_URL", "")
    fb_model = os.getenv("FINAGENT_FALLBACK_MODEL", "")
    if not (fb_key and fb_url and fb_model):
        console.print(
            "[red]错误: --use-fallback 需要 .env 中配置 "
            "OPENAI_COMPAT_FALLBACK_API_KEY / OPENAI_COMPAT_FALLBACK_BASE_URL / "
            "FINAGENT_FALLBACK_MODEL[/]"
        )
        sys.exit(1)
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_COMPAT_API_KEY"] = fb_key
    os.environ["OPENAI_COMPAT_BASE_URL"] = fb_url
    os.environ["FINAGENT_MODEL"] = fb_model
    # Disable secondary fallback — we're already on it, no further escalation.
    os.environ["OPENAI_COMPAT_FALLBACK_API_KEY"] = ""
    os.environ["OPENAI_COMPAT_FALLBACK_BASE_URL"] = ""
    os.environ["FINAGENT_FALLBACK_MODEL"] = ""
    import finagent.config as _cfg
    importlib.reload(_cfg)
    console.print(f"[cyan]✦ --use-fallback: 所有 LLM 调用改走 fallback API ({fb_model})[/]")


_BANNER_HEADER = [
    "╔════════════════════════════════════════════════════╗",
    "║░░█░█░█░█░█▀▀░█░█░█▀█░█▀▀░█▀▀░░░█▀█░█▀▀░█▀▀░█▀█░▀█▀░║",
    "║░░█▄█░░█░░█░░░█▀▄░█░█░█▀▀░█▀▀░░░█▀█░█░█░█▀▀░█░█░░█░░║",
    "║░░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀░░░▀░░░░░▀░▀░▀▀▀░▀▀▀░▀░▀░░▀░░║",
    "║░░░░░░░░░░Self-Evolving░Wyckoff░AI░Engine░░░░░░░░░░░║",
    "╠════════════════════════════════════════════════════╣",
]

_BOX_W = 52  # inner content width — must match ═ count in header


def _build_info_box(model: str, profile: str) -> list:
    """Build info rows below the header, as list of (plain, rich) tuples."""
    W = _BOX_W

    def row(plain_content: str, rich_content: str = None) -> tuple:
        p = f"║{plain_content:<{W}}║"
        r = f"[dim]║[/]{rich_content or plain_content:<{W}}[dim]║[/]"
        return p, r

    def sep() -> tuple:
        s = f"╠{'═' * W}╣"
        return s, f"[dim]{s}[/]"

    def bot() -> tuple:
        s = f"╚{'═' * W}╝"
        return s, f"[dim]{s}[/]"

    def rrow(plain: str, rich_inner: str) -> tuple:
        pad = W - len(plain)
        return f"║{plain}{' ' * pad}║", f"[dim]║[/]{rich_inner}{' ' * pad}[dim]║[/]"

    m = model[:38] if len(model) > 38 else model
    p = profile[:36] if len(profile) > 36 else profile
    m_plain = f"  ● Model    : {m}"
    p_plain = f"  ● Strategy : {p}"
    m_rich  = f"  [green]●[/] Model    : [cyan]{m}[/]"
    p_rich  = f"  [green]●[/] Strategy : [cyan]{p}[/]"

    return [
        row(" " * W),
        rrow(m_plain, m_rich),
        rrow(p_plain, p_rich),
        rrow("  ● Data     : Wind WDS / AkShare",
             "  [green]●[/] Data     : Wind WDS / AkShare"),
        row(" " * W),
        sep(),
        row(" " * W),
        rrow("   Author  :  GLMS_FinEng Team",
             "   [bold yellow]Author  :  GLMS_FinEng Team[/]"),
        rrow("   Powered by Claude Code",
             "   [dim]Powered by Claude Code[/]"),
        row(" " * W),
        sep(),
        rrow("   Version : v1.0  │  Data : Wind WDS",
             "   [dim]Version : v1.0  │  Data : Wind WDS[/]"),
        bot(),
    ]


def _print_banner() -> None:
    """Print startup banner."""
    import os
    from finagent.config import DEFAULT_MODEL
    from finagent.storage.profile_store import ProfileStore

    os.system("clear")

    try:
        active_name, _ = ProfileStore().load_active()
    except Exception:
        active_name = "default"

    for line in _BANNER_HEADER:
        console.print(f"[dim]{line}[/]")

    for _plain, rich in _build_info_box(model=DEFAULT_MODEL, profile=active_name):
        console.print(rich)
    console.print()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _print_banner()

    parser = argparse.ArgumentParser(
        prog="finagent",
        description="自进化威科夫技术分析 Agent",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # new-profile
    np_ = sub.add_parser("new-profile", help="新建档案并设为当前档")
    np_.add_argument("name", nargs="?", default=None, help="档案名（留空则自动用时间戳）")

    # use-profile
    up = sub.add_parser("use-profile", help="切换当前档")
    up.add_argument("name", help="档案名")

    # list-profiles
    sub.add_parser("list-profiles", help="列出所有档案")

    # evolve
    ev = sub.add_parser("evolve", help="Mode 1: 滚动历史回测 + 进化当前档")
    ev.add_argument("symbol", help="股票代码，如 000001.SZ 或 AAPL")
    ev.add_argument("--profile", default=None, help="指定使用某个档案（不改变当前档设置）")
    ev.add_argument("--model", default=None, help="Claude 模型名")
    ev.add_argument("--step", type=int, default=STEP_SIZE, help=f"窗口步长（交易日，默认{STEP_SIZE}）")
    ev.add_argument("--max-windows", type=int, default=None, help="最多处理N个窗口（调试用）")
    ev.add_argument("--no-auto-apply", action="store_true", help="不自动部署候选，保存为候选档等待手动确认")
    ev.add_argument("--skip-evolution", action="store_true", help="只跑预测+批判，跳过进化")
    ev.add_argument("--force-evolution", action="store_true", help="强制进入进化阶段（即使无新窗口）")
    ev.add_argument("--evolve-every", type=int, default=None,
                    metavar="N", help="至少处理N个新窗口才触发进化（0=总是，覆盖档案设置）")
    ev.add_argument("--concurrency", type=int, default=None,
                    metavar="N", help="并发窗口数（预测阶段+holdout阶段共用，默认读 config.TRAIN_CONCURRENCY）")
    ev.add_argument("--use-fallback", action="store_true",
                    help="全局改走 fallback API（OpenAI-compat），跳过主 API。当主 API 不可用时使用。")
    ev.add_argument("-v", "--verbose", action="store_true")

    # predict
    pr = sub.add_parser("predict", help="Mode 2: 对指定股票给出当前预测")
    pr.add_argument("symbol", help="股票代码，如 000001.SZ 或 AAPL")
    pr.add_argument("--profile", default=None, help="指定使用某个档案")
    pr.add_argument("--date", default=None, help="截止日期 YYYY-MM-DD（默认最新）")
    pr.add_argument("--model", default=None, help="Claude 模型名")
    pr.add_argument("--json", action="store_true", dest="output_json", help="以 JSON 格式输出")
    pr.add_argument("--use-fallback", action="store_true",
                    help="全局改走 fallback API（OpenAI-compat），跳过主 API。")
    pr.add_argument("-v", "--verbose", action="store_true")

    # batch-evolve
    be = sub.add_parser("batch-evolve", help="批量训练：按顺序对多支股票进化，结果归档到 _log/ 目录")
    be.add_argument("symbols", nargs="+", help="股票代码列表，如 000001.SZ 002714.SZ")
    be.add_argument("--profile", default=None, help="指定使用某个档案")
    be.add_argument("--model", default=None, help="Claude 模型名")
    be.add_argument("--step", type=int, default=STEP_SIZE, help=f"窗口步长（默认{STEP_SIZE}）")
    be.add_argument("--max-windows", type=int, default=None, help="每支股票最多处理N个窗口")
    be.add_argument("--no-auto-apply", action="store_true", help="不自动部署候选")
    be.add_argument("--skip-evolution", action="store_true", help="只跑预测+批判，跳过进化")
    be.add_argument("--concurrency", type=int, default=None, metavar="N")
    be.add_argument("--use-fallback", action="store_true",
                    help="全局改走 fallback API")
    be.add_argument("-v", "--verbose", action="store_true")

    # rebuild-embeddings
    rb = sub.add_parser("rebuild-embeddings", help="为记忆笔记重建语义检索向量索引")
    rb.add_argument("--profile", default=None, help="指定档案名（默认当前档）")

    # compress-memory
    cm = sub.add_parser("compress-memory", help="强制合并相似记忆笔记（跳过数量阈值）")
    cm.add_argument("--profile", default=None, help="指定档案名（默认当前档）")
    cm.add_argument("--model", default=None, help="LLM 模型名")

    # apply
    ap = sub.add_parser("apply", help="将当前档的候选版正式部署")
    ap.add_argument("--profile", default=None, help="指定档案名（默认当前档）")

    # status
    st = sub.add_parser("status", help="查看当前档状态和历史统计")
    st.add_argument("symbol", nargs="?", default=None, help="可选：只显示该股票的统计")
    st.add_argument("--profile", default=None, help="指定档案名（默认当前档）")

    args = parser.parse_args()

    if hasattr(args, "verbose") and args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("finagent").setLevel(logging.DEBUG)

    if args.mode in ("evolve", "predict", "batch-evolve") and getattr(args, "use_fallback", False):
        _apply_fallback_override()

    if args.mode in ("evolve", "predict", "batch-evolve"):
        _check_api_key()

    # ── dispatch ────────────────────────────────────────────────────────────

    if args.mode == "new-profile":
        _cmd_new_profile(args.name)

    elif args.mode == "use-profile":
        _cmd_use_profile(args.name)

    elif args.mode == "list-profiles":
        _cmd_list_profiles()

    elif args.mode == "evolve":
        import datetime
        from finagent.config import DEFAULT_MODEL, PROFILES_DIR
        from finagent.engine.evolution import run_evolution, run_evolution_index_batches
        from finagent.storage.profile_store import ProfileStore

        _store = ProfileStore()
        _profile = args.profile or _store.get_active_name() or "default"
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_path = PROFILES_DIR / f"{_profile}_log" / f"{args.symbol}_{_ts}.log"

        # Detect index profile → run multi-batch driver
        try:
            _strategy_peek = _store.load(_profile)
            _is_index = _strategy_peek.get("data_source_type") == "index"
        except Exception:
            _is_index = False

        with _log_console(_log_path):
            if _is_index:
                result = asyncio.run(
                    run_evolution_index_batches(
                        symbol=args.symbol,
                        model=args.model or DEFAULT_MODEL,
                        auto_apply=not args.no_auto_apply,
                        profile_name=args.profile,
                        concurrency=args.concurrency,
                        max_batches=args.max_windows,  # repurposed: cap batch count
                    )
                )
                # CLI exit code: success if any batch succeeded
                _batches = result.get("batch_results", [])
                if _batches and any("error" not in b for b in _batches):
                    result.pop("error", None)
            else:
                result = asyncio.run(
                    run_evolution(
                        symbol=args.symbol,
                        model=args.model or DEFAULT_MODEL,
                        step_size=args.step,
                        max_windows=args.max_windows,
                        auto_apply=not args.no_auto_apply,
                        skip_evolution=args.skip_evolution,
                        force_evolution=args.force_evolution,
                        profile_name=args.profile,
                        evolve_every_n_new_windows=args.evolve_every,
                        concurrency=args.concurrency,
                    )
                )
        console.print(f"[dim]日志已保存: {_log_path}[/]")
        if "error" in result:
            sys.exit(1)

    elif args.mode == "predict":
        from finagent.config import DEFAULT_MODEL
        from finagent.engine.prediction import run_prediction
        result = asyncio.run(
            run_prediction(
                symbol=args.symbol,
                as_of_date=args.date,
                model=args.model or DEFAULT_MODEL,
                profile_name=args.profile,
            )
        )
        if args.output_json:
            out = {k: v for k, v in result.items() if k != "raw_llm_response"}
            console.print_json(json.dumps(out, ensure_ascii=False))

    elif args.mode == "batch-evolve":
        from finagent.config import DEFAULT_MODEL
        asyncio.run(
            _cmd_batch_evolve(
                symbols=args.symbols,
                profile_name=args.profile,
                model=args.model or DEFAULT_MODEL,
                step_size=args.step,
                max_windows=args.max_windows,
                auto_apply=not args.no_auto_apply,
                skip_evolution=args.skip_evolution,
                concurrency=args.concurrency,
            )
        )

    elif args.mode == "rebuild-embeddings":
        asyncio.run(_cmd_rebuild_embeddings(args.profile))

    elif args.mode == "compress-memory":
        from finagent.config import DEFAULT_MODEL
        asyncio.run(_cmd_compress_memory(args.profile, args.model or DEFAULT_MODEL))

    elif args.mode == "apply":
        _cmd_apply(args.profile)

    elif args.mode == "status":
        asyncio.run(_cmd_status(args.symbol, args.profile))


# ── sub-command implementations ──────────────────────────────────────────────

async def _cmd_batch_evolve(
    symbols: list,
    profile_name,
    model: str,
    step_size: int,
    max_windows,
    auto_apply: bool,
    skip_evolution: bool,
    concurrency,
) -> None:
    import datetime
    from finagent.config import PROFILES_DIR
    from finagent.engine.evolution import run_evolution, run_evolution_index_batches
    from finagent.storage.profile_store import ProfileStore

    store = ProfileStore()
    name = profile_name or store.get_active_name()
    if name is None:
        console.print("[red]没有当前档案，请用 --profile 指定[/]")
        return

    # Detect index profile → each symbol runs ALL 6 batches (not just last 60 months)
    strategy_peek = store.load(name)
    is_index = strategy_peek.get("data_source_type") == "index"

    log_dir = PROFILES_DIR / f"{name}_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(
        f"\n[bold cyan]batch-evolve[/]  档案=[magenta]{name}[/]  "
        f"股票数={len(symbols)}  model={model}"
        f"{'  [yellow](指数模式：每只跑全部 6 批次)[/]' if is_index else ''}"
    )
    console.print(f"  日志目录: {log_dir}\n")

    results = []
    for i, symbol in enumerate(symbols, 1):
        console.print(f"[bold white]── [{i}/{len(symbols)}] {symbol} ──[/]")
        log_path = log_dir / f"{symbol}_{ts}.log"
        try:
            with _log_console(log_path):
                if is_index:
                    batch_result = await run_evolution_index_batches(
                        symbol=symbol,
                        model=model,
                        auto_apply=auto_apply,
                        profile_name=name,
                        concurrency=concurrency,
                        max_batches=max_windows,  # repurpose: cap batch count
                    )
                    # Flatten last batch's stats for summary table
                    _batches = batch_result.get("batch_results", []) or []
                    _last_ok = next((b for b in reversed(_batches) if "error" not in b), {})
                    result = _last_ok
                else:
                    result = await run_evolution(
                        symbol=symbol,
                        model=model,
                        step_size=step_size,
                        max_windows=max_windows,
                        auto_apply=auto_apply,
                        skip_evolution=skip_evolution,
                        force_evolution=False,
                        profile_name=name,
                        concurrency=concurrency,
                    )
            console.print(f"  [dim]日志已保存: {log_path.name}[/]")
        except Exception as e:
            console.print(f"[red]  {symbol} 训练失败: {e}[/]")
            results.append({"symbol": symbol, "error": str(e)})
            continue

        results.append({
            "symbol": symbol,
            "new_windows": result.get("new_windows", 0),
            "direction_accuracy": result.get("direction_accuracy", 0),
            "avg_score": result.get("avg_score", 0),
        })

    console.print(f"\n[bold white]批量训练完成[/]")
    tbl = Table(show_header=True, header_style="bold cyan", box=None)
    tbl.add_column("股票", width=14)
    tbl.add_column("新窗口", justify="right", width=6)
    tbl.add_column("方向准确率", justify="right", width=10)
    tbl.add_column("均分", justify="right", width=7)
    tbl.add_column("状态")
    for r in results:
        if "error" in r:
            tbl.add_row(r["symbol"], "—", "—", "—", f"[red]{r['error'][:30]}[/]")
        else:
            acc = r.get("direction_accuracy", 0)
            tbl.add_row(
                r["symbol"],
                str(r.get("new_windows", 0)),
                f"[{'green' if acc >= 0.5 else 'red'}]{acc:.1%}[/]",
                f"{r.get('avg_score', 0):.3f}",
                "[green]✓[/]",
            )
    console.print(tbl)

def _cmd_new_profile(name) -> None:
    from finagent.storage.profile_store import ProfileStore
    store = ProfileStore()
    try:
        actual_name, path = store.new_profile(name)
        store.set_active(actual_name)
        console.print(f"[green]新档案已创建:[/] {actual_name}")
        console.print(f"  路径: {path}")
        console.print(f"  已设为当前档")
    except FileExistsError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)


def _cmd_use_profile(name: str) -> None:
    from finagent.storage.profile_store import ProfileStore, ProfileNotFoundError
    store = ProfileStore()
    try:
        store.set_active(name)
        console.print(f"[green]当前档已切换为:[/] {name}")
    except ProfileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        console.print("运行 [cyan]finagent list-profiles[/] 查看可用档案")
        sys.exit(1)


def _cmd_list_profiles() -> None:
    from finagent.storage.profile_store import ProfileStore
    store = ProfileStore()
    profiles = store.list_profiles()
    if not profiles:
        console.print("[yellow]暂无档案。运行 finagent new-profile 新建一个。[/]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("", width=2)
    table.add_column("档案名", style="cyan")
    table.add_column("版本", justify="right")
    table.add_column("更新时间")
    table.add_column("候选", justify="center")
    table.add_column("备注")

    for p in profiles:
        marker = "[bold green]▶[/]" if p["is_active"] else ""
        cand = "[yellow]待部署[/]" if p["has_candidate"] else ""
        table.add_row(
            marker,
            p["name"],
            f"v{p['version']}",
            p["updated_at"][:19].replace("T", " "),
            cand,
            p["notes"],
        )
    console.print(table)


async def _cmd_rebuild_embeddings(profile_name) -> None:
    from finagent.storage.profile_store import ProfileStore
    from finagent.storage.memory_store import MemoryManager
    from finagent.storage.embedder import get_default_embedder

    store = ProfileStore()
    name = profile_name or store.get_active_name()
    if name is None:
        console.print("[red]没有当前档案，请用 --profile 指定[/]")
        return

    embedder = get_default_embedder()
    if not embedder.configured:
        console.print(
            "[red]Embedder 未配置：请在 .env 中设置 EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL "
            "（或 OPENAI_COMPAT_API_KEY / OPENAI_COMPAT_BASE_URL）[/]"
        )
        return

    console.print(f"[bold cyan]重建向量索引[/]  档案=[magenta]{name}[/]  model={embedder.model}")
    mem = MemoryManager(name, embedder=embedder)
    count = await mem.rebuild_embeddings()
    if count:
        console.print(f"[green]✓ 已嵌入 {count} 条记忆笔记 → {mem._embed_path}[/]")
    else:
        console.print("[yellow]没有找到任何记忆笔记。[/]")


async def _cmd_compress_memory(profile_name, model: str) -> None:
    from finagent.storage.profile_store import ProfileStore
    from finagent.storage.memory_store import MemoryManager
    from finagent.storage.embedder import get_default_embedder
    from finagent.agents.predictor import PredictorAgent

    store = ProfileStore()
    name = profile_name or store.get_active_name()
    if name is None:
        console.print("[red]没有当前档案，请用 --profile 指定[/]")
        return

    strategy = store.load(name)
    embedder = get_default_embedder()
    mem = MemoryManager(name, embedder=embedder)
    await mem.ensure_embeddings_built()

    notes = mem.load_all_notes()
    console.print(
        f"[bold cyan]记忆合并[/]  档案=[magenta]{name}[/]  "
        f"笔记数={len(notes)}  model={model}"
    )

    predictor = PredictorAgent(strategy, model=model)
    merged = await mem._run_compression(llm_fn=lambda p: predictor.call(p))
    if merged:
        console.print(f"[green]✓ 合并完成，净减少 {merged} 条笔记[/]")
    else:
        console.print("[yellow]未发现相似度 > 0.85 的笔记组，无需合并。[/]")


def _cmd_apply(profile_name) -> None:
    from finagent.storage.profile_store import ProfileStore, ProfileNotFoundError
    store = ProfileStore()
    name = profile_name or store.get_active_name()
    if name is None:
        console.print("[red]没有当前档案[/]")
        sys.exit(1)
    if not store.has_candidate(name):
        console.print(f"[red]档案 '{name}' 没有待部署的候选版[/]")
        console.print("请先运行 finagent evolve <symbol> 生成候选")
        sys.exit(1)
    try:
        path = store.promote_candidate(name)
        console.print(f"[green]档案 '{name}' 候选版已正式部署 → {path}[/]")
    except ProfileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)


async def _cmd_status(symbol, profile_name) -> None:
    from finagent.storage.profile_store import ProfileStore
    from finagent.storage.database import Database

    store = ProfileStore()
    name = profile_name or store.get_active_name()
    if name is None:
        console.print("[yellow]暂无档案[/]")
        return

    profile = store.load(name)
    active_marker = " [bold green](当前档)[/]" if name == store.get_active_name() else ""
    console.print(f"\n[bold]档案: {name}{active_marker}[/]")
    console.print(f"  版本: v{profile.get('profile_version', 0)}")
    console.print(f"  更新时间: {profile.get('updated_at', '—')[:19].replace('T', ' ')}")

    notes = profile.get("notes", "")
    if notes:
        console.print(f"  备注: {notes[:200]}")

    if store.has_candidate(name):
        console.print(f"  [bold yellow]⚠ 有待部署的候选版[/] — 运行 finagent apply 部署")

    db = Database()
    await db.init_schema()
    stats = await db.get_summary_stats(name, symbol=symbol)

    scope = f"({symbol})" if symbol else "(全部股票)"
    if stats["total_windows"] > 0:
        console.print(f"\n[bold]训练统计 {scope}[/]")
        console.print(f"  已处理窗口: {stats['total_windows']}")
        console.print(f"  平均评分:   {stats['avg_score']:.3f}")
        console.print(f"  方向准确率: {stats['direction_accuracy']:.1%}")
        console.print(f"  平均涨跌幅: {stats['avg_return_pct']:+.2f}%")
    else:
        console.print(f"\n[yellow]暂无训练数据 {scope}[/]")

    # Win-rate log
    win_rate_log = profile.get("win_rate_log", [])
    if win_rate_log:
        from rich.table import Table
        console.print(f"\n[bold]胜率日志（每次进化快照）[/]")
        tbl = Table(show_header=True, header_style="bold cyan", box=None)
        tbl.add_column("版本", justify="right", width=4)
        tbl.add_column("胜率", justify="right", width=7)
        tbl.add_column("均分", justify="right", width=6)
        tbl.add_column("窗口", justify="right", width=5)
        tbl.add_column("日期范围", width=24)
        tbl.add_column("品种")
        for entry in win_rate_log:
            wr = entry.get("win_rate", 0)
            wr_color = "green" if wr >= 0.5 else "red"
            tbl.add_row(
                f"v{entry.get('profile_version', '?')}",
                f"[{wr_color}]{wr:.1%}[/]",
                f"{entry.get('avg_score', 0):.3f}",
                str(entry.get("windows", 0)),
                f"{entry.get('date_from','')[:10]} ~ {entry.get('date_to','')[:10]}",
                ", ".join(entry.get("symbols", [])),
            )
        console.print(tbl)


if __name__ == "__main__":
    main()
