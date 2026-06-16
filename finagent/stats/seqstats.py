"""
Sequence-probability stats lookup.

Given a symbol + recent_events list (由远及近), produces per-source best-prefix
probability records suitable for rendering into the snapshot's 【统计概率】 block.

Sources: ALL + tag_size + tag_style + tag_sector (from stockinfo/tags.csv).
"""
from __future__ import annotations
import csv
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from finagent.config import DB_PATH, ROOT_DIR

logger = logging.getLogger(__name__)

WYCKOFFSTATS_DIR = ROOT_DIR / "wyckoffstats"
TAGS_CSV = ROOT_DIR / "stockinfo" / "tags.csv"


# ── First-run auto-build of the seqstats reference table ──────────────────
# The 727k-row historical sequence stats ship as wyckoffstats/*.csv (git-friendly).
# On first use we ingest them into the `seqstats` table of finagent.db. This is
# idempotent and cheap once built (a row-count check short-circuits later runs).
_SEQSTATS_BUILD_CHECKED = False

_SEQSTATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS seqstats (
    bucket           TEXT NOT NULL,
    sequence         TEXT NOT NULL,
    length           INTEGER NOT NULL,
    count            INTEGER NOT NULL,
    up_count         INTEGER NOT NULL,
    down_count       INTEGER NOT NULL,
    up_probability   REAL NOT NULL,
    PRIMARY KEY (bucket, sequence)
);
CREATE INDEX IF NOT EXISTS idx_seqstats_bucket ON seqstats(bucket);
"""

_SEQSTATS_TAG_FILES = [
    "seqstats_大盘.csv", "seqstats_中盘.csv", "seqstats_小盘.csv",
    "seqstats_成长.csv", "seqstats_价值.csv", "seqstats_红利.csv",
    "seqstats_金融.csv", "seqstats_消费.csv", "seqstats_科技.csv",
    "seqstats_制造.csv", "seqstats_周期.csv", "seqstats_基础设施.csv",
]


def _ingest_seqstats_csv(conn: sqlite3.Connection, path: Path, bucket: str) -> int:
    rows: list[tuple] = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((
                bucket, r["sequence"], int(r["length"]), int(r["count"]),
                int(r["up_count"]), int(r["down_count"]), float(r["up_probability"]),
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO seqstats "
        "(bucket, sequence, length, count, up_count, down_count, up_probability) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def ensure_seqstats_built() -> int:
    """Build the seqstats table from wyckoffstats/*.csv if it is missing/empty.

    Safe to call from any entry point: it self-checks and only builds once.
    Returns the number of rows ingested this call (0 if already built or no CSVs).
    """
    global _SEQSTATS_BUILD_CHECKED
    if _SEQSTATS_BUILD_CHECKED:
        return 0
    _SEQSTATS_BUILD_CHECKED = True

    all_csv = WYCKOFFSTATS_DIR / "sequence_stats.csv"
    if not all_csv.exists():
        return 0  # no source CSVs shipped — sequence-match feature simply stays off

    try:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(DB_PATH)) as conn:
            tbl = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='seqstats'"
            ).fetchone()
            if tbl:
                cnt = conn.execute("SELECT COUNT(*) FROM seqstats").fetchone()[0]
                if cnt and cnt > 0:
                    return 0  # already built
            print("[finagent] 首次运行：正在从 CSV 构建 seqstats 历史参考表（一次性，约数秒）…")
            conn.executescript(_SEQSTATS_SCHEMA)
            total = _ingest_seqstats_csv(conn, all_csv, "ALL")
            for fname in _SEQSTATS_TAG_FILES:
                p = WYCKOFFSTATS_DIR / fname
                if p.exists():
                    total += _ingest_seqstats_csv(conn, p, fname[len("seqstats_"):-len(".csv")])
            conn.commit()
            print(f"[finagent] seqstats 构建完成：{total} 行。")
            return total
    except (sqlite3.Error, OSError, KeyError, ValueError) as e:
        logger.warning(f"seqstats auto-build failed: {e}")
        return 0

# Merge/drop raw wyckoff events into the 25-code mapping space.
# Raw event -> canonical event_mapping key, or None (skip entirely, do not truncate).
_EVENT_NORMALIZE: dict[str, Optional[str]] = {
    "ST_down": "AR&ST_down",
    "ST_up":   "AR&ST_up",
    "AR_down": None,
    "AR_up":   None,
    "OKR_up":   None,
    "OKR_down": None,
    "SOS": None,
    "SOW": None,
    "ABS": None,
    "COB": None,
}

_EVENT_MAP: Optional[dict[str, int]] = None
_REVERSE_EVENT_MAP: Optional[dict[int, str]] = None
_TAG_MAP: Optional[dict[str, tuple[str, str, str]]] = None

MAX_PREFIX_LEN = 10


def _load_event_mapping() -> dict[str, int]:
    global _EVENT_MAP, _REVERSE_EVENT_MAP
    if _EVENT_MAP is not None:
        return _EVENT_MAP
    mapping: dict[str, int] = {}
    path = WYCKOFFSTATS_DIR / "event_mapping.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["event_name"]] = int(row["event_id"])
    _EVENT_MAP = mapping
    _REVERSE_EVENT_MAP = {v: k for k, v in mapping.items()}
    return mapping


def _load_reverse_event_mapping() -> dict[int, str]:
    if _REVERSE_EVENT_MAP is None:
        _load_event_mapping()
    assert _REVERSE_EVENT_MAP is not None
    return _REVERSE_EVENT_MAP


def _load_tags() -> dict[str, tuple[str, str, str]]:
    global _TAG_MAP
    if _TAG_MAP is not None:
        return _TAG_MAP
    tags: dict[str, tuple[str, str, str]] = {}
    if not TAGS_CSV.exists():
        _TAG_MAP = tags
        return tags
    with open(TAGS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("status") or "").strip() != "ok":
                continue
            sym = (row.get("symbol") or "").strip()
            size = (row.get("tag_size") or "").strip()
            style = (row.get("tag_style") or "").strip()
            sector = (row.get("tag_sector") or "").strip()
            if sym:
                tags[sym] = (size, style, sector)
    _TAG_MAP = tags
    return tags


def _normalize_events(recent_events: list[dict]) -> list[int]:
    """
    Input: recent_events ordered 由远及近 (as returned by wyckoff engine).
    Output: int codes ordered 由近及远, with normalization/drop rules applied.
    """
    event_map = _load_event_mapping()
    # Reverse to near -> far
    near_first = list(reversed(recent_events))
    codes: list[int] = []
    for ev in near_first:
        raw = (ev.get("event") or "").strip()
        if not raw:
            continue
        if raw in _EVENT_NORMALIZE:
            canonical = _EVENT_NORMALIZE[raw]
            if canonical is None:
                continue  # skip, don't truncate
        else:
            canonical = raw
        code = event_map.get(canonical)
        if code is None:
            continue  # unknown event -> skip
        codes.append(code)
    return codes


def _codes_to_names(codes_csv: str) -> list[str]:
    rev = _load_reverse_event_mapping()
    out: list[str] = []
    for part in codes_csv.split(","):
        try:
            n = int(part)
        except ValueError:
            continue
        name = rev.get(n)
        if name:
            out.append(name)
    return out


def _query_bucket_best(
    conn: sqlite3.Connection,
    bucket: str,
    prefix_sequences: list[str],
) -> Optional[dict]:
    """Returns the row with max |up_prob - 0.5| among matching prefixes for this bucket, or None."""
    if not prefix_sequences:
        return None
    placeholders = ",".join("?" for _ in prefix_sequences)
    sql = (
        f"SELECT sequence, count, up_probability FROM seqstats "
        f"WHERE bucket = ? AND sequence IN ({placeholders})"
    )
    params = [bucket, *prefix_sequences]
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        return None
    best = max(rows, key=lambda r: abs((r[2] or 0.0) - 0.5))
    seq, cnt, up_prob = best
    return {
        "sequence_codes": seq,
        "sequence_names": _codes_to_names(seq),
        "count": int(cnt),
        "up_prob": float(up_prob),
    }


def get_seqstats_probs(symbol: str, recent_events: list[dict]) -> list[dict]:
    """
    Returns up to 4 dicts, one per source with at least one match:
      [{"source_label": "整体" | "大盘(规模)" | ...,
        "sequence_codes": "5,3,8",
        "sequence_names": ["AS", "AR&ST_down", "BOI"],
        "count": 450, "up_prob": 0.402}, ...]

    Order: 整体 → size → style → sector. Sources with no match are omitted.
    Empty list if no events / no mappable codes / db unavailable.
    """
    if not recent_events:
        return []
    codes = _normalize_events(recent_events)
    if not codes:
        return []

    # Build prefix sequences for lengths 1..MAX_PREFIX_LEN.
    max_len = min(MAX_PREFIX_LEN, len(codes))
    prefix_sequences = [",".join(str(c) for c in codes[:k]) for k in range(1, max_len + 1)]

    tags = _load_tags()
    tag_triple = tags.get(symbol)

    sources: list[tuple[str, str]] = [("ALL", "整体")]
    if tag_triple:
        size, style, sector = tag_triple
        if size:
            sources.append((size, f"{size}(规模)"))
        if style:
            sources.append((style, f"{style}(风格)"))
        if sector:
            sources.append((sector, f"{sector}(行业)"))

    if not Path(DB_PATH).exists():
        return []

    out: list[dict] = []
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            for bucket, label in sources:
                best = _query_bucket_best(conn, bucket, prefix_sequences)
                if best is None:
                    continue
                best["source_label"] = label
                out.append(best)
    except sqlite3.Error:
        return []
    return out


def format_seqstats_block(symbol: str, recent_events: list[dict]) -> list[str]:
    """
    Returns a list of lines to append to the snapshot, or [] to skip the block entirely.
    """
    records = get_seqstats_probs(symbol, recent_events)
    if not records:
        return []

    lines: list[str] = []
    lines.append("【序列匹配概率（历史数据库）】")
    lines.append("  说明: 将当前威科夫事件序列与历史数据库比对，统计相同前缀序列之后的涨跌结果。")
    lines.append("        从长度1-10的前缀中，每个维度取|上涨概率-50%|最大的匹配序列作代表。")
    lines.append("        此为实证历史频率，优先级高于【阶段概率（规则引擎）】，引用概率时应以此为准。")
    for rec in records:
        names = rec.get("sequence_names") or []
        if not names:
            continue
        # sequence_names is near→far; reverse to chronological (far→near) for intuitive "→" reading
        seq_str = " → ".join(reversed(names))
        up_pct = rec["up_prob"] * 100.0
        lines.append(
            f"  {rec['source_label']}: {seq_str} — {rec['count']}次, 上涨 {up_pct:.1f}%"
        )
    lines.append("")
    return lines
