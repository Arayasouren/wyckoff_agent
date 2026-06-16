# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
One-shot ingest of wyckoffstats/*.csv into the SQLite `seqstats` table.

CSV schema (both sequence_stats.csv and seqstats_{tag}.csv):
  sequence, length, count, up_count, down_count, up_probability

Rows are loaded into `seqstats` keyed by (bucket, sequence):
  - bucket = "ALL" for sequence_stats.csv
  - bucket = "{tag}" for seqstats_{tag}.csv  (e.g., 大盘, 成长, 金融, ...)

Idempotent: truncates table before inserting.

Run from repo root:  python3 scripts/ingest_seqstats.py
"""
from __future__ import annotations
import csv
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from finagent.config import DB_PATH, ROOT_DIR

WYCKOFFSTATS_DIR = ROOT_DIR / "wyckoffstats"

ALL_CSV = WYCKOFFSTATS_DIR / "sequence_stats.csv"
TAG_FILES = [
    "seqstats_大盘.csv", "seqstats_中盘.csv", "seqstats_小盘.csv",
    "seqstats_成长.csv", "seqstats_价值.csv", "seqstats_红利.csv",
    "seqstats_金融.csv", "seqstats_消费.csv", "seqstats_科技.csv",
    "seqstats_制造.csv", "seqstats_周期.csv", "seqstats_基础设施.csv",
]

SCHEMA_SQL = """
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

BATCH_SIZE = 5000


def _iter_rows(path: Path, bucket: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield (
                bucket,
                row["sequence"],
                int(row["length"]),
                int(row["count"]),
                int(row["up_count"]),
                int(row["down_count"]),
                float(row["up_probability"]),
            )


def ingest_file(conn: sqlite3.Connection, path: Path, bucket: str) -> int:
    total = 0
    batch: list[tuple] = []
    for row in _iter_rows(path, bucket):
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            conn.executemany(
                "INSERT OR REPLACE INTO seqstats "
                "(bucket, sequence, length, count, up_count, down_count, up_probability) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            total += len(batch)
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO seqstats "
            "(bucket, sequence, length, count, up_count, down_count, up_probability) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        total += len(batch)
    return total


def main() -> None:
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not ALL_CSV.exists():
        raise SystemExit(f"Missing {ALL_CSV}")
    missing = [f for f in TAG_FILES if not (WYCKOFFSTATS_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing tag files: {missing}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM seqstats")
        conn.commit()

        print(f"Ingesting {ALL_CSV.name} as bucket=ALL ...")
        n = ingest_file(conn, ALL_CSV, "ALL")
        conn.commit()
        print(f"  inserted {n} rows")

        for fname in TAG_FILES:
            tag = fname[len("seqstats_"):-len(".csv")]
            path = WYCKOFFSTATS_DIR / fname
            print(f"Ingesting {fname} as bucket={tag} ...")
            n = ingest_file(conn, path, tag)
            conn.commit()
            print(f"  inserted {n} rows")

        total = conn.execute("SELECT COUNT(*) FROM seqstats").fetchone()[0]
        print(f"\nDone. Total rows in seqstats: {total}")
        for bucket in ["ALL"] + [f[len("seqstats_"):-len(".csv")] for f in TAG_FILES]:
            c = conn.execute(
                "SELECT COUNT(*) FROM seqstats WHERE bucket = ?", (bucket,)
            ).fetchone()[0]
            print(f"  {bucket}: {c}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
