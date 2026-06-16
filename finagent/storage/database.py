# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SQLite async database for storing predictions, critiques, and evolution runs.
"""
from __future__ import annotations
import json
import aiosqlite
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path

from finagent.config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    profile_name        TEXT NOT NULL DEFAULT '',
    strategy_version    INTEGER NOT NULL DEFAULT 0,
    window_end_date     TEXT NOT NULL,
    horizon_start_date  TEXT NOT NULL,
    horizon_end_date    TEXT NOT NULL,
    direction           TEXT NOT NULL,
    confidence          REAL NOT NULL,
    key_support         REAL,
    key_resistance      REAL,
    target_price        REAL,
    rationale           TEXT,
    key_signals         TEXT,
    risk_factors        TEXT,
    raw_llm_response    TEXT,
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(profile_name, symbol, window_end_date)
);

CREATE TABLE IF NOT EXISTS critiques (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id       INTEGER NOT NULL REFERENCES predictions(id),
    symbol              TEXT NOT NULL,
    actual_direction    TEXT NOT NULL,
    actual_return_pct   REAL NOT NULL,
    max_drawdown_pct    REAL NOT NULL,
    max_gain_pct        REAL NOT NULL,
    direction_correct   INTEGER NOT NULL,
    support_hit         INTEGER,
    resistance_hit      INTEGER,
    target_hit          INTEGER,
    score               REAL NOT NULL,
    critique_text       TEXT,
    what_worked         TEXT,
    what_failed         TEXT,
    improvement_hints   TEXT,
    raw_llm_response    TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evolution_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name            TEXT NOT NULL DEFAULT '',
    symbol                  TEXT NOT NULL,
    from_strategy_version   INTEGER,
    to_strategy_version     INTEGER,
    windows_processed       INTEGER,
    avg_score               REAL,
    direction_accuracy_pct  REAL,
    candidate_chosen        TEXT,
    evolution_notes         TEXT,
    created_at              TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_crit_symbol ON critiques(symbol);
CREATE INDEX IF NOT EXISTS idx_crit_pred ON critiques(prediction_id);

CREATE TABLE IF NOT EXISTS memory_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    note_filename       TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    window_end_date     TEXT NOT NULL,
    direction_correct   INTEGER NOT NULL,
    score               REAL NOT NULL,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_mem_out_note ON memory_outcomes(note_filename);
CREATE INDEX IF NOT EXISTS idx_mem_out_symbol ON memory_outcomes(symbol);

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

# Migration statements: add new columns to existing DBs that predate profile support
MIGRATIONS = [
    "ALTER TABLE predictions ADD COLUMN profile_name TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE evolution_runs ADD COLUMN profile_name TEXT NOT NULL DEFAULT ''",
]

# Post-migration indexes (depend on columns added by MIGRATIONS)
POST_MIGRATION_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pred_profile ON predictions(profile_name, symbol)",
]


@dataclass
class PredictionRecord:
    symbol: str
    profile_name: str
    strategy_version: int
    window_end_date: str
    horizon_start_date: str
    horizon_end_date: str
    direction: str
    confidence: float
    rationale: str = ""
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None
    target_price: Optional[float] = None
    key_signals: Optional[list] = field(default_factory=list)
    risk_factors: Optional[list] = field(default_factory=list)
    raw_llm_response: str = ""
    id: Optional[int] = None


@dataclass
class CritiqueRecord:
    prediction_id: int
    symbol: str
    actual_direction: str
    actual_return_pct: float
    max_drawdown_pct: float
    max_gain_pct: float
    direction_correct: bool
    score: float
    critique_text: str = ""
    what_worked: str = ""
    what_failed: str = ""
    support_hit: Optional[bool] = None
    resistance_hit: Optional[bool] = None
    target_hit: Optional[bool] = None
    improvement_hints: Optional[list] = field(default_factory=list)
    raw_llm_response: str = ""
    id: Optional[int] = None


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_schema(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            # Apply migrations idempotently (ignore "duplicate column" errors)
            for stmt in MIGRATIONS:
                try:
                    await db.execute(stmt)
                except Exception:
                    pass
            # Create indexes that depend on migrated columns
            for stmt in POST_MIGRATION_INDEXES:
                try:
                    await db.execute(stmt)
                except Exception:
                    pass
            await db.commit()

    async def save_prediction(self, rec: PredictionRecord) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT OR IGNORE INTO predictions
                   (profile_name, symbol, strategy_version, window_end_date,
                    horizon_start_date, horizon_end_date, direction, confidence,
                    key_support, key_resistance, target_price, rationale,
                    key_signals, risk_factors, raw_llm_response)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.profile_name, rec.symbol, rec.strategy_version,
                    rec.window_end_date, rec.horizon_start_date, rec.horizon_end_date,
                    rec.direction, rec.confidence, rec.key_support, rec.key_resistance,
                    rec.target_price, rec.rationale,
                    json.dumps(rec.key_signals or [], ensure_ascii=False),
                    json.dumps(rec.risk_factors or [], ensure_ascii=False),
                    rec.raw_llm_response,
                ),
            )
            await db.commit()
            # If IGNORE triggered, fetch existing id
            if cursor.lastrowid == 0:
                async with db.execute(
                    "SELECT id FROM predictions WHERE profile_name=? AND symbol=? AND window_end_date=?",
                    (rec.profile_name, rec.symbol, rec.window_end_date),
                ) as cur:
                    row = await cur.fetchone()
                    return row[0] if row else -1
            return cursor.lastrowid

    async def save_critique(self, rec: CritiqueRecord) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO critiques
                   (prediction_id, symbol, actual_direction, actual_return_pct,
                    max_drawdown_pct, max_gain_pct, direction_correct, support_hit,
                    resistance_hit, target_hit, score, critique_text, what_worked,
                    what_failed, improvement_hints, raw_llm_response)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.prediction_id, rec.symbol, rec.actual_direction,
                    rec.actual_return_pct, rec.max_drawdown_pct, rec.max_gain_pct,
                    int(rec.direction_correct),
                    int(rec.support_hit) if rec.support_hit is not None else None,
                    int(rec.resistance_hit) if rec.resistance_hit is not None else None,
                    int(rec.target_hit) if rec.target_hit is not None else None,
                    rec.score, rec.critique_text, rec.what_worked, rec.what_failed,
                    json.dumps(rec.improvement_hints or [], ensure_ascii=False),
                    rec.raw_llm_response,
                ),
            )
            await db.commit()

    async def prediction_exists(self, profile_name: str, symbol: str, window_end_date: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id FROM predictions WHERE profile_name=? AND symbol=? AND window_end_date=?",
                (profile_name, symbol, window_end_date),
            ) as cur:
                row = await cur.fetchone()
                return row is not None

    async def critique_exists(self, prediction_id: int) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id FROM critiques WHERE prediction_id=?", (prediction_id,)
            ) as cur:
                row = await cur.fetchone()
                return row is not None

    async def get_prediction_by_window(self, symbol: str, window_end_date: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM predictions WHERE symbol=? AND window_end_date=?",
                (symbol, window_end_date),
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def get_summary_stats(self, profile_name: str, symbol: Optional[str] = None) -> dict:
        """Stats for a profile, optionally filtered to a single symbol."""
        if symbol:
            where = "WHERE p.profile_name = ? AND p.symbol = ?"
            params = (profile_name, symbol)
        else:
            where = "WHERE p.profile_name = ?"
            params = (profile_name,)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"""SELECT COUNT(*) as total,
                          AVG(c.score) as avg_score,
                          SUM(c.direction_correct) as correct_count,
                          AVG(c.actual_return_pct) as avg_return
                   FROM predictions p
                   JOIN critiques c ON c.prediction_id = p.id
                   {where}""",
                params,
            ) as cur:
                row = await cur.fetchone()
                total = row[0] or 0
                avg_score = row[1] or 0.0
                correct = row[2] or 0
                avg_return = row[3] or 0.0

        return {
            "total_windows": total,
            "avg_score": round(avg_score, 4),
            "direction_accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "avg_return_pct": round(avg_return, 4),
        }

    async def get_evolution_dataset(
        self, profile_name: str, symbol: Optional[str] = None, limit: int = 500
    ) -> list:
        """Return joined prediction+critique rows, newest first."""
        if symbol:
            where = "WHERE p.profile_name = ? AND p.symbol = ?"
            params = (profile_name, symbol, limit)
        else:
            where = "WHERE p.profile_name = ?"
            params = (profile_name, limit)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"""SELECT p.symbol, p.window_end_date, p.direction, p.confidence,
                          p.key_support, p.key_resistance, p.target_price,
                          p.rationale, p.key_signals, p.risk_factors,
                          c.actual_direction, c.actual_return_pct,
                          c.max_drawdown_pct, c.direction_correct,
                          c.score, c.critique_text, c.what_worked,
                          c.what_failed, c.improvement_hints
                   FROM predictions p
                   JOIN critiques c ON c.prediction_id = p.id
                   {where}
                   ORDER BY p.window_end_date DESC
                   LIMIT ?""",
                params,
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_worst_predictions(
        self, profile_name: str, symbol: Optional[str] = None,
        n: int = 25, max_score: Optional[float] = None,
        since_date: Optional[str] = None, until_date: Optional[str] = None,
    ) -> list:
        if symbol:
            where = "WHERE p.profile_name = ? AND p.symbol = ?"
            params: tuple = (profile_name, symbol)
        else:
            where = "WHERE p.profile_name = ?"
            params = (profile_name,)
        if max_score is not None:
            where += " AND c.score < ?"
            params = params + (max_score,)
        if since_date is not None:
            where += " AND p.window_end_date >= ?"
            params = params + (since_date,)
        if until_date is not None:
            where += " AND p.window_end_date <= ?"
            params = params + (until_date,)
        params = params + (n,)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"""SELECT p.symbol, p.window_end_date, p.direction, p.confidence,
                          p.rationale, p.key_signals,
                          c.actual_direction, c.actual_return_pct,
                          c.score, c.critique_text, c.what_failed,
                          c.improvement_hints
                   FROM predictions p
                   JOIN critiques c ON c.prediction_id = p.id
                   {where}
                   ORDER BY c.score ASC
                   LIMIT ?""",
                params,
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_best_predictions(
        self, profile_name: str, symbol: Optional[str] = None, n: int = 25,
        since_date: Optional[str] = None, until_date: Optional[str] = None,
    ) -> list:
        if symbol:
            where = "WHERE p.profile_name = ? AND p.symbol = ?"
            params: tuple = (profile_name, symbol)
        else:
            where = "WHERE p.profile_name = ?"
            params = (profile_name,)
        if since_date is not None:
            where += " AND p.window_end_date >= ?"
            params = params + (since_date,)
        if until_date is not None:
            where += " AND p.window_end_date <= ?"
            params = params + (until_date,)
        params = params + (n,)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"""SELECT p.symbol, p.window_end_date, p.direction, p.confidence,
                          p.rationale, p.key_signals,
                          c.actual_direction, c.actual_return_pct,
                          c.score, c.critique_text, c.what_worked
                   FROM predictions p
                   JOIN critiques c ON c.prediction_id = p.id
                   {where}
                   ORDER BY c.score DESC
                   LIMIT ?""",
                params,
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_all_predictions(self, profile_name: str, symbol: str) -> list:
        """Return all prediction+critique rows for chart generation, oldest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT p.window_end_date, p.direction, p.confidence,
                          c.score, c.direction_correct, c.actual_direction
                   FROM predictions p
                   LEFT JOIN critiques c ON c.prediction_id = p.id
                   WHERE p.profile_name = ? AND p.symbol = ?
                   ORDER BY p.window_end_date ASC""",
                (profile_name, symbol),
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_recent_predictions(
        self, profile_name: str, n: int = 20,
        symbol: Optional[str] = None,
        since_date: Optional[str] = None, until_date: Optional[str] = None,
    ) -> list:
        """Return the N most recent prediction+critique records, newest first."""
        where = "WHERE p.profile_name = ?"
        params: tuple = (profile_name,)
        if symbol:
            where += " AND p.symbol = ?"
            params = params + (symbol,)
        if since_date is not None:
            where += " AND p.window_end_date >= ?"
            params = params + (since_date,)
        if until_date is not None:
            where += " AND p.window_end_date <= ?"
            params = params + (until_date,)
        params = params + (n,)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"""SELECT p.symbol, p.window_end_date, p.direction, p.confidence,
                          c.actual_direction, c.actual_return_pct, c.score,
                          c.direction_correct
                   FROM predictions p
                   JOIN critiques c ON c.prediction_id = p.id
                   {where}
                   ORDER BY p.window_end_date DESC
                   LIMIT ?""",
                params,
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_direction_stats(
        self, profile_name: str, symbol: Optional[str] = None, last_n: Optional[int] = None
    ) -> dict:
        """Return direction distribution and per-direction win rates for a profile/symbol."""
        if symbol and last_n:
            where = "WHERE p.profile_name = ? AND p.symbol = ?"
            base_params: tuple = (profile_name, symbol)
        elif symbol:
            where = "WHERE p.profile_name = ? AND p.symbol = ?"
            base_params = (profile_name, symbol)
        else:
            where = "WHERE p.profile_name = ?"
            base_params = (profile_name,)

        async with aiosqlite.connect(self.db_path) as db:
            # If last_n, we use a subquery to get the most recent N first
            if last_n:
                sql = f"""
                    SELECT p.direction, c.direction_correct, c.actual_direction
                    FROM (
                        SELECT p2.id, p2.direction, p2.window_end_date
                        FROM predictions p2
                        JOIN critiques c2 ON c2.prediction_id = p2.id
                        {where}
                        ORDER BY p2.window_end_date DESC
                        LIMIT {last_n}
                    ) p
                    JOIN critiques c ON c.prediction_id = p.id
                """
                params: tuple = base_params
            else:
                sql = f"""
                    SELECT p.direction, c.direction_correct, c.actual_direction
                    FROM predictions p
                    JOIN critiques c ON c.prediction_id = p.id
                    {where}
                """
                params = base_params

            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()

        total = len(rows)
        if total == 0:
            return {
                "total": 0, "bullish_cnt": 0, "bearish_cnt": 0, "neutral_cnt": 0,
                "neutral_pct": 0.0, "bullish_win_pct": 0.0, "bearish_win_pct": 0.0,
                "neutral_win_pct": 0.0, "overall_win_pct": 0.0,
            }

        bullish_total = bearish_total = neutral_total = 0
        bullish_correct = bearish_correct = neutral_correct = 0
        overall_correct = 0

        for direction, direction_correct, actual_direction in rows:
            d = (direction or "").lower()
            correct = int(direction_correct or 0)
            overall_correct += correct
            if d == "bullish":
                bullish_total += 1
                bullish_correct += correct
            elif d == "bearish":
                bearish_total += 1
                bearish_correct += correct
            else:
                neutral_total += 1
                # neutral "correct" = actual also neutral
                if (actual_direction or "").lower() == "neutral":
                    neutral_correct += 1

        return {
            "total": total,
            "bullish_cnt": bullish_total,
            "bearish_cnt": bearish_total,
            "neutral_cnt": neutral_total,
            "neutral_pct": round(neutral_total / total, 4),
            "bullish_win_pct": round(bullish_correct / bullish_total, 4) if bullish_total else 0.0,
            "bearish_win_pct": round(bearish_correct / bearish_total, 4) if bearish_total else 0.0,
            "neutral_win_pct": round(neutral_correct / neutral_total, 4) if neutral_total else 0.0,
            "overall_win_pct": round(overall_correct / total, 4),
        }

    async def get_stats_for_version(self, profile_name: str, profile_version: int) -> dict:
        """
        Win-rate stats for all predictions made with a specific profile_version.
        Used to snapshot performance before each evolution.
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT
                       COUNT(*)                   AS total,
                       SUM(c.direction_correct)   AS correct,
                       AVG(c.score)               AS avg_score,
                       AVG(c.actual_return_pct)   AS avg_return,
                       MIN(p.window_end_date)      AS date_from,
                       MAX(p.window_end_date)      AS date_to,
                       GROUP_CONCAT(DISTINCT p.symbol) AS symbols
                   FROM predictions p
                   JOIN critiques c ON c.prediction_id = p.id
                   WHERE p.profile_name = ? AND p.strategy_version = ?""",
                (profile_name, profile_version),
            ) as cur:
                row = await cur.fetchone()
                total    = row[0] or 0
                correct  = row[1] or 0
                avg_score = row[2] or 0.0
                avg_ret  = row[3] or 0.0
                date_from = row[4] or ""
                date_to   = row[5] or ""
                symbols   = sorted((row[6] or "").split(",")) if row[6] else []

        return {
            "total": total,
            "win_rate": round(correct / total, 4) if total > 0 else 0.0,
            "avg_score": round(avg_score, 4),
            "avg_return_pct": round(avg_ret, 4),
            "date_from": date_from,
            "date_to": date_to,
            "symbols": symbols,
        }

    async def save_memory_outcome(
        self,
        note_filename: str,
        symbol: str,
        window_end_date: str,
        direction_correct: bool,
        score: float,
    ) -> None:
        """B3: Persist memory note firing outcome for cross-stock analysis."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO memory_outcomes
                   (note_filename, symbol, window_end_date, direction_correct, score)
                   VALUES (?, ?, ?, ?, ?)""",
                (note_filename, symbol, window_end_date, int(direction_correct), score),
            )
            await db.commit()

    async def save_evolution_run(
        self,
        profile_name: str,
        symbol: str,
        from_version: int,
        to_version: int,
        windows_processed: int,
        avg_score: float,
        direction_accuracy: float,
        candidate_chosen: str,
        evolution_notes: str,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO evolution_runs
                   (profile_name, symbol, from_strategy_version, to_strategy_version,
                    windows_processed, avg_score, direction_accuracy_pct,
                    candidate_chosen, evolution_notes)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (profile_name, symbol, from_version, to_version, windows_processed,
                 avg_score, direction_accuracy, candidate_chosen, evolution_notes),
            )
            await db.commit()
