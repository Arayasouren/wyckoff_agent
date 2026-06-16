# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Situation-aware memory system for evolution profiles.
Each profile gets a memory folder with:
  - MEMORY.md: index of all situation notes (always loaded into predictor context)
  - *.md files: individual situation notes with frontmatter
  - _embeddings.npz: cached embedding vectors for semantic retrieval
"""
from __future__ import annotations
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from finagent.config import PROFILES_DIR

if TYPE_CHECKING:
    from finagent.storage.embedder import Embedder

logger = logging.getLogger(__name__)

MAX_INDEX_LINES = 100           # MEMORY.md line cap
COMPRESS_THRESHOLD_NOTES = 50  # note count >= 50 → compress
MAX_REFINE_COUNT = 5            # max L2 refinement passes per note
MAX_EXCEPTION_BRANCHES = 3      # max "例外分支" sections per note
_EMBED_CONFLICT_TOP_K = 6       # candidate existing notes checked by resolve_new_memory_conflicts


class MemoryManager:
    def __init__(
        self,
        profile_name: str,
        profiles_dir: Path = PROFILES_DIR,
        embedder: Optional["Embedder"] = None,
    ):
        self.profile_name = profile_name
        self.memory_dir = profiles_dir / f"{profile_name}_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / "MEMORY.md"
        self._archive_dir = self.memory_dir / "_archive"
        self._embed_path = self.memory_dir / "_embeddings.npz"
        if not self.index_path.exists():
            self.index_path.write_text("# 情境记忆索引\n", encoding="utf-8")
        # Runtime hit counters — reset per MemoryManager instance (one per evolve run)
        self.match_counts: dict[str, int] = {}   # filename → times matched
        self.match_attempts: int = 0             # total calls to match_by_similarity
        # Embeddings in memory: filename → np.ndarray[float32, DIM]
        self._embeddings: dict[str, np.ndarray] = {}
        self._embeddings_loaded: bool = False
        # Embedder instance (shared across all calls)
        self._embedder = embedder

    def _get_embedder(self) -> "Embedder":
        if self._embedder is None:
            from finagent.storage.embedder import get_default_embedder
            self._embedder = get_default_embedder()
        return self._embedder

    # ── Embeddings I/O ──────────────────────────────────────────────────────

    def _load_embeddings(self) -> None:
        if self._embeddings_loaded:
            return
        self._embeddings_loaded = True
        if not self._embed_path.exists():
            return
        try:
            data = np.load(str(self._embed_path), allow_pickle=True)
            names = data["names"].tolist()
            vectors = data["vectors"]
            self._embeddings = {n: vectors[i] for i, n in enumerate(names)}
            logger.debug(f"Loaded {len(self._embeddings)} embeddings for {self.profile_name}")
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            self._embeddings = {}

    def _save_embeddings(self) -> None:
        if not self._embeddings:
            if self._embed_path.exists():
                self._embed_path.unlink()
            return
        names = np.array(list(self._embeddings.keys()), dtype=object)
        vectors = np.stack(list(self._embeddings.values())).astype(np.float32)
        np.savez_compressed(str(self._embed_path), names=names, vectors=vectors)

    async def _embed_note(self, filename: str, retrieval_text: str) -> None:
        """Embed retrieval_text and update in-memory index + disk."""
        embedder = self._get_embedder()
        if not embedder.configured:
            return
        self._load_embeddings()
        try:
            vec = await embedder.aembed(retrieval_text)
            self._embeddings[filename] = vec
            self._save_embeddings()
        except Exception as e:
            logger.warning(f"Failed to embed note {filename}: {e}")

    async def rebuild_embeddings(self) -> int:
        """
        Batch-rebuild _embeddings.npz from all current notes.
        For notes without retrieval_text in frontmatter, synthesise from situation+body.
        Returns number of notes embedded.
        """
        embedder = self._get_embedder()
        if not embedder.configured:
            logger.warning("Embedder not configured — rebuild_embeddings skipped.")
            return 0

        notes = self.load_all_notes()
        if not notes:
            return 0

        texts = []
        filenames = []
        for n in notes:
            rt = n.get("retrieval_text", "").strip()
            if not rt:
                rt = n.get("situation", "") + " " + n.get("content", "")[:400]
            texts.append(rt.strip())
            filenames.append(n["filename"])

        vecs = await embedder.aembed_batch(texts)
        self._embeddings = {fn: v for fn, v in zip(filenames, vecs)}
        self._embeddings_loaded = True
        self._save_embeddings()
        logger.info(f"Rebuilt {len(filenames)} embeddings for {self.profile_name}")
        return len(filenames)

    async def ensure_embeddings_built(self) -> None:
        """Auto-rebuild _embeddings.npz if it's missing or stale (new notes not embedded)."""
        self._load_embeddings()
        all_files = {p.name for p in self.memory_dir.glob("*.md") if p.name != "MEMORY.md"}
        missing = all_files - set(self._embeddings.keys())
        if not self._embed_path.exists() or missing:
            logger.info(
                f"Embedding index missing or stale ({len(missing)} notes unembedded). Rebuilding..."
            )
            await self.rebuild_embeddings()

    # ── Semantic retrieval ──────────────────────────────────────────────────

    def _build_query_text(self, snapshot: dict, symbol_tags: list) -> str:
        """Build a structured query string from snapshot signals for embedding.

        Uses structured fields instead of raw text slice, so that specific signals
        (event names, phase, probability) are always present regardless of text layout.
        """
        raw = snapshot.get("raw", {})
        phase = snapshot.get("market_phase", raw.get("market_phase", {}))
        prob = snapshot.get("probability", raw.get("probability", {}))

        parts: list[str] = []

        if symbol_tags:
            parts.append(f"股票类型: {'/'.join(symbol_tags)}")

        phase_name = phase.get("phase_name", "")
        phase_id = phase.get("phase_id", "")
        if phase_name:
            parts.append(f"市场阶段: {phase_name} (phase {phase_id})")

        up = prob.get("up", 0)
        down = prob.get("down", 0)
        flat = prob.get("flat", 0)
        parts.append(f"概率: 上涨{up:.0%} 下跌{down:.0%} 横盘{flat:.0%}")

        phase_label = prob.get("phase_label", "")
        phase_desc = prob.get("phase_description", "")
        if phase_label:
            parts.append(f"微观阶段: {phase_label}")
        if phase_desc:
            parts.append(f"阶段描述: {phase_desc[:80]}")

        # Derived probability feature: flag near-equal up/down
        prob_diff = abs(up - down)
        if prob_diff < 0.08:
            parts.append(f"多空概率差极小({prob_diff:.0%})，方向信号模糊")
        elif up > down + 0.15:
            parts.append(f"上涨概率显著偏高(+{up - down:.0%})")
        elif down > up + 0.15:
            parts.append(f"下跌概率显著偏高(+{down - up:.0%})")

        events = raw.get("recent_events", [])
        event_names: list[str] = []
        if events:
            event_names = [e.get("event", "") for e in events[-15:] if e.get("event")]
            if event_names:
                parts.append(f"最近事件: {', '.join(event_names)}")

        # Derived event features: consecutive streaks
        if event_names:
            for sig in ("OB_up", "OB_down", "3H", "3L", "SOT_up", "SOT_down"):
                count = 0
                for name in reversed(event_names):
                    if name == sig:
                        count += 1
                    else:
                        break
                if count >= 3:
                    parts.append(f"连续{count}次{sig}信号")

        sup = phase.get("support_lines") or []
        res = phase.get("resistance_lines") or []
        cur = snapshot.get("current_price") or raw.get("current_price")
        if cur:
            parts.append(f"当前价: {cur:.2f}")
        if sup:
            parts.append(f"支撑: {sup[0]:.2f}")
        if res:
            parts.append(f"阻力: {res[0]:.2f}")
            # Derived position feature: price relative to resistance
            if cur and res[0] > 0:
                dist_pct = (res[0] - cur) / res[0]
                if dist_pct < 0.05:
                    parts.append("价格紧贴阻力位（距阻力<5%）")
                elif dist_pct > 0.20:
                    parts.append(f"价格远低于阻力位（距阻力{dist_pct:.0%}，中低位区间）")

        return "\n".join(parts)

    def _passes_sector_gate(self, note: dict, symbol_tags: list) -> bool:
        """Return False if the note's sector_scope/sector_excluded excludes these symbol_tags."""
        sector_scope = _parse_list_field(note.get("sector_scope", "['all']"))
        if sector_scope and sector_scope != ["all"]:
            if symbol_tags and not any(t in sector_scope for t in symbol_tags):
                return False
        sector_excluded = _parse_list_field(note.get("sector_excluded", "[]"))
        if sector_excluded and symbol_tags:
            if any(t in sector_excluded for t in symbol_tags):
                return False
        return True

    def _get_cross_score(self, note: dict) -> float:
        """Pseudocount-adjusted cross_stock_score for a note dict."""
        validated = _parse_list_field(note.get("stocks_validated", "[]"))
        failed = _parse_list_field(note.get("stocks_failed", "[]"))
        total = len(validated) + len(failed)
        if total < 3:
            return (len(validated) + 0.3) / (total + 1.0)
        return float(note.get("cross_stock_score", 0.3))

    async def match_by_similarity(
        self, snapshot: dict, symbol_tags: list = [], k: int = 5,
        llm_fn=None,
    ) -> list[str]:
        """
        Retrieve the top-k most relevant memory notes for the given snapshot.

        If llm_fn is provided: LLM reads all notes and selects the most relevant.
        Otherwise: falls back to embedding cosine similarity (used for compression clustering).
        """
        notes = self.load_all_notes()
        candidates = [n for n in notes if self._passes_sector_gate(n, symbol_tags)]
        if not candidates:
            return []

        if llm_fn is not None:
            top = await self._match_by_llm(snapshot, symbol_tags, candidates, k, llm_fn)
        else:
            top = await self._match_by_embedding(snapshot, symbol_tags, candidates, k)

        self.match_attempts += 1
        for fn in top:
            self.match_counts[fn] = self.match_counts.get(fn, 0) + 1
        return top

    async def _match_by_llm(
        self, snapshot: dict, symbol_tags: list, candidates: list, k: int, llm_fn
    ) -> list[str]:
        """LLM-based note selection: send all notes + snapshot summary, get ranked filenames."""
        import json, re as _re

        summary = self._build_query_text(snapshot, symbol_tags)

        index_map = {i: n["filename"] for i, n in enumerate(candidates, 1)}

        note_lines = []
        for i, n in enumerate(candidates, 1):
            situation = n.get("situation", n["filename"].replace(".md", ""))
            sig = (n.get("retrieval_text") or "")[:150]
            note_lines.append(
                f"{i}. 情境: {situation}\n"
                f"   签名: {sig}"
            )

        prompt = f"""你是威科夫记忆检索助手。根据当前市场快照，从记忆笔记库中选出最相关的笔记。

## 当前市场快照
{summary}

## 记忆笔记库（共{len(candidates)}条）
{chr(10).join(note_lines)}

## 选择标准
选出与当前快照**最可能相关的陷阱或失败模式**，即：笔记描述的情境在当前信号组合下有可能发生。
- 优先选择：与当前事件序列、概率结构、价格位置直接匹配的笔记
- 最多选{k}条，可以少于{k}条（若无明显相关笔记则只选1-2条）
- 不要为了凑数选不相关的笔记

直接输出编号数组，按相关性从高到低，例如 [3, 1, 5]："""

        try:
            raw = await llm_fn(prompt)
            m = _re.search(r'\[[\d,\s]+\]', raw)
            if m:
                indices = json.loads(m.group())
                return [
                    index_map[i] for i in indices
                    if isinstance(i, int) and i in index_map
                ][:k]
        except Exception as e:
            logger.warning(f"LLM memory selection failed: {e}, falling back to embedding")
        return await self._match_by_embedding(snapshot, symbol_tags, candidates, k)

    async def _match_by_embedding(
        self, snapshot: dict, symbol_tags: list, candidates: list, k: int
    ) -> list[str]:
        """Embedding cosine similarity fallback (also used for compression clustering)."""
        embedder = self._get_embedder()
        if not embedder.configured:
            return []

        self._load_embeddings()
        if not self._embeddings:
            return []

        candidates = [n for n in candidates if n["filename"] in self._embeddings]
        if not candidates:
            return []

        query_text = self._build_query_text(snapshot, symbol_tags)
        try:
            q = await embedder.aembed(query_text)
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        names = [n["filename"] for n in candidates]
        M = np.stack([self._embeddings[fn] for fn in names]).astype(np.float32)
        q32 = q.astype(np.float32)
        norm_M = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        norm_q = np.linalg.norm(q32) + 1e-9
        sims = (M / norm_M) @ (q32 / norm_q)

        note_by_fn = {n["filename"]: n for n in candidates}
        scores = {
            fn: float(sims[i]) * (0.7 + 0.3 * self._get_cross_score(note_by_fn[fn]))
            for i, fn in enumerate(names)
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]

    # ── Index I/O ───────────────────────────────────────────────────────────

    def load_index(self) -> str:
        """Return full MEMORY.md content (for injection into prompts)."""
        if not self.index_path.exists():
            return ""
        return self.index_path.read_text(encoding="utf-8")

    def _rewrite_index(self, entries: list[dict]) -> None:
        lines = ["# 情境记忆索引\n"]
        for e in entries[:MAX_INDEX_LINES]:
            lines.append(f"- [{e['title']}]({e['filename']}) — {e['summary']}")
        self.index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _parse_index(self) -> list[dict]:
        text = self.load_index()
        entries = []
        for line in text.splitlines():
            m = re.match(r"^- \[(.+?)\]\((.+?)\)\s*—\s*(.+)$", line)
            if m:
                entries.append({
                    "title": m.group(1),
                    "filename": m.group(2),
                    "summary": m.group(3).strip(),
                })
        return entries

    # ── Note I/O ────────────────────────────────────────────────────────────

    def load_notes(self, filenames: list[str]) -> list[dict]:
        """Load specific notes by filename. Returns list of note dicts."""
        results = []
        for fn in filenames:
            path = self.memory_dir / fn
            if not path.exists():
                continue
            text = _safe_read_text(path, "load_notes")
            fm, body = _parse_frontmatter(text)
            results.append({
                "filename": fn,
                "situation": fm.get("situation", ""),
                "retrieval_text": fm.get("retrieval_text", ""),
                "confidence": fm.get("confidence", 0.0),
                "cross_stock_score": float(fm.get("cross_stock_score", 0.5)),
                "sector_scope": fm.get("sector_scope", "['all']"),
                "sector_excluded": fm.get("sector_excluded", "[]"),
                "stocks_validated": fm.get("stocks_validated", "[]"),
                "stocks_failed": fm.get("stocks_failed", "[]"),
                "refined_count": int(fm.get("refined_count", 0)),
                "content": body.strip(),
            })
        return results

    def load_all_notes(self) -> list[dict]:
        filenames = [
            p.name for p in self.memory_dir.glob("*.md")
            if p.name != "MEMORY.md"
        ]
        return self.load_notes(filenames)

    # ── CRUD ────────────────────────────────────────────────────────────────

    async def add_memory(
        self,
        situation: str,
        content: str,
        retrieval_text: str = "",
        summary: str = "",
        confidence: float = 0.5,
        source_windows: int = 0,
        sector_scope: list = None,
        sector_excluded: list = None,
    ) -> Path:
        """Create a new memory note, update MEMORY.md index, and embed retrieval_text."""
        slug = _slugify(situation)
        filename = f"{slug}.md"
        path = self.memory_dir / filename
        counter = 1
        while path.exists():
            filename = f"{slug}_{counter}.md"
            path = self.memory_dir / filename
            counter += 1

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        scope_str = str(sector_scope or ["all"]).replace('"', "'")
        excluded_str = str(sector_excluded or []).replace('"', "'")

        # Synthesise retrieval_text if not provided
        if not retrieval_text:
            retrieval_text = f"{situation} {content[:300]}"

        note_text = (
            f"---\n"
            f"situation: {situation}\n"
            f"retrieval_text: {retrieval_text}\n"
            f"confidence: {confidence}\n"
            f"created_at: {now}\n"
            f"evolved_at: {now}\n"
            f"source_windows: {source_windows}\n"
            f"stocks_validated: []\n"
            f"stocks_failed: []\n"
            f"cross_stock_score: 0.3\n"
            f"sector_scope: {scope_str}\n"
            f"sector_excluded: {excluded_str}\n"
            f"---\n\n"
            f"{content}\n"
        )
        path.write_text(note_text, encoding="utf-8")

        entries = self._parse_index()
        if not summary:
            summary = content[:80].replace("\n", " ")
        entries.append({"title": situation, "filename": filename, "summary": summary})
        self._rewrite_index(entries)

        # Embed the retrieval_text
        await self._embed_note(filename, retrieval_text)

        logger.info(f"Memory added: {filename}")
        return path

    def update_memory(self, filename: str, new_content: str,
                      new_retrieval_text: str = "") -> Path:
        """Overwrite a memory note's body, preserving frontmatter. Re-embeds if rt changed."""
        path = self.memory_dir / filename
        if not path.exists():
            import re as _re
            def _norm(s: str) -> str:
                return _re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", s.lower())
            target = _norm(filename)
            candidates = [p for p in self.memory_dir.glob("*.md")
                          if _norm(p.name) == target]
            if candidates:
                path = candidates[0]
                filename = path.name
                logger.info(f"update_memory: fuzzy-matched '{filename}' → '{path.name}'")
            else:
                raise FileNotFoundError(f"Memory note not found: {filename}")

        text = _safe_read_text(path, "update_memory")
        fm, _ = _parse_frontmatter(text)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fm["evolved_at"] = now

        old_rt = str(fm.get("retrieval_text", ""))
        if new_retrieval_text:
            fm["retrieval_text"] = new_retrieval_text

        fm_lines = ["---"]
        for k, v in fm.items():
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        path.write_text("\n".join(fm_lines) + f"\n\n{new_content}\n", encoding="utf-8")
        logger.info(f"Memory updated: {filename}")

        # Re-embed if retrieval_text changed (fire-and-forget via stored reference)
        new_rt = str(fm.get("retrieval_text", ""))
        if new_rt and new_rt != old_rt:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._embed_note(filename, new_rt))
            except RuntimeError:
                pass  # not in async context — caller should call ensure_embeddings_built() later

        return path

    def deprecate_memory(self, filename: str) -> None:
        """Move a memory note to _archive/ and remove from index and embeddings."""
        path = self.memory_dir / filename
        if not path.exists():
            import re as _re
            def _norm(s: str) -> str:
                return _re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", s.lower())
            candidates = [p for p in self.memory_dir.glob("*.md")
                          if _norm(p.name) == _norm(filename)]
            if candidates:
                path = candidates[0]
                filename = path.name
                logger.info(f"deprecate_memory: fuzzy-matched → '{filename}'")
            else:
                logger.warning(f"Cannot deprecate: {filename} not found")
                return

        self._archive_dir.mkdir(exist_ok=True)
        shutil.move(str(path), str(self._archive_dir / filename))

        entries = [e for e in self._parse_index() if e["filename"] != filename]
        self._rewrite_index(entries)

        # Remove from embedding index
        self._load_embeddings()
        if filename in self._embeddings:
            del self._embeddings[filename]
            self._save_embeddings()

        logger.info(f"Memory deprecated: {filename} → _archive/")

    def cleanup_ghost_entries(self) -> int:
        """Remove MEMORY.md entries and embeddings whose files no longer exist."""
        entries = self._parse_index()
        valid = [e for e in entries if (self.memory_dir / e["filename"]).exists()]
        removed = len(entries) - len(valid)
        if removed:
            self._rewrite_index(valid)
            logger.info(f"Cleaned up {removed} ghost memory entries from index")

        # Also clean orphan embeddings
        self._load_embeddings()
        orphan_keys = [fn for fn in list(self._embeddings)
                       if not (self.memory_dir / fn).exists()]
        if orphan_keys:
            for fn in orphan_keys:
                del self._embeddings[fn]
            self._save_embeddings()
            logger.info(f"Cleaned up {len(orphan_keys)} orphan embedding entries")

        return removed

    # ── Cross-stock validation scoring ─────────────────────────────────────

    def update_note_outcome(self, filename: str, symbol: str, was_correct: bool,
                            symbol_tags: list = []) -> None:
        """
        Record whether a retrieved memory note led to a correct prediction.
        Updates stocks_validated / stocks_failed lists and cross_stock_score.
        """
        path = self.memory_dir / filename
        if not path.exists():
            return
        text = _safe_read_text(path, "update_note_outcome")
        fm, body = _parse_frontmatter(text)

        validated = _parse_list_field(fm.get("stocks_validated", "[]"))
        failed = _parse_list_field(fm.get("stocks_failed", "[]"))

        if was_correct:
            if symbol not in validated:
                validated.append(symbol)
            failed = [s for s in failed if s != symbol]
        else:
            if symbol not in failed:
                failed.append(symbol)
            validated = [s for s in validated if s != symbol]

        total = len(validated) + len(failed)
        cross_score = round(len(validated) / total, 3) if total > 0 else 0.5

        fm["stocks_validated"] = "[" + ", ".join(f"'{s}'" for s in validated) + "]"
        fm["stocks_failed"] = "[" + ", ".join(f"'{s}'" for s in failed) + "]"
        fm["cross_stock_score"] = cross_score

        # Track per-sector failures; auto-narrow sector_scope on persistent failures
        if symbol_tags and not was_correct:
            sector_failures_raw = fm.get("sector_failures", "{}")
            try:
                import json as _json
                sector_failures: dict = _json.loads(str(sector_failures_raw).replace("'", '"'))
            except Exception:
                sector_failures = {}
            for tag in symbol_tags:
                sector_failures[tag] = sector_failures.get(tag, 0) + 1
            fm["sector_failures"] = str(sector_failures).replace('"', "'")

            current_scope = _parse_list_field(fm.get("sector_scope", "['all']"))
            if current_scope == ["all"] and len(failed) >= 3:
                sectors_to_exclude = [t for t, cnt in sector_failures.items() if cnt >= 3]
                if sectors_to_exclude:
                    excluded_raw = fm.get("sector_excluded", "[]")
                    excluded = _parse_list_field(excluded_raw)
                    for t in sectors_to_exclude:
                        if t not in excluded:
                            excluded.append(t)
                    fm["sector_excluded"] = "[" + ", ".join(f"'{t}'" for t in excluded) + "]"

        fm_lines = ["---"]
        for k, v in fm.items():
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        path.write_text("\n".join(fm_lines) + f"\n\n{body.lstrip()}", encoding="utf-8")

    # ── L2 Deepening ────────────────────────────────────────────────────────

    async def refine_note_if_needed(
        self,
        filename: str,
        snapshot: dict,
        prediction: dict,
        critique: dict,
        llm_fn,
    ) -> bool:
        """
        Called after critique when this note fired but outcome still contradicted it.
        Refines the note body and optionally updates retrieval_text.
        Returns True if note was updated.
        """
        direction_correct = critique.get("direction_correct", True)
        if direction_correct:
            return False

        path_check = self.memory_dir / filename
        if path_check.exists():
            fm_check, _ = _parse_frontmatter(_safe_read_text(path_check, "refine_note_if_needed"))
            if int(fm_check.get("refined_count", 0)) >= MAX_REFINE_COUNT:
                return False

        path = self.memory_dir / filename
        if not path.exists():
            return False

        note_text = _safe_read_text(path, "refine_note_if_needed")
        fm, body = _parse_frontmatter(note_text)

        pred_dir = prediction.get("direction", "?")
        actual_dir = critique.get("actual_direction", "?")
        actual_ret = critique.get("actual_return_pct", 0.0)
        what_failed = critique.get("what_failed", "")
        existing_branches = len(re.findall(r"^##\s*例外分支", note_text, re.MULTILINE))

        prompt = f"""你是威科夫策略改进专家。以下是一条情境记忆笔记，它在这次预测中被检索到，但预测仍然错了。

## 当前笔记内容
```
{note_text}
```

## 本次预测情况
- 预测方向：{pred_dir}
- 实际方向：{actual_dir}
- 实际涨跌幅：{actual_ret:.2f}%
- 失败原因（Critic评估）：{what_failed}

## 快照摘要（检索时的市场状态）
{snapshot.get("text", "")[:800]}

## 你的任务
笔记内容已正确描述了某类失败情境，但这次它没能帮到我们。请分析：
1. 这次失败是否属于笔记已描述情境的**例外分支**？（如：同样情境但有某个额外信号使结论相反）
2. 还是笔记的**建议本身需要修正**？（如：建议在某条件下反而错误）

输出格式（只输出笔记的新正文，不含frontmatter，markdown格式）：
- 保留原笔记的核心描述和建议
- 在末尾增加「## 例外分支」或「## 补充条件」小节，描述本次新发现的条件
- 可选：在「## retrieval_text更新建议」中写出更准确的检索签名（80-150字自然语言），描述适用情境
- **重要**：例外分支总数不超过 {MAX_EXCEPTION_BRANCHES} 个。当前已有 {existing_branches} 个。如已达上限，将新发现与现有最相似例外合并
- 简洁，不超过400字
"""
        try:
            new_body = await llm_fn(prompt)
        except Exception as e:
            logger.warning(f"refine_note LLM failed for {filename}: {e}")
            return False

        new_body = new_body.strip()
        if not new_body or len(new_body) < 50:
            return False

        # Check if there's a retrieval_text update suggestion
        rt_match = re.search(
            r"##\s*retrieval_text更新建议\s*\n+(.+?)(?=\n##|\Z)", new_body, re.DOTALL
        )
        new_rt = ""
        if rt_match:
            new_rt = rt_match.group(1).strip()
            if new_rt:
                fm["retrieval_text"] = new_rt

        fm["source_windows"] = int(fm.get("source_windows", 0)) + 1
        fm["evolved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fm["refined_count"] = int(fm.get("refined_count", 0)) + 1

        fm_lines = ["---"]
        for k, v in fm.items():
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        path.write_text("\n".join(fm_lines) + f"\n\n{new_body}\n", encoding="utf-8")

        # Re-embed if retrieval_text was updated
        if new_rt:
            await self._embed_note(filename, new_rt)

        logger.info(f"L2 refined: {filename} (refined_count={fm['refined_count']})")
        return True

    # ── Memory Compression ──────────────────────────────────────────────────

    async def compress_if_needed(self, llm_fn) -> int:
        """
        If note count >= COMPRESS_THRESHOLD_NOTES, identify similar note groups
        via embedding clustering + LLM merge. Returns number of notes merged away.
        """
        notes = self.load_all_notes()
        if len(notes) < COMPRESS_THRESHOLD_NOTES:
            return 0
        logger.info(f"{len(notes)} notes >= threshold {COMPRESS_THRESHOLD_NOTES}, running compression")
        return await self._run_compression(llm_fn)

    async def _run_compression(self, llm_fn) -> int:
        """Use embedding clustering to find merge candidates, then LLM-merge each cluster."""
        notes = self.load_all_notes()
        if len(notes) < 6:
            return 0

        self._load_embeddings()

        # Step 1: Embedding-based pre-clustering (cosine > 0.85 threshold)
        embedded_notes = [n for n in notes if n["filename"] in self._embeddings]
        if len(embedded_notes) < 2:
            # Fallback to LLM-only if embeddings unavailable
            return await self._run_compression_llm_only(llm_fn, notes)

        fns = [n["filename"] for n in embedded_notes]
        M = np.stack([self._embeddings[fn] for fn in fns]).astype(np.float32)
        norm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        M_norm = M / norm
        sim_matrix = M_norm @ M_norm.T  # shape (N, N)

        # Greedy clustering: group notes with pairwise sim > 0.85
        visited = set()
        clusters = []
        for i in range(len(fns)):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, len(fns)):
                if j not in visited and sim_matrix[i, j] > 0.85:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) >= 2:
                clusters.append([fns[k] for k in cluster])

        if not clusters:
            return 0

        merged_total = 0
        for group in clusters:
            valid = [fn for fn in group if (self.memory_dir / fn).exists()]
            if len(valid) >= 2:
                merged = await self._merge_notes(valid, llm_fn)
                if merged:
                    merged_total += len(valid) - 1

        return merged_total

    async def _run_compression_llm_only(self, llm_fn, notes) -> int:
        """LLM-only compression fallback (used when embeddings unavailable)."""
        note_summaries = [
            f"[{n['filename']}]\nsituation: {n.get('situation', '')}\n"
            f"body: {n.get('content', '')[:200]}"
            for n in notes
        ]
        prompt = f"""你是威科夫策略知识库管理员。以下是 {len(notes)} 条情境记忆笔记的摘要。

{chr(10).join(note_summaries)}

## 任务
找出相似度极高（≥0.80）、可以合并为一条的笔记组。

输出 JSON 列表（只输出 JSON，不加说明）：
[
  {{
    "merge_group": ["filename_a.md", "filename_b.md"],
    "reason": "相似原因说明"
  }}
]

如果没有可合并的，输出空列表 []。
"""
        try:
            raw = await llm_fn(prompt)
        except Exception as e:
            logger.warning(f"compress LLM failed: {e}")
            return 0

        import json
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return 0
        try:
            groups = json.loads(json_match.group())
        except Exception:
            return 0

        merged_total = 0
        for group in (groups or []):
            fns = group.get("merge_group", [])
            valid = [fn for fn in fns if (self.memory_dir / fn).exists()]
            if len(valid) >= 2:
                if await self._merge_notes(valid, llm_fn):
                    merged_total += len(valid) - 1
        return merged_total

    async def _merge_notes(self, filenames: list, llm_fn) -> bool:
        """Merge multiple notes into one, deprecate the originals."""
        import json

        note_texts = []
        for fn in filenames:
            p = self.memory_dir / fn
            if p.exists():
                note_texts.append(f"=== {fn} ===\n{_safe_read_text(p, '_merge_notes')}")

        all_notes_text = "\n\n".join(note_texts)
        notes_data = self.load_notes(filenames)

        prompt = f"""将以下威科夫策略情境记忆笔记合并为一条更全面的笔记。

{all_notes_text}

要求：
- situation（标题）：用一句话概括合并后的核心情境
- retrieval_text：80-150字的自然语言情境签名，涵盖合并后最重要的失败模式与适用情境，供embedding检索用
- 正文：保留所有重要知识点，整合成连贯说明，包含「## 触发条件」「## 失败模式」「## 建议调整」「## 例外分支」（如有）
- confidence：取各笔记 confidence 的最大值
- sector_scope：取各笔记 sector_scope 的并集（合并后规律适用的股票类型）

输出严格 JSON（不加说明）：
{{
  "situation": "...",
  "retrieval_text": "...",
  "confidence": 0.0,
  "summary": "一行摘要（用于索引）",
  "body": "笔记正文（markdown）",
  "sector_scope": ["all"]
}}
"""
        try:
            raw = await llm_fn(prompt)
        except Exception as e:
            logger.warning(f"compress LLM (merge) failed: {e}")
            return False

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return False
        try:
            merged_data = json.loads(json_match.group())
        except Exception:
            return False

        situation = merged_data.get("situation", "").strip()
        retrieval_text = merged_data.get("retrieval_text", "").strip()
        confidence = float(merged_data.get("confidence", 0.5))
        summary = merged_data.get("summary", "")
        body = merged_data.get("body", "").strip()
        sector_scope = merged_data.get("sector_scope", ["all"])

        if not situation or not body:
            return False

        total_windows = sum(int(n.get("source_windows", 0)) for n in notes_data)

        for fn in filenames:
            self.deprecate_memory(fn)

        await self.add_memory(
            situation=situation,
            content=body,
            retrieval_text=retrieval_text,
            summary=summary,
            confidence=confidence,
            source_windows=total_windows,
            sector_scope=sector_scope,
        )
        logger.info(f"Merged {filenames} → {_slugify(situation)}.md")
        return True

    # ── Contradiction detection for new memories ────────────────────────────

    async def resolve_new_memory_conflicts(self, new_mem: dict, llm_fn) -> dict:
        """
        Check a proposed new memory against semantically similar existing memories.
        Uses embedding Top-K to limit LLM context to the most relevant candidates.

        Returns a decision dict with action: "add" | "skip" | "replace" | "branch".
        Falls back to {"action":"add"} on LLM failure.
        """
        import json as _json

        all_notes = self.load_all_notes()
        if not all_notes:
            return {"action": "add"}

        # Find Top-K semantically similar notes via embedding
        candidate_notes = all_notes
        self._load_embeddings()
        if self._embeddings:
            new_rt = new_mem.get("retrieval_text", "") or new_mem.get("situation", "")
            embedder = self._get_embedder()
            if embedder.configured and new_rt:
                try:
                    q = await embedder.aembed(new_rt)
                    fns = [n["filename"] for n in all_notes if n["filename"] in self._embeddings]
                    if fns:
                        M = np.stack([self._embeddings[fn] for fn in fns]).astype(np.float32)
                        norm_M = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
                        norm_q = np.linalg.norm(q.astype(np.float32)) + 1e-9
                        sims = (M / norm_M) @ (q.astype(np.float32) / norm_q)
                        top_idx = np.argsort(sims)[::-1][:_EMBED_CONFLICT_TOP_K]
                        top_fns = {fns[i] for i in top_idx}
                        candidate_notes = [n for n in all_notes if n["filename"] in top_fns]
                except Exception:
                    pass  # fall through to all notes

        existing_summaries = [
            f"[{n['filename']}]\nsituation: {n.get('situation', '')}\n"
            f"retrieval_text: {n.get('retrieval_text', '')}\n"
            f"body: {n.get('content', '')[:220]}"
            for n in candidate_notes
        ]

        new_situation = new_mem.get("situation", "")
        new_rt = new_mem.get("retrieval_text", "")
        new_content = new_mem.get("content", "")
        new_summary = new_mem.get("summary", "")

        prompt = f"""你是威科夫策略知识库管理员。现有若干最相关的情境记忆。
Reflector 刚产出一条新笔记，你需要判断它与现有笔记的关系，做出操作决策。

## 最相关的现有笔记（按语义相似度排序）
{chr(10).join(existing_summaries)}

## 新提交的笔记
situation: {new_situation}
retrieval_text: {new_rt}
summary: {new_summary}
body:
{new_content}

## 决策规则
逐条比较，尤其关注 situation / retrieval_text 有显著重叠的那几条：
  A. 无冲突：新笔记覆盖现有笔记尚未涉及的情境 → action="add"
  B. 被覆盖：某条现有笔记已以更完整方式覆盖新笔记，新笔记是冗余 → action="skip"
  C. 更完备：新笔记与某条/几条现有笔记讨论同一情境，但新笔记逻辑更完整/条件更准确/样本更新
     → action="replace"，整合为一条，废弃原件
  D. 可分支：新笔记与某条现有笔记在同一情境下结论相反，但可通过明确区分条件（stock_tags、phase、信号）区分
     → action="branch"，在新笔记body中写「## 例外分支」段落说明与sibling的区别，不废弃sibling

## 输出 JSON 之一（只输出 JSON，不加说明）
{{"action": "add"}}

{{"action": "skip", "existing_file": "xxx.md", "reason": "..."}}

{{"action": "replace",
  "replace_files": ["a.md"],
  "situation": "合并后标题",
  "retrieval_text": "合并后检索签名（80-150字）",
  "summary": "一行摘要",
  "content": "markdown 正文",
  "confidence": 0.65}}

{{"action": "branch",
  "sibling_file": "existing.md",
  "situation": "新笔记标题",
  "retrieval_text": "此分支的检索签名（80-150字）",
  "summary": "一行摘要",
  "content": "markdown 正文，必须含「## 例外分支」段",
  "confidence": 0.6}}
"""
        try:
            raw = await llm_fn(prompt)
        except Exception as e:
            logger.warning(f"resolve_new_memory_conflicts LLM failed: {e}")
            return {"action": "add"}

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"action": "add"}
        try:
            decision = _json.loads(m.group())
        except Exception:
            return {"action": "add"}

        action = decision.get("action")
        if action not in {"add", "skip", "replace", "branch"}:
            return {"action": "add"}
        return decision


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_list_field(raw) -> list:
    """Parse a YAML-style list field like "['a', 'b']" or "[]" into a Python list."""
    if not raw or str(raw).strip() in ("[]", ""):
        return []
    raw = str(raw).strip().strip("[]")
    return [s.strip().strip("'\"") for s in raw.split(",") if s.strip()]


def _slugify(text: str) -> str:
    """Convert a situation title to a safe filename slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return slug[:60] if slug else "memory"


def _safe_read_text(path: "Path", warn_prefix: str = "_safe_read_text") -> str:
    """
    UTF-8-tolerant file read. If the file has corrupted bytes (rare but happens
    when LLM-streamed text gets mangled mid-multibyte-char), replace + strip
    U+FFFD rather than raise — preserves whatever readable content remains.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_bytes()
        recovered = raw.decode("utf-8", errors="replace").replace("�", "")
        logger.warning(f"{warn_prefix}: {path.name} 含损坏 UTF-8 字节，已剥离恢复 ({len(raw)} bytes → {len(recovered)} chars)")
        return recovered


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from a markdown note."""
    fm = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    try:
                        val = float(val)
                        if val == int(val):
                            val = int(val)
                    except (ValueError, TypeError):
                        pass
                    fm[key] = val
    return fm, body
