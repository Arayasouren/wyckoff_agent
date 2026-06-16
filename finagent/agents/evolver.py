"""
Evolver agent: generate 3 Pareto candidate strategies and select the best.
GEPA-inspired: diagnosis → multi-candidate → holdout evaluation → select winner.
"""
from __future__ import annotations
import copy
import logging
from typing import Optional

from finagent.agents.base import BaseLLMAgent
from finagent.config import MAX_TOKENS_EVOLVER, DEFAULT_MODEL

logger = logging.getLogger(__name__)

SAFE_BOUNDS: dict = {}

IMMUTABLE_FIELDS = {"schema_version", "symbol", "created_at", "performance_history"}


class EvolverAgent(BaseLLMAgent):
    def __init__(self, current_strategy: dict, model: str = DEFAULT_MODEL):
        system_prompt = current_strategy.get("evolver_system_prompt", "")
        super().__init__(
            system_prompt=system_prompt,
            max_tokens=MAX_TOKENS_EVOLVER,
            model=model,
            temperature=0.5,
        )
        self.current_strategy = current_strategy

    async def generate_candidates(
        self,
        diagnosis: str,
        stats: dict,
        best_records: list[dict],
        worst_records: list[dict],
        symbol: Optional[str] = None,
        symbol_tags: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate 3 candidate strategies based on reflector diagnosis.
        Returns {"candidate_a": {...}, "candidate_b": {...}, "candidate_c": {...},
                 "evolution_notes": "..."}
        """
        from finagent.prompts.templates import build_evolver_user_prompt
        user_msg = build_evolver_user_prompt(
            self.current_strategy, diagnosis, stats, best_records, worst_records,
            symbol=symbol, symbol_tags=symbol_tags,
        )

        raw = await self.call(user_msg)
        try:
            result = self.extract_json(raw)
        except ValueError as e:
            logger.error(f"Evolver response unparseable: {e}")
            # Return 3 copies of current strategy with minor note
            fallback = copy.deepcopy(self.current_strategy)
            fallback["evolution_notes"] = "Evolution LLM call failed, returning current strategy."
            return {
                "candidate_a": fallback,
                "candidate_b": fallback,
                "candidate_c": fallback,
                "evolution_notes": "Fallback: LLM parse error",
            }

        candidates = {}
        notes = result.get("evolution_notes", "")
        for key in ("candidate_a", "candidate_b", "candidate_c"):
            cand = result.get(key)
            if not isinstance(cand, dict):
                cand = copy.deepcopy(self.current_strategy)
            candidates[key] = self._validate_candidate(cand)

        candidates["evolution_notes"] = notes
        return candidates

    def _validate_candidate(self, candidate: dict) -> dict:
        """
        Validate and sanitize a candidate strategy.
        - evolvable_fields dict in current strategy controls per-field enable/disable
        - Invalid values silently fall back to current strategy values
        - Immutable fields are always restored from current strategy
        """
        result = copy.deepcopy(self.current_strategy)
        ef = self.current_strategy.get("evolvable_fields", {})

        # Helper: is this field allowed to evolve?
        def _allowed(field: str) -> bool:
            return ef.get(field, True)  # default True if key absent (backward compat)

        # Copy evolvable text fields if present, non-empty, and unlocked
        for field in ("predictor_system_prompt", "critic_system_prompt",
                      "reflector_system_prompt", "evolver_system_prompt",
                      "symbol_specific_notes", "evolution_notes"):
            if not _allowed(field):
                continue
            val = candidate.get(field)
            if isinstance(val, str) and val.strip():
                result[field] = val
            elif field in ("predictor_system_prompt",) and field not in candidate:
                logger.debug(
                    f"_validate_candidate: '{field}' absent from LLM candidate — "
                    "keeping current value (LLM may have omitted large field to save tokens)"
                )

        # Pass-through: memory notes the candidate declares can be absorbed into predictor_system_prompt.
        # Only filenames (strings) are kept. Actual deletion happens in evolution.py after adoption.
        absorbed = candidate.get("absorbed_memory_notes")
        if isinstance(absorbed, list):
            result["absorbed_memory_notes"] = [f for f in absorbed if isinstance(f, str) and f.strip()]

        # Restore immutable fields
        for field in IMMUTABLE_FIELDS:
            if field in self.current_strategy:
                result[field] = self.current_strategy[field]

        return result

    def select_best_candidate(
        self,
        candidates: dict,
        holdout_scores: dict,
    ) -> tuple[str, dict]:
        """
        Select the best candidate based on holdout evaluation scores.
        holdout_scores: {"candidate_a": float, "candidate_b": float, "candidate_c": float}
        Returns (candidate_key, strategy_dict).
        """
        best_key = max(
            ("candidate_a", "candidate_b", "candidate_c"),
            key=lambda k: holdout_scores.get(k, 0.0),
        )
        logger.info(f"Candidate scores: {holdout_scores} → selected {best_key}")
        return best_key, candidates[best_key]
