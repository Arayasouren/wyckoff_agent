# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Critic agent: evaluate prediction quality against actual outcomes.
"""
from __future__ import annotations
import logging
from typing import Optional

from finagent.agents.base import BaseLLMAgent, LLMCallError
from finagent.config import (
    MAX_TOKENS_CRITIC, DEFAULT_MODEL,
    SCORE_DIRECTION, SCORE_TARGET_HIT, SCORE_SUPPORT,
    SCORE_RESISTANCE, SCORE_CONFIDENCE_CALIB, SCORE_INSIGHT,
    TARGET_HIT_TOLERANCE,
)

logger = logging.getLogger(__name__)


class CriticAgent(BaseLLMAgent):
    def __init__(self, strategy: dict, model: str = DEFAULT_MODEL):
        system_prompt = strategy.get("critic_system_prompt", "")
        super().__init__(
            system_prompt=system_prompt,
            max_tokens=MAX_TOKENS_CRITIC,
            model=model,
            temperature=0.2,
        )

    async def evaluate(self, prediction: dict, actual_outcome: dict) -> dict:
        """
        Evaluate prediction vs actual outcome.
        Returns critique dict with score and qualitative feedback.
        """
        from finagent.prompts.templates import build_critic_user_prompt
        user_msg = build_critic_user_prompt(prediction, actual_outcome)

        # Pre-compute objective metrics (don't rely on LLM for factual checks)
        obj = self._compute_objective_metrics(prediction, actual_outcome)

        raw = await self.call(user_msg)
        try:
            llm_result = self.extract_json(raw)
        except ValueError:
            # Fallback to objective scoring only
            logger.warning("Critic LLM response unparseable, using objective metrics only. Raw: %s", raw[:400])
            llm_result = {
                "score": obj["objective_score"],
                "what_worked": "",
                "what_failed": "LLM evaluation failed",
                "improvement_hints": [],
                "critique": "Objective metrics only.",
            }

        # Merge: use LLM's qualitative text, but anchor score to objective metrics
        final_score = 0.7 * obj["objective_score"] + 0.3 * float(llm_result.get("score", 0))
        final_score = round(max(0.0, min(1.0, final_score)), 4)

        return {
            "score": final_score,
            "actual_direction": obj["actual_direction"],
            "actual_return_pct": obj["actual_return_pct"],
            "max_drawdown_pct": obj["max_drawdown_pct"],
            "max_gain_pct": obj["max_gain_pct"],
            "direction_correct": obj["direction_correct"],
            "support_hit": obj.get("support_hit"),
            "resistance_hit": obj.get("resistance_hit"),
            "target_hit": obj.get("target_hit"),
            "critique_text": llm_result.get("critique", ""),
            "what_worked": llm_result.get("what_worked", ""),
            "what_failed": llm_result.get("what_failed", ""),
            "improvement_hints": llm_result.get("improvement_hints", []),
            "raw_llm_response": raw,
        }

    def _compute_objective_metrics(self, prediction: dict, actual: dict) -> dict:
        pred_dir = prediction.get("direction", "neutral")
        actual_dir = actual.get("actual_direction", "neutral")
        actual_ret = actual.get("actual_return_pct", 0.0)
        max_dd = actual.get("max_drawdown_pct", 0.0)
        max_gain = actual.get("max_gain_pct", 0.0)
        entry = actual.get("entry_price", 0.0)
        exit_p = actual.get("exit_price", 0.0)

        direction_correct = pred_dir == actual_dir
        score = SCORE_DIRECTION if direction_correct else 0.0

        # Target hit
        target_hit = None
        pred_target = prediction.get("target_price")
        if pred_target and entry > 0:
            target_hit = abs(exit_p - pred_target) / max(abs(pred_target), 1e-9) <= TARGET_HIT_TOLERANCE
            if target_hit:
                score += SCORE_TARGET_HIT

        # Support held (bullish prediction: price didn't fall through support)
        support_hit = None
        pred_support = prediction.get("key_support")
        if pred_support and pred_dir == "bullish" and entry > 0:
            min_price = entry * (1 - actual.get("max_drawdown_pct", 0) / 100)
            support_hit = min_price >= pred_support * 0.98
            if support_hit:
                score += SCORE_SUPPORT

        # Resistance broken (bullish prediction: price exceeded resistance)
        resistance_hit = None
        pred_resistance = prediction.get("key_resistance")
        if pred_resistance and pred_dir == "bullish" and entry > 0:
            resistance_hit = exit_p >= pred_resistance * 0.98
            if resistance_hit:
                score += SCORE_RESISTANCE

        # Confidence calibration: high conf + correct, or low conf + wrong
        conf = prediction.get("confidence", 0.5)
        if (conf >= 0.7 and direction_correct) or (conf < 0.5 and not direction_correct):
            score += SCORE_CONFIDENCE_CALIB

        # Insight quality: placeholder, full score if other signals are consistent
        score += SCORE_INSIGHT * 0.5  # partial credit always

        # Opportunity cost penalty: neutral prediction that missed a strong directional move.
        # This prevents the Evolver from converging on "always neutral" as a safe floor.
        # Penalty scales with the magnitude of the missed move: 0 at ±3%, full at ±18%+.
        if pred_dir == "neutral" and actual_dir != "neutral":
            missed_pct = abs(actual_ret)
            if missed_pct > 3.0:
                opp_penalty = min((missed_pct - 3.0) / 15.0, 1.0) * SCORE_DIRECTION
                score = max(0.0, score - opp_penalty)

        return {
            "objective_score": round(min(score, 1.0), 4),
            "direction_correct": direction_correct,
            "actual_direction": actual_dir,
            "actual_return_pct": actual_ret,
            "max_drawdown_pct": max_dd,
            "max_gain_pct": max_gain,
            "target_hit": target_hit,
            "support_hit": support_hit,
            "resistance_hit": resistance_hit,
        }
