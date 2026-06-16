# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Reflector agent (GEPA-inspired): diagnose failure patterns from worst predictions.
Outputs both a diagnosis report and memory suggestions for the MemoryManager.
"""
from __future__ import annotations
import logging
from typing import Optional

from finagent.agents.base import BaseLLMAgent
from finagent.config import MAX_TOKENS_REFLECTOR, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Appended to reflector system prompt to request memory suggestions
_MEMORY_ADDENDUM = """

## 记忆建议
在诊断报告之后，请额外输出一个JSON块，用于更新情境记忆系统。
格式如下（用```json包裹）：
```json
{
  "memory_suggestions": {
    "new_memories": [
      {
        "situation": "情境标题（如：下跌趋势中成长股诱多）",
        "retrieval_text": "情境语义签名（80-150字），用自然语言概括：在什么市场阶段、什么股票类型、什么信号出现时、会发生什么失败模式。这段文本会被embedding后用于未来相似情境的语义检索，请写得具体、可检索——包含关键词：威科夫阶段名称、事件名、股票标签、关键价量特征。",
        "insight": "经验总结（2-3句话，描述为什么会这样、下次该怎么看）",
        "suggested_adjustments": "建议的软性调整（confidence ±0.05-0.10 / 列为风险项 / 优先核对某信号）",
        "sector_scope": ["成长"],
        "sector_excluded": []
      }
    ],
    "update_memories": [
      {
        "file": "已有笔记的文件名.md",
        "revision": "需要修改或补充的内容"
      }
    ],
    "deprecate_memories": ["已过时的笔记文件名.md"]
  }
}
```

注意：
- 只在发现明确的情境性失败模式时建议新增记忆（不要为泛化的建议创建记忆）
- 若此失败模式与股票类型高度相关（如仅在成长股中出现），在 sector_scope 填写对应标签，在 retrieval_text 中明确写入该类型名称
- 如果没有需要新增/修改/废弃的记忆，对应数组留空即可

## retrieval_text 写作要点（决定未来能否被正确检索）

retrieval_text 是这条笔记的**语义检索签名**——未来预测时，系统会将当前市场状态embedding后与所有笔记的retrieval_text做相似度匹配，相似度最高的笔记会被注入predictor。

**写作模板**（参考，不必照搬格式）：
> "在[威科夫阶段，如下跌趋势中段 phase 4-5]的[股票类型，如大盘成长股]，当[关键信号，如放量反弹+Spring形态]出现时，[常见失败模式，如被误判为吸筹反转]。关键细节：[额外区分条件，如冰线未破、上方压力密集]。"

**包含以下关键词可提高检索精度**：
- 威科夫阶段名称（吸筹/派发/上涨/下跌，或具体 phase 编号）
- 关键事件名（SPRING/OB_up/OS_down/SOS/UT/BC 等）
- 股票类型（大盘/成长/价值/红利/周期等）
- 失败模式描述（诱多/诱空/量价背离/派发误判等）

## 记忆笔记的定位与纪律（严格遵守）

- 情境记忆是**次要参考**，其作用是"个别案例级"的提醒，**不是规则**。系统性规律应由 evolver 写入 predictor_system_prompt。
- situation：描述**可复现的具体情境**，不是泛化的市场评论。
- insight：写"当时判断为何失败"与"下次遇到类似情境应检查哪些细节"，**禁止**写成"方向必须 bullish/bearish"之类的硬性方向指令。
- suggested_adjustments：只能是**小幅、定性**的微调——例如"置信度下调 0.05–0.10""将该情境列为风险项""优先核对量价背离"等。**禁止**写"将 direction 强制设为 X"之类硬性指令。
- 笔记不得命令模型覆盖框架判断方向；只能影响风险描述与置信度的边际调整。
"""


class ReflectorAgent(BaseLLMAgent):
    def __init__(self, strategy: dict, model: str = DEFAULT_MODEL):
        system_prompt = strategy.get("reflector_system_prompt", "")
        system_prompt += _MEMORY_ADDENDUM
        super().__init__(
            system_prompt=system_prompt,
            max_tokens=MAX_TOKENS_REFLECTOR,
            model=model,
            temperature=0.4,
        )

    async def diagnose(
        self,
        worst_records: list[dict],
        memory_index: str = "",
        recent_records: Optional[list[dict]] = None,
        direction_stats: Optional[dict] = None,
        symbol: Optional[str] = None,
        symbol_tags: Optional[list[str]] = None,
        best_records: Optional[list[dict]] = None,
        predictor_system_prompt: str = "",
        memory_triggers: str = "",
    ) -> tuple[str, dict]:
        """
        Analyze worst predictions and return:
          - diagnosis text (natural language report)
          - memory_suggestions dict (new/update/deprecate operations)
        """
        from finagent.prompts.templates import build_reflector_user_prompt
        user_msg = build_reflector_user_prompt(
            worst_records, memory_index=memory_index, recent_records=recent_records,
            direction_stats=direction_stats, symbol=symbol, symbol_tags=symbol_tags,
            best_records=best_records, predictor_system_prompt=predictor_system_prompt,
            memory_triggers=memory_triggers,
        )
        raw = await self.call(user_msg)
        logger.info(f"Reflector diagnosis ({len(raw)} chars)")

        # Try to extract memory_suggestions JSON from the response
        memory_suggestions = self._extract_memory_suggestions(raw)

        # The diagnosis text is everything before the JSON block
        diagnosis = raw
        import re
        json_match = re.search(r"```json\s*\{", raw)
        if json_match:
            diagnosis = raw[:json_match.start()].strip()

        return diagnosis, memory_suggestions

    def _extract_memory_suggestions(self, raw: str) -> dict:
        """Extract memory_suggestions JSON from reflector output."""
        import re
        import json

        default = {"new_memories": [], "update_memories": [], "deprecate_memories": []}

        # Look for ```json { ... } ``` block
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if not m:
            return default

        try:
            parsed = json.loads(m.group(1))
            suggestions = parsed.get("memory_suggestions", parsed)
            return {
                "new_memories": suggestions.get("new_memories", []),
                "update_memories": suggestions.get("update_memories", []),
                "deprecate_memories": suggestions.get("deprecate_memories", []),
            }
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse memory suggestions: {e}")
            return default
