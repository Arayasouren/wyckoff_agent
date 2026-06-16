"""
Predictor agent: given a Wyckoff snapshot, predict the next 20 trading days.
"""
from __future__ import annotations
import logging
from typing import Optional

from finagent.agents.base import BaseLLMAgent
from finagent.config import MAX_TOKENS_PREDICTOR, DEFAULT_MODEL
from finagent.storage.memory_store import MemoryManager

logger = logging.getLogger(__name__)

# Fixed Wyckoff base prompt — prepended to every predictor system prompt,
# never modified by evolution.
_WYCKOFF_BASE_PROMPT = """\
## 威科夫分析基础框架（不变原则）

### 威科夫三大原则（每次分析的出发点）
1. **供求关系原则**：股票涨跌本质是供求关系的变化，关注技术特征的目的是识别出吸筹和派发阶段和供需强弱，而非利用技术形态本身进行交易。
2. **因果关系原则**：横盘整理（因）决定后续运动幅度（果）。P&F目标是估算"因"积累多少
3. **努力与结果原则**：成交量（努力）应与价格变化（结果）成正比。放量小涨=供压大；缩量大涨=需求强

### 主力视角
所有分析从"主力"思维出发，通过价格/成交量形态判断市场供需强弱和主力行为逻辑：
- **吸筹阶段**：PS后低位放量止跌（SC/AR/ST），随后缩量窄幅整理，主力悄悄建仓
- **派发阶段**：PSY后高位放量冲高受阻（BC/AR/ST）或上涨波段收窄（SOT），随后缩量反弹乏力顶部震荡，主力悄悄出货
- **测试行为**：成交量明显萎缩的回踩/反弹（No Supply / No Demand），验证供求是否已转换
- **吸筹完成标志**：Spring/TSO洗盘结束后，JOC（跳过阻力）/3H拉升测试，紧接缩量小幅回踩（LPS），标志着洗盘干净，可能会上涨加速，是标准买点
- **派发完成标志**：UT/UTAD假突破套人后，3L砸盘测试，紧接LPSY后跌破ICE线（BOI），可能放量加速下跌，LPSY是强卖点

### 事件名的本质：程序标签，不是市场结论
snapshot 里的威科夫事件名（OB_up / OS_down / 3H / 3L / SPRING / BC / SOS / UT / ST / JOC / ASY / BOI / AS / SOT_up / SOT_down / OKR / ABS / COB 等）是程序基于价量形态规则自动识别出的标签，不是对市场真实状态的客观判定，程序识别有机械性，需灵活理解和运用。例如：
- `OB_up(超买)` 只表示"收盘接近上轨+放量"的形态组合，不等于真超买；在强趋势中它通常只是动能加速。
- `OS_down(超卖)` 只表示"收盘跌破下轨+放量"，不等于真超卖；下跌趋势里它常是中继信号。
- `SPRING(弹簧形态)` 只表示"微幅跌破支撑后快速收回"，不等于真主力洗盘，更常见的是假突破后的弱反弹。
- `BC(抢购高潮)` 只表示"极量大阳"，不等于需求耗尽；在突破初期它可能是真需求进入。
- `SOS(强势信号)` 只表示"震荡区右侧放量长阳"，不等于真实需求接管；缺乏后续 LPS 回测时常是诱多。
- `3H/3L` 只是连续3根高/低点形态，在趋势中出现非常普遍，本身不具备反转预警价值。

**分析原则**：把事件名当作"原始价量信号的速记符号"，回到它背后的原始形态（放量/缩量、上轨/下轨、收回/未收回、位置/阶段），结合波段结构、阶段概率、多级支撑阻力自行判断真实含义。不要把事件字面意义当结论——"出现 OB_up 就警惕顶部"、"出现 SPRING 就考虑反转"这类条件反射会放大假信号权重，必须避免。

### 多级支撑/阻力的使用方式
snapshot 可能给出一条或多条"支撑位"/"阻力位"(由近及远)。
- 离现价最近的一档用于即时决策(突破/测试判定)、填入 key_support / key_resistance。
- 更远的层级用于推理:若近档被吃穿,空间打开到下一档;作为 rationale 里"下档目标"或"延伸目标"的依据。
- 列表为空说明现阶段无明确水平,应在 rationale 中说明依靠概率与 P&F 目标判断。

### 关键价量深度解读
- **高位放量长阳**：警惕！可能是主力拉高出货（派发），而非真正突破
- **低位跳空高开放量收短K线**：警惕！可能是拉高为继续派发创造空间，不代表真正做多
- **低位放量止跌后缩量回踩**：吸筹信号，测试浮筹是否充分换手
- **趋势中段放量滞涨**：动能衰减；须区分"整固"（缩量后继续）vs"派发开始"（持续放量滞涨）

### 情境记忆的使用方式
当用户消息中出现【情境记忆】区段时：
- 这些笔记记录的是历史上若干具体案例中的失败复盘或经验片段，属于"个别案例观察"，长期验证后才能成为系统性规律。
- **判断优先级：本文档中的威科夫分析框架为主，profile 中的方向判断框架为辅，情境记忆笔记作为重要补充。**
- 笔记用来提醒你关注某个易忽视的细节或陷阱，主要还是参考框架分析结果。
- 正确用法：先按框架独立完成方向/置信度判断 → 再检查是否有匹配的笔记 → 若笔记提示的细节确实存在于当前 snapshot，明确其确实是框架考虑的漏洞或者不周全的地方并且论据充分，再用于修正。
- 类比：框架是你经过系统训练形成的"交易体系"，笔记是"过去踩过的几个坑的便签"。便签能提醒你,但不能取代体系。
- 已被 evolver 吸收进框架的经验会直接反映在默认分析框架里；笔记里残留的都是尚未被系统化、尚未证明稳定有效的观察。若多条笔记建议矛盾，以框架为准。
- 若参考了笔记，需在 rationale 中说明"参考了笔记 X，其与框架判断一致/冲突，经过审慎思考，最终以框架/笔记为主"。
"""


class PredictionParseError(Exception):
    pass


REQUIRED_FIELDS = {"direction", "confidence", "key_support", "key_resistance",
                   "time_horizon_days", "rationale", "key_signals", "risk_factors"}
VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}


class PredictorAgent(BaseLLMAgent):
    def __init__(
        self,
        strategy: dict,
        model: str = DEFAULT_MODEL,
        memory_manager: Optional[MemoryManager] = None,
        symbol_tags: Optional[list] = None,
    ):
        # Fixed base always prepended; evolved prompt follows
        system_prompt = _WYCKOFF_BASE_PROMPT + "\n\n" + strategy.get("predictor_system_prompt", "")
        self._symbol_tags: list = symbol_tags or []
        # A2: Replace full MEMORY.md injection with compact one-liner to save tokens.
        # The L2 matched notes are injected in the user message 【情境记忆】section.
        if memory_manager:
            entries = memory_manager._parse_index()
            n = len(entries)
            if n > 0:
                system_prompt += (
                    f"\n\n情境记忆系统已加载 {n} 条经验笔记。"
                    "匹配到的相关笔记会在用户消息的【情境记忆】区段中自动提供，请认真参考。"
                )
        super().__init__(
            system_prompt=system_prompt,
            max_tokens=MAX_TOKENS_PREDICTOR,
            model=model,
            temperature=0.3,
        )
        self.strategy = strategy
        self.memory_manager = memory_manager

    async def get_memory_notes(self, snapshot: dict) -> tuple[list[dict], list[str]]:
        """Match and load relevant memory notes for the given snapshot.
        Returns (notes_list, matched_filenames)."""
        if not self.memory_manager:
            return [], []
        matched = await self.memory_manager.match_by_similarity(
            snapshot, symbol_tags=self._symbol_tags, llm_fn=self.call
        )
        if not matched:
            return [], []
        top = matched[:5]
        return self.memory_manager.load_notes(top), top

    async def predict(self, user_message: str, snapshot: Optional[dict] = None) -> dict:
        """
        Call LLM with the formatted snapshot and return structured prediction.
        If snapshot is provided and memory_manager exists, matching memory notes
        are injected into the user message.
        result["triggered_memory"] contains list of filenames that fired (may be empty).
        Raises PredictionParseError if response cannot be parsed.
        """
        triggered_memory: list[str] = []
        if snapshot is not None:
            memory_notes, triggered_memory = await self.get_memory_notes(snapshot)
            if memory_notes:
                from finagent.prompts.templates import build_predictor_user_prompt
                # Re-build user message with memory notes injected
                user_message = build_predictor_user_prompt(
                    snapshot.get("text", ""),
                    self.strategy,
                    memory_notes=memory_notes,
                    symbol_tags=self._symbol_tags,
                )
        raw = await self.call(user_message)
        try:
            result = self.extract_json(raw)
        except ValueError as e:
            raise PredictionParseError(f"Cannot parse predictor response: {e}") from e

        result = self._validate_and_fill(result)
        result["raw_llm_response"] = raw
        result["triggered_memory"] = triggered_memory
        return result

    def _validate_and_fill(self, data: dict) -> dict:
        """Validate required fields and fill defaults for missing optional ones."""
        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            logger.debug(f"Prediction missing fields: {missing}, using defaults")

        direction = data.get("direction", "neutral")
        if direction not in VALID_DIRECTIONS:
            direction = "neutral"
        data["direction"] = direction

        conf = data.get("confidence", 0.5)
        try:
            conf = float(conf)
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.5
        data["confidence"] = conf

        data.setdefault("target_price", None)
        data.setdefault("key_support", None)
        data.setdefault("key_resistance", None)
        data.setdefault("time_horizon_days", 20)
        data.setdefault("rationale", "")
        data.setdefault("key_signals", [])
        data.setdefault("risk_factors", [])

        # Ensure lists
        if isinstance(data["key_signals"], str):
            data["key_signals"] = [data["key_signals"]]
        if isinstance(data["risk_factors"], str):
            data["risk_factors"] = [data["risk_factors"]]

        return data
