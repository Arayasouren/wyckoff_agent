# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Helper functions that build final prompts by merging strategy config with templates.
The actual system prompt text lives in the strategy JSON (so it can evolve).
These helpers inject dynamic context into the evolved prompts.
"""
from __future__ import annotations
from typing import Optional


def build_predictor_user_prompt(
    snapshot_text: str,
    strategy: dict,
    memory_notes: Optional[list[dict]] = None,
    symbol_tags: Optional[list[str]] = None,
) -> str:
    """Build the user-turn message for the Predictor agent."""
    notes = strategy.get("symbol_specific_notes", "")

    lines = []
    if symbol_tags or notes:
        tag_line = f"股票类型: {'/'.join(symbol_tags)}\n" if symbol_tags else ""
        lines.append(f"【品种特性备注】\n{tag_line}{notes}\n" if notes else f"【品种特性备注】\n{tag_line}")

    # Situation memory injection (L2 — loaded on demand)
    if memory_notes:
        lines.append(
            "【情境记忆】以下是过去与当前 snapshot 匹配的个别案例笔记（次要参考）。"
            "请先按分析框架独立得出主判断，再用这些笔记微调置信度或补充风险项，"
            "不得让笔记凌驾于框架之上："
        )
        for note in memory_notes[:5]:
            lines.append(f"\n◆ {note.get('situation', '未知情境')}")
            content = note.get("content", "")
            if len(content) > 400:
                content = content[:400] + "…"
            lines.append(content)
        lines.append("")

    lines.append(snapshot_text)
    lines.append(
        "\n请基于以上分析，给出未来20个交易日的走势预测。"
        "严格按以下字段顺序返回JSON（rationale先于direction，让推理驱动结论），"
        "不要任何markdown包裹，直接输出JSON：\n"
        "{\n"
        '  "rationale": "<综合分析推理，先完整写完分析再提炼方向，2-4句>",\n'
        '  "key_signals": ["<最关键信号1>", "<信号2>", "<信号3>"],\n'
        '  "risk_factors": ["<主要风险1>", "<风险2>"],\n'
        '  "direction": "bullish" | "bearish" | "neutral",\n'
        '  "confidence": <0.0~1.0，根据rationale得出>,\n'
        '  "target_price": <浮点数或null>,\n'
        '  "key_support": <浮点数>,\n'
        '  "key_resistance": <浮点数>,\n'
        '  "time_horizon_days": 20\n'
        "}"
    )
    return "\n".join(lines)


def build_critic_user_prompt(prediction: dict, actual_outcome: dict) -> str:
    """Build the user-turn message for the Critic agent."""
    pred_dir = prediction.get("direction", "")
    pred_conf = prediction.get("confidence", 0)
    pred_target = prediction.get("target_price")
    pred_support = prediction.get("key_support")
    pred_resistance = prediction.get("key_resistance")
    rationale = prediction.get("rationale", "")
    signals = prediction.get("key_signals", [])

    actual_dir = actual_outcome.get("actual_direction", "")
    actual_ret = actual_outcome.get("actual_return_pct", 0)
    actual_dd = actual_outcome.get("max_drawdown_pct", 0)
    actual_gain = actual_outcome.get("max_gain_pct", 0)
    entry = actual_outcome.get("entry_price", 0)
    exit_p = actual_outcome.get("exit_price", 0)

    lines = [
        "【预测内容】",
        f"  方向: {pred_dir}  置信度: {pred_conf:.2f}",
        f"  目标价: {pred_target}  支撑: {pred_support}  阻力: {pred_resistance}",
        f"  理由: {rationale}",
        f"  关键信号: {', '.join(signals)}",
        "",
        "【实际结果（20个交易日后）】",
        f"  入场价: {entry:.4f}  出场价: {exit_p:.4f}",
        f"  实际方向: {actual_dir}  实际涨跌幅: {actual_ret:+.2f}%",
        f"  最大回撤: {actual_dd:.2f}%  最大涨幅: {actual_gain:.2f}%",
        "",
        "请评估预测质量并返回JSON评分。",
    ]
    return "\n".join(lines)


def build_reflector_user_prompt(
    worst_records: list[dict],
    memory_index: str = "",
    recent_records: Optional[list[dict]] = None,
    direction_stats: Optional[dict] = None,
    symbol: Optional[str] = None,
    symbol_tags: Optional[list[str]] = None,
    best_records: Optional[list[dict]] = None,
    predictor_system_prompt: str = "",
    memory_triggers: str = "",
) -> str:
    """Build the user-turn message for the Reflector agent."""
    import json as _json

    lines = []

    if symbol or symbol_tags:
        head = f"【当前诊断股票】{symbol or ''}"
        if symbol_tags:
            head += f"  标签: {'/'.join(symbol_tags)}"
        lines.append(head)
        lines.append("")

    # Aggregated direction stats — injected first so LLM calibrates diagnosis to reality
    if direction_stats and direction_stats.get("total", 0) > 0:
        ds = direction_stats
        lines.append("【当前股票实际行为统计】")
        lines.append(
            f"总窗口: {ds['total']}  方向准确率: {ds['overall_win_pct']:.1%}  "
            f"neutral率: {ds['neutral_pct']:.1%}"
        )
        lines.append(
            f"多头: {ds['bullish_cnt']}次 胜率 {ds['bullish_win_pct']:.1%}  "
            f"空头: {ds['bearish_cnt']}次 胜率 {ds['bearish_win_pct']:.1%}  "
            f"中性: {ds['neutral_cnt']}次"
        )
        lines.append("")
        lines.append("请首先对照以上统计确认当前实际失败模式。")
        if ds["neutral_pct"] < 0.05:
            lines.append('注意：neutral率 < 5%，请不要将"neutral过多"列为主要失败原因。')
        bullish_weak = ds["bullish_win_pct"] < 0.35 and ds["bullish_cnt"] >= 3
        bearish_weak = ds["bearish_win_pct"] < 0.35 and ds["bearish_cnt"] >= 3
        if bullish_weak:
            lines.append(f"注意：多头胜率仅 {ds['bullish_win_pct']:.1%}，请重点分析多头预测为何系统性错误。")
        if bearish_weak:
            lines.append(f"注意：空头胜率仅 {ds['bearish_win_pct']:.1%}，请重点分析空头预测为何系统性错误。")
        lines.append("")

    lines.append(f"以下是评分最低的 {len(worst_records)} 条预测记录，请分析失败根因：\n")
    for i, rec in enumerate(worst_records, 1):
        lines.append(f"--- 记录 {i} (评分: {rec.get('score', 0):.2f}) ---")
        lines.append(f"  日期: {rec.get('window_end_date', '')}  品种: {rec.get('symbol', '')}")
        lines.append(f"  预测方向: {rec.get('direction', '')}  置信度: {rec.get('confidence', 0):.2f}")
        lines.append(f"  实际方向: {rec.get('actual_direction', '')}  实际涨跌: {rec.get('actual_return_pct', 0):+.2f}%")
        lines.append(f"  预测理由: {rec.get('rationale', '')[:200]}")
        lines.append(f"  失败点: {rec.get('what_failed', '')[:200]}")
        hints = rec.get("improvement_hints", "[]")
        if isinstance(hints, str):
            try:
                hints = _json.loads(hints)
            except Exception:
                hints = []
        if hints:
            lines.append(f"  改进建议: {'; '.join(hints[:3])}")
        lines.append("")

    if best_records:
        top = best_records[:10]
        lines.append(f"【成功预测记录（评分最高 {len(top)} 条）】\n")
        for i, rec in enumerate(top, 1):
            lines.append(f"--- 成功记录 {i} (评分: {rec.get('score', 0):.2f}) ---")
            lines.append(f"  日期: {rec.get('window_end_date', '')}  品种: {rec.get('symbol', '')}")
            lines.append(f"  预测方向: {rec.get('direction', '')}  置信度: {rec.get('confidence', 0):.2f}")
            lines.append(f"  实际方向: {rec.get('actual_direction', '')}  实际涨跌: {rec.get('actual_return_pct', 0):+.2f}%")
            lines.append(f"  预测理由: {rec.get('rationale', '')[:200]}")
            lines.append(f"  成功点: {rec.get('what_worked', '')[:200]}")
            lines.append("")

    if recent_records:
        lines.append(f"【近期预测记录（最新 {len(recent_records)} 条，按时间倒序）】")
        lines.append("  日期  品种  预测→实际  涨跌  评分  方向对否")
        for r in recent_records:
            correct = "✓" if r.get("direction_correct") else "✗"
            lines.append(
                f"  {r.get('window_end_date','')[:10]}  {r.get('symbol','')}  "
                f"{r.get('direction','')[:4]}→{r.get('actual_direction','')[:4]}  "
                f"{r.get('actual_return_pct', 0):+.1f}%  "
                f"{r.get('score', 0):.2f}  {correct}"
            )
        lines.append("")

    if memory_index and memory_index.strip() != "# 情境记忆索引":
        lines.append("【当前情境记忆索引】")
        lines.append(memory_index)
        lines.append("")

    if memory_triggers:
        lines.append("【现有情境记忆检索签名列表】（撰写新 retrieval_text 时参考，避免重复造轮子）")
        lines.append(memory_triggers)
        lines.append("")

    if predictor_system_prompt:
        lines.append("【Predictor System Prompt（当前版本）】")
        lines.append(predictor_system_prompt)
        lines.append("")

    lines.append("请输出诊断报告，分条列举系统性失败模式及改进方向。")
    return "\n".join(lines)


def build_evolver_user_prompt(
    current_strategy: dict,
    diagnosis: str,
    stats: dict,
    best_records: list[dict],
    worst_records: list[dict],
    symbol: Optional[str] = None,
    symbol_tags: Optional[list[str]] = None,
) -> str:
    """Build the user-turn message for the Evolver agent."""
    import json

    # Sanitize strategy for prompt: drop large fields AND the full per-symbol tags map
    # (we inject the current symbol's tags explicitly below, to avoid dumping 30+ unrelated entries).
    safe_strategy = {k: v for k, v in current_strategy.items()
                     if k not in ("symbol_tags", "per_symbol_overrides",
                                  "performance_history", "win_rate_log")}

    lines = []
    if symbol or symbol_tags:
        head = f"【本次进化目标股票】{symbol or ''}"
        if symbol_tags:
            head += f"  标签: {'/'.join(symbol_tags)}"
        lines.append(head)
        lines.append("")

    lines += [
        "【当前策略】",
        json.dumps(safe_strategy, ensure_ascii=False, indent=2),
        "",
        "【Reflector 诊断报告】",
        diagnosis,
        "",
        "【绩效统计】",
        f"  总窗口数: {stats.get('total_windows', 0)}",
        f"  平均评分: {stats.get('avg_score', 0):.3f}",
        f"  方向准确率: {stats.get('direction_accuracy', 0):.1%}",
        f"  平均涨跌幅: {stats.get('avg_return_pct', 0):+.2f}%",
        "  注意：holdout验证评分以方向准确率为主（权重80%）。候选策略必须优先提升方向判断正确率，"
        "目标价、关键位等为次要优化目标。若某候选策略不能显著改善方向准确率，不会被采纳。",
        "",
        f"【最优预测样例（前{len(best_records)}条）】",
    ]
    for r in best_records[:5]:
        lines.append(
            f"  {r.get('window_end_date','')} {r.get('direction','')} "
            f"conf={r.get('confidence',0):.2f} score={r.get('score',0):.2f} "
            f"- {r.get('what_worked','')[:100]}"
        )
    lines.append("")
    lines.append(f"【最差预测样例（前{len(worst_records)}条）】")
    for r in worst_records[:5]:
        signals = r.get("key_signals", "[]")
        if isinstance(signals, str):
            try:
                import json as _j
                signals = _j.loads(signals)
            except Exception:
                signals = []
        lines.append(
            f"  {r.get('window_end_date','')} {r.get('direction','')} "
            f"conf={r.get('confidence',0):.2f} score={r.get('score',0):.2f} "
            f"实际方向={r.get('actual_direction','')} 涨跌={r.get('actual_return_pct',0):+.1f}%"
        )
        lines.append(f"    预测理由: {r.get('rationale','')[:200]}")
        lines.append(f"    关键信号: {', '.join(signals) if isinstance(signals,list) else str(signals)}")
        lines.append(f"    失败点: {r.get('what_failed','')[:150]}")
        lines.append("")
    lines += [
        "【进化授权与要求】",
        "你的核心任务是改进 predictor_system_prompt，使预测模型在保持原有正确判断不受影响的前提下，在当前失败模式上表现更好。",
        "可（次要）更新 symbol_specific_notes（针对本股的特化提示，不超过150字）。",
        "**长度约束**：输出的 predictor_system_prompt 控制在 10000 字以内，可在此范围内充分表达；"
        "请精简冗余表述，但严禁为压缩篇幅而删除既有的完整段落或结尾内容。务必输出完整的 prompt 全文"
        "（从开头到结尾的「## 输出格式」段落都要完整保留），不得中途截断或省略收尾。",
        "",
        "【关于 predictor_system_prompt 的修改权限】",
        "你被授权修改 predictor_system_prompt，但必须遵循以下纪律：",
        "  - 硬编码的威科夫基础框架（在代码 _WYCKOFF_BASE_PROMPT 中）位于 profile 之外，无论如何修改都不会被触及，"
        "可视为永久不可变的「基本原则」——不要尝试重写它。",
        "  - profile 中 predictor_system_prompt 开头的 profile 级「基本原则」段落同样应保持结构完整，不要整段删除或颠覆。",
        "  - 「## 输出格式」段落及其中的 JSON schema 结构严禁修改：所有字段名（direction/confidence/target_price/"
        "key_support/key_resistance/time_horizon_days/rationale/key_signals/risk_factors）必须原样保留，"
        "不得删除、重命名或改变类型。该段落是系统解析的硬约束，修改会导致输出无法被程序读取。",
        "  - 修改必须经过深思熟虑：优先用软性指导替换硬性强制；规则体系应给模型留有根据证据权衡的空间；"
        "不允许用一个绝对规则替换另一个绝对规则。",
        "  - 可在核心约束规则中加入基于 stock_tags 的分支判断（例如「若 stock_tags 包含 XX，则在 Spring 形态上降低置信度…」），"
        "以便针对不同股票类型给出差异化处理；但新分支也必须是软性指引，不能变成新的硬性强制。",
        "  - 每一处修改都应在 evolution_notes 中给出证据链（来自 worst/best 样例或 diagnosis 的哪些观察支持这条修改）。"
        "若没有充分证据，请保持 predictor_system_prompt 不变。",
        "",
        "三个候选的差异聚焦点：",
        "  candidate_a：最保守，仅修复最主要的系统性失败模式，改动最小",
        "  candidate_b：中等力度，针对最常见方向错误做针对性规则调整",
        "  candidate_c：最激进，着重纠正系统性偏差，允许较大幅度重写相关段落",
        "",
        "【记忆吸收（严格）】",
        "情境记忆笔记是「次要参考」，只应保留那些尚未被系统化成规则的个别观察。若你在本次进化中写入 predictor_system_prompt 的新规则"
        "已经**完整、体系化**地覆盖了某条现有笔记的职能，则该笔记应被废弃；请在对应候选的输出中声明：",
        '  "absorbed_memory_notes": ["filename1.md", "filename2.md"]',
        "严格吸收标准（必须全部满足，否则不得声明吸收）：",
        "  1. 笔记的 situation / retrieval_text 所描述的情境已被新规则中明确、可识别的条件覆盖（不是隐含、不是「类似情境」）；",
        "  2. 笔记的 insight 与 suggested_adjustments 的要点已以更通用、体系化的措辞写入 predictor_system_prompt，"
        "     且 LLM 仅凭新规则即可在该情境下做出正确判断，不再需要该笔记作为提示；",
        "  3. 新规则是用自己的体系化语言撰写的，而不是把笔记原文直接复制进去；",
        "  4. 部分覆盖、疑似覆盖、方向相反、措辞含糊 → 一律不得吸收；",
        "  5. 吸收是**单向永久**的——一旦候选被采纳，被吸收的笔记将从记忆库中删除，无法回退。",
        "若没有符合上述标准的笔记，absorbed_memory_notes 应为空列表 []。宁缺毋滥。",
        "",
        "【输出格式要求（严格执行）】",
        "返回合法 JSON，结构如下：",
        '  {"candidate_a": {...}, "candidate_b": {...}, "candidate_c": {...}, "evolution_notes": "..."}',
        "每个候选对象中，无论是否修改，都必须包含以下字段：",
        "  - predictor_system_prompt：完整的 prompt 全文（不得省略、截断或以占位符代替）。",
        "    即使该候选策略对 predictor_system_prompt 无任何改动，也必须原文照抄当前版本。",
        "    LLM 省略此字段将导致本轮进化无效，所有修改丢失。",
        "  - evolution_notes：本候选的修改依据与证据链（若无修改则说明原因）。",
        "  - symbol_specific_notes：若有针对本股的特化更新则填写，否则原文照抄。",
    ]
    return "\n".join(lines)
