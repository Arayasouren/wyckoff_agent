---
situation: 强趋势中OB_up连续出现≥5次仍看空
confidence: 0.5
created_at: 2026-04-09
evolved_at: 2026-04-12
source_windows: 85
refined_count: 85
stocks_validated: ['002580.SZ', '300131.SZ', '002245.SZ', '000333.SZ', '600435.SH', '601100.SH', '605499.SH', '000792.SZ', '600176.SH', '000596.SZ', '601012.SH']
stocks_failed: ['600537.SH', '600337.SH', '600547.SH', '300015.SZ', '002050.SZ', '688111.SH', '600900.SH']
cross_stock_score: 0.611
sector_failures: {'小盘': 25, '成长': 33, '制造': 24, '中盘': 12, '科技': 9, '价值': 4, '消费': 4, '大盘': 4, '红利': 4, '基础设施': 4}
sector_excluded: ['小盘', '成长', '制造', '中盘', '科技', '价值', '消费', '大盘', '红利', '基础设施']
---

## 经验总结
强上升趋势中OB_up连续出现或3H/分布预警但未确认见顶时，通常为动能延续，需右侧结构破位才构成顶部依据。

## 建议调整
上升趋势+尚未确认见顶+OB_up/3H，禁止看空；若需反转，需SOW/LPSY右侧确认。

## 例外分支
当连续OB_up（≥3次）与高置信度3H（≥0.95）共振，且多空概率差极小（<10%）时，属于强烈的左侧派发预警。此时无需等待右侧确认即可偏向bearish防守，严禁盲目看多。**注意：若价格远离阻力位（当前价 < 阻力位 * 0.90），此共振属于上涨中继的强势蓄势，例外分支失效，仍严禁看空；若价格处于中间区域（阻力位 * 0.90 <= 当前价 < 阻力位 * 0.97），此共振构成派发预警，极易演变为宽幅震荡或动能衰竭，严禁强行看多；同时因缺乏右侧破位驱动，严禁直接看空，应强制输出neutral防守。**

## 补充条件
59-65（保留原内容）
66. 当阶段标签包含“警示/分布”且满足3H+OB_up共振、中间区域（0.90-0.97）及波浪回落>5%时，局部派发已实质发生。此时出现的ASY_up/JOC/VDB等突破信号极大概率是UTAD或多头陷阱，补充条件61在此条件下失效，严禁视为动能延续，必须直接转为bearish防守。

## Trigger更新建议
`event contains "OB_up" AND prob_up > 0.38 AND latest_OB_up_age <= 15 AND event contains "3H" AND 3H_confidence >= 0.95 AND current_price >= resistance * 0.90 AND pnf_target <= current_price * 1.10 AND recent_wave_drop <= 0.05 AND not (stage contains "警示" AND recent_wave_drop > 0.05)`
