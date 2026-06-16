# FinAgent — 自进化威科夫技术分析 Agent

FinAgent 是一个**自我进化**的威科夫（Wyckoff）技术分析智能体，面向 A 股个股与指数。
它用大语言模型（LLM）对市场快照做方向预测，再以历史滚动回测的"对错"反馈，
持续地**自动改写自己的策略提示词与情境记忆库**——形成
`预测 → 批判 → 反思 → 进化（Predict → Critique → Reflect → Evolve）`的闭环。

> 内置一份已训练的策略档案 `mywyckoff`，开箱即可预测；也可从默认模板新建档案自行训练。

---

## 功能概览

- **威科夫分析内核**（`wyckoff_analyzer/`，已内置）：波段划分、阶段识别、事件检测、
  点数图（P&F）目标位、8 阶段概率打分。
- **四智能体进化闭环**：Predictor / Critic / Reflector / Evolver。
- **情境记忆库**：LLM 优先选择 + bge-m3 向量兜底的记忆检索，带跨股验证与自动精炼/合并。
- **序列匹配概率**：将当前事件序列与 72.7 万行历史数据库比对，给出实证涨跌频率
  （首次运行自动从内置 CSV 构建）。
- **多档案管理**：版本化、候选/部署、滚动回测。
- **多数据源**：akshare（A 股，免费）/ yfinance（全球，免费）/ Wind WDS（需自有授权）。

---

## 环境要求

- Python **3.9+**
- 一个 LLM API Key（Anthropic，或任意 OpenAI 兼容端点，如 DeepSeek / SiliconFlow / 本地 vLLM）
- 联网以拉取行情数据

---

## 安装

```bash
# 1) 克隆
git clone <your-repo-url> finagent && cd finagent

# 2) 建虚拟环境
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) 安装依赖（二选一）
pip install -e ".[all]"          # 推荐：含 akshare/yfinance/openai/matplotlib 全部可选项
# 或仅核心 + 按需：
pip install -r requirements.txt
```

可选依赖分组（`pyproject.toml` 的 extras）：

| extra | 作用 |
|---|---|
| `akshare` | A 股免费行情（推荐） |
| `yfinance` | 全球行情兜底 |
| `openai` | OpenAI 兼容 LLM 端点 + 向量嵌入 |
| `wind` | Wind WDS（Oracle，需 `oracledb` 与自有账号） |
| `chart` | 画图（`matplotlib`，仅在开启画图时需要） |
| `all` | 除 Wind 外的全部 |

---

## 配置

复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
```

最少只需一个 LLM Key：

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
FINAGENT_MODEL=claude-sonnet-4-6
DATA_SOURCE=akshare
```

使用 OpenAI 兼容端点（如 DeepSeek）：

```dotenv
LLM_PROVIDER=openai
OPENAI_COMPAT_API_KEY=sk-...
OPENAI_COMPAT_BASE_URL=https://api.deepseek.com/v1
FINAGENT_MODEL=deepseek-chat
```

其余可选项（嵌入、fallback 端点、Wind、画图）见 `.env.example` 注释。

---

## 数据与数据库

- **首次运行会自动**把内置的 `wyckoffstats/*.csv`（约 27MB）构建进 `data/finagent.db`
  的 `seqstats` 表（72.7 万行，一次性，约数秒），用于【序列匹配概率】。无需手动操作。
  - 如需手动重建：`python3 scripts/ingest_seqstats.py`
- **预测/批判历史从空表起步**（首次运行自动建表）。`mywyckoff` 已学到的策略
  完整保存在 `data/profiles/mywyckoff.json` 与 `data/profiles/mywyckoff_memory/` 中，
  不依赖历史表；`status` 里的胜率日志也来自档案本身。
- `data/finagent.db` 是**本地生成产物**，已在 `.gitignore` 中，不纳入版本库。

### 关于 Wind WDS（可选）

Wind 是商业授权数据服务，本分发版**已清空所有 Wind 凭据**，仅保留接入代码作为接口。
若你有自己的 Wind 账号，在 `.env` 中填好 `WIND_*` 并设 `DATA_SOURCE=wind`、
`pip install oracledb` 即可启用。否则默认走 akshare / yfinance，无需任何凭据。

---

## 使用

入口统一为 `python -m finagent <子命令>`（或安装后直接 `finagent <子命令>`）。

```bash
# 预测：用当前档案对某标的给出未来约 20 日方向
python -m finagent predict 600519.SH
python -m finagent predict 000300.SH --date 2026-03-31 --json

# 查看档案状态与历史统计
python -m finagent status
python -m finagent status 600519.SH

# 档案管理
python -m finagent list-profiles
python -m finagent new-profile my_strategy     # 从 default 模板新建并设为当前档
python -m finagent use-profile mywyckoff        # 切换当前档

# 训练（滚动历史回测 + 进化当前档）
python -m finagent evolve 600519.SH
python -m finagent evolve 600519.SH --no-auto-apply   # 仅生成候选，待手动部署
python -m finagent apply                              # 部署候选版

# 批量训练
python -m finagent batch-evolve 600519.SH 000001.SZ 600036.SH

# 记忆库维护
python -m finagent rebuild-embeddings     # 重建向量索引（需嵌入端点）
python -m finagent compress-memory        # 强制合并相似记忆
```

常用开关：`--profile <name>` 指定档案、`--model <name>` 指定模型、
`-v` 详细日志、`--use-fallback` 改走备用 LLM 端点。

> **画图默认关闭**。如需在 `evolve` 时输出 K 线进化图，设 `FINAGENT_ENABLE_CHART=1`
> 并安装 `matplotlib`，图保存在 `data/figure/`。

---

## 目录结构

```
finagent/            # 主程序包（CLI、四智能体、引擎、存储、数据、记忆、序列统计）
wyckoff_analyzer/    # 内置威科夫计算内核
data/
  profiles/
    mywyckoff.json         # 已训练的个股策略档案（当前档）
    mywyckoff_memory/      # 其情境记忆库（笔记 + 索引 + 向量）
    default.json           # 新建档案的模板
  figure/                  # 进化图输出（默认不产出）
wyckoffstats/        # seqstats 源 CSV（首次运行据此构建参考表）
stockinfo/tags.csv   # 标的的规模/风格/行业标签（供序列匹配分桶）
scripts/             # 辅助脚本（如手动 ingest seqstats）
```

---

## 许可证

**Copyright (C) 2026 Araya**

本项目采用 **GNU Affero 通用公共许可证 v3（AGPL-3.0-or-later）**。完整条款见 [LICENSE](LICENSE)。

简言之：你可以自由使用、修改、分发本项目，但——
- 任何修改/衍生作品在分发时**必须同样以 AGPL-3.0 开源**；
- 即使你只是把它**架成在线服务**对外提供（不分发代码），也**必须向用户公开完整源码**；
- 必须保留版权声明，且本软件**不提供任何担保**。

```
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. This program is distributed WITHOUT ANY WARRANTY. See the GNU
Affero General Public License <https://www.gnu.org/licenses/> for more details.
```

## 免责声明

本项目仅用于技术研究与学习，**不构成任何投资建议**。据此交易，盈亏自负。
