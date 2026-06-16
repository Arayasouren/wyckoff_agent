# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROFILES_DIR = DATA_DIR / "profiles"
DB_PATH = DATA_DIR / "finagent.db"
FIGURE_DIR = DATA_DIR / "figure"
PROFILE_HISTORY_DIR = PROFILES_DIR / "history"

# Chart generation is OFF by default (avoids a hard matplotlib dependency and
# keeps `evolve` headless-friendly). Enable with FINAGENT_ENABLE_CHART=1.
ENABLE_CHART = os.getenv("FINAGENT_ENABLE_CHART", "0").strip() == "1"

# Wyckoff computation kernel. The `wyckoff_analyzer` package is bundled at the
# repo root (next to `finagent/`), so the default points there. Override via env
# only if you keep the kernel elsewhere.
WYCKOFF_PATH = Path(os.getenv("WYCKOFF_PATH", ROOT_DIR))

# Market data source: "akshare" | "yfinance" | "wind"
#   akshare  — free A-share data (recommended default for CN markets)
#   yfinance — free global fallback
#   wind     — Wind WDS (Oracle); requires your own licensed credentials (see below)
DATA_SOURCE = os.getenv("DATA_SOURCE", "akshare")

# Wind WDS (Oracle) connection — only used when DATA_SOURCE=wind.
# Credentials are intentionally BLANK in this distribution: Wind is a licensed
# commercial data service. Supply your own endpoint/account via .env to enable it.
WIND_HOST     = os.getenv("WIND_HOST", "")
WIND_PORT     = int(os.getenv("WIND_PORT", "1521"))
WIND_SERVICE  = os.getenv("WIND_SERVICE", "")
WIND_USER     = os.getenv("WIND_USER", "")
WIND_PASSWORD = os.getenv("WIND_PASSWORD", "")
WIND_TABLE         = os.getenv("WIND_TABLE", "winddb.ashareeodprices")
ORACLE_CLIENT_DIR  = os.getenv("ORACLE_CLIENT_DIR", "")   # path to Instant Client dylibs

# LLM config
DEFAULT_MODEL = os.getenv("FINAGENT_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")  # e.g. https://aia.linglong521.cn
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai"
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "")

# Optional fallback (OpenAI-compat): used only after primary exhausts all retries.
OPENAI_COMPAT_FALLBACK_API_KEY = os.getenv("OPENAI_COMPAT_FALLBACK_API_KEY", "")
OPENAI_COMPAT_FALLBACK_BASE_URL = os.getenv("OPENAI_COMPAT_FALLBACK_BASE_URL", "")
FALLBACK_MODEL = os.getenv("FINAGENT_FALLBACK_MODEL", "")

# Embedding provider (OpenAI-compat /embeddings endpoint) — used for memory semantic retrieval.
# Defaults fall through to the OPENAI_COMPAT_* config so a single SiliconFlow/etc. account works.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENAI_COMPAT_BASE_URL)
EMBEDDING_API_KEY  = os.getenv("EMBEDDING_API_KEY", OPENAI_COMPAT_API_KEY)
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-m3 = 1024

# Agent token limits
MAX_TOKENS_PREDICTOR = 2048
MAX_TOKENS_CRITIC = 1024
MAX_TOKENS_REFLECTOR = 2048
MAX_TOKENS_EVOLVER = 32768

# Retry config
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

# Rolling window defaults
CONTEXT_DAYS = 500       # bars of history fed to wyckoff engine
PREDICTION_HORIZON = 20  # trading days to predict
STEP_SIZE = 20           # roll forward N days per window (monthly)
HOLDOUT_RATIO = 0.20     # last 20% of windows used for candidate evaluation

# Index-mode (data_source_type="index") monthly batch
INDEX_BATCH_MONTHS   = 60    # evolve every 60 monthly windows
INDEX_TRAIN_MONTHS   = 40    # first 40 months of batch = training
INDEX_HOLDOUT_MONTHS = 20    # last 20 months of batch = holdout

# Concurrency
TRAIN_CONCURRENCY = 5    # max simultaneous windows in Phase 1 training loop

# Evolution
NUM_CANDIDATES = 3       # Pareto candidates per evolution run
WORST_N_FOR_REFLECTOR = 10
WORST_SCORE_CAP_FOR_REFLECTOR = 0.5  # only predictions scoring below this are shown to reflector
BEST_N_FOR_EVOLVER = 25

# Critic scoring weights — direction is primary (0.70), others share remaining 0.30
SCORE_DIRECTION = 0.70
SCORE_TARGET_HIT = 0.12
SCORE_SUPPORT = 0.05
SCORE_RESISTANCE = 0.05
SCORE_CONFIDENCE_CALIB = 0.04
SCORE_INSIGHT = 0.04
TARGET_HIT_TOLERANCE = 0.02  # ±2%
