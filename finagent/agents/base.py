"""
Base LLM agent supporting Anthropic SDK and OpenAI-compatible providers.
Provider is selected via LLM_PROVIDER env var ("anthropic" | "openai").
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import time
import logging
from datetime import datetime
from typing import Optional

import anthropic

from finagent.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, DEFAULT_MODEL,
    MAX_RETRIES, RETRY_BASE_DELAY,
    LLM_PROVIDER, OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_FALLBACK_API_KEY, OPENAI_COMPAT_FALLBACK_BASE_URL,
    FALLBACK_MODEL,
)

logger = logging.getLogger(__name__)

# Module-level API error counter — only counts calls that exhausted all retries.
# Reset each run via get_api_error_summary().
_api_error_counts: dict[str, int] = {}


# ──────────────────────────────────────────────────────────────────────────────
# LLM I/O markdown logger (opt-in via enable_llm_log or FINAGENT_LLM_LOG env var).
# Writes serialised under an asyncio.Lock so 20-way concurrent calls don't interleave.
# Set FINAGENT_LLM_LOG_FIRST_ONLY=1 to record only the first call per agent type.
# ──────────────────────────────────────────────────────────────────────────────
_llm_log_path: Optional[str] = None
_llm_log_lock: Optional[asyncio.Lock] = None
_llm_log_counter = 0
_llm_log_first_only: bool = False
_llm_log_seen_agents: set = set()


def enable_llm_log(path: str, first_only: bool = False) -> None:
    """Enable per-call LLM I/O logging to the given markdown file (truncates existing).
    If first_only=True, only the first call per agent class name is recorded."""
    global _llm_log_path, _llm_log_lock, _llm_log_counter, _llm_log_first_only, _llm_log_seen_agents
    _llm_log_path = path
    _llm_log_lock = None  # created lazily inside running loop
    _llm_log_counter = 0
    _llm_log_first_only = first_only
    _llm_log_seen_agents = set()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# LLM I/O Log\n\nStarted: {datetime.now().isoformat()}\n\n")


def _env_log_path() -> Optional[str]:
    return _llm_log_path or os.environ.get("FINAGENT_LLM_LOG") or None


def _env_first_only() -> bool:
    return _llm_log_first_only or os.environ.get("FINAGENT_LLM_LOG_FIRST_ONLY", "") == "1"


async def _log_llm_call(
    agent_name: str,
    model: str,
    system_prompt: str,
    user_message: str,
    response: Optional[str],
    error: Optional[str],
    duration_ms: int,
) -> None:
    global _llm_log_lock, _llm_log_counter, _llm_log_seen_agents
    path = _env_log_path()
    if not path:
        return
    if _llm_log_lock is None:
        _llm_log_lock = asyncio.Lock()
    async with _llm_log_lock:
        # When first_only mode: skip if this agent type was already logged.
        if _env_first_only():
            base_name = agent_name.replace("[fallback]", "")
            if base_name in _llm_log_seen_agents:
                return
            _llm_log_seen_agents.add(base_name)
        _llm_log_counter += 1
        n = _llm_log_counter
        ts = datetime.now().isoformat(timespec="milliseconds")
        parts = [
            f"\n---\n\n## [{n}] {agent_name}  ·  {ts}  ·  {duration_ms}ms  ·  model={model}\n",
            f"\n### System Prompt\n\n```\n{system_prompt}\n```\n",
            f"\n### User Message\n\n```\n{user_message}\n```\n",
        ]
        if error is not None:
            parts.append(f"\n### ERROR\n\n```\n{error}\n```\n")
        else:
            parts.append(f"\n### Response\n\n```\n{response}\n```\n")
        block = "".join(parts)
        # Single open+write+close; buffered I/O flushes on close, so concurrent
        # calls can't tear blocks (lock already prevents interleaving anyway).
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)


def _record_api_error(exc: BaseException) -> None:
    msg = str(exc)
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        key = "Request timed out"
    elif "rate limit" in msg.lower():
        key = "Rate limit"
    elif "connection" in msg.lower():
        key = "Connection error"
    elif "internal server" in msg.lower() or "500" in msg:
        key = "Internal server error"
    else:
        key = type(exc).__name__
    _api_error_counts[key] = _api_error_counts.get(key, 0) + 1


def get_api_error_summary() -> dict[str, int]:
    """Return and reset the accumulated API error counts (final failures only)."""
    summary = dict(_api_error_counts)
    _api_error_counts.clear()
    return summary


RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class LLMCallError(Exception):
    pass


def _repair_truncated_json(text: str) -> str:
    """
    Attempt to repair a truncated JSON string by:
    1. Finding the opening {
    2. Scanning through tracking in_string / escape state / brace depth
    3. At end of text: close any open string, then close open braces/brackets
    Returns repaired string, or empty string if no { found.
    """
    start = text.find('{')
    if start == -1:
        return ""

    chunk = text[start:]
    in_string = False
    escape_next = False
    depth_stack = []  # '{' or '['

    for ch in chunk:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            depth_stack.append(ch)
        elif ch == '}':
            if depth_stack and depth_stack[-1] == '{':
                depth_stack.pop()
        elif ch == ']':
            if depth_stack and depth_stack[-1] == '[':
                depth_stack.pop()

    # Build closing suffix
    suffix = ""
    if in_string:
        suffix += '"'  # close the open string
    # Drop any trailing comma before closing (common in truncated JSON)
    repaired = (chunk + suffix).rstrip()
    if repaired.endswith(','):
        repaired = repaired[:-1]
    # Close open structures in reverse order
    for opener in reversed(depth_stack):
        repaired += '}' if opener == '{' else ']'

    return repaired


def _sanitize_json_strings(text: str) -> str:
    """
    Escape unescaped double quotes inside JSON string values.

    LLMs sometimes embed quoted phrases without escaping, e.g.:
        "rationale": "阶段提示"分布信号出现"，综合判断..."
    This function detects such inner quotes by checking whether the
    character following the quote (skipping spaces/tabs) is a JSON
    structural character (`,` `}` `]` `\n` `\r` `:`).  If it is not,
    the quote is an unescaped inner quote and gets replaced with `\"`.

    Also escapes bare newlines / carriage returns inside string values,
    which likewise break JSON parsing.
    """
    result = []
    i = 0
    n = len(text)
    in_string = False
    escape_next = False

    while i < n:
        ch = text[i]

        if escape_next:
            escape_next = False
            result.append(ch)
            i += 1
            continue

        if ch == '\\' and in_string:
            escape_next = True
            result.append(ch)
            i += 1
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
                i += 1
                continue
            # Inside a string: decide whether this is a closing quote or an
            # unescaped inner quote by peeking at the next non-space char.
            j = i + 1
            while j < n and text[j] in (' ', '\t'):
                j += 1
            next_ch = text[j] if j < n else None
            if next_ch is None or next_ch in (',', '}', ']', '\n', '\r', ':'):
                # Legitimate closing quote
                in_string = False
                result.append(ch)
            else:
                # Unescaped inner quote — escape it
                result.append('\\')
                result.append('"')
            i += 1
            continue

        if in_string and ch == '\n':
            result.append('\\')
            result.append('n')
            i += 1
            continue

        if in_string and ch == '\r':
            result.append('\\')
            result.append('r')
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


class BaseLLMAgent:
    def __init__(
        self,
        system_prompt: str,
        max_tokens: int,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_retries: int = MAX_RETRIES,
    ):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._provider = LLM_PROVIDER

        self._fallback_client = None
        self._fallback_model = ""

        # httpx timeouts: connect 10s, read 240s, total max 300s per call.
        # Prevents indefinite hangs when LLM API silently drops connections.
        import httpx
        _http_limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
            keepalive_expiry=30.0,
        )
        _http_timeout = httpx.Timeout(300.0, connect=10.0, read=240.0)

        if LLM_PROVIDER == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=OPENAI_COMPAT_API_KEY,
                base_url=OPENAI_COMPAT_BASE_URL,
                timeout=_http_timeout,
                max_retries=0,  # our own retry loop handles retries
                http_client=httpx.AsyncClient(limits=_http_limits, timeout=_http_timeout),
            )
        else:
            client_kwargs = {"api_key": ANTHROPIC_API_KEY}
            if ANTHROPIC_BASE_URL:
                client_kwargs["base_url"] = ANTHROPIC_BASE_URL
            client_kwargs["timeout"] = _http_timeout
            client_kwargs["max_retries"] = 0
            client_kwargs["http_client"] = httpx.AsyncClient(limits=_http_limits, timeout=_http_timeout)
            self._client = anthropic.AsyncAnthropic(**client_kwargs)

        # Fallback (always OpenAI-compat) — independent of primary provider.
        if OPENAI_COMPAT_FALLBACK_API_KEY and OPENAI_COMPAT_FALLBACK_BASE_URL:
            from openai import AsyncOpenAI
            self._fallback_client = AsyncOpenAI(
                api_key=OPENAI_COMPAT_FALLBACK_API_KEY,
                base_url=OPENAI_COMPAT_FALLBACK_BASE_URL,
                timeout=_http_timeout,
                max_retries=0,
                http_client=httpx.AsyncClient(limits=_http_limits, timeout=_http_timeout),
            )
            self._fallback_model = FALLBACK_MODEL or model

    # Above this max_tokens, the Anthropic SDK refuses a non-streaming call
    # ("Streaming is required for operations that may take longer than 10 minutes").
    # Large evolver/merge calls (MAX_TOKENS_EVOLVER) must stream to avoid that guard
    # and to avoid truncating multi-candidate JSON. Threshold left below the ~21k SDK
    # cutoff with margin.
    _STREAM_MIN_MAX_TOKENS = 16000

    async def _call_openai(self, client, model, user_message, max_tokens, stream: bool) -> str:
        if not stream:
            response = await client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=self.temperature,
                messages=[{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": user_message}],
            )
            return response.choices[0].message.content
        parts = []
        s = await client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=self.temperature, stream=True,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": user_message}],
        )
        async for chunk in s:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        return "".join(parts)

    async def _call_anthropic(self, client, model, user_message, max_tokens, stream: bool) -> str:
        if not stream:
            response = await client.messages.create(
                model=model, max_tokens=max_tokens, temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        parts = []
        async with client.messages.stream(
            model=model, max_tokens=max_tokens, temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as s:
            async for txt in s.text_stream:
                parts.append(txt)
        return "".join(parts)

    async def call(self, user_message: str, max_tokens_override: int = 0) -> str:
        """Call LLM and return raw text response, with exponential backoff retry."""
        _max_tokens = max_tokens_override or self.max_tokens
        _stream = _max_tokens > self._STREAM_MIN_MAX_TOKENS
        last_err = None
        _t0 = time.monotonic()
        _agent_name = type(self).__name__
        for attempt in range(self.max_retries):
            try:
                if self._provider == "openai":
                    from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
                    try:
                        _text = await self._call_openai(
                            self._client, self.model, user_message, _max_tokens, _stream)
                        await _log_llm_call(
                            _agent_name, self.model, self.system_prompt,
                            user_message, _text, None,
                            int((time.monotonic() - _t0) * 1000),
                        )
                        return _text
                    except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                        raise  # caught by outer except below as generic Exception for retry
                else:
                    _text = await self._call_anthropic(
                        self._client, self.model, user_message, _max_tokens, _stream)
                    await _log_llm_call(
                        _agent_name, self.model, self.system_prompt,
                        user_message, _text, None,
                        int((time.monotonic() - _t0) * 1000),
                    )
                    return _text
            except RETRYABLE_ERRORS as e:
                last_err = e
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            except (anthropic.AuthenticationError, anthropic.BadRequestError):
                raise
            except Exception as e:
                # Catches OpenAI retryable errors when provider=openai
                last_err = e
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        # Primary exhausted — try fallback provider once if configured.
        if self._fallback_client is not None:
            logger.debug(
                f"Primary LLM exhausted after {self.max_retries} attempts ({last_err}); "
                f"failing over to {self._fallback_model}"
            )
            try:
                _text = await self._call_openai(
                    self._fallback_client, self._fallback_model,
                    user_message, _max_tokens, _stream)
                await _log_llm_call(
                    f"{_agent_name}[fallback]", self._fallback_model, self.system_prompt,
                    user_message, _text, None,
                    int((time.monotonic() - _t0) * 1000),
                )
                return _text
            except Exception as fb_err:
                logger.warning(
                    f"Primary LLM exhausted ({last_err}); fallback {self._fallback_model} "
                    f"also failed: {fb_err}"
                )
                last_err = fb_err

        _record_api_error(last_err)
        await _log_llm_call(
            _agent_name, self.model, self.system_prompt,
            user_message, None, f"{type(last_err).__name__}: {last_err}",
            int((time.monotonic() - _t0) * 1000),
        )
        raise LLMCallError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    @staticmethod
    def extract_json(text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks and nested structures."""
        text = text.strip()

        # 1. Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip ```json ... ``` fences and retry
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 2.5 Sanitize unescaped inner quotes / bare newlines, then retry
        sanitized = _sanitize_json_strings(text)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        if fence_match:
            try:
                return json.loads(_sanitize_json_strings(fence_match.group(1).strip()))
            except json.JSONDecodeError:
                pass

        # 3. Find the outermost { ... } by brace counting (handles nested JSON)
        start = text.find('{')
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False
            for i, ch in enumerate(text[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        # 4. Attempt repair: close any unclosed string then close unclosed braces/brackets
        repaired = _repair_truncated_json(text)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Cannot extract JSON from response: {text[:300]}")
