# Copyright (C) 2026 Araya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Thin HTTP client for the Wyckoff computation service.

All market data (prices, fundamentals) and Wyckoff analysis are provided by the
remote service, authorized by WYCKOFF_API_KEY. This client holds no data backend
of its own — it only orchestrates the LLM evolution loop around the service.
"""
from __future__ import annotations
import logging
from typing import Optional

import requests

from finagent.config import WYCKOFF_API_URL, WYCKOFF_API_KEY, WYCKOFF_API_TIMEOUT

logger = logging.getLogger(__name__)


class WyckoffServiceError(RuntimeError):
    """The Wyckoff service is unreachable, unauthorized, or returned an error."""


def service_post(path: str, body: dict, timeout: Optional[float] = None) -> dict:
    """POST to the Wyckoff service with the auth code; return the JSON body.

    Raises WyckoffServiceError on connection failure, auth failure, or non-200.
    """
    if not WYCKOFF_API_URL or not WYCKOFF_API_KEY:
        raise WyckoffServiceError(
            "Wyckoff 服务未配置：请在 .env 中设置 WYCKOFF_API_URL 与 WYCKOFF_API_KEY（授权码）。"
        )
    url = f"{WYCKOFF_API_URL}{path}"
    headers = {"Authorization": f"Bearer {WYCKOFF_API_KEY}"}
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=timeout or WYCKOFF_API_TIMEOUT)
    except requests.RequestException as e:
        raise WyckoffServiceError(f"无法连接 Wyckoff 服务 {url}: {e}") from e

    if resp.status_code in (401, 403):
        raise WyckoffServiceError(
            "授权码无效、已过期或被吊销（HTTP %d）。请检查 WYCKOFF_API_KEY。" % resp.status_code
        )
    if resp.status_code != 200:
        raise WyckoffServiceError(f"Wyckoff 服务返回 HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as e:
        raise WyckoffServiceError(f"Wyckoff 服务返回非 JSON 响应: {e}") from e
