#!/usr/bin/env python
"""
Client for Factor Evaluation REST API (template, updated for new server).

This module exposes a simple function:

    from api.factor_eval_client import batch_evaluate_factors_via_api

which your factor search code can use.

It assumes the server you showed is running with endpoints:
- GET  /health
- POST /check
- GET/POST /eval
- POST /batch_eval
"""

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ------------------------------
# Configuration
# ------------------------------

DEFAULT_API_URL = os.environ.get("FACTOR_EVAL_API_URL", "http://localhost:19889")
DEFAULT_TIMEOUT = int(os.environ.get("FACTOR_EVAL_CLIENT_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("FACTOR_EVAL_CLIENT_MAX_RETRIES", "5"))
RETRY_DELAY = float(os.environ.get("FACTOR_EVAL_CLIENT_RETRY_DELAY", "1.0"))


class FactorEvalClient:
    """
    Thin HTTP client for your Factor Evaluation API.

    It wraps:
    - /health
    - /check
    - /eval
    - /batch_eval
    """

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # -------------------------- #
    # Internal request helper
    # -------------------------- #

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request with simple retry logic.

        Returns:
            Parsed JSON dict on success, or None if all retries fail.
        """
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    params=params,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        logger.error("Failed to parse JSON from %s %s: %r", method, url, resp.text)
                        return None
                else:
                    logger.warning(
                        "API %s %s failed with status %s: %s",
                        method,
                        url,
                        resp.status_code,
                        resp.text[:500],
                    )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Connection error talking to %s (attempt %d/%d)",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout talking to %s (attempt %d/%d)",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                )
            except Exception as e:
                logger.error("Unexpected error calling %s %s: %s", method, url, e)
                break

            # Backoff between retries
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

        return None

    # -------------------------- #
    # Public methods
    # -------------------------- #

    def health_check(self) -> bool:
        """
        Check if the API server is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        data = self._request("GET", "/health")
        return bool(data and data.get("status") == "healthy")

    def check_factor(
        self,
        expr: str,
        *,
        instruments: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call POST /check to validate an expression quickly.

        Returns:
            The server's JSON response on success, or a default error dict.
        """
        payload: Dict[str, Any] = {"expression": expr}
        if instruments is not None:
            payload["instruments"] = instruments
        if start is not None:
            payload["start"] = start
        if end is not None:
            payload["end"] = end
        if timeout is not None:
            payload["timeout"] = timeout

        res = self._request("POST", "/check", json_body=payload, timeout=timeout)
        if res is None:
            return {"success": False, "error_message": "Check request failed", "error_type": "CLIENT"}
        return res

    def evaluate_factor(
        self,
        expr: str,
        *,
        market: str = "csi300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        label: str = "close_return",
        use_cache: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single expression (wraps /eval).

        Returns:
            Server JSON response, or a default failure structure.
        """
        # Use POST for simplicity and robustness with long expressions.
        payload = {
            "expression": expr,
            "start": start_date,
            "end": end_date,
            "market": market,
            "label": label,
            "use_cache": use_cache,
        }
        res = self._request("POST", "/eval", json_body=payload, timeout=timeout)
        if res is None:
            return {
                "success": False,
                "error": "evaluate_factor request failed",
                "expression": expr,
                "market": market,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": {
                    "ic": 0.0,
                    "rank_ic": 0.0,
                    "ir": 0.0,
                    "icir": 0.0,
                    "rank_icir": 0.0,
                    "turnover": 1.0,
                    "n_dates": 0,
                },
            }
        return res

    def batch_evaluate_factors(
        self,
        factors: List[Dict[str, Any]],
        *,
        market: str = "csi300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        label: str = "close_return",
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple factors individually. If an error occurs while evaluating
        any factor, set all metrics to 0 and continue evaluating the next factor.

        Args:
            factors: List[{"name": str, "expression": str}]
            market: Market identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            label: Label type for evaluation
            timeout: Timeout for the request

        Returns:
            List of result dicts, one per factor (order-preserving).
            Each result dict will include at least a "metrics" field (which is
            set to zero if an error occurs).
        """
        results = []

        for f in factors:
            try:
                payload = {
                    "expression": f.get("expression", ""),
                    "start": start_date,
                    "end": end_date,
                    "market": market,
                    "label": label,
                }
                if timeout is not None:
                    payload["timeout"] = timeout

                # Make the request for a single factor
                resp = self._request("POST", "/eval", json_body=payload, timeout=timeout)

                # If response is successful, process the result
                if resp and resp.get("success"):
                    result = resp
                else:
                    result = {
                        "success": False,
                        "error": "Factor evaluation failed",
                        "name": f.get("name"),
                        "expression": f.get("expression"),
                        "metrics": {
                            "ic": 0.0,
                            "rank_ic": 0.0,
                            "ir": 0.0,
                            "icir": 0.0,
                            "rank_icir": 0.0,
                            "turnover": 1.0,
                            "n_dates": 0,
                        },
                    }

            except Exception as e:
                logger.error(f"Error evaluating factor {f.get('name', 'unknown')}: {e}")
                result = {
                    "success": False,
                    "error": f"Error evaluating factor: {str(e)}",
                    "name": f.get("name"),
                    "expression": f.get("expression"),
                    "metrics": {
                        "ic": 0.0,
                        "rank_ic": 0.0,
                        "ir": 0.0,
                        "icir": 0.0,
                        "rank_icir": 0.0,
                        "turnover": 1.0,
                        "n_dates": 0,
                    },
                }

            # Append the result for this factor to the list of results
            results.append(result)

        return results



# ----------------------------------------------------------------------
# Convenience functions (what your searcher imports)
# ----------------------------------------------------------------------

_global_client: Optional[FactorEvalClient] = None


def _get_client(api_url: Optional[str] = None) -> FactorEvalClient:
    global _global_client
    base = (api_url or DEFAULT_API_URL).rstrip("/")
    if _global_client is None or _global_client.base_url != base:
        _global_client = FactorEvalClient(base_url=base)
    return _global_client


def check_factor_via_api(expr: str, api_url: Optional[str] = None) -> Dict[str, Any]:
    client = _get_client(api_url)
    return client.check_factor(expr)


def evaluate_factor_via_api(
    expr: str,
    *,
    market: str = "csi300",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    label: str = "close_return",
    use_cache: bool = True,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    client = _get_client(api_url)
    return client.evaluate_factor(
        expr,
        market=market,
        start_date=start_date,
        end_date=end_date,
        label=label,
        use_cache=use_cache,
    )


def batch_evaluate_factors_via_api(
    factors: List[Dict[str, Any]],
    *,
    market: str = "csi300",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    label: str = "close_return",
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    What your multi-agent searcher uses.

    This function returns a list of result dicts, one per input factor,
    each with at least a "metrics" dict.
    """
    client = _get_client(api_url)
    return client.batch_evaluate_factors(
        factors,
        market=market,
        start_date=start_date,
        end_date=end_date,
        label=label,
    )
