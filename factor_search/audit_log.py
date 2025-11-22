# factor_search/audit_log.py

import os
import json
import threading
from datetime import datetime
from typing import Any, Dict, Optional


class AuditLogger:
    """
    Simple JSONL logger for all API calls and generated factors.

    Each line in the log file is a JSON object:

        {
          "ts": "2025-01-01T12:34:56.789012Z",
          "event": "api_request" | "api_response" | "llm_call" |
                   "llm_response" | "llm_candidates" | "round_summary",
          "data": { ... arbitrary payload ... }
        }

    Log directory can be configured with env FACTOR_SEARCH_LOG_DIR.
    """

    def __init__(
        self,
        log_dir: str = None,
        filename_prefix: str = "factor_search_events",
    ) -> None:
        self.log_dir = log_dir or os.environ.get("FACTOR_SEARCH_LOG_DIR", "./logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.filename = os.path.join(self.log_dir, f"{filename_prefix}.jsonl")
        self._lock = threading.Lock()

    def log_event(self, event: str, data: Dict[str, Any]) -> None:
        """
        Append a single event line to the JSONL file.
        Never raises (errors are swallowed).
        """
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "data": data,
        }
        try:
            line = json.dumps(record, ensure_ascii=False)
        except Exception:
            # Fallback: best-effort string
            line = json.dumps(
                {
                    "ts": record["ts"],
                    "event": event,
                    "data": str(data),
                    "warning": "Failed to JSON-encode data cleanly",
                },
                ensure_ascii=False,
            )

        try:
            with self._lock:
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Logging should never crash the main process
            pass


_global_logger: Optional[AuditLogger] = None
_global_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """
    Get a process-wide singleton AuditLogger.
    """
    global _global_logger
    if _global_logger is None:
        with _global_lock:
            if _global_logger is None:
                _global_logger = AuditLogger()
    return _global_logger
