# factor_search/run_logger.py

import json
import os
from typing import Any, Dict, List


class RunLogger:
    """
    Per-run logger that writes:

    - Raw LLM calls per round + searcher:
        <save_dir>/raw/round_<round_id>/searcher_<agent_id>.json

    - Factor records (metrics + accepted flag) per round + searcher:
        <save_dir>/record/round_<round_id>/searcher_<agent_id>.json
    """

    def __init__(self, save_dir: str = ".") -> None:
        # save_dir is the base directory for this run, e.g. "./logs/exp_001"
        self.base_dir = os.path.abspath(save_dir)
        self.raw_root = os.path.join(self.base_dir, "raw")
        self.record_root = os.path.join(self.base_dir, "record")
        os.makedirs(self.raw_root, exist_ok=True)
        os.makedirs(self.record_root, exist_ok=True)

    def _ensure_round_dir(self, root: str, round_id: int) -> str:
        """
        Make sure round directory exists under root and return its path.
        Example: root='.../raw', round_id=1 -> '.../raw/round_1'
        """
        round_dir = os.path.join(root, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)
        return round_dir

    # ---------------- Raw LLM calls ---------------- #

    def log_llm_round(
        self,
        *,
        round_id: int,
        agent_id: str,
        mode: str,
        persona: str,
        attempts: List[Dict[str, Any]],
    ) -> None:
        """
        Save all LLM attempts of a given searcher in a given round to:

            <save_dir>/raw/round_<round_id>/searcher_<agent_id>.json

        Payload shape:
        {
          "round": ...,
          "agent_id": ...,
          "mode": ...,
          "persona": ...,
          "attempts": [
            {
              "attempt": 1,
              "system_prompt": "...",
              "user_prompt": "...",
              "response": "...",
              "elapsed_sec": 0.123
            },
            ...
          ]
        }
        """
        round_dir = self._ensure_round_dir(self.raw_root, round_id)
        path = os.path.join(round_dir, f"searcher_{agent_id}.json")

        data = {
            "round": round_id,
            "agent_id": agent_id,
            "mode": mode,
            "persona": persona,
            "attempts": attempts,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------------- Factor records ---------------- #

    def log_factor_round(
        self,
        *,
        round_id: int,
        per_agent_records: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """
        Save factor records (with metrics + accepted flag) per searcher for a round:

            <save_dir>/record/round_<round_id>/searcher_<agent_id>.json

        Each file payload shape:
        {
          "round": ...,
          "agent_id": "...",
          "factors": [
            {
              "name": "...",
              "expression": "...",
              "doc_type": "search" | "origin",
              "meta": {...},         # mutation / crossover / origin classification
              "tags": {...},
              "provenance": {...},   # includes agent_id, round, etc.
              "metrics": {...},      # IC, RankIC, etc.
              "accepted": true|false
            },
            ...
          ]
        }
        """
        round_dir = self._ensure_round_dir(self.record_root, round_id)

        for agent_id, factors in per_agent_records.items():
            path = os.path.join(round_dir, f"searcher_{agent_id}.json")
            payload = {
                "round": round_id,
                "agent_id": agent_id,
                "factors": factors,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
