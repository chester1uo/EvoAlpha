import time
import json
from typing import Any, Callable, Dict, List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import build_mutation_prompt, build_crossover_prompt
from .schemas import FactorCandidate, SearcherReport
from .utils import extract_json_array, seed_block_json
from .run_logger import RunLogger

QualityCheckFn = Callable[[Dict[str, Any], Dict[str, Any]], bool]


class SearcherAgent:
    """
    A single-role searcher (either mutation or crossover).

    Responsibilities:
    - Build a strong prompt based on persona, user task, and seed pool.
    - Call the LLM to get factor candidates.
    - Run a quality check on each candidate via an external API.
    - Retry up to max_retries times to fill the requested number of factors.
    - Compute a reliability score that reflects call cost per accepted factor.
    - Optionally log raw LLM attempts per round via RunLogger.
    """

    def __init__(
        self,
        *,
        mode: str,  # "mutation" or "crossover"
        persona_name: str,
        persona_description: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 1.1,
        max_retries: int = 5,
        enable_reason: bool = True,
        quality_check_fn: QualityCheckFn,
    ) -> None:
        if mode not in ("mutation", "crossover"):
            raise ValueError(f"mode must be 'mutation' or 'crossover', got {mode!r}")
        self.mode = mode
        self.persona_name = persona_name
        self.persona_description = persona_description
        self.max_retries = max_retries
        self.enable_reason = enable_reason
        self.quality_check_fn = quality_check_fn

        # Configure your proxy/base URL as needed
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://api.openai-proxy.com/v1",
        )

        # Short random id for logging
        import uuid as _uuid
        self.agent_id = str(_uuid.uuid4())[:8]

    # ---------------------------- Prompt builders ---------------------------- #

    def _build_system_prompt(self, n: int) -> str:
        """
        Build the system prompt with instructions (using the original prompt builders).
        We keep the instruction body from build_mutation_prompt/build_crossover_prompt,
        but the *data* (task, seeds, metrics) is fed via the user prompt below.
        """
        # We call the original builders to keep the instruction intent,
        # but their seed embedding is ignored because the user message will carry data.
        # We pass empty/noisy seeds and small n here; the JSONL/output format is enforced
        # in the *user* prompt below to avoid conflicts.
        dummy_common = dict(
            persona_name=self.persona_name,
            persona_description=self.persona_description,
            user_request="(provided in user prompt)",
            seeds=[],  # data will be in user prompt
            required_components=[],
            avoided_ops=[],
            market="(user prompt)",
            universe="(user prompt)",
            style="(user prompt)",
            horizon="(user prompt)",
            n=n,
            round_id=-1,
            enable_reason=self.enable_reason,
        )
        if self.mode == "mutation":
            return build_mutation_prompt(**dummy_common)
        return build_crossover_prompt(**dummy_common)

    def _build_user_prompt(
        self,
        *,
        user_request: str,
        seeds: List[Dict[str, Any]],
        required_components: List[str],
        avoided_operators: List[str],
        market: str,
        universe: str,
        style: str,
        horizon: str,
        n: int,
        round_id: int,
    ) -> str:
        """
        User prompt carries the concrete task, constraints, and seeds+metrics.
        It also *enforces JSONL output* explicitly.
        """
        seed_block = seed_block_json(seeds)
        req_str = ", ".join(required_components) if required_components else "none"
        avoid_str = ", ".join(avoided_operators) if avoided_operators else "none"

        # Reason field guidance (kept optional)
        reason_hint = (
            '- Include a short "reason" field (1–2 sentences).' if self.enable_reason else
            "- Do NOT include a 'reason' field."
        )

        return f"""
Round: {round_id}
Target market: {market}
Universe: {universe}
Style focus: {style}
Typical holding horizon: {horizon}

User search request (natural language):
\"\"\"{user_request}\"\"\"

Constraints:
- Preferred components (soft): {req_str}
- Banned operators: {avoid_str}

Seed factors (JSON array; each has `name`, `expression`, `metrics`):
{seed_block}

OUTPUT FORMAT (JSONL):
- Output **exactly {n} lines**.
- **Each line** must be one valid JSON object (no array, no extra text).
- Each object MUST contain:
  - "name": short unique name (string)
  - "expression": full Qlib expression (string)
  - "meta": object describing generation:
      {{"type": "{self.mode}", "from": "<seed/subexpr>"}}      # for mutation
      or
      {{"type": "crossover", "from_A": "<...>", "from_B": "<...>"}}  # for crossover
  - "tags": object:
      {{"mode": "{self.mode}", "persona": "{self.persona_name}"}}
  {reason_hint}

Self-check before answering:
- Only allowed variables ($close, $open, $high, $low, $volume) are used.
- Banned operators are not used.
- Parentheses are balanced; arities correct.
- Denominators that can be small include +1e-12 for safety.
- None of the outputs are identical to seeds (non-trivial changes).
"""

    # ------------------------------ Parsing utils --------------------------- #

    @staticmethod
    def _parse_jsonl_or_array(text: str) -> List[Dict[str, Any]]:
        """
        Accept JSONL (preferred) or fallback to JSON array.
        """
        items: List[Dict[str, Any]] = []

        # Try JSONL
        for line in text.splitlines():
            ln = line.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                # ignore malformed lines
                continue
        if items:
            return items

        # Fallback: JSON array (reuse helper)
        try:
            arr = extract_json_array(text)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except Exception:
            pass
        return []

    # ------------------------------ Main search ----------------------------- #

    def search(
        self,
        *,
        user_request: str,
        seeds: List[Dict[str, Any]],
        required_components: List[str],
        avoided_operators: List[str],
        market: str,
        universe: str,
        style: str,
        horizon: str,
        n_factors: int,
        round_id: int,
        context: Dict[str, Any],
        run_logger: Optional[RunLogger] = None,  # <-- for raw LLM logging
    ) -> Tuple[List[FactorCandidate], SearcherReport]:
        """
        Run the searcher to produce up to n_factors candidates that pass quality checks.
        Optionally records raw LLM calls (prompts + raw response) via RunLogger.
        """
        accepted: List[FactorCandidate] = []
        attempts = 0
        retries = 0
        llm_attempt_logs: List[Dict[str, Any]] = []

        start_time = time.time()

        while len(accepted) < n_factors and retries < self.max_retries:
            need = n_factors - len(accepted)

            system_prompt = self._build_system_prompt(need)
            user_prompt = self._build_user_prompt(
                user_request=user_request,
                seeds=seeds,
                required_components=required_components,
                avoided_operators=avoided_operators,
                market=market,
                universe=universe,
                style=style,
                horizon=horizon,
                n=need,
                round_id=round_id,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            attempts += 1
            t0 = time.time()
            response = self.llm.invoke(messages)
            call_elapsed = time.time() - t0
            raw_text = response.content if hasattr(response, "content") else str(response)

            # Record raw LLM attempt for this searcher/round
            if run_logger is not None:
                llm_attempt_logs.append(
                    {
                        "attempt": attempts,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "response": raw_text,
                        "elapsed_sec": call_elapsed,
                    }
                )

            items = self._parse_jsonl_or_array(raw_text)

            for item in items:
                name = item.get("name")
                expr = item.get("expression")
                if not name or not expr:
                    continue

                meta = item.get("meta") or {"type": self.mode}
                candidate_dict: Dict[str, Any] = {
                    "name": name,
                    "expression": expr,
                    "doc_type": "search",
                    "meta": meta,
                    "reason": item.get("reason", ""),
                    "tags": item.get("tags", {}),
                    "provenance": {
                        "agent_id": self.agent_id,
                        "mode": self.mode,
                        "persona": self.persona_name,
                        "round": round_id,
                        "attempt": attempts,
                    },
                }

                ok = False
                try:
                    ok = bool(self.quality_check_fn(candidate_dict, context))
                except Exception:
                    ok = False
                if not ok:
                    continue

                accepted.append(
                    FactorCandidate(
                        name=name,
                        expression=expr,
                        reason=candidate_dict.get("reason", ""),
                        tags=candidate_dict.get("tags", {}),
                        provenance=candidate_dict.get("provenance", {}),
                        meta=meta,
                        doc_type="search",
                    )
                )

                if len(accepted) >= n_factors:
                    break

            if len(accepted) < n_factors:
                retries += 1

        # Persist raw LLM attempts for this round/searcher
        if run_logger is not None and llm_attempt_logs:
            run_logger.log_llm_round(
                round_id=round_id,
                agent_id=self.agent_id,
                mode=self.mode,
                persona=self.persona_name,
                attempts=llm_attempt_logs,
            )

        elapsed = time.time() - start_time
        if accepted:
            calls_per_factor = attempts / float(len(accepted))
            reliability = 1.0 / (1.0 + calls_per_factor)
        else:
            calls_per_factor = float("inf")
            reliability = 0.0

        report = SearcherReport(
            agent_id=self.agent_id,
            character=self.persona_name,
            mode=self.mode,
            requested=n_factors,
            accepted=len(accepted),
            attempts=attempts,
            retries=retries,
            elapsed_sec=elapsed,
            calls_per_factor=calls_per_factor,
            reliability_score=reliability,
            extra={"persona_description": self.persona_description},
        )

        return accepted, report
        