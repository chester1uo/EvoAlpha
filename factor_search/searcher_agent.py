import time
from typing import Any, Callable, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import build_mutation_prompt, build_crossover_prompt
from .schemas import FactorCandidate, SearcherReport
from .utils import extract_json_array
from .utils import seed_block_json

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

        self.llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.openai-proxy.com/v1")
        # Short random id for logging
        import uuid as _uuid

        self.agent_id = str(_uuid.uuid4())[:8]

    def _build_user_prompt(self, seeds):
        seed_block = seed_block_json(seeds)
        
        user_prompts =  f"""
        Seed factors (JSON array with name, expression, metrics):
        {seed_block}
        """
    
        return user_prompts

    def _build_prompt(
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
        common = dict(
            persona_name=self.persona_name,
            persona_description=self.persona_description,
            user_request=user_request,
            seeds=seeds,
            required_components=required_components,
            avoided_ops=avoided_operators,
            market=market,
            universe=universe,
            style=style,
            horizon=horizon,
            n=n,
            round_id=round_id,
            enable_reason=self.enable_reason,
        )
        if self.mode == "mutation":
            return build_mutation_prompt(**common)
        return build_crossover_prompt(**common)

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
    ) -> Tuple[List[FactorCandidate], SearcherReport]:
        """
        Run the searcher to produce up to n_factors candidates that pass quality checks.
        """
        accepted: List[FactorCandidate] = []
        attempts = 0
        retries = 0

        start_time = time.time()

        while len(accepted) < n_factors and retries < self.max_retries:
            need = n_factors - len(accepted)
            system_prompt = self._build_prompt(
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

            user_prompt = self._build_user_prompt(seeds)
            messages = [
                SystemMessage(
                    content=(system_prompt)
                ),
                HumanMessage(content=user_prompt),
            ]

            attempts += 1
            response = self.llm.invoke(messages)
            items = extract_json_array(response.content)

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
                
                # import pdb; pdb.set_trace() 

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
            
            # import pdb;pdb.set_trace()
            if len(accepted) < n_factors:
                retries += 1

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
