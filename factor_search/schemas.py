from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class FactorCandidate:
    """
    In-memory representation of a factor.

    Fields:
        name       : factor identifier
        expression : Qlib expression string
        doc_type   : "origin" or "search"
        meta       : classification info, e.g.

            {"type": "origin"}

            {"type": "mutation", "from": "Sign(Slope(close, 10))"}

            {
              "type": "crossover",
              "from_A": "Sign(Slope(close, 10))",
              "from_B": "Log(Std(volume, 30))"
            }

        tags/provenance/reason : extra annotations.
    """

    name: str
    expression: str
    reason: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    doc_type: str = "search"  # "origin" or "search"


@dataclass
class SearcherReport:
    """
    Diagnostics for a single SearcherAgent.
    """
    agent_id: str
    character: str
    mode: str                     # "mutation" or "crossover"
    requested: int
    accepted: int
    attempts: int                 # number of LLM calls
    retries: int
    elapsed_sec: float
    calls_per_factor: float
    reliability_score: float
    extra: Dict[str, Any] = field(default_factory=dict)
