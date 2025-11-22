from typing import Any, Dict, List
import json
import re


def safe_metric(f: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(f.get("metrics", {}).get(key, default))
    except (TypeError, ValueError):
        return float(default)


def rank_by_ic(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort a pool of factor dicts by metrics.ic descending.
    """
    return sorted(pool, key=lambda x: safe_metric(x, "ic", -1e9), reverse=True)


def select_seed_pool(pool: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
    """
    Pick the top_k factors from a pool by IC. If top_k <= 0, returns full ranked pool.
    """
    ranked = rank_by_ic(pool)
    if top_k <= 0:
        return ranked
    k = max(1, min(top_k, len(ranked)))
    return ranked[:k]


def seed_block_json(seeds: List[Dict[str, Any]]) -> str:
    """
    Convert a list of seeds to a compact JSON block for prompts.
    """
    compact = []
    for s in seeds:
        compact.append(
            {
                "name": s.get("name"),
                "expression": s.get("expression") or s.get("qlib_expression_default"),
                "metrics": s.get("metrics", {}),
            }
        )
    return json.dumps(compact, ensure_ascii=False, separators=(",", ": "), indent=2)


def dedup_by_expression(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate dict factors by normalized expression.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for f in factors:
        expr = f.get("expression", "")
        if not expr:
            continue
        key = re.sub(r"\s+", "", expr)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def extract_json_array(text: str):
    """
    Try to extract the first valid JSON array from a model output.
    Be forgiving about stray commentary or code fences.
    """
    text = text.strip()
    # Fast path
    if text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Remove common Markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        stripped = "".join(
            p for p in parts if p.strip() and not p.strip().startswith("json")
        )
        text = stripped.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                pass

    # Fallback: regex for an array
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        return []
