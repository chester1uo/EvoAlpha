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
    # TODO: use new strategy to select factor.
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


def get_factor_parents_and_paths(factors: List[Dict[str, Any]], factor_name: str):
    """
    Given a list of factor dicts (each may contain a 'meta' with 'from', 'from_A', 'from_B'),
    return information about the named factor's direct parents and all ancestor paths.

    Returns a dict with:
      - name: the queried factor name
      - parents: list of direct parent identifiers (strings)
      - ancestor_chains: list of lists, each chain is [root_parent, ..., direct_parent, factor_name]

    Parent identifiers that exactly match another factor's 'name' will be expanded recursively.
    Non-matching parent strings are treated as leaf nodes.
    """

    # build map name -> factor
    name_map = {f.get("name"): f for f in factors if isinstance(f, dict) and f.get("name")}

    if factor_name not in name_map:
        return {"name": factor_name, "parents": [], "ancestor_chains": []}

    def extract_parents(meta: Dict[str, Any]) -> List[str]:
        parents = []
        if not isinstance(meta, dict):
            return parents
        for k in ("from_A", "from_B", "from"):
            v = meta.get(k)
            if isinstance(v, str) and v:
                parents.append(v)
        return parents

    # use an ordered recursion stack (list) rather than an unordered set
    # so ancestor chains preserve the path order (root -> ... -> node)
    def dfs(node_name: str, stack: List[str]):
        # return list of ancestor chains for node_name (chains ending with node_name)
        if node_name not in name_map:
            # leaf
            return [[node_name]]

        if node_name in stack:
            # cycle detected; return the cycle path (preserve order)
            idx = stack.index(node_name)
            # stack[idx:] is the part of the path forming the cycle
            return [stack[idx:] + [node_name]]
        meta = name_map[node_name].get("meta", {})
        parents = extract_parents(meta)
        if not parents:
            return [[node_name]]

        chains = []
        for p in parents:
            subchains = dfs(p, stack + [node_name])
            for sc in subchains:
                chains.append(sc + [node_name])

        # no need to mutate a global visited set; stack is passed down
        return chains

    direct_meta = name_map[factor_name].get("meta", {})
    direct_parents = extract_parents(direct_meta)

    ancestor_chains = []
    for chain in dfs(factor_name, []):
        # only include chains that end with factor_name (dfs returns these)
        if chain[-1] == factor_name:
            ancestor_chains.append(chain)

    # dedupe parents preserving order
    seen = set()
    uniq_parents = []
    for p in direct_parents:
        if p not in seen:
            uniq_parents.append(p)
            seen.add(p)

    return {"name": factor_name, "parents": uniq_parents, "ancestor_chains": ancestor_chains}
