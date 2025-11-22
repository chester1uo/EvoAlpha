from typing import Any, Dict


def default_quality_check(candidate: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Placeholder for your real 'factor requirement' API.

    Expected input:
        candidate: {"name": str, "expression": str, ...}
        context:   arbitrary dict with task/controller information

    Return:
        True  -> this factor passes quality checks and should be kept
        False -> discard this factor

    Replace this implementation with a call to your own service.
    """
    expr = candidate.get("expression", "")
    if not expr or "$" not in expr:
        return False

    # Very light sanity checks; your real API should be stricter.
    if len(expr) < 10:
        return False

    banned_fragments = ["NaN", "Inf"]
    if any(bad in expr for bad in banned_fragments):
        return False

    return True
