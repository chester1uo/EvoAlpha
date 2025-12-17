import json

from factor_search.personas import Persona, tune_persona


def test_tune_persona_returns_parsed():
    # pytest factor_search/tests/test_personas_tune.py  -vs
    old = Persona(name="Mean-Reverter", description="targets short-term mean reversion")
    perf = {"ic": 0.045, "stability": 0.10}

    res = tune_persona(old_persona=old, performance_stats=perf, n=3)

    # Should return a PersonaLibrary-like object with PERSONA_LIBRARY attribute
    assert hasattr(res, "PERSONA_LIBRARY")
    assert len(res.PERSONA_LIBRARY) == 3
    print(f"res.PERSONA_LIBRARY: {res.PERSONA_LIBRARY}")
