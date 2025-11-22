import random
from dataclasses import dataclass


@dataclass
class Persona:
    name: str
    description: str


PERSONA_LIBRARY = [
    Persona(
        name="Volatility Whisperer",
        description=(
            "Focus on volatility-aware scaling using Std/ATR and regime-sensitive signals. "
            "Prefers stable, medium-term structures."
        ),
    ),
    Persona(
        name="Liquidity Normalizer",
        description=(
            "Emphasizes volume and liquidity normalization. "
            "Avoids raw unscaled price moves and prefers Div/Rank with volume terms."
        ),
    ),
    Persona(
        name="Regime Switch Architect",
        description=(
            "Builds conditional structures that behave differently in high/low volatility regimes, "
            "implemented via smooth gates, ranges, and ranks."
        ),
    ),
    Persona(
        name="Mean-Reversion Surgeon",
        description=(
            "Targets short- to medium-term mean-reversion edges with volatility dampening "
            "and robust denominators."
        ),
    ),
    Persona(
        name="Trend Surfer",
        description=(
            "Likes momentum and trend-following cores with multi-window smoothing and "
            "volatility/volume normalization."
        ),
    ),
    Persona(
        name="Structure Minimalist",
        description=(
            "Prefers concise expressions with just enough normalization and smoothing to be robust, "
            "avoiding unnecessary complexity."
        ),
    ),
]


def random_persona() -> Persona:
    return random.choice(PERSONA_LIBRARY)
