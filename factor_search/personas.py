import random
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import build_persona_generator_prompt, build_persona_tuner_prompt

# @dataclass
class Persona(BaseModel):
    """Represents a unique financial persona or quantitative strategy profile."""
    name: str = Field(..., description="A short, memorable, and professional English name for the persona.")
    description: str = Field(..., description="A concise description of the persona's core strategy, technical focus, and preferred time horizon.")
    
class PersonaLibrary(BaseModel):
    """The complete list of generated financial personas."""
    PERSONA_LIBRARY: List[Persona] = Field(..., description="A list containing the newly generated financial personas.")


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

# generate by llm(gemini)
PERSONA_LIBRARY_CLASSIC = [
    Persona(
        name="Momentum-Maximizer",
        description=(
            "Focuses on momentum-oriented patterns, trend-following elements, and "
            "long-horizon structures. Seeks to capture persistent price trends."
        ),
    ),
    Persona(
        name="Mean-Reverter",
        description=(
            "Prefers reversal-oriented signals, shorter windows, and z-score-based "
            "normalization to measure and exploit short-term overextensions from the mean."
        ),
    ),
    Persona(
        name="Volatility-Engineer",
        description=(
            "Constructs factors emphasizing volatility features using price ranges "
            "and ATR-like structures. Signals are focused on measuring market uncertainty and risk."
        ),
    ),
]

NEW_PERSONA_LIBRARY = [
    Persona(
        name="Statistical Arbitrageur",
        description=(
            "Focuses on modeling time-series and cross-sectional residuals. "
            "Prioritizes factors based on linear regression residuals, cointegration, "
            "or paired trading logic (Spread/Residual). Encourages use of multivariate "
            "statistical functions like LinReg, Covariance, and Corr."
        ),
    ),
    Persona(
        name="Lag & Lead Explorer",
        description=(
            "Emphasizes the integration of non-synchronous time-series data and "
            "exploration of lead-lag relationships across different time windows. "
            "Heavily uses time-series operators (Delay, Shift, TsMax/Min/ArgMax/ArgMin) "
            "with varying parameters to create non-smooth conditional jumps."
        ),
    ),
    Persona(
        name="Non-Linear Transformist",
        description=(
            "Assumes non-linear market relationships. Prefers using non-linear activation "
            "functions, Log/Exp transformations, and Clipping/Winsorization to "
            "enhance robustness against outliers and non-linear signal expression."
        ),
    ),
    Persona(
        name="Fundamental Data Integrator",
        description=(
            "Combines trading data (Price/Volume) with fundamental or financial "
            "data (e.g., PE Ratio, Book Value changes) as the factor core. "
            "Focuses on the Delta or RateOfChange of fundamental metrics combined "
            "with market dynamics."
        ),
    ),
    Persona(
        name="Factor Decay Analyst",
        description=(
            "Aims to build factors with controlled signal decay rates. Utilizes "
            "adaptive weighting or exponential decay smoothing (EWMA, DecayLinear) "
            "to balance signal timeliness and stability. Seeks to match smoothing "
            "periods with optimal factor IC decay rates."
        ),
    ),
    Persona(
        name="Complexity Optimizer",
        description=(
            "Strictly adheres to the principle of parsimony (simplicity). "
            "Uses complexity constraints (e.g., expression length/node count) "
            "as a penalty term in fitness, seeking maximum information gain "
            "with minimal structural complexity."
        ),
    ),
]

def generate_new_personas(
    *,
    user_request: str = "",
    model: str = "gpt-4.1-mini",
    temperature: float = 1.1,
    n: int = 5,
) -> str:
    prompt = build_persona_generator_prompt(num_to_generate=n)
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url="https://api.openai-proxy.com/v1",
    )
    structured_llm = llm.with_structured_output(
        PersonaLibrary,
        method="json_schema",
        strict=True
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_request),
    ]
    persona_schema = PersonaLibrary.model_json_schema()
    response = structured_llm.invoke(
        messages,
    )
    return response

def random_persona() -> Persona:
    return random.choice(PERSONA_LIBRARY)


def tune_persona(
    old_persona: Persona,
    performance_stats: Dict[str, float],
    model: str = "gpt-4.1-mini",
    temperature: float = 1.0,
    n: int = 3,
) -> Any:
    """
    Produce tuned persona variants from an existing persona and performance statistics.

    Args:
        old_persona: the persona to tune (a Persona instance)
        performance_stats: dictionary of measured performance values (e.g., {'ic': 0.045, 'stability': 0.10})
        model: LLM model name
        temperature: sampling temperature for the LLM
        n: number of tuned personas to generate

    Returns:
        Parsed structured output representing the new personas (typically a PersonaLibrary instance or dict).

    Note: This function calls the project's configured LLM via langchain_openai; in tests you may want to monkeypatch
    ChatOpenAI to avoid network calls.
    """

    prompt = build_persona_tuner_prompt(
        old_persona=old_persona.model_dump() if hasattr(old_persona, "model_dump") else old_persona.dict(),
        performance_stats=performance_stats,
        num_to_generate=n,
    )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url="https://api.openai-proxy.com/v1",
    )

    structured_llm = llm.with_structured_output(
        PersonaLibrary,
        method="json_schema",
        strict=True,
    )

    messages = [
        SystemMessage(content=prompt),
    ]

    response = structured_llm.invoke(messages)

    return response

if __name__ == "__main__":
    # python -m factor_search.personas
    # PERSONA_LIBRARY=[Persona(name='MeanReverter', description='Focuses on identifying and trading asset price deviations from their historical mean using statistical band models, typically operating on intraday and daily timeframes, with heavy reliance on Ornstein-Uhlenbeck process modeling and cointegration techniques.'), Persona(name='VolatilityArbitrageur', description='Capitalizes on discrepancies in implied versus realized volatility by employing options spread strategies and GARCH volatility forecasting models, favoring mid-term horizons and leveraging advanced volatility surfaces and Greeks for risk management.'), Persona(name='MomentumNavigator', description='Exploits persistent price trends across multiple asset classes using adaptive moving average filters and trend-strength indicators like the Average Directional Index, preferring weekly to monthly holding periods with dynamic position sizing based on volatility-adjusted weights.'), Persona(name='FactorAllocator', description='Implements multi-factor equity style investing by quantitatively weighting exposure to value, quality, momentum, and low-volatility factors, utilizing robust cross-sectional regression analyses and portfolio optimization frameworks over medium to long investment horizons.'), Persona(name='HighFrequencyStatArb', description='Engages in ultra-short-term statistical arbitrage by modeling microstructural price dynamics and execution latency, deploying machine learning classifiers for signal generation with sub-second position turnovers and stringent risk controls on order flow imbalance and price impact.')
    # res = generate_new_personas()
    # print(f"res of generate_new_personas {res}")
    
    res = tune_persona(PERSONA_LIBRARY[0], {"ic": 0.045, "stability": 0.10}, n=3)
    print(f"res of tune_persona {res}")