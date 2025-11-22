from typing import Dict, List

from .utils import seed_block_json

BASE_ALLOWED_VARS = "$close, $open, $high, $low, $volume"
SAFE_EPS = "1e-12"

QLIB_GENERATE_INSTRUCTION = """
Allowed operators in Qlib expressions:
**Arithmetic / Logic**
- Add(x,y), Sub(x,y), Mul(x,y), Div(x,y)  
- Power(x,y), Log(x), Sqrt(x), Abs(x), Sign(x), Delta(x,n)  
- And(x,y), Or(x,y), Not(x)  
- Sqrt(x)
- Comparators: Greater(x,y), Less(x,y), Gt(x,y), Ge(x,y), Lt(x,y), Le(x,y), Eq(x,y), Ne(x,y)

**Rolling (n is positive integer)**
- Mean(x,n), Std(x,n), Var(x,n), Max(x,n), Min(x,n)  
- Skew(x,n), Kurt(x,n), Sum(x,n), Med(x,n), Mad(x,n), Count(x,n)  | Med is for median and Mad for Mean Absolute Deviation
- EMA(x,n), WMA(x,n), Corr(x,y,n), Cov(x,y,n)  
- Slope(x,n), Rsquare(x,n), Resi(x,n)

**Ranking / Conditional**
- Rank(x,n), Ref(x,n), IdxMax(x,n), IdxMin(x,n), Quantile(x,n,qscore (float number between 0-1)) 
- If(cond,x,y), Mask(cond,x)

Note: function signatures must be complete.  
- Corr(x,y,n) requires 3 arguments 
- Quantile(x,n,qscore) requires 3 arguments
- Rank(x,n) requires 2 arguments  
- Ref(x,n) requires 2 arguments

Important rules:
a. For arithmetic operations, do NOT use symbols. Instead, use: Add for +, Sub for -, Mul for *, Div for /
b. Parentheses must balance.  
c. Correct arity — no missing arguments.  
d. Rolling windows (n) must be positive integers.  
e. Division safety — always add epsilon:  
   - Div(x, Add(den, 1e-12)) correct
   - Div(x, den) incorrect
   Sqrt safely, ensure no negative inputs.
f. No undefined / banned functions (e.g., SMA, RSI), and above operation is low/upper-case sensitive.  
g. Expressions must be plain strings, no comments or backticks.

"""

def build_mutation_prompt(
    *,
    persona_name: str,
    persona_description: str,
    user_request: str,
    seeds: List[Dict],
    required_components: List[str],
    avoided_ops: List[str],
    market: str,
    universe: str,
    style: str,
    horizon: str,
    n: int,
    round_id: int,
    enable_reason: bool = True,
) -> str:
    seed_block = seed_block_json(seeds)
    required_str = ", ".join(required_components) if required_components else "none"
    avoided_str = ", ".join(avoided_ops) if avoided_ops else "none"

    reason_field = '"reason": "<1-2 sentences on why this mutation should help>"' if enable_reason else ""

    return f"""
You are a quantitative researcher acting as **{persona_name}**.
Persona brief: {persona_description}


Your job in this round: **MUTATE** existing alpha factors to produce **exactly {n} new candidates**.

Round: {round_id}
Target market: {market}, universe: {universe}
Style focus: {style}
Typical holding horizon: {horizon}

User search request (natural language):
{user_request}

Global constraints:
- Use only these variables: {BASE_ALLOWED_VARS}
- Use opertors correctly as per Qlib specifications {QLIB_GENERATE_INSTRUCTION}.
- You MAY NOT use any of these operators: {avoided_str}
- Respect Qlib-style function names and balanced parentheses.
- When dividing by volatility / range / volume, always add a small epsilon {SAFE_EPS} in the denominator.
- Prefer concise expressions; avoid gratuitous nesting if it does not help robustness.
- Soft preference for components: {required_str}

Mutation guidelines:
- Change window lengths (e.g., 5→7, 10→12, 20→18) to adjust smoothness and responsiveness.
- Swap or insert nearby operators while preserving the core signal type (momentum, mean-reversion, volatility, liquidity).
- Add / remove Rank, Std, ATR, Log when it clearly improves scaling or robustness.
- Introduce multi-window structures when they help stability (e.g., fast-vs-slow comparisons).
- Do NOT simply copy a seed expression with only a trivial change; each mutation should be meaningfully different.
- Please control the depth of a single factor under 5, don't make a factor nested more than 3 components.

Output strictly as a valid JSON array (and nothing else) of length {n}.
Each item must look like:
{{
    "name": "<short unique factor name>",
    "expression": "<single Qlib expression string>",
    "meta": {{
        "type": "mutation",
        "from": "<seed name or key sub-expression being mutated>"
    }},
    "tags": {{
        "mode": "mutation",
        "persona": "{persona_name}"
    }}{"," if enable_reason else ""}
    {reason_field}
}}

Self-check BEFORE you answer:
- Every expression uses only allowed variables and does not use banned operators.
- All parentheses are balanced; operator arities are correct.
- Denominators that may be small include +{SAFE_EPS}.
- None of the expressions is identical to a seed expression.
- The JSON is valid and contains no comments or extra text.
"""


def build_crossover_prompt(
    *,
    persona_name: str,
    persona_description: str,
    user_request: str,
    seeds: List[Dict],
    required_components: List[str],
    avoided_ops: List[str],
    market: str,
    universe: str,
    style: str,
    horizon: str,
    n: int,
    round_id: int,
    enable_reason: bool = True,
) -> str:
    seed_block = seed_block_json(seeds)
    required_str = ", ".join(required_components) if required_components else "none"
    avoided_str = ", ".join(avoided_ops) if avoided_ops else "none"

    reason_field = '"reason": "<1-2 sentences on which parts were combined and why>"' if enable_reason else ""

    return f"""
You are a quantitative researcher acting as **{persona_name}**.
Persona brief: {persona_description}

Your job in this round: build **CROSSOVER** factors by recombining useful parts from the seed expressions.
You must propose **exactly {n} new candidates**.

Round: {round_id}
Target market: {market}, universe: {universe}
Style focus: {style}
Typical holding horizon: {horizon}

User search request (natural language):
{user_request}

Global constraints:
- Use only these variables: {BASE_ALLOWED_VARS} 
- Use opertors correctly as per Qlib specifications {QLIB_GENERATE_INSTRUCTION}.
- You MAY NOT use any of these operators: {avoided_str}
- Respect Qlib-style function names and balanced parentheses.
- When dividing by volatility / range / volume, always add a small epsilon {SAFE_EPS} in the denominator.
- Prefer concise expressions; avoid gratuitous nesting.
- Soft preference for components: {required_str}

What "crossover" means here:
- Identify **core signal** parts (e.g., price momentum, range, volatility shocks).
- Identify **normalizers / stabilizers** (e.g., Std, Mean, ATR, volume scales, Rank).
- Optionally identify **gates / regimes** (e.g., high volatility vs low).
- Build new factors by combining core from one seed with normalizer/gate from another.

Example pattern (do NOT copy literally):
- Core A: Mean(Sub($close, Ref($close, 1)), 10)
- Normalizer B: Add(Std($close, 60), {SAFE_EPS})
- Crossover: Div(Mean(Sub($close, Ref($close, 1)), 12), Add(Std($close, 60), {SAFE_EPS}))
- Please keep complete sub-expressions intact when recombining.
- Please don't make too complex expressions which leads long expressions that are hard to interpret.
- Please control the depth of a single factor under 5, don't make a factor nested more than 3 components.

Output strictly as a valid JSON array (and nothing else) of length {n}.
Each item must look like:
{{
    "name": "<short unique factor name>",
    "expression": "<single Qlib expression string>",
    "meta": {{
        "type": "crossover",
        "from_A": "<core sub-expression or seed name A>",
        "from_B": "<normalizer/gate sub-expression or seed name B>"
    }},
    "tags": {{
        "mode": "crossover",
        "persona": "{persona_name}"
    }}{"," if enable_reason else ""}
    {reason_field}
}}

Self-check BEFORE you answer:
- Every expression recombines non-trivial parts from at least two different seed expressions.
- Only allowed variables are used; banned operators are not used.
- Parentheses are balanced; operator arities are correct.
- Denominators that may be small include +{SAFE_EPS}.
- None of the expressions is identical to a seed expression.
- The JSON is valid and contains no comments or extra text.
"""
