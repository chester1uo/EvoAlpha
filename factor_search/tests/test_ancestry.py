from factor_search.utils import get_factor_parents_and_paths


SAMPLE_FACTORS = [
    {
        "name": "SUMD",
        "meta": {"type": "seed"},
    },
    {
        "name": "Std($close, 10)",
        "meta": {"type": "seed"},
    },
    {
        "name": "SUMD_div_STDclose",
        "meta": {"type": "crossover", "from_A": "SUMD", "from_B": "Std($close, 10)"},
    },
    {
        "name": "BETA",
        "meta": {"type": "seed"},
    },
    {
        "name": "RankBETA_div_STDclose",
        "meta": {"type": "crossover", "from_A": "BETA", "from_B": "Std($close, 20)"},
    },
    {
        "name": "TopCompound",
        "meta": {"type": "crossover", "from_A": "SUMD_div_STDclose", "from_B": "RankBETA_div_STDclose"},
    },
]


def test_direct_parents_and_chains():
    res = get_factor_parents_and_paths(SAMPLE_FACTORS, "SUMD_div_STDclose")
    assert res["name"] == "SUMD_div_STDclose"
    assert set(res["parents"]) == {"SUMD", "Std($close, 10)"}
    # expected ancestor chains: SUMD -> SUMD_div_STDclose and Std($close,10) -> SUMD_div_STDclose
    print(f'res["ancestor_chains"]: {res["ancestor_chains"]}')
    assert ["SUMD", "SUMD_div_STDclose"] in res["ancestor_chains"]
    assert ["Std($close, 10)", "SUMD_div_STDclose"] in res["ancestor_chains"]


def test_recursive_chains_to_root():
    # TopCompound uses two children that themselves refer to seeds
    res = get_factor_parents_and_paths(SAMPLE_FACTORS, "TopCompound")
    assert res["name"] == "TopCompound"
    assert set(res["parents"]) == {"SUMD_div_STDclose", "RankBETA_div_STDclose"}
    # check there are chains that include deeper nodes
    # examples: SUMD -> SUMD_div_STDclose -> TopCompound
    print(f'res["ancestor_chains"]: {res["ancestor_chains"]}')
    assert ["SUMD", "SUMD_div_STDclose", "TopCompound"] in res["ancestor_chains"]
    assert ["Std($close, 10)", "SUMD_div_STDclose", "TopCompound"] in res["ancestor_chains"]
    assert ["BETA", "RankBETA_div_STDclose", "TopCompound"] in res["ancestor_chains"]
