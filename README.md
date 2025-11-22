# Quant Factor Multi-Agent Search

This project contains a multi-agent factor search system with:
- Controller / Searcher agents / Validator
- MongoDB-backed factor database with `origin` + `search` factors
- CLI tools for initializing the DB from JSON and running the search

Folders:
- `factor_search/` : Python package with core logic
- `apps/`          : CLI entry points
- `data/`          : Sample input JSON

See `apps/init_factors_from_json.py` and `apps/run_search.py` for usage.
