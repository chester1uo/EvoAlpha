"""
CLI tool: initialize the MongoDB factor database from a JSON file.

Example usage:

    export MONGODB_URI="mongodb://localhost:27017"
    python apps/init_factors_from_json.py data/origin_factors_sample.json
"""

import argparse
import json
import os
from typing import Any, Dict, List

from factor_search.db import FactorRepository


def load_factors_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "factors" in data:
        factors = data["factors"]
    elif isinstance(data, list):
        factors = data
    else:
        raise ValueError("JSON must be a list or an object with a 'factors' key.")

    normalized: List[Dict[str, Any]] = []
    for item in factors:
        if "name" not in item or "expression" not in item:
            raise ValueError("Each factor must have 'name' and 'expression'.")
        normalized.append(
            {
                "name": item["name"],
                "expression": item["expression"],
                "type": item.get("type", "origin"),
                "operations": item.get("operations"),
                "metrics": item.get("metrics", {}),
                "tags": item.get("tags", {}),
                "provenance": item.get("provenance", {}),
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize factor database from JSON.")
    parser.add_argument("json_path", help="Path to JSON file with factors.")
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        help="MongoDB URI (default: env MONGODB_URI or mongodb://localhost:27017)",
    )
    parser.add_argument(
        "--db-name",
        default="factor_search",
        help="MongoDB database name (default: factor_search)",
    )
    parser.add_argument(
        "--collection",
        default="factors",
        help="MongoDB collection name (default: factors)",
    )

    args = parser.parse_args()

    factors = load_factors_from_json(args.json_path)
    origin_factors = [f for f in factors if f.get("type", "origin") == "origin"]

    repo = FactorRepository(
        uri=args.mongo_uri, db_name=args.db_name, collection_name=args.collection
    )
    inserted = repo.insert_origin_factors(origin_factors)
    print(
        f"Inserted {inserted} origin factors into "
        f"{args.db_name}.{args.collection} (MongoDB URI: {args.mongo_uri})"
    )


if __name__ == "__main__":
    main()
