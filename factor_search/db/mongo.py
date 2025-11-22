from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from pymongo import MongoClient, DESCENDING


@dataclass
class FactorRepository:
    """
    Thin wrapper over a MongoDB collection that stores factors.

    Document schema (per factor):

    {
        "name": <str>,                     # unique identifier
        "expression": <str>,               # Qlib expression
        "type": "origin" | "search",       # origin or searched factor
        "operations": {                    # only for searched factors
            "type": "mutation" | "crossover",
            ...                            # e.g., from_A, from_B, notes, etc.
        } | null,
        "metrics": {                       # performance metrics
            "ic": <float>,
            "rank_ic": <float>,
            "icir": <float>,
            "winrate": <float>,
            "stability": <float>,
            ...
        },
        "tags": { ... },
        "provenance": { ... },
        "created_at": <datetime>,
        "updated_at": <datetime>
    }
    """

    uri: str
    db_name: str = "factor_search"
    collection_name: str = "factors"

    def __post_init__(self) -> None:
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.col = self.db[self.collection_name]
        self.ensure_indexes()

    # ------------------------------------------------------------------ #
    # Indexes
    # ------------------------------------------------------------------ #

    def ensure_indexes(self) -> None:
        """
        Create a few helpful indexes if they do not exist already.
        """
        self.col.create_index("name", unique=True)
        self.col.create_index([("metrics.ic", DESCENDING)])
        self.col.create_index("type")

    # ------------------------------------------------------------------ #
    # Basic operations
    # ------------------------------------------------------------------ #

    def insert_origin_factors(self, factors: List[Dict[str, Any]]) -> int:
        """
        Insert or upsert origin factors as the initial dataset.

        Only the name and expression fields are required; other fields are optional.
        If a factor with the same name already exists, it is left unchanged.
        """
        now = datetime.utcnow()
        inserted = 0

        for f in factors:
            name = f["name"]
            expr = f["expression"]
            # origin factors have no operations by design
            doc = {
                "name": name,
                "expression": expr,
                "type": "origin",
                "meta": f.get("meta", {"type": "origin"}),
                "metrics": f.get("metrics", {}),
                "tags": f.get("tags", {}),
                "provenance": f.get("provenance", {}),
                "created_at": now,
                "updated_at": now,
            }
            res = self.col.update_one(
                {"name": name},
                {"$setOnInsert": doc},
                upsert=True,
            )
            if res.upserted_id is not None:
                inserted += 1

        return inserted

    def get_seeds(self, limit: int = 100, include_search: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch seed factors sorted by metrics.ic descending (falling back to 0).
        By default it returns both origin and previously accepted search factors.
        """
        types = ["origin"]
        if include_search:
            types.append("search")

        cursor = (
            self.col.find({"type": {"$in": types}}, {"_id": False})
            .sort([("metrics.ic", -1), ("name", 1)])
            .limit(limit)
        )
        return list(cursor)

    def update_metrics_bulk(self, factors: List[Dict[str, Any]]) -> None:
        """
        Update metrics for a list of factors by name.
        """
        now = datetime.utcnow()
        for f in factors:
            name = f["name"]
            metrics = f.get("metrics", {})
            self.col.update_one(
                {"name": name},
                {"$set": {"metrics": metrics, "updated_at": now}},
                upsert=False,
            )

    def store_search_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Upsert searched factors with full metadata.
        """
        now = datetime.utcnow()
        for r in results:
            name = r["name"]
            expr = r["expression"]
            doc = {
                "name": name,
                "expression": expr,
                "type": r.get("type", "search"),
                "meta": r.get("meta", {"type": "search"}),
                "metrics": r.get("metrics", {}),
                "tags": r.get("tags", {}),
                "provenance": r.get("provenance", {}),
                "created_at": now,
                "updated_at": now,
            }
            self.col.update_one(
                {"name": name},
                {"$set": doc},
                upsert=True,
            )
