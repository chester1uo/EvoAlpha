from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

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


@dataclass
class PersonaRepository:
    """
    Simple Mongo-backed repository for Persona objects.

    Stored document schema:

    {
        "name": <str>,
        "description": <str>,
        "meta": {...},
        "created_at": <datetime>,
        "updated_at": <datetime>
    }
    """

    uri: str
    db_name: str = "factor_search"
    collection_name: str = "personas"

    def __post_init__(self) -> None:
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.col = self.db[self.collection_name]
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        """Ensure a unique index on `name`."""
        self.col.create_index("name", unique=True)

    def insert_personas(self, personas: List[Dict[str, Any]]) -> int:
        """Insert or upsert a list of personas. Returns number inserted."""
        now = datetime.utcnow()
        inserted = 0
        for p in personas:
            name = p["name"]
            doc = {
                "name": name,
                "description": p.get("description", ""),
                "meta": p.get("meta", {}),
                "created_at": now,
                "updated_at": now,
            }
            res = self.col.update_one({"name": name}, {"$setOnInsert": doc}, upsert=True)
            if getattr(res, "upserted_id", None) is not None:
                inserted += 1
        return inserted

    def list_personas(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return personas sorted by name."""
        cursor = self.col.find({}, {"_id": False}).sort([("name", 1)]).limit(limit)
        return list(cursor)

    def get_persona(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a persona by name (returns None if missing)."""
        doc = self.col.find_one({"name": name}, {"_id": False})
        return doc

    def upsert_persona(self, persona: Dict[str, Any]) -> None:
        """Create or replace a persona document."""
        now = datetime.utcnow()
        name = persona["name"]
        doc = {
            "name": name,
            "description": persona.get("description", ""),
            "meta": persona.get("meta", {}),
            "created_at": now,
            "updated_at": now,
        }
        self.col.update_one({"name": name}, {"$set": doc}, upsert=True)

    def delete_persona(self, name: str) -> bool:
        """Delete a persona by name. Returns True if deleted."""
        res = self.col.delete_one({"name": name})
        return getattr(res, "deleted_count", 0) > 0
