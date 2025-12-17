import pytest

from factor_search.db.mongo import PersonaRepository

# pytest factor_search/db/tests/test_persona_repository.py
@pytest.fixture
def repo():
    """Provide a PersonaRepository pointing at a real local Mongo and cleanup after test.

    Uses a dedicated test database and collection to avoid touching production data.
    """
    repo = PersonaRepository(
        uri="mongodb://localhost:27017",
        db_name="factor_search_test",
        collection_name="personas_test",
    )
    # ensure clean state before test
    try:
        repo.db.drop_collection(repo.collection_name)
    except Exception:
        pass

    yield repo

    # cleanup after test
    try:
        repo.db.drop_collection(repo.collection_name)
    except Exception:
        pass
    try:
        repo.client.close()
    except Exception:
        pass


def test_insert_and_get_persona(repo):
    # insert one persona
    inserted = repo.insert_personas([{"name": "Tester", "description": "A test persona"}])
    assert inserted == 1

    p = repo.get_persona("Tester")
    assert p is not None
    assert p["name"] == "Tester"
    assert p["description"] == "A test persona"


def test_upsert_and_delete_persona(repo):
    # upsert (create)
    repo.upsert_persona({"name": "Upserter", "description": "first"})
    p1 = repo.get_persona("Upserter")
    assert p1 is not None and p1["description"] == "first"

    # upsert (update)
    repo.upsert_persona({"name": "Upserter", "description": "second"})
    p2 = repo.get_persona("Upserter")
    assert p2 is not None and p2["description"] == "second"

    # delete
    deleted = repo.delete_persona("Upserter")
    assert deleted is True
    assert repo.get_persona("Upserter") is None


def test_list_personas_ordering_and_limit(repo):
    items = [
        {"name": "B", "description": "b"},
        {"name": "A", "description": "a"},
        {"name": "C", "description": "c"},
    ]
    repo.insert_personas(items)

    listed = repo.list_personas(limit=2)
    # should be sorted by name A, B, C and limited to 2
    assert len(listed) == 2
    assert listed[0]["name"] == "A"
    assert listed[1]["name"] == "B"
