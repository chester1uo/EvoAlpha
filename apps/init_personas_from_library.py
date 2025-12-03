"""
Initialization script: batch import PERSONA_LIBRARY from personas.py into MongoDB.
"""

from factor_search.personas import PERSONA_LIBRARY
from factor_search.db import PersonaRepository

# You can modify the MongoDB connection parameters as needed
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "factor_search"
COLLECTION_NAME = "personas"

def main():
    repo = PersonaRepository(
        uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
    )
    # Convert to list of dicts
    personas = [p.__dict__ for p in PERSONA_LIBRARY]
    inserted = repo.insert_personas(personas)
    print(f"insert {inserted} personas to MongoDB ({DB_NAME}.{COLLECTION_NAME})")

if __name__ == "__main__":
    main()
