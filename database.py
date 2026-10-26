"""
PetPulse database.

Stores pet profiles in a local SQLite file (petpulse.db) using Python's
built-in sqlite3 module. Every function returns plain dictionaries so the
data is easy to use in Streamlit and, later, by the AI agent.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# The database file lives next to this script.
DB_PATH = Path(__file__).parent / "petpulse.db"

SPECIES_OPTIONS = ["Dog", "Cat", "Other"]

# All pet fields an owner can fill in (everything except id and timestamps).
PET_FIELDS = [
    # Basic information
    "name",
    "species",
    "breed",
    "age",
    # Diet and routine
    "diet",
    "feeding_schedule",
    "daily_routine",
    "habits",
    # Behavior and sensitivities
    "allergies",
    "sensitivities",
    "fears",
    "behavior_notes",
    # Medical information
    "medical_conditions",
    "medications",
    "medical_history",
    # Emergency information
    "emergency_instructions",
    "owner_contact",
    "vet_name",
    "vet_contact",
]

CREATE_PETS_TABLE = """
CREATE TABLE IF NOT EXISTS pets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    species TEXT NOT NULL CHECK (species IN ('Dog', 'Cat', 'Other')),
    breed TEXT,
    age INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    diet TEXT,
    feeding_schedule TEXT,
    daily_routine TEXT,
    habits TEXT,

    allergies TEXT,
    sensitivities TEXT,
    fears TEXT,
    behavior_notes TEXT,

    medical_conditions TEXT,
    medications TEXT,
    medical_history TEXT,

    emergency_instructions TEXT,
    owner_contact TEXT,
    vet_name TEXT,
    vet_contact TEXT
)
"""

# Observations saved by sitters (Pet Memory). Each belongs to exactly one pet.
# embedding holds the local sentence-transformers vector (float32 bytes), or NULL.
CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS pet_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pet_id INTEGER NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL,
    embedding BLOB
)
"""

CREATE_MEMORIES_INDEX = "CREATE INDEX IF NOT EXISTS idx_pet_memories_pet_id ON pet_memories (pet_id)"

# Fictional demo pet, inserted only when the database is empty.
DEMO_PET = {
    "name": "Milo",
    "species": "Dog",
    "breed": "Golden Retriever",
    "age": 5,
    "diet": "Dry dog food without chicken",
    "feeding_schedule": "8 AM and 6 PM",
    "habits": "Likes playing with a tennis ball and sleeping with a blue blanket",
    "allergies": "Chicken",
    "sensitivities": "Loud noises",
    "fears": "Thunderstorms",
    "behavior_notes": "DEMO DATA: Milo is a fictional pet. All of his information is made up for testing.",
    "medical_history": "Previous seizure episode (fictional demo data)",
    "emergency_instructions": (
        "If a seizure occurs, keep Milo safe from nearby hazards "
        "and seek veterinary assistance immediately."
    ),
    "owner_contact": "Demo Owner (fictional) - 555-0100",
    "vet_name": "Dr. Demo Vet (fictional)",
    "vet_contact": "555-0199 (fictional)",
}


def get_connection():
    """Open a connection that returns rows which behave like dictionaries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _clean(data):
    """Keep only known pet fields, trim text, and store blank values as None."""
    cleaned = {}
    for field in PET_FIELDS:
        if field not in data:
            continue
        value = data[field]
        if isinstance(value, str):
            value = value.strip() or None
        cleaned[field] = value

    if "name" in cleaned and not cleaned["name"]:
        raise ValueError("Pet name is required.")
    if "species" in cleaned and cleaned["species"] not in SPECIES_OPTIONS:
        raise ValueError(f"Species must be one of: {', '.join(SPECIES_OPTIONS)}.")
    return cleaned


def init_db():
    """Create the database file and its tables if they don't exist. Safe to run every time."""
    with get_connection() as conn:
        conn.execute(CREATE_PETS_TABLE)
        conn.execute(CREATE_MEMORIES_TABLE)
        conn.execute(CREATE_MEMORIES_INDEX)
    conn.close()


def create_pet(data):
    """Save a new pet profile and return its ID."""
    pet = _clean(data)
    if "name" not in pet or "species" not in pet:
        raise ValueError("Pet name and species are required.")

    now = _now()
    pet["created_at"] = now
    pet["updated_at"] = now

    columns = ", ".join(pet.keys())
    placeholders = ", ".join("?" for _ in pet)
    with get_connection() as conn:
        cursor = conn.execute(
            f"INSERT INTO pets ({columns}) VALUES ({placeholders})",
            list(pet.values()),
        )
        pet_id = cursor.lastrowid
    conn.close()
    return pet_id


def get_all_pets():
    """Return all pet profiles as a list of dictionaries, sorted by name."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM pets ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_pet(pet_id):
    """Return one pet profile as a dictionary, or None if it doesn't exist."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM pets WHERE id = ?", (pet_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_pet(pet_id, data):
    """Update a pet profile. Returns True if the pet was found and updated."""
    pet = _clean(data)
    pet["updated_at"] = _now()

    assignments = ", ".join(f"{column} = ?" for column in pet)
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE pets SET {assignments} WHERE id = ?",
            [*pet.values(), pet_id],
        )
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def add_demo_pet_if_empty():
    """Insert the fictional demo pet Milo, but only if there are no pets yet."""
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM pets").fetchone()[0]
    conn.close()
    if count == 0:
        create_pet(DEMO_PET)


# ----- Pet Memory (used by memory.py) -----
# Every query filters by pet_id, so one pet's memories are never read or changed
# through another pet.

def add_memory(pet_id, content, source, embedding):
    """Save one observation and return its ID. embedding is bytes or None."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO pet_memories (pet_id, content, source, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (pet_id, content, source, _now(), embedding),
        )
        memory_id = cursor.lastrowid
    conn.close()
    return memory_id


def get_memories(pet_id, with_embeddings=False):
    """Return one pet's memories, oldest first, as dictionaries."""
    columns = "id, pet_id, content, source, created_at" + (", embedding" if with_embeddings else "")
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT {columns} FROM pet_memories WHERE pet_id = ? ORDER BY created_at, id", (pet_id,)
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def set_memory_embedding(pet_id, memory_id, embedding):
    """Store an embedding that was missing (e.g. the model wasn't available when it was saved)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE pet_memories SET embedding = ? WHERE id = ? AND pet_id = ?", (embedding, memory_id, pet_id)
        )
    conn.close()


def delete_memory(pet_id, memory_id):
    """Delete a memory only if it belongs to this pet. Returns True if one was deleted."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM pet_memories WHERE id = ? AND pet_id = ?", (memory_id, pet_id))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted
