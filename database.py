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
    """Create the database file and the pets table if they don't exist."""
    with get_connection() as conn:
        conn.execute(CREATE_PETS_TABLE)
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
