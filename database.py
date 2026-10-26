"""
PetPulse database.

Stores pet profiles in a local SQLite file (petpulse.db) using Python's
built-in sqlite3 module. Every function returns plain dictionaries so the
data is easy to use in Streamlit and, later, by the AI agent.

These are low-level data functions without permission checks. The app and
the agent go through access.py, which checks who may do what first.
"""

import shutil
import sqlite3
from contextlib import contextmanager
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

# ----- Accounts and access control -----
# role is set when the account is created and never changed by the app.
CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('owner', 'sitter')),
    created_at TEXT NOT NULL
)
"""

# A sitter's access to one pet. Partial unique indexes below allow at most one
# active sitter per pet and one active pet per sitter.
CREATE_ACCESS_TABLE = """
CREATE TABLE IF NOT EXISTS pet_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pet_id INTEGER NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
    sitter_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
    granted_at TEXT NOT NULL,
    revoked_at TEXT
)
"""

# Invitation codes. Only a SHA-256 hash of the code is stored.
CREATE_INVITES_TABLE = """
CREATE TABLE IF NOT EXISTS pet_invites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pet_id INTEGER NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    used_by INTEGER REFERENCES users(id),
    revoked_at TEXT,
    created_at TEXT NOT NULL
)
"""

# Failed logins and code redemptions, for basic rate limiting.
CREATE_FAILURES_TABLE = """
CREATE TABLE IF NOT EXISTS auth_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

ACCESS_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS one_active_sitter_per_pet ON pet_access (pet_id) WHERE status = 'active'",
    "CREATE UNIQUE INDEX IF NOT EXISTS one_active_pet_per_sitter ON pet_access (sitter_id) WHERE status = 'active'",
    "CREATE INDEX IF NOT EXISTS idx_pet_invites_pet_id ON pet_invites (pet_id)",
    "CREATE INDEX IF NOT EXISTS idx_auth_failures_subject ON auth_failures (kind, subject, created_at)",
]

BACKUP_DIR = None  # None: a "backups" folder next to the database file

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


@contextmanager
def transaction():
    """
    A connection inside one write transaction (BEGIN IMMEDIATE): everything in
    the block is saved together or not at all, and no other writer can slip in.
    """
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def backup_database(reason):
    """Copy the database file into backups/ and return the copy's path (None if there's no file yet)."""
    if not Path(DB_PATH).exists():
        return None
    folder = Path(BACKUP_DIR) if BACKUP_DIR else Path(DB_PATH).parent / "backups"
    folder.mkdir(exist_ok=True)
    target = folder / f"{Path(DB_PATH).stem}-{reason}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.db"
    shutil.copy2(DB_PATH, target)
    return target


def init_db():
    """
    Create the database file and its tables if they don't exist, and add new
    columns to older databases. Safe to run every time; existing data is kept.
    Before an older database is changed, a backup copy is saved in backups/.
    """
    with get_connection() as conn:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        needs_migration = ("pets" in tables and "owner_id" not in _columns(conn, "pets")) or (
            "pet_memories" in tables and "author_id" not in _columns(conn, "pet_memories"))
    conn.close()
    if needs_migration:
        backup_database("before-migration")

    with get_connection() as conn:
        conn.execute(CREATE_PETS_TABLE)
        conn.execute(CREATE_MEMORIES_TABLE)
        conn.execute(CREATE_MEMORIES_INDEX)
        conn.execute(CREATE_USERS_TABLE)
        conn.execute(CREATE_ACCESS_TABLE)
        conn.execute(CREATE_INVITES_TABLE)
        conn.execute(CREATE_FAILURES_TABLE)
        # New columns for older databases. Existing pets keep owner_id NULL ("unassigned")
        # until someone assigns them explicitly with manage.py.
        if "owner_id" not in _columns(conn, "pets"):
            conn.execute("ALTER TABLE pets ADD COLUMN owner_id INTEGER REFERENCES users(id)")
        if "author_id" not in _columns(conn, "pet_memories"):
            conn.execute("ALTER TABLE pet_memories ADD COLUMN author_id INTEGER REFERENCES users(id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pets_owner_id ON pets (owner_id)")
        for statement in ACCESS_INDEXES:
            conn.execute(statement)
    conn.close()


def create_pet(data, owner_id=None):
    """Save a new pet profile and return its ID. owner_id is never taken from data."""
    pet = _clean(data)
    if "name" not in pet or "species" not in pet:
        raise ValueError("Pet name and species are required.")

    now = _now()
    pet["created_at"] = now
    pet["updated_at"] = now
    pet["owner_id"] = owner_id

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


def get_pets_for_owner(owner_id):
    """Return one owner's pets, sorted by name."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM pets WHERE owner_id = ? ORDER BY name COLLATE NOCASE", (owner_id,)
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_unassigned_pets():
    """Pets created before accounts existed (no owner yet)."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM pets WHERE owner_id IS NULL ORDER BY id").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def assign_pet_owner(pet_id, owner_id):
    """Give an unassigned pet to an owner (used only by manage.py). Returns True if it was unassigned."""
    with get_connection() as conn:
        cursor = conn.execute("UPDATE pets SET owner_id = ? WHERE id = ? AND owner_id IS NULL", (owner_id, pet_id))
        changed = cursor.rowcount > 0
    conn.close()
    return changed


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

def add_memory(pet_id, content, source, embedding, author_id=None):
    """Save one observation and return its ID. embedding is bytes or None."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO pet_memories (pet_id, content, source, created_at, embedding, author_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pet_id, content, source, _now(), embedding, author_id),
        )
        memory_id = cursor.lastrowid
    conn.close()
    return memory_id


def get_memories(pet_id, with_embeddings=False):
    """Return one pet's memories, oldest first, as dictionaries."""
    columns = "id, pet_id, content, source, created_at, author_id" + (", embedding" if with_embeddings else "")
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


# ----- Users -----

def create_user(email, password_hash, role):
    """Create an account and return its ID. Raises sqlite3.IntegrityError if the email is taken."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (email, password_hash, role, _now()),
        )
        user_id = cursor.lastrowid
    conn.close()
    return user_id


def get_user(user_id):
    with get_connection() as conn:
        row = conn.execute("SELECT id, email, role, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_with_password(email):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_password_hash(user_id, password_hash):
    """Store a re-hashed password (when the hashing settings get stronger)."""
    with get_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.close()


# ----- Rate limiting -----

def add_auth_failure(kind, subject):
    with get_connection() as conn:
        conn.execute("INSERT INTO auth_failures (kind, subject, created_at) VALUES (?, ?, ?)", (kind, subject, _now()))
    conn.close()


def count_auth_failures(kind, subject, since):
    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM auth_failures WHERE kind = ? AND subject = ? AND created_at >= ?",
            (kind, subject, since),
        ).fetchone()[0]
    conn.close()
    return count


def clear_auth_failures(kind, subject):
    with get_connection() as conn:
        conn.execute("DELETE FROM auth_failures WHERE kind = ? AND subject = ?", (kind, subject))
    conn.close()


# ----- Sitter access and invitations -----

def get_active_grant(pet_id=None, sitter_id=None):
    """The active access grant for a pet or for a sitter, or None."""
    column, value = ("pet_id", pet_id) if pet_id is not None else ("sitter_id", sitter_id)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM pet_access WHERE {column} = ? AND status = 'active'", (value,)
        ).fetchone()
    conn.close()
    return dict(row) if row else None


def has_active_grant(pet_id, sitter_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM pet_access WHERE pet_id = ? AND sitter_id = ? AND status = 'active'", (pet_id, sitter_id)
        ).fetchone()
    conn.close()
    return row is not None


def get_open_invite(pet_id, now):
    """The pet's unused, unrevoked, unexpired invitation (without the code), or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, pet_id, expires_at, created_at FROM pet_invites "
            "WHERE pet_id = ? AND used_at IS NULL AND revoked_at IS NULL AND expires_at > ? "
            "ORDER BY id DESC LIMIT 1",
            (pet_id, now),
        ).fetchone()
    conn.close()
    return dict(row) if row else None


def _revoke_pet(conn, pet_id, now, revoke_grant):
    conn.execute(
        "UPDATE pet_invites SET revoked_at = ? WHERE pet_id = ? AND used_at IS NULL AND revoked_at IS NULL",
        (now, pet_id),
    )
    if revoke_grant:
        conn.execute(
            "UPDATE pet_access SET status = 'revoked', revoked_at = ? WHERE pet_id = ? AND status = 'active'",
            (now, pet_id),
        )


def replace_invite(pet_id, token_hash, expires_at, revoke_grant):
    """
    In one transaction: cancel the pet's unused invitations, optionally end the
    current sitter's access, and store a new invitation.
    """
    now = _now()
    with transaction() as conn:
        _revoke_pet(conn, pet_id, now, revoke_grant)
        conn.execute(
            "INSERT INTO pet_invites (pet_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (pet_id, token_hash, expires_at, now),
        )


def revoke_pet_access(pet_id):
    """End the current sitter's access and cancel unused invitations, together."""
    with transaction() as conn:
        _revoke_pet(conn, pet_id, _now(), revoke_grant=True)


class RedeemError(Exception):
    """Why an invitation couldn't be redeemed (the app shows a generic message)."""


def redeem_invite(token_hash, sitter_id, now):
    """
    Atomically check an invitation and turn it into an access grant.
    Returns the pet ID, or raises RedeemError.
    """
    with transaction() as conn:
        invite = conn.execute("SELECT * FROM pet_invites WHERE token_hash = ?", (token_hash,)).fetchone()
        if invite is None:
            raise RedeemError("unknown")
        if invite["used_at"] is not None or invite["revoked_at"] is not None:
            raise RedeemError("used or revoked")
        if invite["expires_at"] <= now:
            raise RedeemError("expired")
        if conn.execute("SELECT 1 FROM pet_access WHERE sitter_id = ? AND status = 'active'", (sitter_id,)).fetchone():
            raise RedeemError("sitter already assigned")
        if conn.execute("SELECT 1 FROM pet_access WHERE pet_id = ? AND status = 'active'",
                        (invite["pet_id"],)).fetchone():
            raise RedeemError("pet already has a sitter")
        updated = conn.execute(
            "UPDATE pet_invites SET used_at = ?, used_by = ? WHERE id = ? AND used_at IS NULL AND revoked_at IS NULL",
            (now, sitter_id, invite["id"]),
        )
        if updated.rowcount != 1:
            raise RedeemError("used")
        conn.execute(
            "INSERT INTO pet_access (pet_id, sitter_id, status, granted_at) VALUES (?, ?, 'active', ?)",
            (invite["pet_id"], sitter_id, now),
        )
        return invite["pet_id"]
