"""
PetPulse Pet Memory: observations from earlier pet-sitting sessions.

Sitters save short notes ("Milo hid under the bed when the doorbell rang").
Each note is stored in SQLite with a local sentence-transformers embedding.
To recall, the question is embedded the same way and compared with that pet's
notes by cosine similarity (NumPy). Everything runs on this computer; the
embedding model is downloaded once, then loaded from the local cache.

Memories are unverified sitter observations, not owner-approved facts.
Every lookup is limited to one pet by the SQL queries in database.py.
"""

import re
import threading
import traceback

import numpy as np

import database as db

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.40  # lower scores are treated as unrelated
RELATIVE_MARGIN = 0.15       # also drop results this far below the best match
TOP_K = 5
MAX_LENGTH = 1000            # characters per observation
DEFAULT_SOURCE = "Sitter observation"

_model = None
_model_lock = threading.Lock()


class PetMemoryError(Exception):
    """A memory problem the sitter should see as a friendly message."""


def get_embedding_model():
    """Load the embedding model once (thread-safe) and reuse it."""
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer  # imported here: it takes a few seconds

            try:
                _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu", local_files_only=True)
            except Exception:
                # First run only: download the model (about 90 MB) from Hugging Face.
                _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return _model


PRONOUNS = r"\b(he|she|him|her|his|hers|it|its|they|them|their)\b"


def _comparable_text(text, pet_name):
    """
    Replace the pet's name and pronouns with "the pet" before embedding.
    Otherwise every note about Milo looks similar to every question about Milo
    just because of the name, and "he" in a question wouldn't match "Milo" in a note.
    """
    if pet_name:
        text = re.sub(r"\b" + re.escape(pet_name) + r"('s)?\b", "the pet", text, flags=re.IGNORECASE)
    return re.sub(PRONOUNS, "the pet", text, flags=re.IGNORECASE)


def embed(text, pet_name=None):
    """Return a normalized float32 embedding for text."""
    vector = get_embedding_model().encode(_comparable_text(text, pet_name), normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def _require_pet(pet_id):
    pet = db.get_pet(pet_id)
    if pet is None:
        raise PetMemoryError("This pet's profile couldn't be found.")
    return pet


def _public(row, score=None):
    memory = {key: row[key] for key in ("id", "content", "source", "created_at", "author_id")}
    if score is not None:
        memory["score"] = round(float(score), 3)
    return memory


def save_pet_memory(pet_id, content, source=DEFAULT_SOURCE, author_id=None):
    """Save an observation for one pet and return its ID. author_id is the account that wrote it."""
    content = (content or "").strip()
    if not content:
        raise PetMemoryError("Please write an observation before saving.")
    if len(content) > MAX_LENGTH:
        raise PetMemoryError(f"Please keep observations under {MAX_LENGTH} characters.")
    pet = _require_pet(pet_id)

    try:
        embedding = embed(content, pet["name"]).tobytes()
    except Exception:
        traceback.print_exc()  # no pet data is printed, only the error
        embedding = None       # saved anyway; recall embeds it later or falls back to keywords
    return db.add_memory(pet_id, content, source or DEFAULT_SOURCE, embedding, author_id=author_id)


def get_pet_memories(pet_id):
    """Return one pet's memories, oldest first."""
    return [_public(row) for row in db.get_memories(pet_id)]


def delete_pet_memory(pet_id, memory_id):
    """Delete a memory only if it belongs to this pet. Returns True if it was deleted."""
    return db.delete_memory(pet_id, memory_id)


def recall_pet_memories(pet_id, query, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD, margin=RELATIVE_MARGIN):
    """
    Return this pet's memories most related to query, best first, each with a score.
    Only memories scoring at least threshold, and within margin of the best one,
    are returned (possibly none).
    """
    query = (query or "").strip()
    if not query:
        return []
    pet = _require_pet(pet_id)
    rows = db.get_memories(pet_id, with_embeddings=True)  # only this pet's rows
    if not rows:
        return []

    try:
        query_vector = embed(query, pet["name"])
        vectors = np.stack([_memory_vector(pet, row, len(query_vector)) for row in rows])
        scores = vectors @ query_vector / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector) + 1e-9)
    except Exception:
        traceback.print_exc()
        scores = np.array([_keyword_score(query, row["content"], pet["name"]) for row in rows])

    ranked = sorted(zip(scores, rows), key=lambda pair: pair[0], reverse=True)
    cutoff = max(threshold, ranked[0][0] - margin)
    return [_public(row, score) for score, row in ranked[:top_k] if score >= cutoff]


def _memory_vector(pet, row, size):
    """The stored embedding, or a new one (saved for next time) if it was missing or damaged."""
    if row["embedding"]:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.shape == (size,):
            return vector
    vector = embed(row["content"], pet["name"])
    db.set_memory_embedding(pet["id"], row["id"], vector.tobytes())
    return vector


STOP_WORDS = set(
    "a an and are at be been did do does for from get gets got had has have he her him his how i if in is it "
    "its me my of on or she so that the their them they this to was were what when where which who why will "
    "with you your pet".split()
)


def _keyword_score(query, content, pet_name):
    """Fallback when the embedding model can't be used: share of the question's words found in the note."""
    def keywords(text):
        return {word for word in re.findall(r"[a-z']+", _comparable_text(text, pet_name).lower())
                if word not in STOP_WORDS and len(word) > 2}

    query_words = keywords(query)
    if not query_words:
        return 0.0
    return len(query_words & keywords(content)) / len(query_words)
