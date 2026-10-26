"""
PetPulse access control: who may do what with which pet.

Every pet operation in the app and the agent goes through
require_pet_access(user_id, pet_id, operation), which checks the database on
every call, never the browser session alone:

- Owners may do everything with pets they own, and nothing with other pets.
- A sitter may view, chat about, use voice and vision for, and add memories to
  the one pet they currently have active access to. They can't edit the
  profile or manage access, and they can only delete memories they wrote.

Sitters get access by redeeming a single-use invitation code from the owner.
Only a SHA-256 hash of each code is stored. Rotating the code or revoking access
ends the current sitter's access at once: their next request is refused.
"""

import hashlib
import secrets
from datetime import datetime, timedelta

import database as db
import memory

VIEW = "view"
EDIT = "edit"
MANAGE_ACCESS = "manage_access"
CHAT = "chat"
VOICE = "voice"
VISION = "vision"
MEMORY_READ = "memory_read"
MEMORY_WRITE = "memory_write"
MEMORY_DELETE = "memory_delete"

OPERATIONS = {VIEW, EDIT, MANAGE_ACCESS, CHAT, VOICE, VISION, MEMORY_READ, MEMORY_WRITE, MEMORY_DELETE}
SITTER_OPERATIONS = {VIEW, CHAT, VOICE, VISION, MEMORY_READ, MEMORY_WRITE, MEMORY_DELETE}

INVITE_TTL_HOURS = 24
MAX_REDEEM_FAILURES = 5
FAILURE_WINDOW_MINUTES = 15
NO_ACCESS = "You don't have access to this pet."
INVALID_CODE = "This access code isn't valid. It may have expired, been used, or been replaced by a newer code."


class AccessDenied(Exception):
    """The user may not do this with this pet."""


def _now():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _hash_code(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _user(user_id, role=None):
    user = db.get_user(user_id) if isinstance(user_id, int) and not isinstance(user_id, bool) else None
    if user is None or (role is not None and user["role"] != role):
        raise AccessDenied(NO_ACCESS)
    return user


# ----- The central check -----

def require_pet_access(user_id, pet_id, operation):
    """
    Return the pet's current profile if user_id may perform operation on it;
    otherwise raise AccessDenied. Checked against the database on every call.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")
    user = _user(user_id)
    if not isinstance(pet_id, int) or isinstance(pet_id, bool):
        raise AccessDenied(NO_ACCESS)  # ids from widgets, URLs, or the model are never trusted as-is
    pet = db.get_pet(pet_id)
    if pet is None:
        raise AccessDenied(NO_ACCESS)  # same message as "not yours", so pet IDs can't be probed

    if user["role"] == "owner" and pet["owner_id"] == user["id"]:
        return pet
    if user["role"] == "sitter" and operation in SITTER_OPERATIONS and db.has_active_grant(pet_id, user["id"]):
        return pet
    raise AccessDenied(NO_ACCESS)


def has_pet_access(user_id, pet_id, operation):
    try:
        require_pet_access(user_id, pet_id, operation)
        return True
    except AccessDenied:
        return False


# ----- Owners: pets -----

def list_owner_pets(owner_id):
    return db.get_pets_for_owner(_user(owner_id, "owner")["id"])


def create_owner_pet(owner_id, data):
    return db.create_pet(data, owner_id=_user(owner_id, "owner")["id"])


def update_owner_pet(owner_id, pet_id, data):
    require_pet_access(owner_id, pet_id, EDIT)
    return db.update_pet(pet_id, data)


# ----- Owners: sitter access -----

def access_status(owner_id, pet_id):
    """Who has access to the pet now, and whether an unused invitation is waiting."""
    require_pet_access(owner_id, pet_id, MANAGE_ACCESS)
    grant = db.get_active_grant(pet_id=pet_id)
    sitter = db.get_user(grant["sitter_id"]) if grant else None
    return {
        "sitter_email": sitter["email"] if sitter else None,
        "granted_at": grant["granted_at"] if grant else None,
        "open_invite": db.get_open_invite(pet_id, _now()),
    }


def _new_invite(pet_id, revoke_grant):
    code = secrets.token_urlsafe(24)  # 32 characters, 192 random bits
    expires_at = (datetime.now() + timedelta(hours=INVITE_TTL_HOURS)).isoformat(sep=" ", timespec="seconds")
    db.replace_invite(pet_id, _hash_code(code), expires_at, revoke_grant=revoke_grant)
    return code


def create_invite(owner_id, pet_id):
    """
    Create an invitation code for a pet that has no sitter yet. Any earlier unused
    code stops working. Returns the code; it is shown once and never stored.
    """
    require_pet_access(owner_id, pet_id, MANAGE_ACCESS)
    if db.get_active_grant(pet_id=pet_id):
        raise AccessDenied("This pet already has a sitter. Rotate the key to replace them.")
    return _new_invite(pet_id, revoke_grant=False)


def rotate_invite(owner_id, pet_id):
    """End the current sitter's access, cancel unused codes, and return a new code."""
    require_pet_access(owner_id, pet_id, MANAGE_ACCESS)
    return _new_invite(pet_id, revoke_grant=True)


def revoke_access(owner_id, pet_id):
    """End the current sitter's access and cancel unused codes, without making a new one."""
    require_pet_access(owner_id, pet_id, MANAGE_ACCESS)
    db.revoke_pet_access(pet_id)


# ----- Sitters -----

def sitter_pet(sitter_id):
    """The pet this sitter currently has access to (fresh from the database), or None."""
    grant = db.get_active_grant(sitter_id=_user(sitter_id, "sitter")["id"])
    if grant is None:
        return None
    return require_pet_access(sitter_id, grant["pet_id"], VIEW)


def redeem_invite(sitter_id, code):
    """Redeem an invitation code. Returns the pet ID, or raises AccessDenied with a message to show."""
    sitter = _user(sitter_id, "sitter")
    subject = str(sitter["id"])
    since = (datetime.now() - timedelta(minutes=FAILURE_WINDOW_MINUTES)).isoformat(sep=" ", timespec="seconds")
    if db.count_auth_failures("redeem", subject, since) >= MAX_REDEEM_FAILURES:
        raise AccessDenied("Too many invalid codes. Please wait a few minutes and try again.")
    code = (code or "").strip()
    if not code:
        raise AccessDenied("Please enter the access code from the pet's owner.")
    try:
        pet_id = db.redeem_invite(_hash_code(code), sitter["id"], _now())
    except db.RedeemError as reason:
        if str(reason) == "sitter already assigned":
            raise AccessDenied("You already have access to a pet. Each sitter can care for one pet at a time.")
        db.add_auth_failure("redeem", subject)
        raise AccessDenied(INVALID_CODE)
    db.clear_auth_failures("redeem", subject)
    return pet_id


# ----- Pet Memory, with permission checks -----

def list_memories(user_id, pet_id):
    require_pet_access(user_id, pet_id, MEMORY_READ)
    return memory.get_pet_memories(pet_id)


def recall_memories(user_id, pet_id, query):
    require_pet_access(user_id, pet_id, MEMORY_READ)
    return memory.recall_pet_memories(pet_id, query)


def save_memory(user_id, pet_id, content):
    require_pet_access(user_id, pet_id, MEMORY_WRITE)
    return memory.save_pet_memory(pet_id, content, author_id=user_id)


def can_delete_memory(user_id, pet_id, item):
    """Owners may delete any of their pet's memories; sitters only the ones they wrote."""
    if not has_pet_access(user_id, pet_id, MEMORY_DELETE):
        return False
    return db.get_user(user_id)["role"] == "owner" or item.get("author_id") == user_id


def delete_memory(user_id, pet_id, memory_id):
    require_pet_access(user_id, pet_id, MEMORY_DELETE)
    item = next((m for m in memory.get_pet_memories(pet_id) if m["id"] == memory_id), None)
    if item is None or not can_delete_memory(user_id, pet_id, item):
        raise AccessDenied("You can only delete observations you recorded.")
    return memory.delete_pet_memory(pet_id, memory_id)
