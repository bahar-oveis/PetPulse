"""
PetPulse accounts: registration and login with email and password.

- Passwords are hashed with Argon2id (argon2-cffi); the plain password is never stored.
- Emails are normalized (trimmed, lowercase) and unique.
- The role (owner or sitter) is chosen at registration and never changed by the app.
- Wrong email and wrong password give the same message, and repeated failures
  are rate limited.

This is a local MVP login. For a public deployment, use HTTPS and a managed
identity provider instead (see README).
"""

import re
import sqlite3
from datetime import datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

import database as db

ROLES = ("owner", "sitter")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
MAX_LOGIN_FAILURES = 5        # per email...
FAILURE_WINDOW_MINUTES = 15   # ...within this many minutes
LOGIN_ERROR = "Incorrect email or password."


def make_password_hasher():
    """Argon2id with argon2-cffi's recommended settings (64 MiB memory, 3 passes, 4 lanes)."""
    return PasswordHasher()


_hasher = make_password_hasher()
_dummy_hash = _hasher.hash("not a real password")  # keeps unknown-email logins as slow as real ones


class AuthError(Exception):
    """A registration or login problem the user should see."""


def normalize_email(email):
    return (email or "").strip().lower()


def _valid_email(email):
    return len(email) <= 254 and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) is not None


def register_user(email, password, role):
    """Create an account and return it as {"id", "email", "role", "created_at"}."""
    email = normalize_email(email)
    if not _valid_email(email):
        raise AuthError("Please enter a valid email address.")
    if role not in ROLES:
        raise AuthError("Please choose Pet Owner or Pet Sitter.")
    if not (MIN_PASSWORD_LENGTH <= len(password or "") <= MAX_PASSWORD_LENGTH):
        raise AuthError(f"Passwords must be {MIN_PASSWORD_LENGTH}–{MAX_PASSWORD_LENGTH} characters long.")
    try:
        user_id = db.create_user(email, _hasher.hash(password), role)
    except sqlite3.IntegrityError:
        raise AuthError("An account with this email already exists. Please log in instead.")
    return db.get_user(user_id)


def _since():
    return (datetime.now() - timedelta(minutes=FAILURE_WINDOW_MINUTES)).isoformat(sep=" ", timespec="seconds")


def login(email, password):
    """Check the email and password; return the user, or raise AuthError with a generic message."""
    email = normalize_email(email)
    if db.count_auth_failures("login", email, _since()) >= MAX_LOGIN_FAILURES:
        raise AuthError("Too many failed attempts. Please wait a few minutes and try again.")

    record = db.get_user_with_password(email)
    try:
        _hasher.verify(record["password_hash"] if record else _dummy_hash, password or "")
        if record is None:
            raise VerifyMismatchError()
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        db.add_auth_failure("login", email)
        raise AuthError(LOGIN_ERROR)

    db.clear_auth_failures("login", email)
    if _hasher.check_needs_rehash(record["password_hash"]):
        db.update_password_hash(record["id"], _hasher.hash(password))
    return db.get_user(record["id"])


def get_user(user_id):
    """The account for a signed-in session, re-read from the database (None if it no longer exists)."""
    return db.get_user(user_id) if user_id is not None else None
