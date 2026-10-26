"""
Test helpers: create accounts and give sitters access the same way the app does.

Imported by the test files (it isn't a test file itself). Password hashing uses
lighter Argon2 settings here so the tests run quickly; the app uses the
library's recommended settings.
"""

from argon2 import PasswordHasher

import access
import auth
import database as db

PASSWORD = "correct horse battery staple"

auth._hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
auth._dummy_hash = auth._hasher.hash("not a real password")


def make_user(email, role):
    return auth.register_user(email, PASSWORD, role)["id"]


def make_owner(email="owner@example.com"):
    return make_user(email, "owner")


def make_sitter(email="sitter@example.com"):
    return make_user(email, "sitter")


def make_pet(owner_id, data=None, **fields):
    return db.create_pet({**(data or {}), **fields}, owner_id=owner_id)


def grant(owner_id, pet_id, sitter_id):
    """The owner invites the sitter to the pet and the sitter redeems the code."""
    code = access.create_invite(owner_id, pet_id)
    access.redeem_invite(sitter_id, code)
    return code


def move_sitter(owner_id, from_pet_id, to_pet_id, sitter_id):
    """End the sitter's access to one pet and give them another (the owner's two pets)."""
    access.revoke_access(owner_id, from_pet_id)
    grant(owner_id, to_pet_id, sitter_id)
