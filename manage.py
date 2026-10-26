"""
PetPulse admin commands, run on this computer only (not part of the web app).

    python manage.py create-user --email owner@example.com --role owner
    python manage.py list-unassigned
    python manage.py assign-pets --owner-email owner@example.com --pet-id 1 --pet-id 2
    python manage.py assign-pets --owner-email owner@example.com --all
    python manage.py seed-demo

Pets created before accounts existed have no owner and are visible to nobody.
assign-pets gives them to an owner account explicitly; it backs up the database
first and asks for confirmation (skip it with --yes).

Passwords are typed in hidden, or read from the PETPULSE_PASSWORD environment
variable for scripted demos. They are never printed or stored in plain text.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

import auth
import database as db

DEMO_ACCOUNTS = [
    ("owner.a@example.com", "owner"),
    ("owner.b@example.com", "owner"),
    ("sitter.a@example.com", "sitter"),
    ("sitter.b@example.com", "sitter"),
]

DEMO_LUNA = {
    "name": "Luna",
    "species": "Cat",
    "breed": "Siamese",
    "age": 3,
    "diet": "Wet cat food twice a day",
    "feeding_schedule": "7 AM and 7 PM",
    "allergies": "Fish",
    "fears": "The vacuum cleaner",
    "behavior_notes": "DEMO DATA: Luna is a fictional cat. All of her information is made up for testing.",
    "owner_contact": "Demo Owner B (fictional) - 555-0200",
    "vet_name": "Dr. Demo Vet (fictional)",
    "vet_contact": "555-0299 (fictional)",
}


def ask_password(prompt="Password: "):
    password = os.environ.get("PETPULSE_PASSWORD")
    if password:
        return password
    password = getpass.getpass(prompt)
    if getpass.getpass("Repeat password: ") != password:
        sys.exit("The passwords don't match.")
    return password


def confirm(question, assume_yes):
    if assume_yes:
        return True
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def find_owner(email):
    record = db.get_user_with_password(auth.normalize_email(email))
    if record is None or record["role"] != "owner":
        sys.exit(f"No owner account with the email {email!r}. Create it first with create-user.")
    return record


def create_user(args):
    try:
        user = auth.register_user(args.email, ask_password(), args.role)
    except auth.AuthError as error:
        sys.exit(str(error))
    print(f"Created {user['role']} account {user['email']} (id {user['id']}).")


def list_unassigned(args):
    pets = db.get_unassigned_pets()
    if not pets:
        print("Every pet has an owner.")
    for pet in pets:
        print(f"  id {pet['id']}: {pet['name']} ({pet['species']})")


def assign_pets(args):
    owner = find_owner(args.owner_email)
    unassigned = {pet["id"]: pet for pet in db.get_unassigned_pets()}
    chosen = list(unassigned) if args.all else args.pet_id
    if not chosen:
        sys.exit("Choose pets with --pet-id N (repeatable) or --all. See list-unassigned.")
    missing = [pet_id for pet_id in chosen if pet_id not in unassigned]
    if missing:
        sys.exit(f"These pet IDs don't exist or already have an owner: {missing}")

    names = ", ".join(f"{unassigned[pet_id]['name']} (id {pet_id})" for pet_id in chosen)
    if not confirm(f"Give {names} to {owner['email']}?", args.yes):
        sys.exit("Nothing changed.")
    backup = db.backup_database("before-assign")
    for pet_id in chosen:
        db.assign_pet_owner(pet_id, owner["id"])
    print(f"Assigned {names} to {owner['email']}. Backup: {backup}")


def seed_demo(args):
    """Fictional demo: Owner A owns Milo, Owner B owns Luna, two sitters without access yet."""
    password = ask_password("Password for all four demo accounts: ")
    for email, role in DEMO_ACCOUNTS:
        if db.get_user_with_password(email):
            print(f"  {email} already exists (password unchanged)")
            continue
        try:
            auth.register_user(email, password, role)
        except auth.AuthError as error:
            sys.exit(str(error))
        print(f"  created {role} {email}")

    owner_a, owner_b = find_owner("owner.a@example.com"), find_owner("owner.b@example.com")
    backup = db.backup_database("before-seed-demo")
    milo = next((pet for pet in db.get_unassigned_pets() if pet["name"].lower() == "milo"), None)
    if any(pet["name"].lower() == "milo" for pet in db.get_pets_for_owner(owner_a["id"])):
        print("  Milo already belongs to owner.a@example.com")
    elif milo and confirm(f"Give the existing unassigned pet Milo (id {milo['id']}) to owner.a@example.com?", args.yes):
        db.assign_pet_owner(milo["id"], owner_a["id"])
        print(f"  assigned existing Milo (id {milo['id']}) to owner.a@example.com")
    elif not milo:
        pet_id = db.create_pet(db.DEMO_PET, owner_id=owner_a["id"])
        print(f"  created demo Milo (id {pet_id}) for owner.a@example.com")
    if not any(pet["name"].lower() == "luna" for pet in db.get_pets_for_owner(owner_b["id"])):
        pet_id = db.create_pet(DEMO_LUNA, owner_id=owner_b["id"])
        print(f"  created demo Luna (id {pet_id}) for owner.b@example.com")
    if backup:
        print(f"Backup: {backup}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PetPulse admin commands (local only).")
    parser.add_argument("--db", help="database file (default: petpulse.db next to this script)")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("create-user", help="create an owner or sitter account")
    command.add_argument("--email", required=True)
    command.add_argument("--role", required=True, choices=auth.ROLES)
    command.set_defaults(run=create_user)

    command = commands.add_parser("list-unassigned", help="list pets that have no owner yet")
    command.set_defaults(run=list_unassigned)

    command = commands.add_parser("assign-pets", help="give unassigned (older) pets to an owner")
    command.add_argument("--owner-email", required=True)
    command.add_argument("--pet-id", type=int, action="append", default=[])
    command.add_argument("--all", action="store_true", help="every unassigned pet")
    command.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    command.set_defaults(run=assign_pets)

    command = commands.add_parser("seed-demo", help="create the fictional demo accounts, Milo, and Luna")
    command.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    command.set_defaults(run=seed_demo)

    args = parser.parse_args(argv)
    if args.db:
        db.DB_PATH = Path(args.db)
    db.init_db()
    args.run(args)


if __name__ == "__main__":
    main()
