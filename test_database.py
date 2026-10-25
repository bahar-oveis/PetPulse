"""
Tests for database.py.

Each test uses a temporary database file, so your real petpulse.db is never touched.
Run with:  python -m unittest -v
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import database as db


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def test_starts_empty(self):
        self.assertEqual(db.get_all_pets(), [])

    def test_init_db_can_run_twice(self):
        db.init_db()
        self.assertEqual(db.get_all_pets(), [])

    def test_create_and_get_pet(self):
        pet_id = db.create_pet({"name": "Luna", "species": "Cat", "breed": "Siamese", "age": 3})
        pet = db.get_pet(pet_id)
        self.assertEqual(pet["name"], "Luna")
        self.assertEqual(pet["species"], "Cat")
        self.assertEqual(pet["age"], 3)
        self.assertIsNotNone(pet["created_at"])
        self.assertEqual(pet["created_at"], pet["updated_at"])

    def test_blank_optional_fields_are_saved_as_none(self):
        pet_id = db.create_pet({"name": "Luna", "species": "Cat", "allergies": "   "})
        self.assertIsNone(db.get_pet(pet_id)["allergies"])

    def test_get_all_pets(self):
        db.create_pet({"name": "Rex", "species": "Dog"})
        db.create_pet({"name": "Bella", "species": "Cat"})
        names = [pet["name"] for pet in db.get_all_pets()]
        self.assertEqual(names, ["Bella", "Rex"])  # sorted by name

    def test_get_missing_pet_returns_none(self):
        self.assertIsNone(db.get_pet(999))

    def test_name_is_required(self):
        with self.assertRaises(ValueError):
            db.create_pet({"name": "  ", "species": "Dog"})

    def test_species_must_be_valid(self):
        with self.assertRaises(ValueError):
            db.create_pet({"name": "Nemo", "species": "Fish"})

    def test_update_pet(self):
        pet_id = db.create_pet({"name": "Rex", "species": "Dog", "diet": "Kibble"})
        # Pretend the pet was created a while ago, so we can see updated_at change.
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute("UPDATE pets SET updated_at = '2000-01-01 00:00:00' WHERE id = ?", (pet_id,))
        conn.close()

        self.assertTrue(db.update_pet(pet_id, {"diet": "Wet food", "fears": "Vacuum cleaner"}))

        pet = db.get_pet(pet_id)
        self.assertEqual(pet["diet"], "Wet food")
        self.assertEqual(pet["fears"], "Vacuum cleaner")
        self.assertEqual(pet["name"], "Rex")  # untouched fields stay the same
        self.assertNotEqual(pet["updated_at"], "2000-01-01 00:00:00")

    def test_update_missing_pet_returns_false(self):
        self.assertFalse(db.update_pet(999, {"diet": "Wet food"}))

    def test_unknown_fields_are_ignored(self):
        pet_id = db.create_pet({"name": "Rex", "species": "Dog", "id": 500, "favorite_color": "red"})
        self.assertNotEqual(pet_id, 500)
        self.assertNotIn("favorite_color", db.get_pet(pet_id))

    def test_demo_pet_added_only_once(self):
        db.add_demo_pet_if_empty()
        db.add_demo_pet_if_empty()
        pets = db.get_all_pets()
        self.assertEqual(len(pets), 1)
        self.assertEqual(pets[0]["name"], "Milo")
        self.assertEqual(pets[0]["allergies"], "Chicken")

    def test_demo_pet_not_added_when_pets_exist(self):
        db.create_pet({"name": "Rex", "species": "Dog"})
        db.add_demo_pet_if_empty()
        self.assertEqual([pet["name"] for pet in db.get_all_pets()], ["Rex"])


if __name__ == "__main__":
    unittest.main()
