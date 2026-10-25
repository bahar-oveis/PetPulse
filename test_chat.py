"""
Tests for chat.py.

A plain dictionary stands in for st.session_state, and each test uses a
temporary database, so your real petpulse.db is never touched.
Run with:  python -m unittest -v
"""

import tempfile
import unittest
from pathlib import Path

import chat
import database as db


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.milo = db.get_pet(db.create_pet({"name": "Milo", "species": "Dog"}))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat"}))
        self.store = {}

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def test_new_chat_starts_with_welcome_message(self):
        history = chat.get_history(self.store, self.milo)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "assistant")
        self.assertIn("taking care of Milo", history[0]["content"])

    def test_send_message_adds_question_and_placeholder_reply(self):
        reply = chat.send_message(self.store, self.milo, "When does Milo eat?")
        history = chat.get_history(self.store, self.milo)
        self.assertEqual([m["role"] for m in history], ["assistant", "user", "assistant"])
        self.assertEqual(history[1]["content"], "When does Milo eat?")
        self.assertEqual(history[2]["content"], reply)
        self.assertIn("Milo's profile", reply)

    def test_pets_have_separate_conversations(self):
        chat.send_message(self.store, self.milo, "Question about Milo")
        luna_history = chat.get_history(self.store, self.luna)
        self.assertEqual(len(luna_history), 1)  # only Luna's welcome message
        self.assertIn("Luna", luna_history[0]["content"])
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 3)

    def test_clear_only_affects_selected_pet(self):
        chat.send_message(self.store, self.milo, "Milo question")
        chat.send_message(self.store, self.luna, "Luna question")
        chat.clear_history(self.store, self.milo["id"])
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 1)  # fresh welcome
        self.assertEqual(len(chat.get_history(self.store, self.luna)), 3)

    def test_clear_unknown_pet_does_nothing(self):
        chat.clear_history(self.store, 999)
        self.assertEqual(self.store, {})

    def test_reply_uses_the_given_pet_id(self):
        reply = chat.get_reply(self.luna["id"], "Hello", [])
        self.assertIn("Luna's profile", reply)

    def test_reply_for_missing_pet(self):
        self.assertIn("couldn't find", chat.get_reply(999, "Hello", []))


if __name__ == "__main__":
    unittest.main()
