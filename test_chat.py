"""
Tests for chat.py.

A plain dictionary stands in for st.session_state, a fake client stands in for
Claude (no paid API calls), and each test uses a temporary database, so your
real petpulse.db is never touched.
Run with:  python -m unittest -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent
import chat
import database as db
import fixtures
from test_agent import FakeClient


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.owner = fixtures.make_owner()
        self.milo = db.get_pet(db.create_pet({"name": "Milo", "species": "Dog"}, owner_id=self.owner))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat"}, owner_id=self.owner))
        self.store = {}

        env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""})
        env.start()
        self.addCleanup(env.stop)
        quiet = mock.patch("agent.print", create=True)
        quiet.start()
        self.addCleanup(quiet.stop)
        self.client = FakeClient(reply="Milo eats dry food without chicken.")
        client_patch = mock.patch.object(agent, "_create_client", return_value=self.client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def test_new_chat_starts_with_welcome_message(self):
        history = chat.get_history(self.store, self.milo)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "assistant")
        self.assertIn("taking care of Milo", history[0]["content"])

    def test_send_message_adds_question_and_claude_reply(self):
        reply = chat.send_message(self.store, self.milo, "When does Milo eat?", user_id=self.owner)
        history = chat.get_history(self.store, self.milo)
        self.assertEqual([m["role"] for m in history], ["assistant", "user", "assistant"])
        self.assertEqual(history[1]["content"], "When does Milo eat?")
        self.assertEqual(history[2]["content"], reply)
        self.assertEqual(reply, "Milo eats dry food without chicken.")

    def test_question_is_sent_to_claude_once(self):
        chat.send_message(self.store, self.milo, "When does Milo eat?", user_id=self.owner)
        messages = self.client.requests[0]["messages"]
        self.assertEqual(messages, [{"role": "user", "content": "When does Milo eat?"}])

    def test_follow_up_question_includes_earlier_turn(self):
        chat.send_message(self.store, self.milo, "What food does Milo eat?", user_id=self.owner)
        chat.send_message(self.store, self.milo, "What should I avoid giving him?", user_id=self.owner)
        messages = self.client.requests[1]["messages"]
        self.assertEqual([m["content"] for m in messages], [
            "What food does Milo eat?",
            "Milo eats dry food without chicken.",
            "What should I avoid giving him?",
        ])

    def test_failed_reply_is_not_saved(self):
        self.client.error = RuntimeError("boom")
        with mock.patch("traceback.print_exc"):
            with self.assertRaises(chat.AgentError):
                chat.send_message(self.store, self.milo, "Hi", user_id=self.owner)
        history = chat.get_history(self.store, self.milo)
        self.assertEqual([m["role"] for m in history], ["assistant", "user"])  # question kept, no fake answer

    def test_cleared_chat_is_not_sent_to_claude(self):
        chat.send_message(self.store, self.milo, "Old question", user_id=self.owner)
        chat.clear_history(self.store, self.milo["id"])
        chat.send_message(self.store, self.milo, "New question", user_id=self.owner)
        self.assertEqual(self.client.requests[1]["messages"], [{"role": "user", "content": "New question"}])

    def test_each_pet_uses_its_own_profile(self):
        chat.send_message(self.store, self.milo, "Allergies?", user_id=self.owner)
        chat.send_message(self.store, self.luna, "Allergies?", user_id=self.owner)
        self.assertIn('"Milo"', self.client.requests[0]["system"])
        self.assertIn('"Luna"', self.client.requests[1]["system"])
        self.assertNotIn("Milo", self.client.requests[1]["system"])

    def test_pets_have_separate_conversations(self):
        chat.send_message(self.store, self.milo, "Question about Milo", user_id=self.owner)
        luna_history = chat.get_history(self.store, self.luna)
        self.assertEqual(len(luna_history), 1)  # only Luna's welcome message
        self.assertIn("Luna", luna_history[0]["content"])
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 3)

    def test_clear_only_affects_selected_pet(self):
        chat.send_message(self.store, self.milo, "Milo question", user_id=self.owner)
        chat.send_message(self.store, self.luna, "Luna question", user_id=self.owner)
        chat.clear_history(self.store, self.milo["id"])
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 1)  # fresh welcome
        self.assertEqual(len(chat.get_history(self.store, self.luna)), 3)

    def test_clear_unknown_pet_does_nothing(self):
        chat.clear_history(self.store, 999)
        self.assertEqual(self.store, {})

    def test_reply_uses_the_given_pet_id(self):
        chat.get_reply(self.luna["id"], "Hello", [], user_id=self.owner)
        self.assertIn('"Luna"', self.client.requests[0]["system"])

    def test_reply_for_missing_pet(self):
        with self.assertRaises(chat.AccessEnded) as caught:
            chat.get_reply(999, "Hello", [], user_id=self.owner)
        self.assertIn("don't have access", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
