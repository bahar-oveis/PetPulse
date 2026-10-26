"""
Tests for agent.py.

Claude is never called: a fake client stands in for the Anthropic SDK, so these
tests are free and work offline. Each test uses a temporary database.
Run with:  python -m unittest -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import anthropic
import httpx2

import agent
import database as db
import fixtures

REAL_CREATE_CLIENT = agent._create_client


def fake_response(text, stop_reason="end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=100, output_tokens=20)
    )


class FakeClient:
    """
    Pretends to be anthropic.Anthropic() and its tool runner.

    It records every request, so tests can check exactly what Claude would see,
    and returns a fixed reply (or raises a fixed error). If memory_searches is
    set, it first calls the memory tool with those queries, like Claude would,
    and records what the tool returned.
    """

    def __init__(self, reply="Here is my answer.", error=None, stop_reason="end_turn", memory_searches=()):
        self.reply = reply
        self.error = error
        self.stop_reason = stop_reason
        self.memory_searches = list(memory_searches)
        self.before_reply = None  # optional hook, called while "Claude is thinking"
        self.requests = []
        self.tool_results = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        responses = []
        for query in self.memory_searches:
            responses.append(fake_response(None, "tool_use"))
            self.tool_results.append(request["tools"][0].call({"query": query}))
        if self.before_reply:
            self.before_reply()
        responses.append(fake_response(self.reply, self.stop_reason))
        return iter(responses)


def api_error(error_class, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return error_class("secret internal details", response=httpx2.Response(status, request=request), body=None)


class AgentTestCase(unittest.TestCase):
    """Sets up a temporary database with two pets and a fake Claude client."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.owner = fixtures.make_owner()
        self.milo_id = db.create_pet({**db.DEMO_PET}, owner_id=self.owner)
        self.luna_id = db.create_pet({"name": "Luna", "species": "Cat", "allergies": "Fish"}, owner_id=self.owner)

        env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""})
        env.start()
        self.addCleanup(env.stop)
        quiet = mock.patch("agent.print", create=True)  # hide the token-usage line
        quiet.start()
        self.addCleanup(quiet.stop)

        self.client = FakeClient()
        patcher = mock.patch.object(agent, "_create_client", return_value=self.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()


class ProfileTests(AgentTestCase):
    def test_system_prompt_contains_full_profile(self):
        system = agent.build_system_prompt(db.get_pet(self.milo_id))
        for text in ["Milo", "Golden Retriever", "Chicken", "Loud noises", "Thunderstorms",
                     "seizure", "Emergency instructions", "Owner contact", "Veterinarian contact"]:
            self.assertIn(text, system)

    def test_missing_fields_are_marked_not_recorded(self):
        system = agent.build_system_prompt(db.get_pet(self.luna_id))
        self.assertIn(f'"Medications": "{agent.NOT_RECORDED}"', system)

    def test_profiles_of_different_pets_stay_separate(self):
        luna_system = agent.build_system_prompt(db.get_pet(self.luna_id))
        self.assertIn("Fish", luna_system)
        self.assertNotIn("Milo", luna_system)
        self.assertNotIn("Chicken", luna_system)


class BuildMessagesTests(unittest.TestCase):
    def test_welcome_message_is_dropped_and_question_added_once(self):
        history = [{"role": "assistant", "content": "Hi! I'm PetPulse..."}]
        self.assertEqual(
            agent.build_messages(history, "When does Milo eat?"),
            [{"role": "user", "content": "When does Milo eat?"}],
        )

    def test_follow_up_keeps_earlier_turns(self):
        history = [
            {"role": "assistant", "content": "Welcome"},
            {"role": "user", "content": "What food does Milo eat?"},
            {"role": "assistant", "content": "Dry food without chicken."},
        ]
        messages = agent.build_messages(history, "What should I avoid?")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "What should I avoid?")

    def test_consecutive_user_messages_are_merged(self):
        # e.g. an earlier question whose answer failed
        history = [{"role": "user", "content": "First question"}]
        messages = agent.build_messages(history, "Second question")
        self.assertEqual(messages, [{"role": "user", "content": "First question\n\nSecond question"}])

    def test_history_is_bounded(self):
        history = []
        for i in range(30):
            history += [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]
        messages = agent.build_messages(history, "Latest")
        self.assertEqual(len(messages), agent.HISTORY_TURNS * 2 + 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Q20")
        self.assertEqual(messages[-1]["content"], "Latest")

    def test_does_not_change_the_apps_history(self):
        history = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        agent.build_messages(history, "Next")
        agent.build_messages(history, "Next")
        self.assertEqual(history, [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])


class AskTests(AgentTestCase):
    def test_returns_claudes_answer(self):
        answer = agent.ask(self.milo_id, "Can I give Milo chicken?", [], user_id=self.owner)
        self.assertEqual(answer, "Here is my answer.")

    def test_one_request_with_profile_and_memory_tool(self):
        agent.ask(self.milo_id, "Can I give Milo chicken?", [], user_id=self.owner)
        self.assertEqual(len(self.client.requests), 1)
        request = self.client.requests[0]
        self.assertIn("<pet_profile>", request["system"])
        self.assertIn("Chicken", request["system"])
        self.assertEqual([tool.name for tool in request["tools"]], ["recall_pet_memories"])
        self.assertEqual(request["max_iterations"], agent.MAX_ITERATIONS)

    def test_request_settings(self):
        agent.ask(self.milo_id, "Hi", [], user_id=self.owner)
        request = self.client.requests[0]
        self.assertEqual(request["model"], "claude-haiku-5-5")
        self.assertEqual(request["max_tokens"], agent.MAX_TOKENS)
        self.assertIn("PetPulse Assistant", request["system"])
        self.assertEqual(request["messages"], [{"role": "user", "content": "Hi"}])

    def test_model_can_be_configured(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-sonnet-5-5"}):
            agent.ask(self.milo_id, "Hi", [], user_id=self.owner)
        self.assertEqual(self.client.requests[0]["model"], "claude-sonnet-5-5")

    def test_only_selected_pet_is_sent_to_claude(self):
        agent.ask(self.milo_id, "Does Luna have allergies?", [], user_id=self.owner)
        system = self.client.requests[0]["system"]
        self.assertIn("Milo", system)
        self.assertNotIn("Fish", system)
        self.assertNotIn('"Luna"', system)

    def test_latest_profile_is_loaded_for_each_question(self):
        agent.ask(self.milo_id, "Allergies?", [], user_id=self.owner)
        db.update_pet(self.milo_id, {"allergies": "Chicken and beef"})
        agent.ask(self.milo_id, "Allergies now?", [], user_id=self.owner)
        self.assertNotIn("Chicken and beef", self.client.requests[0]["system"])
        self.assertIn("Chicken and beef", self.client.requests[1]["system"])

    def test_missing_pet_does_not_call_claude(self):
        with self.assertRaises(agent.AgentError):
            agent.ask(999, "Hi", [], user_id=self.owner)
        self.assertEqual(self.client.requests, [])

    def test_database_error_is_friendly_and_does_not_call_claude(self):
        with mock.patch.object(db, "get_pet", side_effect=RuntimeError("disk path /secret")), \
                mock.patch("traceback.print_exc"):
            with self.assertRaises(agent.AgentError) as caught:
                agent.ask(self.milo_id, "Hi", [], user_id=self.owner)
        self.assertNotIn("/secret", str(caught.exception))
        self.assertEqual(self.client.requests, [])

    def test_missing_api_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), \
                mock.patch.object(agent, "_create_client", REAL_CREATE_CLIENT):
            with self.assertRaises(agent.AgentError) as caught:
                agent.ask(self.milo_id, "Hi", [], user_id=self.owner)
        self.assertIn("ANTHROPIC_API_KEY", str(caught.exception))

    def test_api_failures_become_friendly_errors(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        errors = [
            api_error(anthropic.AuthenticationError, 401),
            api_error(anthropic.PermissionDeniedError, 403),
            api_error(anthropic.NotFoundError, 404),
            api_error(anthropic.RateLimitError, 429),
            api_error(anthropic.InternalServerError, 500),
            anthropic.APIConnectionError(request=request),
            RuntimeError("unexpected secret internal details"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.client.error = error
                with mock.patch("traceback.print_exc"):
                    with self.assertRaises(agent.AgentError) as caught:
                        agent.ask(self.milo_id, "Hi", [], user_id=self.owner)
                self.assertNotIn("secret", str(caught.exception))

    def test_empty_answer_is_an_error(self):
        self.client.reply = "   "
        with self.assertRaises(agent.AgentError):
            agent.ask(self.milo_id, "Hi", [], user_id=self.owner)

    def test_refusal_is_an_error(self):
        self.client.stop_reason = "refusal"
        with self.assertRaises(agent.AgentError):
            agent.ask(self.milo_id, "Hi", [], user_id=self.owner)


if __name__ == "__main__":
    unittest.main()
