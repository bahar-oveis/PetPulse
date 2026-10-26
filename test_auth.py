"""
Tests for accounts, access control, invitations, and revocation
(auth.py, access.py, the access checks in the agent and app, and manage.py).

Claude is mocked (no paid API calls); every test uses a temporary database.
Run with:  python -m unittest -v
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import access
import agent
import auth
import chat
import database as db
import fixtures
import live_voice
import manage
import memory
import vision
from test_agent import FakeClient
from test_live_voice import FakeClock, FakeSpeaker, fake_vad, silence, speech
from test_memory import FakeEmbeddingModel

APP = str(Path(__file__).parent / "app.py")


class SecurityTestCase(unittest.TestCase):
    """Owner A owns Milo, Owner B owns Luna; Sitter A and Sitter B have no access yet."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.owner_a = fixtures.make_owner("owner.a@example.com")
        self.owner_b = fixtures.make_owner("owner.b@example.com")
        self.sitter_a = fixtures.make_sitter("sitter.a@example.com")
        self.sitter_b = fixtures.make_sitter("sitter.b@example.com")
        self.milo = fixtures.make_pet(self.owner_a, db.DEMO_PET)
        self.luna = fixtures.make_pet(self.owner_b, name="Luna", species="Cat", allergies="Fish")

        self.client = FakeClient(reply="Milo eats dry food without chicken.")
        for patcher in (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""}),
            mock.patch("agent.print", create=True),
            mock.patch.object(agent, "_create_client", return_value=self.client),
            mock.patch.object(memory, "get_embedding_model", return_value=FakeEmbeddingModel()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def assert_denied(self, user_id, pet_id, operation):
        with self.assertRaises(access.AccessDenied):
            access.require_pet_access(user_id, pet_id, operation)


# ----- 1–6: accounts and passwords -----

class AccountTests(SecurityTestCase):
    def test_owner_registration(self):
        user = auth.register_user("  New.Owner@Example.COM ", fixtures.PASSWORD, "owner")
        self.assertEqual((user["email"], user["role"]), ("new.owner@example.com", "owner"))
        self.assertIsInstance(user["id"], int)

    def test_sitter_registration(self):
        user = auth.register_user("new.sitter@example.com", fixtures.PASSWORD, "sitter")
        self.assertEqual(user["role"], "sitter")
        self.assertNotIn("password_hash", user)  # account details never include the hash

    def test_correct_login(self):
        user = auth.login(" OWNER.A@example.com", fixtures.PASSWORD)
        self.assertEqual(user["id"], self.owner_a)

    def test_incorrect_password_and_unknown_email_look_the_same(self):
        for email, password in (("owner.a@example.com", "wrong password"), ("nobody@example.com", fixtures.PASSWORD)):
            with self.assertRaises(auth.AuthError) as caught:
                auth.login(email, password)
            self.assertEqual(str(caught.exception), auth.LOGIN_ERROR)

    def test_duplicate_email(self):
        with self.assertRaises(auth.AuthError):
            auth.register_user("Owner.A@Example.com", fixtures.PASSWORD, "sitter")

    def test_password_is_hashed_with_argon2id(self):
        auth.register_user("twin1@example.com", fixtures.PASSWORD, "owner")
        auth.register_user("twin2@example.com", fixtures.PASSWORD, "owner")
        with sqlite3.connect(db.DB_PATH) as conn:
            hashes = [row[0] for row in conn.execute("SELECT password_hash FROM users WHERE email LIKE 'twin%'")]
        conn.close()
        self.assertTrue(all(h.startswith("$argon2id$") for h in hashes))
        self.assertNotEqual(hashes[0], hashes[1])  # salted
        self.assertTrue(all(fixtures.PASSWORD not in h for h in hashes))
        self.assertNotIn(fixtures.PASSWORD.encode(), Path(db.DB_PATH).read_bytes())

    def test_app_uses_recommended_argon2_settings(self):
        # The tests use lighter settings for speed (fixtures.py); the app uses these.
        self.assertTrue(auth.make_password_hasher().hash("x").startswith("$argon2id$v=19$m=65536,t=3,p=4$"))

    def test_validation(self):
        for email, password, role in (("not-an-email", fixtures.PASSWORD, "owner"),
                                      ("ok@example.com", "short", "owner"),
                                      ("ok@example.com", fixtures.PASSWORD, "admin")):
            with self.assertRaises(auth.AuthError):
                auth.register_user(email, password, role)

    def test_login_rate_limit(self):
        for _ in range(auth.MAX_LOGIN_FAILURES):
            with self.assertRaises(auth.AuthError):
                auth.login("owner.a@example.com", "wrong password")
        with self.assertRaises(auth.AuthError) as caught:
            auth.login("owner.a@example.com", fixtures.PASSWORD)  # even the right password waits
        self.assertIn("Too many", str(caught.exception))


# ----- 7–8, 18: owners and pet IDs -----

class OwnerIsolationTests(SecurityTestCase):
    def test_owner_sees_only_own_pets(self):
        self.assertEqual([p["name"] for p in access.list_owner_pets(self.owner_a)], ["Milo"])
        self.assertEqual([p["name"] for p in access.list_owner_pets(self.owner_b)], ["Luna"])
        self.assert_denied(self.owner_a, self.luna, access.VIEW)

    def test_owner_cannot_edit_other_owners_pet(self):
        with self.assertRaises(access.AccessDenied):
            access.update_owner_pet(self.owner_a, self.luna, {"allergies": "None"})
        self.assertEqual(db.get_pet(self.luna)["allergies"], "Fish")
        with self.assertRaises(access.AccessDenied):
            access.create_invite(self.owner_a, self.luna)

    def test_owner_id_cannot_be_set_through_pet_data(self):
        pet_id = access.create_owner_pet(self.owner_a, {"name": "Rex", "species": "Dog", "owner_id": self.owner_b})
        self.assertEqual(db.get_pet(pet_id)["owner_id"], self.owner_a)
        access.update_owner_pet(self.owner_a, pet_id, {"owner_id": self.owner_b, "breed": "Lab"})
        self.assertEqual(db.get_pet(pet_id)["owner_id"], self.owner_a)

    def test_changed_or_invalid_pet_ids_are_refused(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        for bad_id in (self.luna, str(self.milo), True, None, 999, -1):
            self.assert_denied(self.sitter_a, bad_id, access.VIEW)
        for bad_user in (None, "1", 999, True):
            self.assert_denied(bad_user, self.milo, access.VIEW)
        with self.assertRaises(chat.AccessEnded):
            chat.get_reply(self.luna, "What does Luna eat?", [], user_id=self.sitter_a)
        self.assertEqual(self.client.requests, [])
        with self.assertRaises(ValueError):
            access.require_pet_access(self.owner_a, self.milo, "delete_everything")

    def test_unassigned_legacy_pets_are_visible_to_nobody(self):
        legacy = db.create_pet({"name": "Old", "species": "Dog"})  # owner_id NULL
        for user in (self.owner_a, self.owner_b, self.sitter_a):
            self.assert_denied(user, legacy, access.VIEW)


# ----- 9–14: invitations -----

class InvitationTests(SecurityTestCase):
    def test_no_access_without_invitation(self):
        self.assertIsNone(access.sitter_pet(self.sitter_a))
        self.assert_denied(self.sitter_a, self.milo, access.VIEW)
        self.assert_denied(self.sitter_a, self.milo, access.CHAT)

    def test_valid_invitation_grants_exactly_one_pet(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        self.assertEqual(access.sitter_pet(self.sitter_a)["name"], "Milo")
        for operation in (access.VIEW, access.CHAT, access.VOICE, access.VISION, access.MEMORY_READ):
            access.require_pet_access(self.sitter_a, self.milo, operation)
        self.assert_denied(self.sitter_a, self.luna, access.VIEW)
        self.assertEqual(access.access_status(self.owner_a, self.milo)["sitter_email"], "sitter.a@example.com")

    def test_sitter_has_limited_permissions(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        self.assert_denied(self.sitter_a, self.milo, access.EDIT)
        self.assert_denied(self.sitter_a, self.milo, access.MANAGE_ACCESS)
        with self.assertRaises(access.AccessDenied):
            access.update_owner_pet(self.sitter_a, self.milo, {"allergies": "None"})
        with self.assertRaises(access.AccessDenied):
            access.rotate_invite(self.sitter_a, self.milo)

    def test_invitation_is_single_use(self):
        code = fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        with self.assertRaises(access.AccessDenied) as caught:
            access.redeem_invite(self.sitter_b, code)
        self.assertEqual(str(caught.exception), access.INVALID_CODE)
        self.assertIsNone(access.sitter_pet(self.sitter_b))

    def test_expired_invitation_fails(self):
        code = access.create_invite(self.owner_a, self.milo)
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute("UPDATE pet_invites SET expires_at = '2000-01-01 00:00:00'")
        conn.close()
        with self.assertRaises(access.AccessDenied):
            access.redeem_invite(self.sitter_a, code)
        self.assertIsNone(access.sitter_pet(self.sitter_a))

    def test_sitter_cannot_access_two_pets(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        luna_code = access.create_invite(self.owner_b, self.luna)
        with self.assertRaises(access.AccessDenied) as caught:
            access.redeem_invite(self.sitter_a, luna_code)
        self.assertIn("already have access", str(caught.exception))
        self.assert_denied(self.sitter_a, self.luna, access.VIEW)
        with sqlite3.connect(db.DB_PATH) as conn, self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO pet_access (pet_id, sitter_id, status, granted_at) "
                         "VALUES (?, ?, 'active', '2026-01-01')", (self.luna, self.sitter_a))  # blocked by the schema too
        conn.close()

    def test_pet_has_at_most_one_active_sitter(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        with self.assertRaises(access.AccessDenied):
            access.create_invite(self.owner_a, self.milo)  # must rotate instead
        with sqlite3.connect(db.DB_PATH) as conn, self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO pet_access (pet_id, sitter_id, status, granted_at) "
                         "VALUES (?, ?, 'active', '2026-01-01')", (self.milo, self.sitter_b))
        conn.close()

    def test_rotation_revokes_previous_sitter_and_code(self):
        first_code = fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        new_code = access.rotate_invite(self.owner_a, self.milo)
        self.assert_denied(self.sitter_a, self.milo, access.VIEW)
        self.assertIsNone(access.sitter_pet(self.sitter_a))
        self.assertNotEqual(first_code, new_code)
        access.redeem_invite(self.sitter_b, new_code)
        self.assertEqual(access.sitter_pet(self.sitter_b)["id"], self.milo)
        with self.assertRaises(access.AccessDenied):
            access.redeem_invite(self.sitter_a, new_code)  # used now

    def test_new_code_cancels_unused_code(self):
        old_code = access.create_invite(self.owner_a, self.milo)
        new_code = access.create_invite(self.owner_a, self.milo)
        with self.assertRaises(access.AccessDenied):
            access.redeem_invite(self.sitter_a, old_code)
        access.redeem_invite(self.sitter_a, new_code)

    def test_revoke_without_new_code(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        access.revoke_access(self.owner_a, self.milo)
        self.assertIsNone(access.sitter_pet(self.sitter_a))
        self.assertIsNone(access.access_status(self.owner_a, self.milo)["sitter_email"])

    def test_codes_are_long_random_and_stored_only_as_hashes(self):
        codes = {access.create_invite(self.owner_a, self.milo) for _ in range(5)}
        self.assertEqual(len(codes), 5)
        self.assertTrue(all(len(code) >= 32 for code in codes))
        contents = Path(db.DB_PATH).read_bytes()
        self.assertFalse(any(code.encode() in contents for code in codes))

    def test_redeem_rate_limit(self):
        for _ in range(access.MAX_REDEEM_FAILURES):
            with self.assertRaises(access.AccessDenied):
                access.redeem_invite(self.sitter_a, "guessing-a-code")
        good_code = access.create_invite(self.owner_a, self.milo)
        with self.assertRaises(access.AccessDenied) as caught:
            access.redeem_invite(self.sitter_a, good_code)
        self.assertIn("Too many", str(caught.exception))

    def test_owner_cannot_redeem_codes(self):
        code = access.create_invite(self.owner_a, self.milo)
        with self.assertRaises(access.AccessDenied):
            access.redeem_invite(self.owner_b, code)


# ----- 15–17: revocation -----

class RevocationTests(SecurityTestCase):
    def setUp(self):
        super().setUp()
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        self.milo_pet = db.get_pet(self.milo)

    def revoke(self):
        access.rotate_invite(self.owner_a, self.milo)

    def test_revoked_sitter_cannot_call_claude(self):
        store = {}
        chat.send_message(store, self.milo_pet, "First question", user_id=self.sitter_a)
        self.revoke()
        with self.assertRaises(chat.AccessEnded):
            chat.send_message(store, self.milo_pet, "Another question", user_id=self.sitter_a)
        self.assertEqual(len(self.client.requests), 1)  # the second question never reached Claude

    def test_answer_is_discarded_if_access_ends_while_claude_answers(self):
        store = {}
        self.client.before_reply = self.revoke  # the owner rotates the key mid-request
        with self.assertRaises(chat.AccessEnded):
            chat.send_message(store, self.milo_pet, "What does Milo eat?", user_id=self.sitter_a)
        self.assertEqual([m["role"] for m in chat.get_history(store, self.milo_pet)], ["assistant", "user"])

    def test_memory_tool_stops_working_after_revocation(self):
        tool = agent.make_memory_tool(self.milo, self.sitter_a)
        tool.call({"query": "visitors"})
        self.revoke()
        with self.assertRaises(agent.ToolError):
            tool.call({"query": "visitors"})

    def test_revoked_sitter_cannot_use_memory(self):
        access.save_memory(self.sitter_a, self.milo, "Milo hid when the doorbell rang.")
        self.revoke()
        for call in (lambda: access.list_memories(self.sitter_a, self.milo),
                     lambda: access.recall_memories(self.sitter_a, self.milo, "doorbell"),
                     lambda: access.save_memory(self.sitter_a, self.milo, "Another note")):
            with self.assertRaises(access.AccessDenied):
                call()

    def test_revoked_sitter_cannot_use_vision(self):
        self.revoke()
        with self.assertRaises(chat.AccessEnded):
            chat.get_label_reply(self.milo, b"jpeg", "image/jpeg", "Ingredients: chicken", {"status": "X",
                                 "restrictions": [], "matches": [], "reasons": []}, "Can he eat this?",
                                 user_id=self.sitter_a)
        self.assertEqual(self.client.requests, [])
        self.assert_denied(self.sitter_a, self.milo, access.VISION)

    def test_revoked_sitter_cannot_use_voice(self):
        voice_state = {}
        with mock.patch.object(vision, "extract_ingredients_text"), \
                mock.patch("voice.transcribe_audio", return_value="Can Milo eat chicken?"):
            chat.transcribe_recording(voice_state, self.milo, b"RIFF" + b"\x00" * 4000)
        self.revoke()
        with self.assertRaises(chat.AccessEnded):
            chat.send_voice_question({}, voice_state, self.milo_pet, "Can Milo eat chicken?", user_id=self.sitter_a)
        self.assert_denied(self.sitter_a, self.milo, access.VOICE)
        self.assertEqual(self.client.requests, [])

    def test_live_session_stops_after_revocation(self):
        clock = FakeClock()
        session = live_voice.LiveSession(self.milo_pet, {}, vad=fake_vad, clock=clock, user_id=self.sitter_a)
        session.speaker = FakeSpeaker()
        session.start(run_worker=False)
        self.revoke()
        with mock.patch("voice.transcribe_samples", return_value="Can Milo eat chicken?"):
            session.feed(np.concatenate([speech(0.8), silence(1.2)]))
            for _ in range(20):
                session.step()
        self.assertTrue(session.access_ended)
        self.assertEqual(session.state, live_voice.STOPPED)
        self.assertEqual(self.client.requests, [])
        self.assertEqual(session.speaker.played, [])

    def test_sitter_deletes_only_own_memories(self):
        own = access.save_memory(self.sitter_a, self.milo, "My note")
        owners = access.save_memory(self.owner_a, self.milo, "Owner's note")
        with self.assertRaises(access.AccessDenied):
            access.delete_memory(self.sitter_a, self.milo, owners)
        self.assertTrue(access.delete_memory(self.sitter_a, self.milo, own))
        self.assertTrue(access.delete_memory(self.owner_a, self.milo, owners))  # owners may delete any


# ----- 15, 18, 20: the app -----

class AppSecurityTests(SecurityTestCase):
    def app(self, user_id=None):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP, default_timeout=30)
        if user_id is not None:
            at.session_state["user_id"] = user_id
        at.run()
        self.assertFalse(at.exception)
        return at

    def page_text(self, at):
        return " ".join(element.value for element in (*at.markdown, *at.caption, *at.error, *at.info)
                        if isinstance(element.value, str))

    def test_logged_out_users_see_only_the_login_page(self):
        at = self.app()
        self.assertIn("Log in", [tab.label for tab in at.tabs])
        self.assertNotIn("Milo", self.page_text(at))

    def test_owner_dashboard_shows_only_own_pets(self):
        at = self.app(self.owner_b)
        self.assertEqual(at.selectbox[0].options, ["Luna"])
        self.assertNotIn("Golden Retriever", self.page_text(at))

    def test_logged_in_sitter_loses_access_without_logging_out(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        at = self.app(self.sitter_a)
        self.assertIn("Golden Retriever", self.page_text(at))
        at.chat_input(key="sitter_chat_input").set_value("What does Milo eat?").run()
        self.assertEqual(len(self.client.requests), 1)

        access.rotate_invite(self.owner_a, self.milo)  # the owner, in another session
        at.run()  # the sitter's next interaction
        text = self.page_text(at)
        self.assertIn("ended your access", text)
        self.assertNotIn("Golden Retriever", text)
        self.assertNotIn(self.milo, at.session_state["chats"])  # cached chat cleared
        self.assertEqual(len(self.client.requests), 1)

    def test_tampered_session_state_does_not_grant_access(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        at = self.app(self.sitter_a)
        at.session_state["active_pet_id"] = self.luna  # pretend to have Luna
        at.session_state["role"] = "owner"             # and to be an owner
        at.run()
        text = self.page_text(at)
        self.assertIn("Golden Retriever", text)  # still only Milo, from the database
        self.assertNotIn("Fish", text)
        self.assertIn("Pet Sitter", text)

    def test_session_for_deleted_or_unknown_account_is_logged_out(self):
        at = self.app(424242)
        self.assertIn("Log in", [tab.label for tab in at.tabs])
        self.assertNotIn("user_id", at.session_state)

    def test_logout_and_new_login_do_not_leak_data(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        at = self.app(self.sitter_a)
        at.chat_input(key="sitter_chat_input").set_value("What does Milo eat?").run()
        self.assertIn(self.milo, at.session_state["chats"])
        next(b for b in at.button if b.label == "Log out").click().run()
        self.assertNotIn("user_id", at.session_state)
        self.assertEqual(at.session_state["chats"], {})  # everything from the sitter's session is gone

        at.text_input[0].input("owner.b@example.com")
        at.text_input[1].input(fixtures.PASSWORD)
        at.button(key="FormSubmitter:login_form-Log in").click().run()
        self.assertEqual(at.session_state["user_id"], self.owner_b)
        self.assertEqual(at.session_state["chats"], {})
        text = self.page_text(at)
        self.assertNotIn("Golden Retriever", text)
        self.assertNotIn("Milo eats dry food", text)

    def test_wrong_login_shows_generic_error(self):
        at = self.app()
        at.text_input[0].input("owner.a@example.com")
        at.text_input[1].input("wrong password")
        at.button(key="FormSubmitter:login_form-Log in").click().run()
        self.assertIn(auth.LOGIN_ERROR, [e.value for e in at.error])
        self.assertNotIn("user_id", at.session_state)

    def test_owner_generates_code_shown_once(self):
        at = self.app(self.owner_a)
        at.button(key=f"invite_{self.milo}").click().run()
        [code] = [element.value for element in at.code]
        at.run()
        self.assertEqual(len(at.code), 0)  # not shown again
        access.redeem_invite(self.sitter_a, code)
        self.assertEqual(access.sitter_pet(self.sitter_a)["id"], self.milo)

    def test_rotation_needs_confirmation(self):
        fixtures.grant(self.owner_a, self.milo, self.sitter_a)
        at = self.app(self.owner_a)
        self.assertTrue(at.button(key=f"rotate_{self.milo}").disabled)
        at.checkbox(key=f"confirm_revoke_{self.milo}").check().run()
        at.button(key=f"rotate_{self.milo}").click().run()
        self.assertIsNone(access.sitter_pet(self.sitter_a))
        self.assertEqual(len(at.code), 1)  # the new code

    def test_sitter_redeems_code_in_app(self):
        code = access.create_invite(self.owner_a, self.milo)
        at = self.app(self.sitter_a)
        next(t for t in at.text_input if t.label == "Access code").input(code)
        at.button(key="FormSubmitter:redeem_form-Redeem Code").click().run()
        self.assertIn("Golden Retriever", self.page_text(at))


# ----- 19: migration -----

class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "petpulse.db"
        with sqlite3.connect(db.DB_PATH) as conn:  # the schema before accounts existed
            conn.execute(db.CREATE_PETS_TABLE)
            conn.execute("""CREATE TABLE pet_memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            pet_id INTEGER NOT NULL REFERENCES pets(id) ON DELETE CASCADE, content TEXT NOT NULL,
                            source TEXT, created_at TEXT NOT NULL, embedding BLOB)""")
            conn.execute("INSERT INTO pets (name, species, created_at, updated_at, allergies) "
                         "VALUES ('Milo', 'Dog', '2026-01-01 00:00:00', '2026-01-01 00:00:00', 'Chicken')")
            conn.execute("INSERT INTO pet_memories (pet_id, content, source, created_at) "
                         "VALUES (1, 'Hid from the doorbell', 'Sitter observation', '2026-01-02 00:00:00')")
        conn.close()

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def test_existing_data_survives_and_backup_is_made(self):
        db.init_db()
        db.init_db()
        [milo] = db.get_all_pets()
        self.assertEqual((milo["name"], milo["allergies"], milo["owner_id"]), ("Milo", "Chicken", None))
        self.assertEqual(db.get_memories(milo["id"])[0]["content"], "Hid from the doorbell")
        backups = list((Path(self.temp_dir.name) / "backups").iterdir())
        self.assertEqual(len(backups), 1)  # one backup, before the first migration only

    def test_old_pets_are_not_given_to_the_first_account(self):
        db.init_db()
        owner = fixtures.make_owner()
        self.assertEqual(access.list_owner_pets(owner), [])

    def test_explicit_assignment_with_manage_py(self):
        db.init_db()
        owner = fixtures.make_owner("owner.a@example.com")
        with mock.patch("builtins.print"):
            manage.main(["--db", str(db.DB_PATH), "assign-pets", "--owner-email", "owner.a@example.com",
                         "--all", "--yes"])
        self.assertEqual([p["name"] for p in access.list_owner_pets(owner)], ["Milo"])
        self.assertTrue(any("before-assign" in p.name for p in (Path(self.temp_dir.name) / "backups").iterdir()))

    def test_assignment_needs_confirmation(self):
        db.init_db()
        owner = fixtures.make_owner("owner.a@example.com")
        with mock.patch("builtins.input", return_value="n"), mock.patch("builtins.print"), \
                self.assertRaises(SystemExit):
            manage.main(["--db", str(db.DB_PATH), "assign-pets", "--owner-email", "owner.a@example.com", "--all"])
        self.assertEqual(access.list_owner_pets(owner), [])


# ----- The demo scenario, end to end -----

class DemoScenarioTests(SecurityTestCase):
    def test_demo_scenario(self):
        store = {}
        milo = db.get_pet(self.milo)
        luna = db.get_pet(self.luna)
        # 1-2. Each owner sees only their pet.
        self.assertEqual([p["name"] for p in access.list_owner_pets(self.owner_a)], ["Milo"])
        self.assertEqual([p["name"] for p in access.list_owner_pets(self.owner_b)], ["Luna"])
        # 3-5. Owner A invites Sitter A, who can chat about Milo.
        code = access.create_invite(self.owner_a, self.milo)
        access.redeem_invite(self.sitter_a, code)
        chat.send_message(store, milo, "What does Milo eat?", user_id=self.sitter_a)
        # 6-7. Owner A rotates the key; Sitter A is denied.
        new_code = access.rotate_invite(self.owner_a, self.milo)
        with self.assertRaises(chat.AccessEnded):
            chat.send_message(store, milo, "And for dinner?", user_id=self.sitter_a)
        # 8-9. Sitter B redeems the new code and can chat about Milo.
        access.redeem_invite(self.sitter_b, new_code)
        chat.send_message({}, milo, "What does Milo eat?", user_id=self.sitter_b)
        # 10. Neither sitter can access Luna.
        for sitter in (self.sitter_a, self.sitter_b):
            self.assert_denied(sitter, self.luna, access.VIEW)
            with self.assertRaises(chat.AccessEnded):
                chat.send_message({}, luna, "What does Luna eat?", user_id=sitter)
        self.assertEqual(len(self.client.requests), 2)


if __name__ == "__main__":
    unittest.main()
