"""
Tests for voice mode (voice.py, the voice functions in chat.py, and the voice UI).

Nothing real runs: Whisper, the macOS `say` command, and Claude are all mocked,
so no model is downloaded, no audio is played, and no paid API call is made.
Run with:  python -m unittest -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent
import chat
import database as db
import voice
from test_agent import FakeClient

RECORDING = b"RIFF" + b"\x00" * 4000  # pretend WAV recording
SPOKEN = b"RIFF-spoken-answer"         # pretend WAV from `say`


class FakeWhisper:
    def __init__(self, text=" Can Milo eat chicken? "):
        self.text = text
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        segments = [SimpleNamespace(text=self.text)] if self.text else []
        return iter(segments), SimpleNamespace(language="en")


class TranscribeTests(unittest.TestCase):
    def test_returns_recognized_text(self):
        whisper = FakeWhisper()
        with mock.patch.object(voice, "get_whisper_model", return_value=whisper):
            self.assertEqual(voice.transcribe_audio(RECORDING), "Can Milo eat chicken?")
        audio, options = whisper.calls[0]
        self.assertEqual(audio.read(), RECORDING)  # passed in memory, not saved to a file
        self.assertEqual(options["language"], "en")

    def test_empty_audio(self):
        with mock.patch.object(voice, "get_whisper_model") as load:
            for audio in (None, b"", b"RIFF"):
                with self.assertRaises(voice.VoiceError):
                    voice.transcribe_audio(audio)
        load.assert_not_called()

    def test_no_speech_detected(self):
        with mock.patch.object(voice, "get_whisper_model", return_value=FakeWhisper(text="")):
            with self.assertRaises(voice.VoiceError) as caught:
                voice.transcribe_audio(RECORDING)
        self.assertIn("No speech", str(caught.exception))

    def test_recognition_error_is_friendly(self):
        broken = mock.Mock()
        broken.transcribe.side_effect = RuntimeError("secret decoder failure")
        with mock.patch.object(voice, "get_whisper_model", return_value=broken), mock.patch("traceback.print_exc"):
            with self.assertRaises(voice.VoiceError) as caught:
                voice.transcribe_audio(RECORDING)
        self.assertNotIn("secret", str(caught.exception))

    def test_model_is_base_cpu_int8_and_loaded_once(self):
        voice.get_whisper_model.cache_clear()
        self.addCleanup(voice.get_whisper_model.cache_clear)
        fake_module = SimpleNamespace(WhisperModel=mock.Mock())
        with mock.patch.dict(sys.modules, {"faster_whisper": fake_module}):
            voice.get_whisper_model()
            voice.get_whisper_model()
        fake_module.WhisperModel.assert_called_once_with(
            "base", device="cpu", compute_type="int8", local_files_only=True
        )

    def test_model_is_downloaded_only_when_missing(self):
        voice.get_whisper_model.cache_clear()
        self.addCleanup(voice.get_whisper_model.cache_clear)
        model_class = mock.Mock(side_effect=[FileNotFoundError("not cached"), "downloaded model"])
        with mock.patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=model_class)}):
            self.assertEqual(voice.get_whisper_model(), "downloaded model")
        self.assertEqual(model_class.call_args_list[1], mock.call("base", device="cpu", compute_type="int8"))


def fake_say(args, **options):
    """Pretends to be `say -o <file> ...`: writes the output file."""
    Path(args[2]).write_bytes(SPOKEN)
    return subprocess.CompletedProcess(args, 0)


class SynthesizeTests(unittest.TestCase):
    def test_returns_audio_and_cleans_up(self):
        with mock.patch("subprocess.run", side_effect=fake_say) as run:
            self.assertEqual(voice.synthesize_speech("Milo is **allergic** to chicken."), SPOKEN)
        args, options = run.call_args
        command = args[0]
        self.assertIsInstance(command, list)  # argument list, never a shell string
        self.assertEqual(command[0], "say")
        self.assertNotIn("shell", options)
        self.assertEqual(options["input"], "Milo is allergic to chicken.")  # Markdown removed, text via stdin
        self.assertFalse(Path(command[2]).parent.exists())  # temporary folder deleted

    def test_say_failure_is_friendly_and_cleans_up(self):
        def failing_say(args, **options):
            Path(args[2]).write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, args)

        with mock.patch("subprocess.run", side_effect=failing_say) as run, mock.patch("traceback.print_exc"):
            with self.assertRaises(voice.VoiceError):
                voice.synthesize_speech("Hello")
        self.assertFalse(Path(run.call_args[0][0][2]).parent.exists())

    def test_say_timeout(self):
        timeout = subprocess.TimeoutExpired("say", 60)
        with mock.patch("subprocess.run", side_effect=timeout), mock.patch("traceback.print_exc"):
            with self.assertRaises(voice.VoiceError):
                voice.synthesize_speech("Hello")

    def test_missing_speech_engine(self):
        with mock.patch("shutil.which", return_value=None), mock.patch("subprocess.run") as run:
            with self.assertRaises(voice.VoiceError):
                voice.synthesize_speech("Hello")
        run.assert_not_called()

    def test_empty_text(self):
        with mock.patch("subprocess.run") as run:
            with self.assertRaises(voice.VoiceError):
                voice.synthesize_speech("  ** ")
        run.assert_not_called()


class VoiceTestCase(unittest.TestCase):
    """Temporary database, fake Claude, fake Whisper, and fake `say`."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.milo = db.get_pet(db.create_pet({**db.DEMO_PET}))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat", "allergies": "Fish"}))

        self.client = FakeClient(reply="No, Milo is allergic to chicken.")
        self.whisper = FakeWhisper()
        patches = [
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""}),
            mock.patch("agent.print", create=True),
            mock.patch.object(agent, "_create_client", return_value=self.client),
            mock.patch.object(voice, "get_whisper_model", return_value=self.whisper),
            mock.patch.object(voice, "synthesize_speech", return_value=SPOKEN),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()


class VoiceChatTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        self.store = {}
        self.voice_state = {}

    def test_transcription_is_pending_until_confirmed(self):
        pending = chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        self.assertEqual(pending["text"], "Can Milo eat chicken?")
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 1)  # only the welcome message
        self.assertEqual(self.client.requests, [])

    def test_confirmed_edited_text_is_sent(self):
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        reply = chat.send_voice_question(self.store, self.voice_state, self.milo, "Can Milo eat chicken treats?")
        self.assertEqual(reply, "No, Milo is allergic to chicken.")
        sent = self.client.requests[0]
        self.assertEqual(sent["messages"], [{"role": "user", "content": "Can Milo eat chicken treats?"}])
        self.assertIn('"Milo"', sent["system"])
        self.assertNotIn("Fish", sent["system"])
        self.assertEqual(self.voice_state["replies"][self.milo["id"]], {"audio": SPOKEN, "error": None})

    def test_voice_and_text_share_one_conversation(self):
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        chat.send_voice_question(self.store, self.voice_state, self.milo, "What food should Milo eat?")
        chat.send_message(self.store, self.milo, "What should I avoid?")
        self.assertEqual([m["content"] for m in self.client.requests[1]["messages"]], [
            "What food should Milo eat?",
            "No, Milo is allergic to chicken.",
            "What should I avoid?",
        ])

    def test_reruns_do_not_repeat_transcription_or_claude_calls(self):
        for _ in range(3):
            chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        self.assertEqual(len(self.whisper.calls), 1)

        chat.send_voice_question(self.store, self.voice_state, self.milo, "Can Milo eat chicken?")
        with self.assertRaises(chat.AgentError):  # nothing pending any more
            chat.send_voice_question(self.store, self.voice_state, self.milo, "Can Milo eat chicken?")
        self.assertEqual(len(self.client.requests), 1)
        voice.synthesize_speech.assert_called_once()

    def test_new_recording_is_transcribed(self):
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING + b"\x01")
        self.assertEqual(len(self.whisper.calls), 2)

    def test_switching_pets_drops_pending_recording(self):
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        self.assertIsNone(chat.pending_voice_question(self.voice_state, self.luna["id"]))
        with self.assertRaises(chat.AgentError):
            chat.send_voice_question(self.store, self.voice_state, self.luna, "Can Milo eat chicken?")
        self.assertIsNone(chat.pending_voice_question(self.voice_state, self.milo["id"]))  # gone for Milo too
        self.assertEqual(self.client.requests, [])

    def test_failed_transcription_is_kept_as_error(self):
        self.whisper.text = ""
        pending = chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        self.assertEqual(pending["text"], "")
        self.assertIn("No speech", pending["error"])

    def test_tts_failure_keeps_text_answer(self):
        voice.synthesize_speech.side_effect = voice.VoiceError("The spoken answer couldn't be created.")
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        reply = chat.send_voice_question(self.store, self.voice_state, self.milo, "Can Milo eat chicken?")
        self.assertEqual(reply, "No, Milo is allergic to chicken.")
        self.assertEqual(chat.get_history(self.store, self.milo)[-1]["content"], reply)
        spoken = self.voice_state["replies"][self.milo["id"]]
        self.assertIsNone(spoken["audio"])
        self.assertIn("couldn't be created", spoken["error"])

    def test_claude_failure_saves_no_answer_and_no_audio(self):
        self.client.error = RuntimeError("boom")
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        with mock.patch("traceback.print_exc"), self.assertRaises(chat.AgentError):
            chat.send_voice_question(self.store, self.voice_state, self.milo, "Can Milo eat chicken?")
        self.assertEqual([m["role"] for m in chat.get_history(self.store, self.milo)], ["assistant", "user"])
        self.assertNotIn(self.milo["id"], self.voice_state.get("replies", {}))
        voice.synthesize_speech.assert_not_called()

    def test_empty_confirmed_text_is_not_sent(self):
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        with self.assertRaises(chat.AgentError):
            chat.send_voice_question(self.store, self.voice_state, self.milo, "   ")
        self.assertEqual(self.client.requests, [])
        self.assertIsNotNone(chat.pending_voice_question(self.voice_state, self.milo["id"]))  # can still edit

    def test_clear_voice_only_affects_one_pet(self):
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING)
        chat.send_voice_question(self.store, self.voice_state, self.milo, "Q")
        self.voice_state["replies"][self.luna["id"]] = {"audio": SPOKEN, "error": None}
        chat.transcribe_recording(self.voice_state, self.milo["id"], RECORDING + b"\x02")
        chat.clear_voice(self.voice_state, self.milo["id"])
        self.assertNotIn("pending", self.voice_state)
        self.assertNotIn(self.milo["id"], self.voice_state["replies"])
        self.assertIn(self.luna["id"], self.voice_state["replies"])


class VoiceAppTests(VoiceTestCase):
    """Drives the real Streamlit app with Streamlit's built-in AppTest."""

    def setUp(self):
        super().setUp()
        from streamlit.testing.v1 import AppTest

        self.at = AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=30).run()
        self.assertFalse(self.at.exception)
        self.at.selectbox(key="sitter_pet_id").set_value(self.milo["id"]).run()

    def record(self, audio=RECORDING):
        self.at.audio_input[0].set_value(("question.wav", audio, "audio/wav")).run()
        self.assertFalse(self.at.exception)

    def chat_texts(self):
        return [m.markdown[0].value for m in self.at.tabs[1].chat_message if m.markdown]

    def test_record_review_send_and_listen(self):
        self.record()
        self.assertEqual(self.at.text_area[0].value, "Can Milo eat chicken?")
        self.at.run()
        self.at.run()
        self.assertEqual(len(self.whisper.calls), 1)  # reruns don't transcribe again
        self.assertEqual(self.client.requests, [])    # nothing sent before confirming

        self.at.text_area[0].input("Can Milo eat chicken treats?")
        self.at.button(key=f"voice_send_{self.milo['id']}").click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.client.requests), 1)
        self.assertIn("Can Milo eat chicken treats?", self.chat_texts())
        self.assertIn("No, Milo is allergic to chicken.", self.chat_texts())
        self.assertEqual(len(self.at.get("audio")), 1)  # spoken answer player
        self.assertEqual(len(self.at.text_area), 0)     # recorder cleared

        self.at.run()
        self.at.run()
        self.assertEqual(len(self.client.requests), 1)  # reruns don't call Claude again
        voice.synthesize_speech.assert_called_once()

    def test_switching_pets_discards_recording(self):
        self.record()
        self.at.selectbox(key="sitter_pet_id").set_value(self.luna["id"]).run()
        self.assertEqual(len(self.at.text_area), 0)
        self.at.selectbox(key="sitter_pet_id").set_value(self.milo["id"]).run()
        self.assertEqual(len(self.at.text_area), 0)  # Milo's old recording is gone too
        self.assertEqual(self.client.requests, [])

    def test_tts_failure_still_shows_text_answer(self):
        voice.synthesize_speech.side_effect = voice.VoiceError("The spoken answer couldn't be created.")
        self.record()
        self.at.button(key=f"voice_send_{self.milo['id']}").click().run()
        self.assertIn("No, Milo is allergic to chicken.", self.chat_texts())
        self.assertTrue(any("couldn't be created" in i.value for i in self.at.info))

    def test_clear_chat_removes_spoken_answer(self):
        self.record()
        self.at.button(key=f"voice_send_{self.milo['id']}").click().run()
        self.at.button(key=f"clear_chat_{self.milo['id']}").click().run()
        self.assertEqual(len(self.at.get("audio")), 0)
        self.assertEqual(len(self.chat_texts()), 1)  # only the welcome message


if __name__ == "__main__":
    unittest.main()
