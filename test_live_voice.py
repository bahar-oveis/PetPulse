"""
Tests for Live Conversation (live_voice.py and the Live mode in app.py).

The microphone is simulated with generated audio, Whisper and Claude are
mocked, and the speaker is a fake that records what it was asked to play.
No model is downloaded and no paid API call is made.
Run with:  python -m unittest -v
"""

import io
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import av
import numpy as np

import agent
import chat
import database as db
import live_voice
import voice
from test_agent import FakeClient

RATE = live_voice.SAMPLE_RATE


def speech(seconds):
    """Loud noise that the fake VAD treats as clear speech."""
    return np.full(int(RATE * seconds), 0.3, dtype=np.float32)


def echo(seconds):
    """Quieter sound, like PetPulse's own voice coming back through the microphone."""
    return np.full(int(RATE * seconds), 0.1, dtype=np.float32)


def silence(seconds):
    return np.zeros(int(RATE * seconds), dtype=np.float32)


def fake_vad(chunk):
    """Speech probability: 1.0 for clear speech, 0.6 for faint echo, 0 for silence."""
    level = np.abs(chunk).mean()
    return 1.0 if level > 0.2 else 0.6 if level > 0.05 else 0.0


def wav_bytes(seconds, rate=live_voice.SPEAKER_RATE):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeSpeaker:
    def __init__(self):
        self.played = []
        self.cleared = 0

    def push(self, pcm):
        self.played.append(pcm)

    def clear(self):
        self.cleared += 1


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = live_voice.UtteranceDetector(fake_vad, live_voice.LiveSettings())

    def test_speech_then_pause_is_one_utterance(self):
        utterances = self.detector.add(np.concatenate([silence(0.5), speech(1.0), silence(1.2)]))
        self.assertEqual(len(utterances), 1)
        seconds = len(utterances[0]) / RATE
        self.assertGreater(seconds, 1.0 + 0.9)  # speech, pre-roll, and the ending pause
        self.assertLess(seconds, 1.0 + 0.3 + 1.1)

    def test_silence_alone_is_not_an_utterance(self):
        self.assertEqual(self.detector.add(silence(3)), [])

    def test_utterance_waits_for_the_pause(self):
        self.assertEqual(self.detector.add(np.concatenate([speech(1.0), silence(0.5)])), [])
        self.assertTrue(self.detector.in_speech)
        self.assertEqual(len(self.detector.add(silence(0.6))), 1)

    def test_short_noise_is_ignored(self):
        self.assertEqual(self.detector.add(np.concatenate([speech(0.1), silence(1.2)])), [])

    def test_two_questions(self):
        audio = np.concatenate([speech(0.8), silence(1.2), speech(0.8), silence(1.2)])
        self.assertEqual(len(self.detector.add(audio)), 2)

    def test_long_speech_is_cut_off(self):
        self.assertEqual(len(self.detector.add(speech(21))), 1)

    def test_small_webrtc_sized_pieces(self):
        audio = np.concatenate([speech(1.0), silence(1.2)])
        found = []
        for start in range(0, len(audio), 320):  # 20 ms frames at 16 kHz
            found += self.detector.add(audio[start:start + 320])
        self.assertEqual(len(found), 1)

    def test_pause_length_is_configurable(self):
        detector = live_voice.UtteranceDetector(fake_vad, live_voice.LiveSettings(end_silence_ms=500))
        self.assertEqual(len(detector.add(np.concatenate([speech(1.0), silence(0.6)]))), 1)


@unittest.skipUnless(shutil.which("say"), "needs macOS `say` to make real speech")
class SileroVadTests(unittest.TestCase):
    """Real Silero VAD (bundled with faster-whisper, no download) on real synthetic speech."""

    def test_detects_real_speech_and_pause(self):
        pcm, rate = voice.wav_to_pcm(voice.synthesize_speech("Can Milo eat chicken?", sample_rate=RATE))
        spoken = pcm.astype(np.float32) / 32768.0
        detector = live_voice.UtteranceDetector(live_voice.SileroVad(), live_voice.LiveSettings())
        self.assertEqual(detector.add(silence(1.0)), [])
        utterances = detector.add(np.concatenate([spoken, silence(1.5)]))
        self.assertEqual(len(utterances), 1)
        self.assertGreater(len(utterances[0]), len(spoken) * 0.8)


class LiveSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.milo = db.get_pet(db.create_pet({**db.DEMO_PET}))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat", "allergies": "Fish"}))
        self.store = {}
        self.client = FakeClient(reply="No, Milo is allergic to chicken.")
        self.transcribe = mock.Mock(return_value="Can Milo eat chicken?")
        self.synthesize = mock.Mock(return_value=wav_bytes(2.0))
        patches = [
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""}),
            mock.patch("agent.print", create=True),
            mock.patch.object(agent, "_create_client", return_value=self.client),
            mock.patch.object(voice, "transcribe_samples", self.transcribe),
            mock.patch.object(voice, "synthesize_speech", self.synthesize),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.clock = FakeClock()
        self.speaker = FakeSpeaker()
        self.session = self.new_session(self.milo)

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def new_session(self, pet):
        session = live_voice.LiveSession(pet, self.store, vad=fake_vad, clock=self.clock)
        session.speaker = self.speaker
        session.start(run_worker=False)
        return session

    def say(self, session=None, seconds=0.8):
        """Simulate the sitter asking one question, then let the worker process it."""
        session = session or self.session
        session.feed(np.concatenate([speech(seconds), silence(1.2)]))
        self.run_worker(session)

    def run_worker(self, session=None):
        session = session or self.session
        for _ in range(20):
            session.step()

    def finish_playback(self, session=None):
        self.clock.now += 10
        self.run_worker(session)

    def test_one_full_turn(self):
        self.assertEqual(self.session.state, live_voice.LISTENING)
        self.say()
        self.transcribe.assert_called_once()
        self.assertEqual(len(self.client.requests), 1)
        request = self.client.requests[0]
        self.assertEqual(request["messages"], [{"role": "user", "content": "Can Milo eat chicken?"}])
        self.assertIn('"Milo"', request["system"])
        history = chat.get_history(self.store, self.milo)
        self.assertEqual([m["content"] for m in history[1:]], ["Can Milo eat chicken?", "No, Milo is allergic to chicken."])
        self.assertEqual(self.synthesize.call_args.kwargs["sample_rate"], live_voice.SPEAKER_RATE)
        self.assertEqual(len(self.speaker.played), 1)
        self.assertEqual(self.session.state, live_voice.SPEAKING)
        self.assertEqual(self.session.heard, "Can Milo eat chicken?")

    def test_listens_again_after_playback(self):
        self.say()
        self.clock.now += 2.0  # 2 s answer, but the extra tail hasn't passed yet
        self.run_worker()
        self.assertEqual(self.session.state, live_voice.SPEAKING)
        self.finish_playback()
        self.assertEqual(self.session.state, live_voice.LISTENING)

    def test_own_voice_is_not_transcribed(self):
        self.say()
        self.clock.now += 1.0
        # The microphone picks up PetPulse's spoken answer while it plays.
        self.feed_in_frames(np.concatenate([echo(1.5), silence(1.2)]))
        self.session.on_audio_frame(self.make_frame())
        self.finish_playback()
        self.run_worker()
        self.transcribe.assert_called_once()
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(self.speaker.cleared, 0)  # the answer was never paused

    def feed_in_frames(self, audio, session=None):
        """Feed audio in 20 ms pieces, like WebRTC, letting the worker keep up."""
        session = session or self.session
        for start in range(0, len(audio), 320):
            session.feed(audio[start:start + 320])
            session.step()
        self.run_worker(session)

    def test_speaking_pauses_answer_and_new_question_is_answered(self):
        self.say()
        self.assertEqual(len(self.speaker.played), 1)
        self.clock.now += 1.0  # one second into the 2-second answer
        self.transcribe.return_value = "Actually, when does he eat?"
        self.client.reply = "At 8 AM and 6 PM."
        self.feed_in_frames(np.concatenate([speech(0.8), silence(1.2)]))
        self.assertGreaterEqual(self.speaker.cleared, 1)  # reading stopped
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(self.client.requests[1]["messages"][-1]["content"], "Actually, when does he eat?")
        self.assertEqual(len(self.speaker.played), 2)  # the new answer, not the old one again
        self.assertEqual(self.session.state, live_voice.SPEAKING)
        self.assertFalse(self.session.paused)
        history = chat.get_history(self.store, self.milo)
        self.assertEqual(history[2]["content"], "No, Milo is allergic to chicken.")  # full first answer kept as text

    def test_whole_interruption_in_one_batch(self):
        self.say()
        self.clock.now += 1.0
        self.transcribe.return_value = "When does he eat?"
        self.session.feed(np.concatenate([speech(0.8), silence(1.2)]))
        self.run_worker()
        self.assertGreaterEqual(self.speaker.cleared, 1)
        self.assertEqual(len(self.client.requests), 2)

    def test_noise_pauses_then_answer_continues(self):
        self.say()
        self.clock.now += 1.0
        self.transcribe.side_effect = voice.VoiceError("No speech was detected.")
        self.feed_in_frames(np.concatenate([speech(0.6), silence(1.2)]))
        self.assertGreaterEqual(self.speaker.cleared, 1)
        self.assertEqual(len(self.client.requests), 1)  # nothing new sent to Claude
        self.assertEqual(len(self.speaker.played), 2)
        first, rest = self.speaker.played
        # 2.0 s answer, paused after 1.0 s, rewound 0.5 s -> about 1.5 s left
        self.assertAlmostEqual(len(rest) / live_voice.SPEAKER_RATE, 1.5, delta=0.05)
        np.testing.assert_array_equal(rest, first[-len(rest):])
        self.assertEqual(self.session.state, live_voice.SPEAKING)

    def test_own_echo_that_interrupts_is_ignored_and_answer_continues(self):
        self.say()
        self.clock.now += 1.0
        self.transcribe.return_value = "Milo is allergic to chicken"  # its own words, heard back
        self.feed_in_frames(np.concatenate([speech(0.8), silence(1.2)]))
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(len(self.speaker.played), 2)  # the paused answer continues
        self.assertEqual(self.session.state, live_voice.SPEAKING)
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 3)

    def test_looks_like_echo(self):
        answer = "No, Milo is allergic to chicken, so please avoid chicken treats."
        self.assertTrue(live_voice.looks_like_echo("Milo is allergic to chicken", answer))
        self.assertFalse(live_voice.looks_like_echo("What time does he eat dinner?", answer))
        self.assertFalse(live_voice.looks_like_echo("Chicken?", answer))  # too short to be sure

    def test_short_noise_does_not_interrupt(self):
        self.say()
        self.clock.now += 1.0
        self.feed_in_frames(np.concatenate([speech(0.2), silence(1.2)]))
        self.assertEqual(self.speaker.cleared, 0)
        self.assertEqual(self.session.state, live_voice.SPEAKING)

    def test_no_interruption_right_as_playback_starts(self):
        self.say()
        self.feed_in_frames(speech(0.6))  # clock hasn't moved: still in the grace period
        self.assertEqual(self.speaker.cleared, 0)
        self.assertEqual(self.session.state, live_voice.SPEAKING)

    def test_interruptions_can_be_turned_off(self):
        self.session.stop()
        session = live_voice.LiveSession(
            self.milo, self.store, settings=live_voice.LiveSettings(barge_in=False), vad=fake_vad, clock=self.clock
        )
        session.speaker = self.speaker
        session.start(run_worker=False)
        self.say(session)
        self.clock.now += 1.0
        self.feed_in_frames(np.concatenate([speech(0.8), silence(1.2)]), session)
        self.assertEqual(self.speaker.cleared, 1)  # only from stopping the first session
        self.assertEqual(len(self.client.requests), 1)

    def test_each_utterance_is_processed_and_played_once(self):
        self.say()
        for _ in range(5):
            self.session.start()  # Streamlit reruns call start() again
            self.run_worker()
        self.finish_playback()
        self.run_worker()
        self.transcribe.assert_called_once()
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(len(self.speaker.played), 1)

    def test_follow_up_question_uses_conversation(self):
        self.say()
        self.finish_playback()
        self.transcribe.return_value = "What should I avoid giving him?"
        self.say()
        self.assertEqual([m["content"] for m in self.client.requests[1]["messages"]], [
            "Can Milo eat chicken?",
            "No, Milo is allergic to chicken.",
            "What should I avoid giving him?",
        ])
        self.assertEqual(len(self.speaker.played), 2)

    def test_shares_history_with_text_chat(self):
        chat.send_message(self.store, self.milo, "What food does Milo eat?")
        self.say()
        sent = [m["content"] for m in self.client.requests[1]["messages"]]
        self.assertEqual(sent[0], "What food does Milo eat?")
        self.assertEqual(sent[-1], "Can Milo eat chicken?")

    def test_stop_ends_session(self):
        self.say()
        self.session.stop()
        self.assertEqual(self.session.state, live_voice.STOPPED)
        self.assertGreaterEqual(self.speaker.cleared, 1)  # playback stopped
        self.session.start()  # can't restart a stopped session
        self.say()
        self.assertEqual(self.session.state, live_voice.STOPPED)
        self.assertEqual(len(self.client.requests), 1)

    def test_answer_is_discarded_if_session_ends_while_claude_answers(self):
        def answer_after_stop(**request):
            self.session.stop()
            return FakeClient._create(self.client, **request)

        self.client.messages.create = answer_after_stop
        self.say()
        history = chat.get_history(self.store, self.milo)
        self.assertEqual([m["role"] for m in history], ["assistant", "user"])  # question visible, no answer
        self.synthesize.assert_not_called()
        self.assertEqual(self.speaker.played, [])

    def test_sessions_for_different_pets_stay_separate(self):
        self.session.stop()
        luna_session = self.new_session(self.luna)
        self.transcribe.return_value = "Does she have allergies?"
        self.say(luna_session)
        system = self.client.requests[0]["system"]
        self.assertIn('"Luna"', system)
        self.assertNotIn("Chicken", system)
        self.assertEqual(len(chat.get_history(self.store, self.milo)), 1)  # Milo's chat untouched
        self.assertEqual(len(chat.get_history(self.store, self.luna)), 3)

    def test_claude_failure_keeps_listening(self):
        self.client.error = RuntimeError("boom")
        with mock.patch("traceback.print_exc"):
            self.say()
        self.assertEqual(self.session.state, live_voice.LISTENING)
        self.assertIn("Something went wrong", self.session.notice)
        self.assertEqual(self.speaker.played, [])
        history = chat.get_history(self.store, self.milo)
        self.assertEqual([m["role"] for m in history], ["assistant", "user"])

    def test_speech_failure_keeps_text_answer(self):
        self.synthesize.side_effect = voice.VoiceError("The spoken answer couldn't be created.")
        self.say()
        self.assertEqual(self.session.state, live_voice.LISTENING)
        self.assertIn("couldn't be created", self.session.notice)
        self.assertEqual(chat.get_history(self.store, self.milo)[-1]["content"], "No, Milo is allergic to chicken.")

    def test_unclear_audio_is_ignored(self):
        self.transcribe.side_effect = voice.VoiceError("No speech was detected.")
        self.say()
        self.transcribe.side_effect = None
        self.transcribe.return_value = " . "
        self.say()
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.session.state, live_voice.LISTENING)

    def make_frame(self, samples=960):
        stereo = np.zeros((1, samples * 2), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
        frame.sample_rate = 48000
        return frame

    def test_webrtc_frames_are_converted_to_16k_mono(self):
        for _ in range(50):  # 1 second of 48 kHz stereo audio
            self.session.on_audio_frame(self.make_frame())
        queued = []
        while not self.session._queue.empty():
            queued.append(self.session._queue.get_nowait())
        total = sum(len(chunk) for chunk in queued)
        self.assertTrue(all(chunk.dtype == np.float32 for chunk in queued))
        self.assertAlmostEqual(total, RATE, delta=RATE * 0.05)
        self.assertTrue(self.session.receiving_audio())


class FakeWebRtc:
    """
    Stands in for streamlit-webrtc, which needs a real browser and Streamlit server.
    `playing` simulates the Start Live Conversation / End Session buttons.
    """

    def __init__(self):
        self.playing = False
        self.mic_callback = None
        self.speaker = FakeSpeaker()
        self.speaker.track = object()

    def create_audio_sink_track(self, callback, key):
        self.mic_callback = callback
        return object()

    def create_pcm_audio_source_track(self, key, sample_rate):
        return self.speaker

    def webrtc_streamer(self, key, **options):
        self.options = options
        return SimpleNamespace(state=SimpleNamespace(playing=self.playing, signalling=False))


class LiveAppTests(unittest.TestCase):
    """The Live mode in the real Streamlit app, with WebRTC faked."""

    def setUp(self):
        import streamlit_webrtc
        from streamlit.testing.v1 import AppTest

        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.milo = db.get_pet(db.create_pet({**db.DEMO_PET}))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat"}))
        self.client = FakeClient(reply="Milo eats dry food.")
        patches = [
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""}),
            mock.patch("agent.print", create=True),
            mock.patch.object(agent, "_create_client", return_value=self.client),
            # The worker thread would load the real Silero model; the fake VAD is enough here.
            mock.patch.object(live_voice, "SileroVad", lambda: fake_vad),
        ]
        self.webrtc = FakeWebRtc()
        for name in ("create_audio_sink_track", "create_pcm_audio_source_track", "webrtc_streamer"):
            patches.append(mock.patch.object(streamlit_webrtc, name, getattr(self.webrtc, name)))
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.at = AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=30).run()
        self.at.selectbox(key="sitter_pet_id").set_value(self.milo["id"]).run()

    def tearDown(self):
        live = self.at.session_state["live_session"] if "live_session" in self.at.session_state else None
        if live is not None:
            live.stop()
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def open_live_mode(self):
        self.at.radio(key="chat_mode").set_value("🔴 Live Conversation").run()
        self.assertFalse(self.at.exception)

    def test_live_mode_shows_ready_state(self):
        self.open_live_mode()
        live = self.at.session_state["live_session"]
        self.assertEqual(live.pet_id, self.milo["id"])
        self.assertEqual(live.state, live_voice.IDLE)
        self.assertTrue(any("Ready" in m.value for m in self.at.markdown))

    def test_start_and_end_session(self):
        self.open_live_mode()
        self.webrtc.playing = True  # sitter clicks Start Live Conversation
        self.at.run()
        live = self.at.session_state["live_session"]
        self.assertEqual(live.state, live_voice.LISTENING)
        self.assertIs(live.speaker, self.webrtc.speaker)
        self.assertEqual(self.webrtc.mic_callback, live.on_audio_frame)
        self.assertEqual(self.webrtc.options["media_stream_constraints"]["video"], False)
        self.assertEqual(self.webrtc.options["translations"]["start"], "Start Live Conversation")
        self.assertTrue(any("Listening" in m.value for m in self.at.markdown))

        self.webrtc.playing = False  # sitter clicks End Session
        self.at.run()
        self.assertEqual(live.state, live_voice.STOPPED)
        self.assertGreaterEqual(self.webrtc.speaker.cleared, 1)

        self.webrtc.playing = True  # a new session can start afterwards
        self.at.run()
        new_live = self.at.session_state["live_session"]
        self.assertIsNot(new_live, live)
        self.assertEqual(new_live.state, live_voice.LISTENING)

    def test_switching_pets_stops_the_session(self):
        self.open_live_mode()
        milo_session = self.at.session_state["live_session"]
        self.at.selectbox(key="sitter_pet_id").set_value(self.luna["id"]).run()
        self.assertEqual(milo_session.state, live_voice.STOPPED)
        self.assertEqual(self.at.session_state["live_session"].pet_id, self.luna["id"])

    def test_leaving_live_mode_stops_the_session(self):
        self.open_live_mode()
        live = self.at.session_state["live_session"]
        self.at.radio(key="chat_mode").set_value("💬 Text Chat").run()
        self.assertEqual(live.state, live_voice.STOPPED)
        self.assertIsNone(self.at.session_state["live_session"])

    def test_text_chat_still_works(self):
        self.at.chat_input(key="sitter_chat_input").set_value("What does Milo eat?").run()
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.client.requests), 1)
        texts = [m.markdown[0].value for m in self.at.tabs[1].chat_message if m.markdown]
        self.assertIn("Milo eats dry food.", texts)


if __name__ == "__main__":
    unittest.main()
