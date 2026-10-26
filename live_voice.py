"""
PetPulse Live Conversation: hands-free, turn-based voice chat.

How one turn works:
1. The browser streams microphone audio to this Python server over WebRTC.
   The audio callback only converts each frame to 16 kHz mono and puts it on a
   queue, so it stays fast.
2. A worker thread runs Silero voice activity detection on 32 ms chunks. When
   the sitter has spoken and then stays quiet for about a second, the collected
   audio is one utterance.
3. The worker transcribes the utterance with the local Whisper model, sends
   the text to the existing Claude agent (chat.get_reply) with this pet's
   conversation, and adds both messages to the shared chat history.
4. The answer is turned into speech with macOS `say` and streamed back to the
   browser over the same WebRTC connection.
5. When playback is over, the session listens again.

Interruptions: while PetPulse speaks, it keeps listening. Clear, sustained
speech (stronger and longer than its own voice coming back through the
microphone usually is) pauses the answer. If that turns out to be a real
question, it's answered next; if it was only a cough or noise, PetPulse goes
back to reading the answer from where it paused. The microphone is ignored
while PetPulse is thinking. Only one turn is handled at a time.
This file has no Streamlit code, so it can be tested with fake audio.
"""

import queue
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np

import chat
import voice

IDLE = "idle"
LISTENING = "listening"
PROCESSING = "processing"
SPEAKING = "speaking"
STOPPED = "stopped"

SAMPLE_RATE = voice.SAMPLE_RATE  # 16 kHz, what Silero VAD and Whisper expect
CHUNK = 512                      # samples per VAD step (32 ms)
SPEAKER_RATE = 48000             # sample rate of the audio sent back to the browser


@dataclass
class LiveSettings:
    speech_threshold: float = 0.5   # Silero speech probability that counts as speech
    end_silence_ms: int = 1000      # pause that ends an utterance
    min_speech_ms: int = 300        # shorter sounds (coughs, clicks) are ignored
    pre_roll_ms: int = 300          # audio kept from just before speech starts
    max_utterance_s: int = 20       # an utterance is cut off after this long
    speaking_tail_ms: int = 500     # extra wait after playback before listening again
    # Interrupting PetPulse while it speaks:
    barge_in: bool = True            # set False to ignore the microphone while speaking
    barge_in_threshold: float = 0.8  # stricter than speech_threshold, so its own echo doesn't count
    barge_in_ms: int = 400           # how long the sitter must speak to pause the answer
    barge_in_grace_ms: int = 300     # ignore the first moments of playback (echo cancellation settling)
    resume_rewind_ms: int = 500      # when resuming, repeat this much of the answer


def words(text):
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def looks_like_echo(heard, answer):
    """
    True if what was heard is mostly words from the answer being read aloud,
    i.e. PetPulse probably heard its own voice through the speakers.
    """
    heard_words = words(heard)
    answer_words = set(words(answer))
    if len(heard_words) < 3 or not answer_words:
        return False
    found = sum(word in answer_words for word in heard_words)
    return found / len(heard_words) >= 0.8


class SileroVad:
    """
    Silero VAD (bundled with faster-whisper) applied one 512-sample chunk at a time.

    faster-whisper's own wrapper processes a whole recording at once, so this
    keeps the model's state between chunks instead.
    """

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self.reset()

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def __call__(self, chunk):
        model_input = np.concatenate([self._context, chunk])[None, :].astype(np.float32)
        output, self._h, self._c = self._session.run(None, {"input": model_input, "h": self._h, "c": self._c})
        self._context = chunk[-64:]
        return float(np.asarray(output).reshape(-1)[0])


class UtteranceDetector:
    """Groups a stream of 16 kHz samples into utterances, using a VAD function."""

    def __init__(self, vad, settings):
        self.vad = vad
        self.settings = settings
        ms_per_chunk = CHUNK * 1000 / SAMPLE_RATE
        self._end_silence_chunks = max(1, round(settings.end_silence_ms / ms_per_chunk))
        self._min_speech_chunks = max(1, round(settings.min_speech_ms / ms_per_chunk))
        self._max_chunks = round(settings.max_utterance_s * 1000 / ms_per_chunk)
        self._pre_roll = deque(maxlen=max(1, round(settings.pre_roll_ms / ms_per_chunk)))
        self.barge_in_chunks = max(1, round(settings.barge_in_ms / ms_per_chunk))
        self.reset()

    def reset(self):
        self._leftover = np.zeros(0, dtype=np.float32)
        self._pre_roll.clear()
        self._chunks = []
        self._speech_chunks = 0
        self._silent_chunks = 0
        self.in_speech = False
        self.strong_streak = 0  # chunks in a row that are clearly speech (used for interruptions)
        self.heard_strong_speech = False  # strong_streak reached barge_in_chunks since the last reset
        if hasattr(self.vad, "reset"):
            self.vad.reset()

    def add(self, samples):
        """Add samples; return a list of finished utterances (usually empty)."""
        samples = np.concatenate([self._leftover, np.asarray(samples, dtype=np.float32)])
        whole = len(samples) // CHUNK * CHUNK
        self._leftover = samples[whole:]
        finished = []
        for start in range(0, whole, CHUNK):
            utterance = self._add_chunk(samples[start:start + CHUNK])
            if utterance is not None:
                finished.append(utterance)
        return finished

    def _add_chunk(self, chunk):
        probability = self.vad(chunk)
        is_speech = probability >= self.settings.speech_threshold
        self.strong_streak = self.strong_streak + 1 if probability >= self.settings.barge_in_threshold else 0
        if self.strong_streak >= self.barge_in_chunks:
            self.heard_strong_speech = True
        if not self.in_speech:
            if is_speech:
                self.in_speech = True
                self._chunks = [*self._pre_roll, chunk]
                self._speech_chunks = 1
                self._silent_chunks = 0
            else:
                self._pre_roll.append(chunk)
            return None

        self._chunks.append(chunk)
        if is_speech:
            self._speech_chunks += 1
            self._silent_chunks = 0
        else:
            self._silent_chunks += 1

        if self._silent_chunks >= self._end_silence_chunks or len(self._chunks) >= self._max_chunks:
            long_enough = self._speech_chunks >= self._min_speech_chunks
            audio = np.concatenate(self._chunks)
            self._pre_roll.clear()
            self._chunks = []
            self.in_speech = False
            return audio if long_enough else None
        return None


class LiveSession:
    """
    One live conversation about one pet.

    The pet is fixed when the session is created; switching pets means stopping
    this session and starting a new one, so audio and answers can't cross pets.
    """

    def __init__(self, pet, store, settings=None, vad=None, clock=time.monotonic, *, user_id):
        self.pet = dict(pet)
        self.pet_id = pet["id"]
        self.user_id = user_id  # the signed-in user; access is checked on every question
        self.access_ended = False
        self.store = store
        self.settings = settings or LiveSettings()
        self.clock = clock
        self.speaker = None  # PcmAudioSource from streamlit-webrtc, set by the app
        self.heard = None    # the last thing the sitter said (for the status line)
        self.notice = None   # the last problem, shown to the sitter
        self.paused = False  # True after the sitter interrupted an answer

        self._vad = vad
        self._detector = None
        self._lock = threading.Lock()
        self._state = IDLE
        self._queue = queue.Queue(maxsize=1000)  # about 30 s of audio chunks
        self._stop = threading.Event()
        self._thread = None
        self._resampler = None
        self._speaking_until = 0.0
        self._playing_since = 0.0
        self._answer_pcm = None  # the answer audio currently playing
        self._answer_text = None  # ...and its text (to recognise PetPulse's own echo)
        self._paused_pcm = None  # the rest of an interrupted answer, in case it should resume
        self._last_audio_at = None

    # ----- State -----

    @property
    def state(self):
        with self._lock:
            return self._state

    def _set_state(self, new_state):
        with self._lock:
            if self._state == STOPPED:
                return False
            self._state = new_state
            return True

    @property
    def active(self):
        return not self._stop.is_set()

    def receiving_audio(self):
        """True if microphone audio arrived in the last 1.5 seconds."""
        return self._last_audio_at is not None and self.clock() - self._last_audio_at < 1.5

    # ----- Control (called by the app) -----

    def start(self, run_worker=True):
        """
        Start listening. Calling it again (Streamlit reruns) does nothing.
        Tests pass run_worker=False and call step() themselves.
        """
        with self._lock:
            if self._state != IDLE:
                return
            self._state = LISTENING
        if run_worker:
            self._thread = threading.Thread(target=self._run, name=f"live-voice-{self.pet_id}", daemon=True)
            self._thread.start()
        else:
            self._setup()

    def stop(self):
        """End the session: stop the worker, drop queued audio, and stop playback."""
        with self._lock:
            self._state = STOPPED
        self._stop.set()
        self._drain_queue()
        if self.speaker is not None:
            self.speaker.clear()

    # ----- Audio input (called by streamlit-webrtc for every frame; must stay fast) -----

    def _accepting_audio(self):
        state = self.state
        return state == LISTENING or (state == SPEAKING and self.settings.barge_in)

    def on_audio_frame(self, frame):
        self._last_audio_at = self.clock()
        if not self._accepting_audio():
            return  # ignore the microphone while thinking (and while speaking, if interruptions are off)
        if self._resampler is None:
            import av

            self._resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for converted in self._resampler.resample(frame):
            samples = converted.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
            self.feed(samples)

    def feed(self, samples):
        """Queue 16 kHz mono float32 samples for the worker."""
        if not self._accepting_audio():
            return
        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            pass  # the worker is behind; dropping audio is better than blocking WebRTC

    # ----- Worker thread -----

    def _setup(self):
        try:
            if self._vad is None:
                self._vad = SileroVad()
            self._detector = UtteranceDetector(self._vad, self.settings)
            return True
        except Exception:
            traceback.print_exc()
            self.notice = "Voice detection couldn't start. Please end the session and try again."
            self.stop()
            return False

    def _run(self):
        if not self._setup():
            return
        while self.active:
            self.step(timeout=0.1)

    def step(self, timeout=0.0):
        """Do one unit of work: finish playback, or process queued audio. Used by tests too."""
        if self.state == SPEAKING and self.clock() >= self._speaking_until:
            self._answer_pcm = None
            self._resume_listening()
            return
        try:
            samples = self._queue.get(timeout=timeout) if timeout else self._queue.get_nowait()
        except queue.Empty:
            return
        state = self.state
        if state == SPEAKING:
            self._listen_for_interruption(samples)
            return
        if state != LISTENING:
            return
        for utterance in self._detector.add(samples):
            self._handle_utterance(utterance)
            break  # the rest of this audio arrived before the turn; it's dropped

    def _listen_for_interruption(self, samples):
        """While PetPulse speaks: pause the answer if the sitter clearly starts talking."""
        utterances = self._detector.add(samples)
        playing_for = self.clock() - self._playing_since
        if playing_for < self.settings.barge_in_grace_ms / 1000:
            self._detector.heard_strong_speech = False
            return
        if not self._detector.heard_strong_speech:
            return  # quiet echo or short noise; utterances here are dropped

        # Keep the rest of the answer (with a short rewind) in case this was only noise.
        rewind = int(self.settings.resume_rewind_ms / 1000 * SPEAKER_RATE)
        played = int(playing_for * SPEAKER_RATE)
        remaining = self._answer_pcm[max(0, played - rewind):] if self._answer_pcm is not None else None
        self._paused_pcm = remaining if remaining is not None and len(remaining) > SPEAKER_RATE // 2 else None
        self._answer_pcm = None
        if self.speaker is not None:
            self.speaker.clear()  # stop reading right away
        self.paused = True
        # The detector keeps the speech it already heard, so the new question isn't cut off.
        self._detector.heard_strong_speech = False
        self._set_state(LISTENING)
        if utterances:  # the whole question already arrived in this batch
            self._handle_utterance(utterances[-1])

    def _start_speaking(self, pcm, text):
        if not self._set_state(SPEAKING):
            return
        self._answer_pcm = pcm
        self._answer_text = text
        self._playing_since = self.clock()
        self._speaking_until = self._playing_since + len(pcm) / SPEAKER_RATE + self.settings.speaking_tail_ms / 1000
        if self._detector is not None:
            self._detector.reset()
        self.speaker.push(pcm)

    def _resume_after_noise(self):
        """No real question was heard: finish a paused answer, or just keep listening."""
        paused, self._paused_pcm = self._paused_pcm, None
        self.paused = False
        if paused is not None and self.active and self.speaker is not None:
            self._drain_queue()
            self._start_speaking(paused, self._answer_text)
        else:
            self._resume_listening()

    def _resume_listening(self):
        self._drain_queue()
        if self._detector is not None:
            self._detector.reset()
        self._set_state(LISTENING)

    def _handle_utterance(self, audio):
        if not self._set_state(PROCESSING):
            return
        self._drain_queue()

        try:
            question = voice.transcribe_samples(audio)
        except voice.VoiceError:
            self._resume_after_noise()  # noise or nothing clear
            return
        if not self.active:
            return
        if sum(character.isalnum() for character in question) < 2:
            self._resume_after_noise()
            return
        if self._paused_pcm is not None and looks_like_echo(question, self._answer_text):
            self._resume_after_noise()  # it heard its own answer; keep reading
            return
        self.heard = question
        self.notice = None
        self._paused_pcm = None  # a new question replaces the interrupted answer
        self.paused = False

        # Same conversation as Text Chat and Voice Message, and the same agent.
        history = chat.get_history(self.store, self.pet)
        earlier = list(history)
        history.append({"role": "user", "content": question})
        try:
            reply = chat.get_reply(self.pet_id, question, earlier, user_id=self.user_id)
        except chat.AccessEnded as error:
            self.notice = str(error)
            self.access_ended = True
            self.stop()  # no more listening or speaking for a pet this user can't access
            return
        except chat.AgentError as error:
            self.notice = str(error)
            self._resume_listening()
            return
        if not self.active:
            return  # ended while Claude was answering; discard the answer
        history.append({"role": "assistant", "content": reply})

        try:
            pcm, _rate = voice.wav_to_pcm(voice.synthesize_speech(reply, sample_rate=SPEAKER_RATE))
        except voice.VoiceError as error:
            self.notice = str(error)
            self._resume_listening()
            return
        if not self.active or self.speaker is None:
            self._resume_listening()
            return
        self._start_speaking(pcm, reply)

    def _drain_queue(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
