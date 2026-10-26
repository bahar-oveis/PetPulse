"""
PetPulse sitter chat.

Keeps a separate conversation for each pet and gets the assistant's replies
from the Claude agent in agent.py. This file has no Streamlit code, so it is
easy to test.

Conversations are stored in a plain dictionary ("store") that maps a pet ID to a
list of messages. In the app, the store lives in st.session_state.
Each message looks like: {"role": "user" or "assistant", "content": "..."}

Voice questions use the same conversations. Their in-progress state lives in a
second dictionary ("voice_state", also in st.session_state):
- "pending": the latest transcribed recording, waiting for the sitter to confirm.
  {"pet_id": ..., "audio_id": ..., "text": ..., "error": ...}
- "replies": {pet_id: {"audio": WAV bytes or None, "error": message or None}}
  the spoken version of the latest voice answer for each pet.
"""

import hashlib

import agent
import voice
from agent import AccessEnded, AgentError  # app.py catches these to show friendly errors


def welcome_message(pet):
    name = pet["name"]
    return (
        f"Hi! I'm PetPulse, your assistant for taking care of {name}. "
        f"You can ask me about {name}'s routines, food, habits, and other care instructions."
    )


def get_history(store, pet):
    """Return this pet's conversation, starting a new one with a welcome message if needed."""
    if pet["id"] not in store:
        store[pet["id"]] = [{"role": "assistant", "content": welcome_message(pet)}]
    return store[pet["id"]]


def clear_history(store, pet_id):
    """Forget the conversation for one pet. Other pets' chats are not touched."""
    store.pop(pet_id, None)


def get_reply(pet_id, question, history, *, user_id):
    """
    Return Claude's answer to the sitter's question.

    pet_id comes from the app, never from the AI, and user_id is the signed-in
    user; the agent checks in the database that this user may use this pet.
    history is the earlier conversation, without the current question.
    Raises AgentError with a friendly message if the answer couldn't be produced.
    """
    return agent.ask(pet_id, question, history, user_id=user_id)


def get_label_reply(pet_id, image_bytes, media_type, label_text, check, question, history=(), *, user_id):
    """Claude's answer about a scanned food label (same agent, with the photo and label data)."""
    return agent.ask_about_label(pet_id, image_bytes, media_type, label_text, check, question, history,
                                 user_id=user_id)


def send_message(store, pet, question, *, user_id):
    """
    Add the sitter's question and the assistant's reply to this pet's conversation.

    If the assistant fails, the question stays in the chat but no reply is saved,
    and the AgentError is passed on so the app can show a friendly error.
    """
    history = get_history(store, pet)
    earlier = list(history)
    history.append({"role": "user", "content": question})
    reply = get_reply(pet["id"], question, earlier, user_id=user_id)
    history.append({"role": "assistant", "content": reply})
    return reply


# ----- Voice questions -----

def transcribe_recording(voice_state, pet_id, audio_bytes):
    """
    Transcribe a new recording for this pet and keep it as the pending question.

    Streamlit reruns the app often; a recording that was already transcribed
    (same pet, same audio) is not transcribed again.
    """
    audio_id = hashlib.sha256(audio_bytes or b"").hexdigest()
    pending = voice_state.get("pending")
    if pending and pending["pet_id"] == pet_id and pending["audio_id"] == audio_id:
        return pending

    try:
        text, error = voice.transcribe_audio(audio_bytes), None
    except voice.VoiceError as problem:
        text, error = "", str(problem)
    pending = {"pet_id": pet_id, "audio_id": audio_id, "text": text, "error": error}
    voice_state["pending"] = pending
    return pending


def pending_voice_question(voice_state, pet_id):
    """
    Return the pending voice question for this pet, or None.

    A pending question recorded for a different pet is thrown away, so it can
    never be sent to the wrong pet's assistant.
    """
    pending = voice_state.get("pending")
    if pending and pending["pet_id"] != pet_id:
        voice_state.pop("pending")
        return None
    return pending


def send_voice_question(store, voice_state, pet, text, *, user_id):
    """
    Send a confirmed (and possibly edited) voice question, then create the spoken answer.

    Returns Claude's text reply. The question goes through the same send_message()
    as typed questions, so both share one conversation. Raises AgentError if
    Claude fails. If speech can't be created, the text reply is still returned
    and the problem is saved in voice_state["replies"].
    """
    pending = pending_voice_question(voice_state, pet["id"])
    if pending is None:
        raise AgentError("There is no recorded question to send. Please record it again.")
    question = (text or "").strip()
    if not question:
        raise AgentError("The question is empty. Please record it again or type it.")

    voice_state.pop("pending")  # sent once; reruns can't send it again
    voice_state.setdefault("replies", {}).pop(pet["id"], None)
    reply = send_message(store, pet, question, user_id=user_id)

    try:
        audio, error = voice.synthesize_speech(reply), None
    except voice.VoiceError as problem:
        audio, error = None, str(problem)
    voice_state["replies"][pet["id"]] = {"audio": audio, "error": error}
    return reply


def clear_voice(voice_state, pet_id):
    """Forget this pet's pending recording and spoken answer (used by Clear Chat)."""
    pending = voice_state.get("pending")
    if pending and pending["pet_id"] == pet_id:
        voice_state.pop("pending")
    voice_state.get("replies", {}).pop(pet_id, None)
