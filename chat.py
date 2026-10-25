"""
PetPulse sitter chat.

Keeps a separate conversation for each pet and produces the assistant's replies.
This file has no Streamlit code, so it is easy to test and, in Phase 4, easy to
connect to the OpenAI agent.

Conversations are stored in a plain dictionary ("store") that maps a pet ID to a
list of messages. In the app, the store lives in st.session_state.
Each message looks like: {"role": "user" or "assistant", "content": "..."}
"""

import database as db


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


def get_reply(pet_id, question, history):
    """
    Return the assistant's answer to the sitter's question.

    pet_id comes from the app (the pet the sitter selected), never from the AI,
    so the assistant can only ever talk about that one pet.

    Phase 4: replace the placeholder below with a call to the agent in agent.py.
    """
    pet = db.get_pet(pet_id)
    if pet is None:
        return "Sorry, I couldn't find this pet's profile."
    return (
        "I'm not connected to the AI agent yet. In Phase 4, I'll be able to "
        f"answer questions using {pet['name']}'s profile."
    )


def send_message(store, pet, question):
    """Add the sitter's question and the assistant's reply to this pet's conversation."""
    history = get_history(store, pet)
    history.append({"role": "user", "content": question})
    reply = get_reply(pet["id"], question, history)
    history.append({"role": "assistant", "content": reply})
    return reply
