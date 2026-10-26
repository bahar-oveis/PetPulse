"""
PetPulse sitter chat.

Keeps a separate conversation for each pet and gets the assistant's replies
from the Claude agent in agent.py. This file has no Streamlit code, so it is
easy to test.

Conversations are stored in a plain dictionary ("store") that maps a pet ID to a
list of messages. In the app, the store lives in st.session_state.
Each message looks like: {"role": "user" or "assistant", "content": "..."}
"""

import agent
from agent import AgentError  # app.py catches chat.AgentError to show friendly errors


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
    Return Claude's answer to the sitter's question.

    pet_id comes from the app (the pet the sitter selected), never from the AI,
    so the assistant can only ever see that one pet's profile.
    history is the earlier conversation, without the current question.
    Raises AgentError with a friendly message if the answer couldn't be produced.
    """
    return agent.ask(pet_id, question, history)


def send_message(store, pet, question):
    """
    Add the sitter's question and the assistant's reply to this pet's conversation.

    If the assistant fails, the question stays in the chat but no reply is saved,
    and the AgentError is passed on so the app can show a friendly error.
    """
    history = get_history(store, pet)
    earlier = list(history)
    history.append({"role": "user", "content": question})
    reply = get_reply(pet["id"], question, earlier)
    history.append({"role": "assistant", "content": reply})
    return reply
