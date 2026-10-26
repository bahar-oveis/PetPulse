"""PetPulse - Streamlit web app."""

import streamlit as st

import chat
import database as db
import live_voice

st.set_page_config(page_title="PetPulse", page_icon="🐾", layout="centered")

# Make sure the database exists, and add the demo pet on first run.
db.init_db()
db.add_demo_pet_if_empty()

# Profile sections: (title, [(field, label), ...]). Used by the forms and the profile view.
SECTIONS = [
    ("Basic Information", [
        ("name", "Name"),
        ("species", "Species"),
        ("breed", "Breed"),
        ("age", "Age (years)"),
    ]),
    ("Food & Daily Routine", [
        ("diet", "Diet"),
        ("feeding_schedule", "Feeding schedule"),
        ("daily_routine", "Daily routine"),
        ("habits", "Habits"),
    ]),
    ("Behavior & Sensitivities", [
        ("allergies", "Allergies"),
        ("sensitivities", "Sensitivities"),
        ("fears", "Fears"),
        ("behavior_notes", "Behavior notes"),
    ]),
    ("Medical Information", [
        ("medical_conditions", "Medical conditions"),
        ("medications", "Medications"),
        ("medical_history", "Medical history"),
    ]),
    ("Emergency Contacts & Instructions", [
        ("emergency_instructions", "Emergency instructions"),
        ("owner_contact", "Owner contact"),
        ("vet_name", "Veterinarian name"),
        ("vet_contact", "Veterinarian contact"),
    ]),
]

# Short single-line fields use text_input; everything else uses text_area.
SHORT_FIELDS = {"name", "breed", "owner_contact", "vet_name", "vet_contact"}

# Session state: which pet is being edited, the form "version" (bumped to clear
# the Add form after saving), and a message to show after the page reloads.
st.session_state.setdefault("editing_pet_id", None)
st.session_state.setdefault("add_form_version", 0)
st.session_state.setdefault("flash", None)
# Sitter chats: {pet_id: [messages]}, so each pet has its own conversation.
st.session_state.setdefault("chats", {})
# Voice questions: pending transcription and spoken answers (see chat.py), the
# recorder "version" per pet (bumped to clear the recorder), and an error to show once.
st.session_state.setdefault("voice", {})
st.session_state.setdefault("voice_versions", {})
st.session_state.setdefault("voice_notice", None)
# Live Conversation: the running live_voice.LiveSession (one at a time), or None.
st.session_state.setdefault("live_session", None)

TEXT_MODE = "💬 Text Chat"
VOICE_MODE = "🎙️ Voice Message"
LIVE_MODE = "🔴 Live Conversation"


def pet_form_fields(key_prefix, pet=None):
    """Draw all pet input fields (inside an st.form) and return the entered values."""
    pet = pet or {}
    values = {}
    for title, fields in SECTIONS:
        st.markdown(f"#### {title}")
        for field, label in fields:
            key = f"{key_prefix}_{field}"
            current = pet.get(field)
            if field == "name":
                values[field] = st.text_input("Name *", value=current or "", key=key)
            elif field == "species":
                index = db.SPECIES_OPTIONS.index(current) if current in db.SPECIES_OPTIONS else None
                values[field] = st.selectbox(
                    "Species *", db.SPECIES_OPTIONS, index=index,
                    placeholder="Choose species", key=key,
                )
            elif field == "age":
                values[field] = st.number_input(
                    label, min_value=0, max_value=50, step=1, value=current, key=key,
                )
            elif field in SHORT_FIELDS:
                values[field] = st.text_input(label, value=current or "", key=key)
            else:
                values[field] = st.text_area(label, value=current or "", height=80, key=key)
    return values


def missing_required(values):
    """Return a list of required fields that are empty."""
    missing = []
    if not values["name"].strip():
        missing.append("Name")
    if not values["species"]:
        missing.append("Species")
    return missing


def show_profile(pet):
    """Show a pet's full profile, grouped into sections."""
    for title, fields in SECTIONS:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            for field, label in fields:
                value = pet.get(field)
                st.markdown(f"{label}: {value if value not in (None, '') else '—'}")
    st.caption(f"Created: {pet['created_at']} · Last updated: {pet['updated_at']}")


def show_edit_form(pet):
    """Show a pre-filled form for editing an existing pet."""
    with st.form(f"edit_pet_form_{pet['id']}"):
        st.markdown(f"### ✏️ Editing {pet['name']}")
        values = pet_form_fields(f"edit_{pet['id']}", pet)
        save_col, cancel_col = st.columns(2)
        saved = save_col.form_submit_button("Save changes", type="primary", width="stretch")
        cancelled = cancel_col.form_submit_button("Cancel", width="stretch")

    if cancelled:
        st.session_state.editing_pet_id = None
        st.rerun()
    if saved:
        missing = missing_required(values)
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            db.update_pet(pet["id"], values)
            st.session_state.editing_pet_id = None
            st.session_state.flash = f"✅ {values['name'].strip()}'s profile was updated."
            st.rerun()


def show_my_pets():
    pets = db.get_all_pets()
    if not pets:
        st.info("No pets yet. Add your first pet in **Add New Pet**.")
        return

    st.dataframe(
        [{"Name": p["name"], "Species": p["species"], "Breed": p["breed"] or "—", "Age": p["age"]} for p in pets],
        hide_index=True,
        width="stretch",
    )

    pets_by_id = {p["id"]: p for p in pets}
    selected_id = st.selectbox(
        "Select a pet to view its full profile",
        list(pets_by_id),
        format_func=lambda pet_id: pets_by_id[pet_id]["name"],
    )
    pet = pets_by_id[selected_id]

    if st.session_state.editing_pet_id == pet["id"]:
        show_edit_form(pet)
    else:
        title_col, button_col = st.columns([3, 1])
        title_col.markdown(f"### 🐾 {pet['name']}")
        if button_col.button("Edit Profile", key=f"edit_button_{pet['id']}", width="stretch"):
            st.session_state.editing_pet_id = pet["id"]
            st.rerun()
        show_profile(pet)


def show_add_pet():
    st.caption("Only **Name** and **Species** are required. You can fill in the rest later.")
    version = st.session_state.add_form_version
    with st.form(f"add_pet_form_{version}"):
        values = pet_form_fields(f"add_{version}")
        submitted = st.form_submit_button("Save pet", type="primary", width="stretch")

    if submitted:
        missing = missing_required(values)
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            db.create_pet(values)
            st.session_state.add_form_version += 1  # gives the next form fresh, empty fields
            st.session_state.flash = f"✅ {values['name'].strip()} was saved! You can find them in **My Pets**."
            st.rerun()


def or_dash(value):
    return value if value not in (None, "") else "—"


def show_pet_summary(pet):
    """Compact card with the most important facts; details are in expanders."""
    with st.container(border=True):
        st.markdown(f"### 🐾 {pet['name']}")
        species_col, breed_col, age_col = st.columns(3)
        species_col.markdown(f"**Species**  \n{pet['species']}")
        breed_col.markdown(f"**Breed**  \n{or_dash(pet['breed'])}")
        age_col.markdown(f"**Age**  \n{or_dash(pet['age'])}")

        st.markdown(f"⚠️ **Allergies:** {or_dash(pet['allergies'])}")
        st.markdown(f"🔊 **Sensitivities:** {or_dash(pet['sensitivities'])}")

        routine = " · ".join(text for text in (pet["feeding_schedule"], pet["daily_routine"]) if text)
        if len(routine) > 150:
            routine = routine[:150].rstrip() + "…"
        st.markdown(f"🕗 **Routine:** {or_dash(routine)}")

    with st.expander("🚨 Emergency contacts & instructions"):
        st.markdown(f"**Emergency instructions:** {or_dash(pet['emergency_instructions'])}")
        st.markdown(f"**Owner contact:** {or_dash(pet['owner_contact'])}")
        st.markdown(f"**Veterinarian:** {or_dash(pet['vet_name'])} ({or_dash(pet['vet_contact'])})")

    with st.expander("🩺 Medical information"):
        st.markdown(f"**Medical conditions:** {or_dash(pet['medical_conditions'])}")
        st.markdown(f"**Medications:** {or_dash(pet['medications'])}")
        st.markdown(f"**Medical history:** {or_dash(pet['medical_history'])}")


def show_chat(pet):
    """Chat with the assistant about the selected pet. Each pet has its own history."""
    chats = st.session_state.chats

    title_col, button_col = st.columns([3, 1])
    title_col.markdown("#### 💬 Ask PetPulse")
    if button_col.button("Clear Chat", key=f"clear_chat_{pet['id']}", width="stretch"):
        chat.clear_history(chats, pet["id"])
        chat.clear_voice(st.session_state.voice, pet["id"])
        reset_recorder(pet["id"])

    mode = st.radio("Chat mode", [TEXT_MODE, VOICE_MODE, LIVE_MODE], horizontal=True,
                    label_visibility="collapsed", key="chat_mode")
    # A live session only runs while its pet is selected and Live mode is open.
    stop_live_session(keep_pet_id=pet["id"] if mode == LIVE_MODE else None)
    if mode == LIVE_MODE:
        show_live_conversation(pet)
        return

    messages = show_messages(chat.get_history(chats, pet))

    if mode == VOICE_MODE:
        spoken = st.session_state.voice.get("replies", {}).get(pet["id"])
        if spoken and spoken["audio"]:
            st.caption("🔊 Spoken answer to your last voice question")
            st.audio(spoken["audio"], format="audio/wav")
        elif spoken:
            st.info(f"🔇 {spoken['error']}")
        show_voice_input(pet)
        return

    question = st.chat_input(f"Ask about {pet['name']}'s care…", key="sitter_chat_input")
    # chat_input only returns the question on the run right after the sitter sends it,
    # so Streamlit reruns (clicks, expanders) never call Claude again.
    if question:
        with messages:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking…"):
                        reply = chat.send_message(chats, pet, question)
                    st.markdown(reply)
                except chat.AgentError as error:
                    st.error(str(error))


def show_messages(history):
    """Fixed-height board: messages scroll inside it instead of making the page longer."""
    messages = st.container(height=420, border=True)
    with messages:
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    return messages


def stop_live_session(keep_pet_id=None):
    """Stop the running live session, unless it belongs to keep_pet_id."""
    live = st.session_state.live_session
    if live is not None and live.pet_id != keep_pet_id:
        live.stop()
        st.session_state.live_session = None


def show_live_conversation(pet):
    """Hands-free, turn-based voice chat over WebRTC (see live_voice.py)."""
    from streamlit_webrtc import (
        WebRtcMode,
        create_audio_sink_track,
        create_pcm_audio_source_track,
        webrtc_streamer,
    )

    live = st.session_state.live_session
    if live is None or live.state == live_voice.STOPPED:
        live = live_voice.LiveSession(pet, st.session_state.chats)
        st.session_state.live_session = live

    st.caption(
        f"Talk with PetPulse about **{pet['name']}** hands-free: start the conversation, ask a question, "
        "then pause. PetPulse answers out loud and then listens again. Speech is turned into text and "
        "back into speech on this computer; only the text is sent to Claude. Headphones work best."
    )

    # Microphone audio goes to the session; the session's spoken answers go back to the browser.
    microphone = create_audio_sink_track(callback=live.on_audio_frame, key=f"live-mic-{pet['id']}")
    speaker = create_pcm_audio_source_track(key=f"live-speaker-{pet['id']}", sample_rate=live_voice.SPEAKER_RATE)
    live.speaker = speaker
    stream = webrtc_streamer(
        key=f"live-{pet['id']}",
        mode=WebRtcMode.SENDRECV,
        sink_audio_track=microphone,
        source_audio_track=speaker.track,
        media_stream_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True},
            "video": False,
        },
        rtc_configuration={"iceServers": []},  # browser and this server connect directly (localhost)
        audio_html_attrs={"autoPlay": True, "controls": False},
        translations={"start": "Start Live Conversation", "stop": "End Session"},
    )
    if stream.state.playing:
        live.start()
    elif live.state != live_voice.IDLE:
        live.stop()

    show_live_status(pet)


LIVE_STATUS = {
    live_voice.IDLE: "⚪ **Ready** — click *Start Live Conversation*.",
    live_voice.LISTENING: "🟢 **Listening** — ask your question, then pause.",
    live_voice.PROCESSING: "🟡 **Processing** — understanding your question and asking PetPulse…",
    live_voice.SPEAKING: "🔵 **Speaking** — PetPulse is answering out loud. Start talking to interrupt.",
    live_voice.STOPPED: "⚫ **Session ended.**",
}


@st.fragment(run_every=1.0)
def show_live_status(pet):
    """Refreshes every second: state, microphone, and the shared conversation."""
    live = st.session_state.live_session
    if live is None:
        return
    state = live.state
    if state in (live_voice.LISTENING, live_voice.PROCESSING, live_voice.SPEAKING):
        microphone = "🎤 Microphone on" if live.receiving_audio() else "🎤 Waiting for microphone audio…"
    else:
        microphone = "🎤 Microphone off"
    st.markdown(f"{LIVE_STATUS[state]} &nbsp;·&nbsp; {microphone}")
    if live.paused and state == live_voice.LISTENING:
        st.caption("⏸️ Answer paused — go ahead and ask. If it was just a noise, PetPulse will continue the answer.")
    if live.heard:
        st.caption(f"Last heard: “{live.heard}”")
    if live.notice:
        st.warning(live.notice)
    show_messages(chat.get_history(st.session_state.chats, pet))


def reset_recorder(pet_id):
    """Give this pet's recorder a new key, so it starts empty on the next run."""
    versions = st.session_state.voice_versions
    versions[pet_id] = versions.get(pet_id, 0) + 1


def show_voice_input(pet):
    """Push-to-talk: record, check the transcription, then send it like a typed question."""
    voice_state = st.session_state.voice
    # Drops a recording made for a different pet, so it can't be sent to this one.
    pending = chat.pending_voice_question(voice_state, pet["id"])

    st.markdown("#### 🎙️ Ask by voice")
    st.caption(
        "Record your question, check the text, then send it. Speech is turned into text "
        "on this computer; only the confirmed text is sent to Claude. "
        "Allow microphone access when your browser asks, or type your question instead."
    )

    notice = st.session_state.voice_notice
    if notice and notice["pet_id"] == pet["id"]:
        st.error(notice["message"])
    st.session_state.voice_notice = None

    version = st.session_state.voice_versions.get(pet["id"], 0)
    recording = st.audio_input(
        "Record a question", key=f"voice_input_{pet['id']}_{version}", label_visibility="collapsed"
    )
    if recording is None:
        if pending:  # the sitter deleted the recording
            voice_state.pop("pending")
        return

    with st.spinner("Transcribing on this computer…"):
        pending = chat.transcribe_recording(voice_state, pet["id"], recording.getvalue())
    if pending["error"]:
        st.warning(pending["error"])
        return

    text = st.text_area(
        "Check the question and fix any mistakes before sending:",
        value=pending["text"],
        key=f"voice_text_{pending['audio_id'][:16]}",
    )
    if st.button("Send to PetPulse", type="primary", key=f"voice_send_{pet['id']}"):
        if not text.strip():
            st.warning("The question is empty. Please record it again or type it.")
            return
        try:
            with st.spinner("Thinking…"):
                chat.send_voice_question(st.session_state.chats, voice_state, pet, text)
        except chat.AgentError as error:
            st.session_state.voice_notice = {"pet_id": pet["id"], "message": str(error)}
        reset_recorder(pet["id"])
        st.rerun()  # show the new messages in the chat and an empty recorder


def show_sitter_dashboard():
    pets = db.get_all_pets()
    if not pets:
        st.info("No pets are available yet. Ask the pet's owner to create a profile in the **Pet Owner** tab.")
        return

    st.badge("Demo Mode", icon="🧪", color="orange")
    st.caption(
        "Selecting a pet simulates being authorized by its owner. "
        "Access control is not implemented yet."
    )

    # The selected pet ID is kept in session state by the app. The chat (and the
    # future AI agent) only ever receives this ID; the AI cannot pick another pet.
    pets_by_id = {p["id"]: p for p in pets}
    selected_id = st.selectbox(
        "Which pet are you caring for?",
        list(pets_by_id),
        format_func=lambda pet_id: f"{pets_by_id[pet_id]['name']} ({pets_by_id[pet_id]['species']})",
        key="sitter_pet_id",
    )
    pet = pets_by_id[selected_id]

    show_pet_summary(pet)
    st.divider()
    show_chat(pet)


# Header
st.title("🐾 PetPulse")
st.caption("AI-powered pet care assistant")
st.divider()

# The key keeps the selected tab when the app reruns (e.g. after a voice question).
owner_tab, sitter_tab = st.tabs(["🏠 Pet Owner", "🧑‍🍼 Pet Sitter"], key="main_tab", on_change="rerun")

# Pet Owner tab
with owner_tab:
    st.subheader("Welcome, Pet Owner!")
    st.write(
        "Create profiles for your pets with their habits, feeding routines, "
        "allergies, medical history, and emergency information."
    )

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    view = st.radio("View", ["My Pets", "Add New Pet"], horizontal=True, label_visibility="collapsed")
    if view == "My Pets":
        show_my_pets()
    else:
        show_add_pet()

# Pet Sitter tab
with sitter_tab:
    st.subheader("Welcome, Pet Sitter!")
    st.write(
        "Get personalized guidance for the pets you care for, "
        "based on each pet's profile."
    )
    show_sitter_dashboard()
