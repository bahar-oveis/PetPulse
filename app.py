"""PetPulse - Streamlit web app."""

import streamlit as st

import hashlib
from datetime import datetime

import access
import auth
import chat
import database as db
import food_checker
import live_voice
import memory
import vision

st.set_page_config(page_title="PetPulse", page_icon="🐾", layout="centered")

# Create or update the database tables (existing data is kept; see database.init_db).
db.init_db()

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
# Food & Allergy Scanner: the current scan (tagged with its pet ID), and an
# uploader "version" per pet (bumped by "Scan another label" to clear the image).
st.session_state.setdefault("scan", None)
st.session_state.setdefault("scan_versions", {})

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


def show_edit_form(user, pet):
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
            access.update_owner_pet(user["id"], pet["id"], values)  # checks that this owner owns the pet
            st.session_state.editing_pet_id = None
            st.session_state.flash = f"✅ {values['name'].strip()}'s profile was updated."
            st.rerun()


def show_my_pets(user):
    pets = access.list_owner_pets(user["id"])  # only this owner's pets
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
        show_edit_form(user, pet)
    else:
        title_col, button_col = st.columns([3, 1])
        title_col.markdown(f"### 🐾 {pet['name']}")
        if button_col.button("Edit Profile", key=f"edit_button_{pet['id']}", width="stretch"):
            st.session_state.editing_pet_id = pet["id"]
            st.rerun()
        show_profile(pet)
        show_access_management(user, pet)
        with st.expander("📝 Sitter observations (unverified)"):
            st.caption("Notes saved by pet sitters. They are not added to the profile above.")
            show_memory_timeline(user, pet)


def show_access_management(user, pet):
    """Owner: see who has access, create a one-time invitation code, rotate it, or revoke access."""
    with st.expander("🔑 Manage Sitter Access", expanded=True):
        new_code = st.session_state.get("new_invite")
        if new_code and new_code["pet_id"] == pet["id"]:
            st.success(
                f"Share this access code with {pet['name']}'s sitter. It works **once**, expires in "
                f"{access.INVITE_TTL_HOURS} hours, and is **shown only now** — PetPulse stores only a scrambled "
                "(hashed) version."
            )
            st.code(new_code["code"], language=None)
        st.session_state.new_invite = None  # shown once

        status = access.access_status(user["id"], pet["id"])
        sitter = status["sitter_email"]
        if sitter:
            st.markdown(f"👤 **Current sitter:** {sitter} (since {status['granted_at']})")
        else:
            st.markdown("👤 **Current sitter:** none")
        if status["open_invite"]:
            st.caption(f"An unused access code is waiting (expires {status['open_invite']['expires_at']}).")

        def issue(make_code):
            st.session_state.new_invite = {"pet_id": pet["id"], "code": make_code(user["id"], pet["id"])}
            st.rerun()

        if not sitter:
            label = "Generate New Access Key" if status["open_invite"] else "Generate Access Key"
            if st.button(label, type="primary", key=f"invite_{pet['id']}"):
                issue(access.create_invite)  # cancels any unused code
            if status["open_invite"] and st.button("Cancel unused code", key=f"cancel_invite_{pet['id']}"):
                access.revoke_access(user["id"], pet["id"])
                st.rerun()
            return

        st.caption(
            f"**Rotate** ends {sitter}'s access immediately and creates a new code for another sitter. "
            "**Revoke** ends their access without a new code."
        )
        confirmed = st.checkbox(f"Yes, end {sitter}'s access to {pet['name']}", key=f"confirm_revoke_{pet['id']}")
        rotate_col, revoke_col = st.columns(2)
        if rotate_col.button("Rotate Access Key", disabled=not confirmed, key=f"rotate_{pet['id']}", width="stretch"):
            issue(access.rotate_invite)
        if revoke_col.button("Revoke Sitter Access", disabled=not confirmed, key=f"revoke_{pet['id']}", width="stretch"):
            access.revoke_access(user["id"], pet["id"])
            st.session_state.flash = f"✅ {sitter} no longer has access to {pet['name']}."
            st.rerun()


def show_add_pet(user):
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
            access.create_owner_pet(user["id"], values)  # owned by this account
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


STATUS_DISPLAY = {
    food_checker.POTENTIAL_CONFLICT: (st.error, "⚠️ **Potential Conflict** — a recorded allergy or diet restriction appears in the ingredients."),
    food_checker.NO_MATCH_DETECTED: (st.info, "ℹ️ **No Match Detected** — none of the recorded allergies were found. This does **not** mean the food is safe."),
    food_checker.UNABLE_TO_VERIFY: (st.warning, "❓ **Unable to Verify** — the ingredients can't be checked reliably."),
}


def current_scan(pet):
    """The scan for this pet, or None. A scan made for another pet is thrown away."""
    scan = st.session_state.scan
    if scan is not None and scan["pet_id"] != pet["id"]:
        st.session_state.scan = scan = None
    return scan


def show_food_scanner(user, pet):
    """Food & Allergy Scanner: local image processing and OCR, rule-based check, then Claude."""
    with st.expander("🔍 Food & Allergy Scanner"):
        scan = current_scan(pet)
        st.caption(
            f"Photograph or upload a food or treat label. The photo is processed and read **on this computer**; "
            f"PetPulse then checks the ingredients against {pet['name']}'s recorded allergies and asks Claude to "
            "explain. Photos and results are not saved."
        )
        version = st.session_state.scan_versions.get(pet["id"], 0)
        source = st.radio("Image source", ["📁 Upload image", "📷 Take a picture"], horizontal=True,
                          key=f"scan_source_{pet['id']}", label_visibility="collapsed")
        if source.startswith("📁"):
            photo = st.file_uploader("Food label photo (JPG or PNG)", type=["jpg", "jpeg", "png"],
                                     key=f"scan_upload_{pet['id']}_{version}")
        else:
            photo = st.camera_input("Take a photo of the label", key=f"scan_camera_{pet['id']}_{version}")
        if photo is None:
            st.session_state.scan = None
            return

        data = photo.getvalue()
        image_id = hashlib.sha256(data).hexdigest()
        if scan is not None and scan["image_id"] != image_id:
            st.session_state.scan = scan = None  # a new photo needs a new analysis
        if scan is None:
            try:
                st.image(data, caption="Preview", width=320)
            except Exception:
                st.warning("This file can't be previewed. It may not be a valid image.")

        # Only this button starts processing, so reruns never repeat it.
        if st.button("Analyze Food Label", type="primary", key=f"scan_analyze_{pet['id']}"):
            access.require_pet_access(user["id"], pet["id"], access.VISION)
            with st.spinner("Processing the photo and reading the label on this computer…"):
                scan = analyze_label_photo(pet, data, image_id)
            st.session_state.scan = scan
        if scan is not None:
            show_scan_results(user, pet, scan)


def analyze_label_photo(pet, data, image_id):
    """Run the local pipeline (validation, preprocessing, OCR). No Claude call here."""
    scan = {"pet_id": pet["id"], "image_id": image_id, "error": None, "warnings": [], "conversation": [],
            "claude_error": None, "evidence": None}
    try:
        prepared = vision.prepare_label_image(data)
        ocr = vision.extract_ingredients_text(prepared.ocr_image)
    except vision.VisionError as error:
        scan["error"] = str(error)
        return scan

    if prepared.is_blurry:
        scan["warnings"].append(
            f"The photo looks blurry (sharpness {prepared.blur_score:.0f}; below {vision.BLUR_THRESHOLD:.0f} is "
            "usually too soft). Please retake it closer, in good light. This check is only a rough guide."
        )
    if not ocr.readable:
        scan["warnings"].append(
            f"Little readable text was found ({ocr.word_count} words, average confidence {ocr.confidence:.0f}%). "
            "Please fix the text below or upload a clearer photo."
        )
    claude_image, media_type = vision.image_for_claude(prepared.color)
    scan.update(
        ocr_text=ocr.text, ocr_confidence=ocr.confidence, ocr_readable=ocr.readable,
        is_blurry=prepared.is_blurry, claude_image=claude_image, media_type=media_type,
    )
    return scan


def show_scan_results(user, pet, scan):
    if scan["error"]:
        st.error(scan["error"])
        return

    st.markdown("##### 1. Image processing")
    st.image(scan["claude_image"], caption="Processed photo (orientation fixed, resized)", width=320)
    st.caption(f"🖥️ Processed and read on this computer · OCR confidence {scan['ocr_confidence']:.0f}%")
    for warning in scan["warnings"]:
        st.warning(warning)
    text = st.text_area(
        "Ingredient text read from the label — please check it and fix any mistakes:",
        value=scan["ocr_text"], height=160, key=f"scan_text_{scan['image_id'][:16]}",
    )
    edited = text.strip() != scan["ocr_text"].strip()
    # A blurry photo is never "verified", even if the text looks fine; otherwise the sitter's edits count.
    reliable = not scan["is_blurry"] and (scan["ocr_readable"] or edited)

    st.markdown("##### 2. Allergy check (rule-based)")
    check = food_checker.check_allergens(text, pet["allergies"], pet["diet"], reliable=reliable)
    st.markdown(f"**{pet['name']}'s recorded allergies:** {pet['allergies'] or 'Not recorded'}  \n"
                f"**Diet:** {pet['diet'] or 'Not recorded'}")
    show_status, message = STATUS_DISPLAY[check["status"]]
    show_status(message)
    for match in check["matches"]:
        st.markdown(f"- **{match['phrase']}** matches *{match['allergy']}* ({match['source'].lower()}) "
                    f"— on the label: “{match['evidence']}”")
    for reason in check["reasons"][1:]:
        st.caption(reason)

    st.markdown("##### 3. PetPulse explanation (Claude)")
    show_label_conversation(user, pet, scan, text, check)

    if st.button("Scan another label", key=f"scan_reset_{pet['id']}"):
        st.session_state.scan = None
        versions = st.session_state.scan_versions
        versions[pet["id"]] = versions.get(pet["id"], 0) + 1
        st.rerun()


def show_label_conversation(user, pet, scan, text, check):
    conversation = scan["conversation"]
    for message in conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if scan["claude_error"]:
        st.error(f"{scan['claude_error']} The rule-based check above still applies.")

    with st.form(f"label_question_{pet['id']}_{scan['image_id'][:16]}", clear_on_submit=True):
        label = "Ask a follow-up about this label" if conversation else "Your question"
        question = st.text_input(label, value="" if conversation else f"Can {pet['name']} eat this?")
        asked = st.form_submit_button("Ask PetPulse about this label")
    # Claude is only called on the run right after this click.
    if asked and question.strip():
        try:
            with st.spinner("Asking PetPulse…"):
                reply = chat.get_label_reply(pet["id"], scan["claude_image"], scan["media_type"], text, check,
                                             question.strip(), conversation, user_id=user["id"])
        except chat.AccessEnded:
            st.rerun()  # the sitter dashboard will show that access has ended
        except chat.AgentError as error:
            scan["claude_error"] = str(error)
        else:
            conversation += [{"role": "user", "content": question.strip()}, {"role": "assistant", "content": reply}]
            scan["claude_error"] = None
            scan["evidence"] = {"text": text, "check": check}
        st.rerun()

    evidence = scan["evidence"]
    if evidence:
        with st.expander("Evidence used for the explanation"):
            st.markdown(f"**Owner-recorded allergies:** {pet['allergies'] or 'Not recorded'}  \n"
                        f"**Diet:** {pet['diet'] or 'Not recorded'}  \n"
                        f"**Rule-based result:** {evidence['check']['status'].replace('_', ' ').title()}")
            phrases = ", ".join(m["phrase"] for m in evidence["check"]["matches"]) or "none"
            st.markdown(f"**Matched ingredient phrases:** {phrases}")
            st.text(evidence["text"] or "(no text)")
            if evidence["text"] != text:
                st.caption("The text has changed since this answer. Ask again to use the new text.")


def show_pet_memory(user, pet):
    """Sitter's Pet Memory: observations that are remembered across visits."""
    with st.expander("📝 Pet Memory"):
        st.caption(
            "Observations saved here are remembered across visits and restarts, and PetPulse can "
            "look them up when answering. They are unverified sitter notes, not owner-approved facts. "
            "This is a local demo: please use fictional records only."
        )
        with st.form(f"memory_form_{pet['id']}", clear_on_submit=True):
            note = st.text_area(
                "Add observation",
                placeholder=f"e.g. {pet['name']} hid under the bed when the doorbell rang.",
                max_chars=memory.MAX_LENGTH,
            )
            saved = st.form_submit_button("Save to Memory", type="primary")
        # A form submit returns True only on the run right after the click, so reruns can't save twice.
        if saved:
            try:
                with st.spinner("Saving…"):
                    access.save_memory(user["id"], pet["id"], note)
                st.success("✅ Saved to memory.")
            except memory.PetMemoryError as error:
                st.warning(str(error))
        show_memory_timeline(user, pet)


def show_memory_timeline(user, pet):
    """List a pet's observations, newest first. Owners can delete any; sitters only their own."""
    memories = list(reversed(access.list_memories(user["id"], pet["id"])))
    if not memories:
        st.caption("No observations recorded yet.")
        return
    for item in memories:
        when = datetime.fromisoformat(item["created_at"]).strftime("%d %b %Y, %H:%M")
        text = f"**{when}** · {item['source']}  \n{item['content']}"
        if not access.can_delete_memory(user["id"], pet["id"], item):
            st.markdown(text)
            continue
        text_col, button_col = st.columns([5, 1])
        text_col.markdown(text)
        if button_col.button("Delete", key=f"delete_memory_{item['id']}"):
            access.delete_memory(user["id"], pet["id"], item["id"])
            st.rerun()


def show_chat(user, pet):
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
        show_live_conversation(user, pet)
        return

    messages = show_messages(chat.get_history(chats, pet))

    if mode == VOICE_MODE:
        spoken = st.session_state.voice.get("replies", {}).get(pet["id"])
        if spoken and spoken["audio"]:
            st.caption("🔊 Spoken answer to your last voice question")
            st.audio(spoken["audio"], format="audio/wav")
        elif spoken:
            st.info(f"🔇 {spoken['error']}")
        show_voice_input(user, pet)
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
                        reply = chat.send_message(chats, pet, question, user_id=user["id"])
                    st.markdown(reply)
                except chat.AccessEnded:
                    st.rerun()  # the sitter dashboard will show that access has ended
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


def show_live_conversation(user, pet):
    """Hands-free, turn-based voice chat over WebRTC (see live_voice.py)."""
    from streamlit_webrtc import (
        WebRtcMode,
        create_audio_sink_track,
        create_pcm_audio_source_track,
        webrtc_streamer,
    )

    live = st.session_state.live_session
    if live is None or live.state == live_voice.STOPPED:
        live = live_voice.LiveSession(pet, st.session_state.chats, user_id=user["id"])
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

    show_live_status(user, pet)


LIVE_STATUS = {
    live_voice.IDLE: "⚪ **Ready** — click *Start Live Conversation*.",
    live_voice.LISTENING: "🟢 **Listening** — ask your question, then pause.",
    live_voice.PROCESSING: "🟡 **Processing** — understanding your question and asking PetPulse…",
    live_voice.SPEAKING: "🔵 **Speaking** — PetPulse is answering out loud. Start talking to interrupt.",
    live_voice.STOPPED: "⚫ **Session ended.**",
}


@st.fragment(run_every=1.0)
def show_live_status(user, pet):
    """Refreshes every second: state, microphone, and the shared conversation."""
    live = st.session_state.live_session
    if live is None:
        return
    # Re-checked every second: if access was revoked, stop listening and show the ended screen.
    if live.access_ended or not access.has_pet_access(user["id"], pet["id"], access.VOICE):
        live.stop()
        st.rerun(scope="app")
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


def show_voice_input(user, pet):
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

    access.require_pet_access(user["id"], pet["id"], access.VOICE)
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
                chat.send_voice_question(st.session_state.chats, voice_state, pet, text, user_id=user["id"])
        except chat.AccessEnded:
            st.rerun()  # the sitter dashboard will show that access has ended
        except chat.AgentError as error:
            st.session_state.voice_notice = {"pet_id": pet["id"], "message": str(error)}
        reset_recorder(pet["id"])
        st.rerun()  # show the new messages in the chat and an empty recorder


def clear_pet_state(pet_id):
    """Forget everything this browser session holds about a pet (chat, voice, live, scan)."""
    st.session_state.chats.pop(pet_id, None)
    chat.clear_voice(st.session_state.voice, pet_id)
    st.session_state.voice_notice = None
    stop_live_session()
    st.session_state.scan = None


def show_sitter_dashboard(user):
    st.subheader("Welcome, Pet Sitter!")
    # Asked from the database on every run, so revoked access takes effect on the next interaction.
    pet = access.sitter_pet(user["id"])
    previous = st.session_state.get("active_pet_id")
    if previous is not None and (pet is None or pet["id"] != previous):
        clear_pet_state(previous)
        st.session_state.access_ended = pet is None
    st.session_state.active_pet_id = pet["id"] if pet else None

    if pet is None:
        if st.session_state.get("access_ended"):
            st.error("🔒 The pet's owner has ended your access. Its information is no longer available here.")
        show_redeem_form(user)
        return
    st.session_state.access_ended = False

    st.caption(f"You have access to **{pet['name']}**, given by the owner.")
    show_pet_summary(pet)
    show_pet_memory(user, pet)
    show_food_scanner(user, pet)
    st.divider()
    show_chat(user, pet)


def show_redeem_form(user):
    st.write("Enter the access code you received from the pet's owner.")
    with st.form("redeem_form", clear_on_submit=True):
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Redeem Code", type="primary")
    if submitted:
        try:
            access.redeem_invite(user["id"], code)
        except access.AccessDenied as error:
            st.error(str(error))
        else:
            st.session_state.access_ended = False
            st.rerun()


def show_owner_dashboard(user):
    st.subheader("Welcome, Pet Owner!")
    st.write(
        "Create profiles for your pets with their habits, feeding routines, "
        "allergies, medical history, and emergency information, and share access with a sitter."
    )
    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    view = st.radio("View", ["My Pets", "Add New Pet"], horizontal=True, label_visibility="collapsed")
    if view == "My Pets":
        show_my_pets(user)
    else:
        show_add_pet(user)


def start_session(user_id):
    """Start a fresh browser session for this account, so nothing from a previous user remains."""
    stop_live_session()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if user_id is not None:
        st.session_state.user_id = user_id


def show_login_page():
    st.write("Log in or create an account to continue.")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            try:
                user = auth.login(email, password)
            except auth.AuthError as error:
                st.error(str(error))
            else:
                start_session(user["id"])
                st.rerun()
    with register_tab:
        with st.form("register_form"):
            role = st.radio("I am a…", ["owner", "sitter"], horizontal=True,
                            format_func=lambda r: "🏠 Pet Owner" if r == "owner" else "🧑‍🍼 Pet Sitter")
            email = st.text_input("Email", key="register_email")
            password = st.text_input(f"Password (at least {auth.MIN_PASSWORD_LENGTH} characters)", type="password",
                                     key="register_password")
            repeat = st.text_input("Repeat password", type="password", key="register_repeat")
            submitted = st.form_submit_button("Create account", type="primary")
        st.caption("The account type can't be changed later.")
        if submitted:
            if password != repeat:
                st.error("The passwords don't match.")
            else:
                try:
                    user = auth.register_user(email, password, role)
                except auth.AuthError as error:
                    st.error(str(error))
                else:
                    start_session(user["id"])
                    st.rerun()


# Header
st.title("🐾 PetPulse")
st.caption("AI-powered pet care assistant")

# The session only remembers the account ID; the account (and its role) is re-read from the database.
current_user = auth.get_user(st.session_state.get("user_id"))
if current_user is None:
    if st.session_state.get("user_id") is not None:
        start_session(None)  # the account no longer exists
    st.divider()
    show_login_page()
    st.stop()

role_label = "Pet Owner" if current_user["role"] == "owner" else "Pet Sitter"
account_col, logout_col = st.columns([4, 1])
account_col.caption(f"Signed in as **{current_user['email']}** · {role_label}")
if logout_col.button("Log out", width="stretch"):
    start_session(None)
    st.rerun()
st.divider()

try:
    if current_user["role"] == "owner":
        show_owner_dashboard(current_user)
    else:
        show_sitter_dashboard(current_user)
except access.AccessDenied as denied:
    # Access changed while this page was being built (e.g. the owner just revoked it).
    if current_user["role"] == "sitter":
        assigned = access.sitter_pet(current_user["id"])
        if assigned is None or assigned["id"] != st.session_state.get("active_pet_id"):
            st.rerun()  # access changed: the next run shows the "access ended" screen
    st.error(str(denied))
