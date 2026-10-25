"""PetPulse - Streamlit web app."""

import streamlit as st

import database as db

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


# Header
st.title("🐾 PetPulse")
st.caption("AI-powered pet care assistant")
st.divider()

owner_tab, sitter_tab = st.tabs(["🏠 Pet Owner", "🧑‍🍼 Pet Sitter"])

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
    st.info("💬 The AI chat assistant is coming soon.")
