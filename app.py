"""PetPulse - Streamlit web app."""

import streamlit as st

st.set_page_config(page_title="PetPulse", page_icon="🐾", layout="centered")

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
    st.info("📋 Pet profiles are coming soon.")

# Pet Sitter tab
with sitter_tab:
    st.subheader("Welcome, Pet Sitter!")
    st.write(
        "Get personalized guidance for the pets you care for, "
        "based on each pet's profile."
    )
    st.info("💬 The AI chat assistant is coming soon.")
