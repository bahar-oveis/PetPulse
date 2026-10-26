# 🐾 PetPulse

PetPulse is an AI-powered pet care assistant for pet owners and pet sitters.

- **Pet owners** create profiles with their pets' habits, feeding routines,
  allergies, sensitivities, medical history, and emergency information.
- **Pet sitters** chat with an AI assistant (Claude) that gives personalized
  guidance based on each pet's profile.

## Dependencies

- [Streamlit](https://streamlit.io) – web interface
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (`anthropic`) – Claude AI assistant for sitters
- [python-dotenv](https://pypi.org/project/python-dotenv/) – loads settings from `.env`
- SQLite (built into Python) – database

## Setup

1. Activate the Conda environment:

   ```bash
   conda activate petpulse
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file from the example and add your Anthropic API key
   (from https://console.anthropic.com):

   ```bash
   cp .env.example .env
   ```

## Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

Log in or create an account first (see **Accounts and access** below). To
set up the fictional demo (two owners, two sitters, Milo, and Luna):

```bash
python manage.py seed-demo
```

## Accounts and access

- **Pet Owners** create pets, edit them, and decide who may help: in
  **🔑 Manage Sitter Access** they generate a one-time access code for a pet,
  rotate it (which ends the current sitter's access immediately), or revoke
  access.
- **Pet Sitters** redeem a code and then see only that one pet: profile
  summary, chat, voice, live conversation, food scanner, and Pet Memory. They
  can't edit the profile. Each pet has at most one sitter, and each sitter
  cares for one pet at a time.
- The account type is chosen when the account is created and can't be changed.
- Passwords are stored only as Argon2id hashes; access codes only as SHA-256
  hashes. Codes are 32 random characters, single-use, and expire after 24 hours.
- Every pet operation is checked against the database (`access.py`), including
  the AI agent's tools. A revoked sitter is locked out on their next click,
  and an answer Claude was still writing for them is discarded.

### Older databases

Starting the app updates older databases automatically (adding the new
tables and columns) after saving a copy in `backups/`. Pets that existed
before accounts have **no owner and are visible to nobody** until you assign
them on purpose:

```bash
python manage.py list-unassigned
python manage.py create-user --email you@example.com --role owner
python manage.py assign-pets --owner-email you@example.com --pet-id 1
```

`assign-pets` saves another backup first and asks for confirmation.

### Security limitations (local MVP)

This login is meant for a local demo, not an internet-facing service:

- The login lasts as long as the Streamlit browser session (the session keeps
  only the account ID; the account and its permissions are re-read from the
  database on every interaction). Closing the tab or refreshing logs you out.
  There is no "remember me", password reset, email verification, or 2FA.
- Rate limiting is basic (per email for logins, per sitter for codes) and
  stored in SQLite.
- Revocation applies to the next request: anything a sitter already saw or
  copied can't be taken back.
- If you deploy it publicly, use HTTPS, a managed identity provider (for
  example Streamlit's built-in OIDC login), and hosted persistent storage
  instead of a local SQLite file.
- Don't commit `petpulse.db` or `backups/`: they contain accounts and pet data.

## Run the tests

```bash
python -m unittest -v
```

The tests use a temporary database, so your saved pets are not affected, and a
fake Claude client, so they never make paid API calls.

## AI assistant settings

Set these in `.env`:

- `ANTHROPIC_API_KEY` – required for the Pet Sitter chat.
- `ANTHROPIC_MODEL` – optional, defaults to `claude-haiku-5-5` (low cost).

Each answer prints its token usage in the terminal, e.g.
`[PetPulse] claude-haiku-5-5: 1450 input / 120 output tokens`.

## Voice mode (push-to-talk)

In the Pet Sitter dashboard, record a question, check or edit the transcribed text,
then click **Send to PetPulse**. The answer appears in the chat and as a spoken
audio player.

- Speech to text runs on this computer with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base` model, CPU,
  INT8). The model (about 140 MB) is downloaded once on the first recording.
- Text to speech uses the macOS `say` command, so spoken answers need macOS.
- Recordings are sent from the browser to the local Streamlit server, kept in
  memory, and never saved. Only the confirmed text goes to Claude.

## Live Conversation (hands-free)

Choose **🔴 Live Conversation** in the Pet Sitter dashboard and click **Start Live
Conversation**. Ask a question and pause for about a second: PetPulse
transcribes it, answers out loud, and listens again. Click **End Session** to
stop. Live, Voice Message, and Text Chat share the same conversation.

- Microphone audio streams from the browser to the local Python server over
  WebRTC ([streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)).
- Silero voice activity detection (bundled with faster-whisper) finds where a
  question starts and ends; Whisper transcribes it; macOS `say` speaks the
  answer, which is streamed back to the browser.
- You can interrupt: start talking while PetPulse is speaking and it pauses the
  answer. A real question is answered next; a cough, noise, or PetPulse's own
  voice coming back through the speakers lets the answer continue. Headphones
  give the best results. Interruptions can be turned off with
  `LiveSettings(barge_in=False)` in `live_voice.py`.
- The microphone is ignored while PetPulse is thinking.
- Only the transcribed text is sent to Claude. Claude itself still needs the
  internet, so the app is not fully offline.
- Use `http://localhost:8501`: browsers only allow the microphone on localhost
  or HTTPS. Running on a remote server would also need STUN/TURN servers.

## Pet Memory

In the Pet Sitter dashboard, open **📝 Pet Memory** to save observations from a
visit (e.g. "Milo hid under the bed when the doorbell rang."). They are stored
in the `pet_memories` table of `petpulse.db`, so they survive restarts and
Clear Chat. When a sitter asks about past behavior, Claude searches them with
its `recall_pet_memories` tool and presents them as unverified sitter
observations, separate from the owner's profile. Owners can review them in the
Pet Owner dashboard. Sitters can delete only observations they wrote.

- Embeddings are computed on this computer with
  [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`, about
  90 MB, downloaded once) and compared with NumPy cosine similarity.
- Searches only ever look at the selected pet's memories.
- Nothing is saved automatically: only what a sitter saves with **Save to Memory**.
- This is a local demo: use fictional records only.

## Food & Allergy Scanner (PetPulse Vision)

In the Pet Sitter dashboard, open **🔍 Food & Allergy Scanner**, upload or take a
photo of a food label, and click **Analyze Food Label**.

1. **On this computer:** the image is validated, its orientation fixed, and
   it's resized (Pillow); sharpness is estimated and contrast enhanced
   (OpenCV); the text is read with Tesseract OCR. You can correct the text.
2. **Rule-based check:** the ingredients are compared with the pet's recorded
   allergies and diet restrictions using a small, explicit list of ingredient
   names (`food_checker.py`). Results: *Potential Conflict*, *No Match
   Detected* (which never means "safe"), or *Unable to Verify*.
3. **Claude:** click **Ask PetPulse about this label** to send the photo, the
   confirmed text, and the rule-based result to the same Claude agent for a
   personalized explanation. Follow-up questions are possible.

Photos and results are kept only in the browser session, never saved.

Tesseract is a system program, installed separately with Homebrew:

```bash
brew install tesseract
tesseract --version   # check it works
```

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface |
| `chat.py` | Sitter chat: per-pet conversations and replies |
| `agent.py` | Claude agent: sends the selected pet's profile with each question |
| `database.py` | SQLite database for pet profiles |
| `test_database.py` | Tests for the database functions |
| `test_chat.py` | Tests for the sitter chat |
| `voice.py` | Local speech to text (faster-whisper) and text to speech (macOS `say`) |
| `test_agent.py` | Tests for the Claude agent (no real API calls) |
| `live_voice.py` | Live Conversation: voice activity detection and the turn-taking session |
| `auth.py` | Accounts: registration, Argon2id password hashing, login, rate limiting |
| `access.py` | Who may do what with which pet; access codes, rotation, and revocation |
| `manage.py` | Local admin commands: create users, assign older pets, set up the demo |
| `test_auth.py` | Tests for accounts, access control, invitations, and revocation |
| `fixtures.py` | Helpers the tests use to create accounts and grant access |
| `memory.py` | Pet Memory: save, list, delete, and semantically search sitter observations |
| `vision.py` | Food label images: validation, orientation, resizing, blur check, contrast enhancement, OCR |
| `food_checker.py` | Rule-based allergy matching for ingredient text |
| `test_vision.py` | Tests for the scanner (OCR and Claude mocked) |
| `test_memory.py` | Tests for Pet Memory (fake embedding model, Claude mocked) |
| `test_voice.py` | Tests for voice mode (Whisper, `say`, and Claude mocked) |
| `test_live_voice.py` | Tests for Live Conversation (simulated microphone, Claude mocked) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment settings |
