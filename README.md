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

On first start, the app creates `petpulse.db` and adds **Milo**, a fictional
demo dog, so there is something to look at. Milo is added only when the
database is empty. To start over, stop the app and delete `petpulse.db`.

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

In the Pet Sitter tab, record a question, check or edit the transcribed text,
then click **Send to PetPulse**. The answer appears in the chat and as a spoken
audio player.

- Speech to text runs on this computer with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base` model, CPU,
  INT8). The model (about 140 MB) is downloaded once on the first recording.
- Text to speech uses the macOS `say` command, so spoken answers need macOS.
- Recordings are sent from the browser to the local Streamlit server, kept in
  memory, and never saved. Only the confirmed text goes to Claude.

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
| `test_voice.py` | Tests for voice mode (Whisper, `say`, and Claude mocked) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment settings |
