# 🐾 PetPulse

PetPulse is an AI-powered pet care assistant for pet owners and pet sitters.

- **Pet owners** create profiles with their pets' habits, feeding routines,
  allergies, sensitivities, medical history, and emergency information.
- **Pet sitters** will chat with an AI assistant that gives personalized
  guidance based on each pet's profile.

## Dependencies

- [Streamlit](https://streamlit.io) – web interface
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) (`openai-agents`) – AI assistant (future phase)
- [python-dotenv](https://pypi.org/project/python-dotenv/) – loads settings from `.env`
- SQLite (built into Python) – database (future phase)

## Setup

1. Activate the Conda environment:

   ```bash
   conda activate petpulse
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file from the example and add your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

## Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface |
| `agent.py` | AI assistant (placeholder) |
| `database.py` | SQLite database (placeholder) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment settings |
