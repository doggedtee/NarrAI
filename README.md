# NarrAI

NarrAI is a multi-agent AI system that analyzes unfinished light novels and books, then generates intelligent continuations in the author's style.

## What it does

NarrAI reads existing chapters of a book and builds a living model of the story world — tracking characters, locations, events, and the author's unique writing style. Using this context, it predicts what happens next and generates a new chapter that feels like a natural continuation.

## How it works

The system uses a multi-agent pipeline built on LangGraph:

- **World Builder** — reads all existing chapters and constructs a structured world state: characters, locations, and global story context. Updates automatically as new chapters are added.
- **Analyzer** — studies the author's narrative voice, pacing, sentence structure, and stylistic patterns.
- **Predictor** — uses the world state and style analysis to generate concise plot predictions for upcoming events.
- **Writer** — generates the next chapter in the author's style, guided by predictions and the current world state.
- **Critic** — reviews the generated chapter for lore consistency and character behavior. If issues are found, it sends the chapter back to the Writer with feedback (up to 3 iterations).

## World State System

Instead of relying purely on vector search, NarrAI maintains a structured world state that grows with the story:

- `world_state.json` — global rules, time, political situation, story-specific mechanics
- `character_state.json` — each character's appearance, personality, goals, relationships, and current situation
- `location_state.json` — each location's description, atmosphere, and current inhabitants

This approach ensures that generated chapters remain logically consistent with the established world, regardless of how long the book is.

## Stack

- Python
- LangGraph + LangChain
- Claude API (claude-sonnet-4-20250514)
- FastAPI
- SQLite

## Installation

```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

## Usage

Place your chapter files in the `chapters/` folder as `.txt` files (e.g. `chapter_01.txt`, `chapter_02.txt`).

Run the pipeline:
```bash
python main.py
```

Or start the API server:
```bash
uvicorn api.main:app --reload
```

### API Endpoints

- `POST /run` — run the full pipeline
- `GET /predictions` — retrieve stored predictions
