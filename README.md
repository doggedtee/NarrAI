# NarrAI

NarrAI is a multi-agent AI system that analyzes unfinished light novels and books, then generates intelligent continuations in the author's style.

## What it does

NarrAI reads existing chapters of a book and builds a living model of the story world — tracking characters, locations, plot threads, and the author's unique writing style. Using this context, it predicts what happens next and generates a new chapter that feels like a natural continuation.

## How it works

The system uses a multi-agent pipeline built on LangGraph:

- **World Builder** — reads all existing chapters and builds a structured world state: characters, locations, world facts, and plot threads.
- **Analyzer** — studies the author's narrative voice, pacing, and stylistic patterns via RAG.
- **Predictor** — uses active plot context and last chapter summary to generate plot predictions.
- **Writer** — generates the next chapter guided by predictions, plot threads, and current scene state.
- **Critic** — reviews the generated chapter for consistency. Sends it back to Writer with feedback if needed (up to 3 iterations).

## World State

NarrAI tracks the story through four components:

- **plot_threads** — main, paused, foreshadowed, and resolved plot lines. The active main thread drives what context is passed to each agent.
- **characters** — appearance, traits, possessions, abilities, and current situation for each character.
- **locations** — features and current events for each location.
- **world** — global facts like currency, magic, technology level, and inhabitants.

Only characters and locations relevant to the main plot thread are passed to the predictor and writer — keeping context focused and token usage fixed regardless of story length.

## Token Efficiency

| Agent | Before (v1) | After (v2) |
|-------|------------|-----------|
| Predictor | ~9.7k | ~1.5k |
| Writer | ~10.7k | ~2.6k |
| **Total (8 chapters)** | **~67.8k** | **~57.9k** |

After the initial world state is built, each new chapter generation costs ~12-16k tokens regardless of how many chapters exist.

## Stack

- Python, LangGraph, LangChain
- Claude API (claude-sonnet-4-20250514)
- Sentence Transformers (cosine similarity for state deduplication)
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

World state is saved to `data/` as JSON files after each run.
