# NarrAI

NarrAI is a multi-agent AI system that analyzes unfinished light novels and books, then generates intelligent continuations in the author's style — chapter after chapter, indefinitely.

## What it does

NarrAI reads existing chapters of a book and builds a living model of the story world — tracking characters, locations, plot threads, and the author's unique writing style. Using this context, it predicts what happens next and generates a new chapter that feels like a natural continuation. Each generated chapter is saved and used as input for the next run, enabling continuous story generation.

## How it works

The system uses a multi-agent pipeline built on LangGraph:

- **World Builder** — reads all existing chapters and builds a structured world state: characters, locations, world facts, and plot threads.
- **Cleaner** — at arc boundaries, removes stale list items from the world state using semantic similarity to keep context lean.
- **Plot Planner** — at arc boundaries, plans long-term story arc milestones stored as `planned` threads for future chapters.
- **Analyzer** — studies the author's narrative voice, pacing, and stylistic patterns via RAG. Result is cached to disk and reused across runs.
- **Predictor** — uses active plot context and last chapter summary to generate plot predictions.
- **Writer** — generates the next chapter guided by predictions, plot threads, and current scene state.
- **Critic** — reviews the generated chapter for consistency. Sends it back to Writer with feedback if needed (up to 3 iterations).

## World State

NarrAI tracks the story through four components:

- **plot_threads** — main, paused, foreshadowed, resolved, and planned plot lines. The active main thread drives what context is passed to each agent.
- **characters** — appearance, traits, possessions, abilities, and current situation for each character.
- **locations** — features and current events for each location.
- **world** — global facts like currency, magic, technology level, and inhabitants.

Only characters and locations relevant to the main plot thread are passed to the predictor and writer — keeping context focused and token usage fixed regardless of story length.

## Token Efficiency

| Agent | v1 | v2 | v3 |
|-------|----|----|-----|
| Predictor | ~9.7k | ~1.5k | ~1.5k |
| Writer | ~10.7k | ~2.6k | ~2.6k |
| **Total per chapter** | **grows linearly** | **~15k** | **~13-17k** |

v3 adds prompt caching on system prompts and static context in the writer→critic loop, reducing repeated token costs on rewrites.

## Stack

- Python, LangGraph, LangChain
- Claude API (claude-sonnet-4-20250514) with prompt caching
- Sentence Transformers (cosine similarity for state deduplication and semantic removal)
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

Each run generates the next chapter. Generated chapters are saved to `predicted_chapters/` with incrementing numbers alongside their plot predictions. World state is saved to `data/` as JSON files after each run.
