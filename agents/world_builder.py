import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.merger import merge_into_whole
from core.schema import get_schema, select_context
from core.llm import llm

GENERATED_WORLD_STATE_PATH = "data/generated/world_state.json"
GENERATED_LOCATION_STATE_PATH = "data/generated/location_state.json"
GENERATED_CHARACTER_STATE_PATH = "data/generated/character_state.json"
GENERATED_PLOT_THREADS_PATH = "data/generated/plot_threads.json"

ORIGINAL_WORLD_STATE_PATH = "data/original/world_state.json"
ORIGINAL_LOCATION_STATE_PATH = "data/original/location_state.json"
ORIGINAL_CHARACTER_STATE_PATH = "data/original/character_state.json"
ORIGINAL_PLOT_THREADS_PATH = "data/original/plot_threads.json"

SYSTEM_PROMPT = """You are a story analyst. Extract structured world state updates from story chapters.

Rules:
- Only track information that is crucial for story continuity and future predictions — skip trivial or one-off details (e.g. "good at waking up", "gave up English in middle school").
- Use lists for fields that can have multiple values (abilities, possessions, traits, features, inhabitants, etc.)
- Use strings only for fields that are always singular (name, age, status, current_location, etc.)
- To remove a list item prefix it with "[remove]" (e.g. "[remove] wallet").
- To remove a field inside world, character or location add it with "[remove]" as value (e.g. "hair": "[remove]").
- To remove an entire character or location add "remove": true to their data.
- If the key of a character or location is not their real name (e.g. "Blonde Girl", "Unknown Man"), add a "name" field with their real name inside their data when it becomes known in the chapter (e.g. "Blonde Girl": {"name": "Emilia", ...}).
- "current_location" and "involved" names must match names already in the schema. If it is a new location or person, use the same name for both the schema entry and these fields.
- Only return fields that appear or change in this chapter. Do NOT copy type indicators ("string", "list") as values — only return actual content.

For plot_threads — always return the FULL current state of all threads:
- main: exactly ONE thread — the most specific and active plot line driving the events of this chapter (not a general background thread like "summoned to another world")
- paused: all other active threads that are not the current focus
- foreshadowed: plot lines mentioned but not started yet
- resolved: list of thread names that are fully completed
- Move threads between statuses as the story progresses (e.g. main → resolved when finished)
- Keep existing threads from the schema unless their status changes
- planned: future story arc milestones — only move them to main/paused/foreshadowed/resolved as they begin, never add new entries to planned

Respond ONLY with valid JSON:
{
    "plot_threads": {
        "main": {
            "thread_name": {"name": "...", "goals": "...", "progress": "...", "involved": [...], "current_location": "..."}
        },
        "paused": {
            "thread_name": {"name": "...", "goals": "...", "progress": "...", "involved": [...]}
        },
        "foreshadowed": {
            "thread_name": {"name": "...", "involved": [...]}
        },
        "resolved": ["thread_name1"]
    },
    "world": {...},
    "characters": {
        "CharacterName": {...}
    },
    "locations": {
        "LocationName": {...}
    }
}"""


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def extract_active_state(chapter_text: str, whole_state: dict, is_last: bool = False) -> dict:
    schema = get_schema(whole_state)

    if schema["plot_threads"] or schema["world"] or schema["characters"] or schema["locations"]:
        schema_section = f"""Current schema — shows existing field names and their types ("string" or "list"). Do NOT copy these type indicators as values.

{json.dumps(schema, indent=2, ensure_ascii=False)}

"""
    else:
        schema_section = "This is the first chapter. Create whatever structure makes sense for tracking story continuity.\n\n"

    summary_instruction = '\n\nThis is the last chapter. Also include "chapter_summary" in your JSON response: a 3-5 sentence summary of what happened in this chapter, then on a new line "[Last sentences]:" followed by the last 3-4 sentences of the chapter copied verbatim.' if is_last else ""
    summary_field = '\n    "chapter_summary": "...",' if is_last else ""

    human_content = f"""{schema_section}Chapter:
{chapter_text}{summary_instruction}"""

    if is_last:
        system_with_summary = SYSTEM_PROMPT.replace(
            '    "locations": {\n        "LocationName": {...}\n    }\n}',
            '    "locations": {\n        "LocationName": {...}\n    },' + summary_field + '\n}'
        )
    else:
        system_with_summary = SYSTEM_PROMPT

    response = llm.invoke([
        SystemMessage(content=system_with_summary),
        HumanMessage(content=human_content)
    ])
    return parse_json_response(response.content)


def world_builder(state: NarrAIState) -> dict:
    print("[0/5] Building world state...")

    has_predicted = any(c["filename"].startswith("predicted") for c in state["chapters"])

    if has_predicted:
        whole_state = {
            "plot_threads": load_json(GENERATED_PLOT_THREADS_PATH),
            "world": load_json(GENERATED_WORLD_STATE_PATH),
            "characters": load_json(GENERATED_CHARACTER_STATE_PATH),
            "locations": load_json(GENERATED_LOCATION_STATE_PATH),
        }
    else:
        whole_state = {
            "plot_threads": {"main": {}, "paused": {}, "foreshadowed": {}, "resolved": []},
            "world": {},
            "characters": {},
            "locations": {}
        }

    active_state = {}
    chapter_summary = None

    for i, chapter in enumerate(state["chapters"]):
        print(f"  Processing chapter {i + 1}/{len(state['chapters'])}: {chapter['filename']}")
        try:
            is_last = i == len(state["chapters"]) - 1
            active_state = extract_active_state(chapter["text"], whole_state, is_last)
            chapter_summary = active_state.pop("chapter_summary", chapter_summary)
            print("  Active state:")
            print(json.dumps(active_state, indent=2, ensure_ascii=False))
            whole_state = merge_into_whole(active_state, whole_state)
        except Exception as e:
            print(f"  Error processing chapter {i + 1}: {e}")
            break

    if not has_predicted:
        os.makedirs("data/original", exist_ok=True)
        save_json(ORIGINAL_WORLD_STATE_PATH, whole_state["world"])
        save_json(ORIGINAL_CHARACTER_STATE_PATH, whole_state["characters"])
        save_json(ORIGINAL_LOCATION_STATE_PATH, whole_state["locations"])
        save_json(ORIGINAL_PLOT_THREADS_PATH, whole_state["plot_threads"])
        os.makedirs("original_chapters", exist_ok=True)
        for chapter in state["chapters"]:
            os.rename(os.path.join("chapters", chapter["filename"]),
                      os.path.join("original_chapters", chapter["filename"]))

    os.makedirs("data/generated", exist_ok=True)
    save_json(GENERATED_WORLD_STATE_PATH, whole_state["world"])
    save_json(GENERATED_CHARACTER_STATE_PATH, whole_state["characters"])
    save_json(GENERATED_LOCATION_STATE_PATH, whole_state["locations"])
    save_json(GENERATED_PLOT_THREADS_PATH, whole_state["plot_threads"])

    selected_context = select_context(whole_state)
    active_state.pop("plot_threads", None)
    active_state.pop("world", None)
    for section in ("characters", "locations"):
        for name in list(active_state.get(section, {}).keys()):
            if name in selected_context.get(section, {}):
                active_state[section].pop(name)

    return {
        "selected_context": selected_context,
        "active_state": active_state,
        "chapter_summary": chapter_summary
    }
