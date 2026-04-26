import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm

GENERATED_WORLD_STATE_PATH = "data/generated/world_state.json"
GENERATED_CHARACTER_STATE_PATH = "data/generated/character_state.json"
GENERATED_LOCATION_STATE_PATH = "data/generated/location_state.json"
GENERATED_PLOT_THREADS_PATH = "data/generated/plot_threads.json"

SYSTEM_PROMPT = """You are a story continuity editor. Your job is to clean up accumulated world state data by removing elements that are no longer relevant to the ongoing story.

Rules:
- Only remove list items that are clearly irrelevant to the current and future story arcs
- To remove a list item prefix it with "[remove]" (e.g. "[remove] merchants from first city")
- If nothing needs removing from a list, do not include it in the response
- Do NOT remove string fields, only list items
- Do NOT add new items — only mark existing ones for removal

Respond ONLY with valid JSON using the same structure as the input, only including fields that need cleaning:
{
    "world": {
        "field_name": ["[remove] item1", "[remove] item2"]
    },
    "characters": {
        "CharacterName": {
            "field_name": ["[remove] item1"]
        }
    },
    "locations": {
        "LocationName": {
            "field_name": ["[remove] item1"]
        }
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


def extract_lists(data: dict) -> dict:
    return {k: v for k, v in data.items() if isinstance(v, list)}


def build_lists_snapshot(world: dict, characters: dict, locations: dict) -> dict:
    snapshot = {}
    world_lists = extract_lists(world)
    if world_lists:
        snapshot["world"] = world_lists
    chars_lists = {name: extract_lists(data) for name, data in characters.items() if extract_lists(data)}
    if chars_lists:
        snapshot["characters"] = chars_lists
    locs_lists = {name: extract_lists(data) for name, data in locations.items() if extract_lists(data)}
    if locs_lists:
        snapshot["locations"] = locs_lists
    return snapshot


def cleaner(state: NarrAIState) -> dict:
    print("[cleaner] Running cleaner...")

    world = load_json(GENERATED_WORLD_STATE_PATH)
    characters = load_json(GENERATED_CHARACTER_STATE_PATH)
    locations = load_json(GENERATED_LOCATION_STATE_PATH)
    plot_threads = load_json(GENERATED_PLOT_THREADS_PATH)

    lists_snapshot = build_lists_snapshot(world, characters, locations)
    if not lists_snapshot:
        print("  Nothing to clean.")
        return {}

    human_content = f"""Current plot threads (use this to judge relevance):
{json.dumps(plot_threads, indent=2, ensure_ascii=False)}

List fields to review:
{json.dumps(lists_snapshot, indent=2, ensure_ascii=False)}"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    result = json.loads(content)

    from core.merger import merge_fields

    if result.get("world"):
        merge_fields(world, result["world"])
        save_json(GENERATED_WORLD_STATE_PATH, world)

    if result.get("characters"):
        for name, data in result["characters"].items():
            if name in characters:
                merge_fields(characters[name], data)
        save_json(GENERATED_CHARACTER_STATE_PATH, characters)

    if result.get("locations"):
        for name, data in result["locations"].items():
            if name in locations:
                merge_fields(locations[name], data)
        save_json(GENERATED_LOCATION_STATE_PATH, locations)

    print("  Cleanup done.")
    return {}
