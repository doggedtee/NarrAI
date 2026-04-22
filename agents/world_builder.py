import json
import os
from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState

WORLD_STATE_PATH = "data/world_state.json"
LOCATION_STATE_PATH = "data/location_state.json"
CHARACTER_STATE_PATH = "data/character_state.json"

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_schema(whole_state: dict) -> dict:
    return {
        "world": list(whole_state.get("world", {}).keys()),
        "characters": {
            name: list(data.keys())
            for name, data in whole_state.get("characters", {}).items()
            if isinstance(data, dict)
        },
        "locations": {
            name: list(data.keys())
            for name, data in whole_state.get("locations", {}).items()
            if isinstance(data, dict)
        }
    }


def merge_fields(existing: dict, new: dict):
    for key, value in new.items():
        if value is None:
            continue
        if key in existing and isinstance(existing[key], list) and isinstance(value, list):
            existing[key] = list(set(existing[key] + value))
        else:
            existing[key] = value


def merge_into_whole(active: dict, whole: dict) -> dict:
    merge_fields(whole.setdefault("world", {}), active.get("world", {}))

    for name, data in active.get("characters", {}).items():
        if name in whole.setdefault("characters", {}):
            merge_fields(whole["characters"][name], data)
        else:
            whole["characters"][name] = data

    for name, data in active.get("locations", {}).items():
        if name in whole.setdefault("locations", {}):
            merge_fields(whole["locations"][name], data)
        else:
            whole["locations"][name] = data

    return whole


def extract_active_state(chapter_text: str, whole_state: dict) -> dict:
    schema = get_schema(whole_state)

    if schema["world"] or schema["characters"] or schema["locations"]:
        schema_section = f"""This is the current world state schema — it shows what fields already exist and their names.
Consider the same field names when analyzing the chapter. Only return fields that appear or change in this chapter.
Do NOT return fields that are not mentioned in this chapter.

Current schema:
{json.dumps(schema, indent=2)}

"""
    else:
        schema_section = "This is the first chapter. Create whatever structure makes sense for tracking story continuity.\n\n"

    prompt = f"""{schema_section}Read this chapter.

Rules:
- Use lists for fields that can have multiple values (abilities, possessions, traits, features, inhabitants, etc.)
- Use strings only for fields that are always singular (name, age, status, current_location, etc.)

Chapter:
{chapter_text}

Respond ONLY with valid JSON:
{{
    "world": {{...}},
    "characters": {{
        "CharacterName": {{...}}
    }},
    "locations": {{
        "LocationName": {{...}}
    }}
}}"""

    response = llm.invoke(prompt)
    return parse_json_response(response.content)


def world_builder(state: NarrAIState) -> dict:
    print("[0/5] Building world state...")

    whole_state = {
        "world": {},
        "characters": {},
        "locations": {}
    }

    active_state = {}

    for i, chapter in enumerate(state["chapters"]):
        print(f"  Processing chapter {i + 1}/{len(state['chapters'])}: {chapter['filename']}")
        try:
            active_state = extract_active_state(chapter["text"], whole_state)
            print("  Active state:")
            print(json.dumps(active_state, indent=2, ensure_ascii=False))
            whole_state = merge_into_whole(active_state, whole_state)
        except Exception as e:
            print(f"  Error processing chapter {i + 1}: {e}")
            break

    save_json(WORLD_STATE_PATH, whole_state["world"])
    save_json(CHARACTER_STATE_PATH, whole_state["characters"])
    save_json(LOCATION_STATE_PATH, whole_state["locations"])

    return {
        "whole_state": whole_state,
        "active_state": active_state
    }


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)

