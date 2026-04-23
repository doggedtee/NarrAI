import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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


def get_schema(whole_state: dict, classified_whole_state: dict) -> dict:
    schema = {
        "world": list(whole_state.get("world", {}).keys()),
        "characters": {},
        "locations": {}
    }

    for tier in ("hot", "warm"):
        for section in ("characters", "locations"):
            for name, data in classified_whole_state.get(tier, {}).get(section, {}).items():
                if isinstance(data, dict):
                    schema[section][name] = [k for k in data.keys() if k != "last_seen"]

    for section in ("characters", "locations"):
        for name in classified_whole_state.get("cold", {}).get(section, {}).keys():
            schema[section][name] = {}

    return schema


def resolve_field_names(new_data: dict, existing_data: dict) -> dict:
    if not existing_data or not new_data:
        return new_data

    existing_keys = list(existing_data.keys())
    existing_embeddings = embedder.encode(existing_keys)

    resolved = {}
    for key, value in new_data.items():
        new_embedding = embedder.encode([key])
        similarities = np.dot(existing_embeddings, new_embedding[0]) / (
            np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(new_embedding[0])
        )
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] > 0.8:
            resolved[existing_keys[best_idx]] = value
        else:
            resolved[key] = value

    return resolved


def merge_fields(existing: dict, new: dict):
    for key, value in new.items():
        if value is None:
            continue
        if key in existing and isinstance(existing[key], list) and isinstance(value, list):
            existing[key] = list(set(existing[key] + value))
        else:
            existing[key] = value


def merge_into_whole(active: dict, whole: dict, chapter_idx: int) -> dict:
    merge_fields(whole.setdefault("world", {}), active.get("world", {}))

    for section in ("characters", "locations"):
        for name, data in active.get(section, {}).items():
            previous_name = data.pop("previous_name", None)
            if previous_name and previous_name in whole.setdefault(section, {}):
                whole[section][name] = whole[section].pop(previous_name)
            if name in whole.setdefault(section, {}):
                merge_fields(whole[section][name], data)
            else:
                whole[section][name] = data
            whole[section][name]["last_seen"] = chapter_idx

    return whole


def classify_elements(whole_state: dict, current_chapter_idx: int) -> dict:
    result = {
        "hot": {"characters": {}, "locations": {}},
        "warm": {"characters": {}, "locations": {}},
        "cold": {"characters": {}, "locations": {}},
    }

    for section in ("characters", "locations"):
        for name, data in whole_state.get(section, {}).items():
            last_seen = data.get("last_seen", 0)
            if last_seen >= current_chapter_idx - 2:
                tier = "hot"
            elif last_seen >= current_chapter_idx - 4:
                tier = "warm"
            else:
                tier = "cold"
            result[tier][section][name] = data

    return result


def extract_active_state(chapter_text: str, whole_state: dict, classified_whole_state: dict) -> dict:
    schema = get_schema(whole_state, classified_whole_state)

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
- If you can identify that a character or location in this chapter is the same as a placeholder in the schema (e.g. "Blonde Girl" is now revealed to be "Emilia"), use the real name and add a field "previous_name" with the placeholder name.

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
    classified_whole_state = {"hot": {"characters": {}, "locations": {}}, "warm": {"characters": {}, "locations": {}}, "cold": {"characters": {}, "locations": {}}}

    for i, chapter in enumerate(state["chapters"]):
        print(f"  Processing chapter {i + 1}/{len(state['chapters'])}: {chapter['filename']}")
        try:
            active_state = extract_active_state(chapter["text"], whole_state, classified_whole_state)

            for section in ("characters", "locations"):
                cold_names = set(classified_whole_state.get("cold", {}).get(section, {}).keys())
                for name, data in active_state.get(section, {}).items():
                    if name in cold_names and name in whole_state.get(section, {}):
                        active_state[section][name] = resolve_field_names(data, whole_state[section][name])

            print("  Active state:")
            print(json.dumps(active_state, indent=2, ensure_ascii=False))
            whole_state = merge_into_whole(active_state, whole_state, i + 1)
            classified_whole_state = classify_elements(whole_state, i + 1)
            print("  Classified state:")
            print(json.dumps(classified_whole_state, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"  Error processing chapter {i + 1}: {e}")
            break

    save_json(WORLD_STATE_PATH, whole_state["world"])
    save_json(CHARACTER_STATE_PATH, whole_state["characters"])
    save_json(LOCATION_STATE_PATH, whole_state["locations"])

    return {
        "classified_whole_state": classified_whole_state,
        "active_state": active_state
    }


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)

