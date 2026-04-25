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
PLOT_THREADS_PATH = "data/plot_threads.json"

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
    plot_threads = whole_state.get("plot_threads", {})

    main_chars = set()
    for thread in plot_threads.get("main", {}).values():
        main_chars.update(thread.get("involved", []))

    secondary_chars = set()
    for status in ("paused", "foreshadowed"):
        for thread in plot_threads.get(status, {}).values():
            secondary_chars.update(thread.get("involved", []))
    secondary_chars -= main_chars

    main_location = None
    main_thread = next(iter(plot_threads.get("main", {}).values()), None)
    if main_thread:
        main_location = main_thread.get("current_location")

    schema = {
        "plot_threads": plot_threads,
        "world": {k: "list" if isinstance(v, list) else "string" for k, v in whole_state.get("world", {}).items()} or {},
        "characters": {},
        "locations": {}
    }

    all_chars = list(whole_state.get("characters", {}).keys())
    all_locs = list(whole_state.get("locations", {}).keys())

    def fuzzy_match(name, candidates):
        if not candidates:
            return None
        if name in candidates:
            return name
        emb_name = embedder.encode([name])
        emb_cands = embedder.encode(candidates)
        sims = np.dot(emb_cands, emb_name[0]) / (np.linalg.norm(emb_cands, axis=1) * np.linalg.norm(emb_name[0]))
        best = int(np.argmax(sims))
        return candidates[best] if sims[best] > 0.8 else None

    resolved_main_chars = {fuzzy_match(n, all_chars) for n in main_chars}
    resolved_main_chars.discard(None)
    resolved_secondary_chars = {fuzzy_match(n, all_chars) for n in secondary_chars}
    resolved_secondary_chars.discard(None)
    resolved_secondary_chars -= resolved_main_chars

    resolved_main_location = fuzzy_match(main_location, all_locs) if main_location else None

    for name, data in whole_state.get("characters", {}).items():
        if name in resolved_main_chars and isinstance(data, dict):
            schema["characters"][name] = {k: "list" if isinstance(v, list) else "string" for k, v in data.items()}
        elif name in resolved_secondary_chars:
            schema["characters"][name] = {}

    for name, data in whole_state.get("locations", {}).items():
        if name == resolved_main_location and isinstance(data, dict):
            schema["locations"][name] = {k: "list" if isinstance(v, list) else "string" for k, v in data.items()}
        else:
            schema["locations"][name] = {}

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
        if value == "[remove]":
            existing.pop(key, None)
        elif key in existing and isinstance(existing[key], list) and isinstance(value, list):
            remove_queries = [v[len("[remove]"):].strip() for v in value if isinstance(v, str) and v.startswith("[remove]")]
            to_add = [v for v in value if not (isinstance(v, str) and v.startswith("[remove]"))]
            kept = list(existing[key])
            if remove_queries and kept:
                kept_embeddings = embedder.encode(kept)
                for query in remove_queries:
                    query_embedding = embedder.encode([query])
                    similarities = np.dot(kept_embeddings, query_embedding[0]) / (
                        np.linalg.norm(kept_embeddings, axis=1) * np.linalg.norm(query_embedding[0])
                    )
                    best_idx = int(np.argmax(similarities))
                    if similarities[best_idx] > 0.7:
                        kept.pop(best_idx)
                        kept_embeddings = np.delete(kept_embeddings, best_idx, axis=0)
            if kept and to_add:
                kept_embeddings = embedder.encode(kept)
                filtered_add = []
                for item in to_add:
                    item_embedding = embedder.encode([item])
                    similarities = np.dot(kept_embeddings, item_embedding[0]) / (
                        np.linalg.norm(kept_embeddings, axis=1) * np.linalg.norm(item_embedding[0])
                    )
                    if float(np.max(similarities)) < 0.85:
                        filtered_add.append(item)
                existing[key] = list(set(kept) | set(filtered_add))
            else:
                existing[key] = list(set(kept) | set(to_add))
        else:
            existing[key] = value


def merge_into_whole(active: dict, whole: dict) -> dict:
    if active.get("plot_threads"):
        whole["plot_threads"] = active["plot_threads"]
    merge_fields(whole.setdefault("world", {}), active.get("world", {}))

    for section in ("characters", "locations"):
        for name, data in active.get(section, {}).items():
            if data.get("remove"):
                whole.get(section, {}).pop(name, None)
                continue
            real_name = data.pop("name", None)
            if real_name and real_name != name and name in whole.setdefault(section, {}):
                whole[section][real_name] = whole[section].pop(name)
                for status in ("main", "paused", "foreshadowed"):
                    for thread in whole.get("plot_threads", {}).get(status, {}).values():
                        if name in thread.get("involved", []):
                            thread["involved"] = [real_name if n == name else n for n in thread["involved"]]
                        if thread.get("current_location") == name:
                            thread["current_location"] = real_name
                name = real_name
            if name in whole.setdefault(section, {}):
                data = resolve_field_names(data, whole[section][name])
                merge_fields(whole[section][name], data)
            else:
                whole[section][name] = data

    return whole



def select_context(whole_state: dict) -> dict:
    plot_threads = whole_state.get("plot_threads", {})
    involved_names = set()
    for thread in plot_threads.get("main", {}).values():
        involved_names.update(thread.get("involved", []))

    context = {"plot_threads": whole_state.get("plot_threads", {}), "world": whole_state.get("world", {}), "characters": {}, "locations": {}}
    for name, data in whole_state.get("characters", {}).items():
        if name in involved_names:
            context["characters"][name] = data

    main_thread = next(iter(plot_threads.get("main", {}).values()), None)
    main_location = main_thread.get("current_location") if main_thread else None
    if main_location and main_location in whole_state.get("locations", {}):
        context["locations"][main_location] = whole_state["locations"][main_location]

    return context


def extract_active_state(chapter_text: str, whole_state: dict, is_last: bool = False) -> dict:
    schema = get_schema(whole_state)

    if schema["plot_threads"] or schema["world"] or schema["characters"] or schema["locations"]:
        schema_section = f"""This is the current world state schema — it shows what fields already exist, their names, and their types ("string" or "list"). The values are type indicators, not actual content.
Consider the same field names and types when analyzing the chapter. Only return fields that appear or change in this chapter.
Do NOT return fields that are not mentioned in this chapter.

Current schema:
{json.dumps(schema, indent=2)}

"""
    else:
        schema_section = "This is the first chapter. Create whatever structure makes sense for tracking story continuity.\n\n"

    summary_rule = '\n- Also return "chapter_summary": a 3-5 sentence summary of what happened in this chapter. Then on a new line add the label "[Last sentences]:" followed by the last 3-4 sentences of the chapter copied verbatim.' if is_last else ""
    summary_field = '\n    "chapter_summary": "...",' if is_last else ""

    prompt = f"""{schema_section}Read this chapter.

Rules:
- Only track information that is crucial for story continuity and future predictions — skip trivial or one-off details (e.g. "good at waking up", "gave up English in middle school").
- Use lists for fields that can have multiple values (abilities, possessions, traits, features, inhabitants, etc.)
- Use strings only for fields that are always singular (name, age, status, current_location, etc.)
- To remove a list item prefix it with "[remove]" (e.g. "[remove] wallet").
- To remove a field inside world, character or location add it with "[remove]" as value (e.g. "hair": "[remove]").
- To remove an entire character or location add "remove": true to their data.
- If the key of a character or location is not their real name (e.g. "Blonde Girl", "Unknown Man"), add a "name" field with their real name inside their data when it becomes known in the chapter (e.g. "Blonde Girl": {{"name": "Emilia", ...}}).
- "current_location" and "involved" names must be similar to the names already in the schema. If it is a new location or person, use the same name for both the schema entry and these fields.{summary_rule}

For plot_threads — always return the FULL current state of all threads:
- main: exactly ONE thread — the most specific and active plot line driving the events of this chapter (not a general background thread like "summoned to another world")
- paused: all other active threads that are not the current focus
- foreshadowed: plot lines mentioned but not started yet
- resolved: list of thread names that are fully completed
- Move threads between statuses as the story progresses (e.g. main → resolved when finished)
- Keep existing threads from the schema unless their status changes

Chapter:
{chapter_text}

Respond ONLY with valid JSON:
{{
    "plot_threads": {{
        "main": {{
            "thread_name": {{"name": "...", "goals": "...", "progress": "...", "involved": [...], "current_location": "..."}}
        }},
        "paused": {{
            "thread_name": {{"name": "...", "goals": "...", "progress": "...", "involved": [...]}}
        }},
        "foreshadowed": {{
            "thread_name": {{"name": "...", "involved": [...]}}
        }},
        "resolved": ["thread_name1"]
    }},
    "world": {{...}},
    "characters": {{
        "CharacterName": {{...}}
    }},
    "locations": {{
        "LocationName": {{...}}
    }}{summary_field}
}}"""

    response = llm.invoke(prompt)
    return parse_json_response(response.content)


def world_builder(state: NarrAIState) -> dict:
    print("[0/5] Building world state...")

    whole_state = {
        "plot_threads": {"main": {}, "paused": {}, "foreshadowed": {}, "resolved": []},
        "world": {},
        "characters": {},
        "locations": {}
    }

    active_state = {}

    for i, chapter in enumerate(state["chapters"]):
        print(f"  Processing chapter {i + 1}/{len(state['chapters'])}: {chapter['filename']}")
        try:
            is_last = i == len(state["chapters"]) - 1
            active_state = extract_active_state(chapter["text"], whole_state, is_last)
            chapter_summary = active_state.pop("chapter_summary", None)
            print("  Active state:")
            print(json.dumps(active_state, indent=2, ensure_ascii=False))
            whole_state = merge_into_whole(active_state, whole_state)
        except Exception as e:
            print(f"  Error processing chapter {i + 1}: {e}")
            break

    save_json(PLOT_THREADS_PATH, whole_state["plot_threads"])
    save_json(WORLD_STATE_PATH, whole_state["world"])
    save_json(CHARACTER_STATE_PATH, whole_state["characters"])
    save_json(LOCATION_STATE_PATH, whole_state["locations"])

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


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)

