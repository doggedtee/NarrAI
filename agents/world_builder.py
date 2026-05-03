import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.merger import merge_into_whole
from core.schema import get_schema, select_context
from core.llm import llm_gpt4o_mini as llm


SYSTEM_PROMPT = """You are a story analyst. Extract structured world state updates from story chapters.

Rules:
- Only track information that is crucial for story continuity and future predictions — skip trivial or one-off details (e.g. "good at waking up", "gave up English in middle school").
- Keep all field values CONCISE and BRIEF.
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
    import json5
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json5.loads(content)


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
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    return parse_json_response(response.content), tokens


def world_builder(state: NarrAIState) -> dict:
    if state.get("on_agent"): state["on_agent"]("world_builder", "active")
    print("[0/5] Building world state...")

    base = state["session_dir"]
    gen = os.path.join(base, "data", "generated")
    orig = os.path.join(base, "data", "original")
    pipeline_state_path = os.path.join(base, "data", "pipeline_state.json")

    if state.get("resume_from") in ("cleaner", "plot_planner", "analyzer", "predictor", "writer"):
        print("  Resuming — skipping world_builder.")
        cached = load_json(pipeline_state_path)
        whole_state = {
            "plot_threads": load_json(os.path.join(gen, "plot_threads.json")),
            "world": load_json(os.path.join(gen, "world_state.json")),
            "characters": load_json(os.path.join(gen, "character_state.json")),
            "locations": load_json(os.path.join(gen, "location_state.json")),
        }
        if state.get("on_agent"): state["on_agent"]("world_builder", "done")
        return {
            "selected_context": select_context(whole_state),
            "active_state": cached.get("active_state", {}),
            "chapter_summary": cached.get("chapter_summary", "")
        }

    has_predicted = any(c["filename"].startswith("predicted") for c in state["chapters"])

    if has_predicted:
        whole_state = {
            "plot_threads": load_json(os.path.join(gen, "plot_threads.json")),
            "world": load_json(os.path.join(gen, "world_state.json")),
            "characters": load_json(os.path.join(gen, "character_state.json")),
            "locations": load_json(os.path.join(gen, "location_state.json")),
        }
    else:
        whole_state = {
            "plot_threads": {"main": {}, "paused": {}, "foreshadowed": {}, "resolved": []},
            "world": {},
            "characters": {},
            "locations": {}
        }

    checkpoint_path = os.path.join(base, "data", "checkpoint.json")
    checkpoint = load_json(checkpoint_path)
    start_from = checkpoint.get("processed", 0) if not has_predicted else 0

    if start_from > 0:
        print(f"  Resuming from chapter {start_from + 1}...")
        whole_state = {
            "plot_threads": load_json(os.path.join(gen, "plot_threads.json")),
            "world": load_json(os.path.join(gen, "world_state.json")),
            "characters": load_json(os.path.join(gen, "character_state.json")),
            "locations": load_json(os.path.join(gen, "location_state.json")),
        }

    active_state = {}
    chapter_summary = None
    total_tokens = 0

    for i, chapter in enumerate(state["chapters"]):
        if i < start_from:
            continue
        print(f"  Processing chapter {i + 1}/{len(state['chapters'])}: {chapter['filename']}")
        try:
            is_last = i == len(state["chapters"]) - 1
            active_state, tokens = extract_active_state(chapter["text"], whole_state, is_last)
            total_tokens += tokens
            chapter_summary = active_state.pop("chapter_summary", chapter_summary)
            print("  Active state:")
            print(json.dumps(active_state, indent=2, ensure_ascii=False))
            whole_state = merge_into_whole(active_state, whole_state)
            os.makedirs(gen, exist_ok=True)
            save_json(os.path.join(gen, "world_state.json"), whole_state["world"])
            save_json(os.path.join(gen, "character_state.json"), whole_state["characters"])
            save_json(os.path.join(gen, "location_state.json"), whole_state["locations"])
            save_json(os.path.join(gen, "plot_threads.json"), whole_state["plot_threads"])
            save_json(checkpoint_path, {"processed": i + 1})
        except Exception as e:
            print(f"  Error processing chapter {i + 1}: {e}")
            if state.get("on_agent"):
                state["on_agent"]("world_builder", f"error:Chapter {i + 1} failed — {e}")
            return {"pipeline_error": True}


    if not has_predicted:
        os.makedirs(orig, exist_ok=True)
        save_json(os.path.join(orig, "world_state.json"), whole_state["world"])
        save_json(os.path.join(orig, "character_state.json"), whole_state["characters"])
        save_json(os.path.join(orig, "location_state.json"), whole_state["locations"])
        save_json(os.path.join(orig, "plot_threads.json"), whole_state["plot_threads"])

    os.makedirs(gen, exist_ok=True)
    save_json(os.path.join(gen, "world_state.json"), whole_state["world"])
    save_json(os.path.join(gen, "character_state.json"), whole_state["characters"])
    save_json(os.path.join(gen, "location_state.json"), whole_state["locations"])
    save_json(os.path.join(gen, "plot_threads.json"), whole_state["plot_threads"])

    selected_context = select_context(whole_state)
    active_state.pop("plot_threads", None)
    active_state.pop("world", None)
    for section in ("characters", "locations"):
        for name in list(active_state.get(section, {}).keys()):
            if name in selected_context.get(section, {}):
                active_state[section].pop(name)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    pipeline_checkpoint_path = os.path.join(base, "data", "pipeline_checkpoint.json")
    if os.path.exists(pipeline_checkpoint_path):
        os.remove(pipeline_checkpoint_path)
    save_json(pipeline_state_path, {"active_state": active_state, "chapter_summary": chapter_summary})
    if state.get("on_agent"): state["on_agent"]("world_builder", "done")
    return {
        "selected_context": selected_context,
        "active_state": active_state,
        "chapter_summary": chapter_summary,
        "total_tokens": total_tokens
    }
