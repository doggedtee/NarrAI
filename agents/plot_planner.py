import json
import json5
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm


SYSTEM_PROMPT = """You are a creative story planner. Design high-level story arc milestones based on the current world state.

Your task:
- Create 2-3 major story arc milestones — broad turning points in the story, not specific scenes or actions
- Each milestone should be significant enough to drive multiple chapters, not just one moment
- You CAN invent new characters and locations that don't exist yet — give them names and a brief role
- Events should feel like a natural and exciting continuation of the current plot threads
- Do NOT resolve existing plot threads — planned events should build on them

Respond ONLY with valid JSON:
{
    "arc_name": "short name for this story arc",
    "planned": {
        "event_key": {
            "name": "broad description of the story milestone"
        }
    }
}"""


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_planner_context(gen: str) -> dict:
    characters = load_json(os.path.join(gen, "character_state.json"))
    locations = load_json(os.path.join(gen, "location_state.json"))
    return {
        "plot_threads": load_json(os.path.join(gen, "plot_threads.json")),
        "world": load_json(os.path.join(gen, "world_state.json")),
        "characters": {
            name: {k: v for k, v in data.items() if k in ("current_location", "status")}
            for name, data in characters.items()
        },
        "locations": {
            name: data.get("description", "")
            for name, data in locations.items()
        }
    }


def plot_planner(state: NarrAIState) -> dict:
    if state.get("on_agent"): state["on_agent"]("plot_planner", "active")
    print("[1/6] Running plot planner...")

    gen = os.path.join(state["session_dir"], "data", "generated")
    context = build_planner_context(gen)

    human_content = f"""Plot threads:
{json.dumps(context["plot_threads"], indent=2, ensure_ascii=False)}

World:
{json.dumps(context["world"], indent=2, ensure_ascii=False)}

Characters (name, location, status):
{json.dumps(context["characters"], indent=2, ensure_ascii=False)}

Locations:
{json.dumps(context["locations"], indent=2, ensure_ascii=False)}

Last chapter summary:
{state["chapter_summary"]}"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ])
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json5.loads(content)
    except Exception as e:
        print(f"  Plot planner error: {e}")
        save_json(os.path.join(state["session_dir"], "data", "pipeline_checkpoint.json"), {"resume_from": "plot_planner"})
        if state.get("on_agent"): state["on_agent"]("plot_planner", f"error:Plot planner failed — {e}")
        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        return {"total_tokens": tokens}

    plan = {"arc_name": result.get("arc_name", ""), "planned": result.get("planned", {})}
    with open(os.path.join(gen, "plot_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    plot_threads = load_json(os.path.join(gen, "plot_threads.json"))
    plot_threads["planned"] = plan["planned"]
    with open(os.path.join(gen, "plot_threads.json"), "w", encoding="utf-8") as f:
        json.dump(plot_threads, f, indent=2, ensure_ascii=False)

    selected_context = dict(state["selected_context"])
    selected_context["plot_threads"] = plot_threads
    if state.get("on_agent"): state["on_agent"]("plot_planner", "done")
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    return {"selected_context": selected_context, "total_tokens": tokens}
