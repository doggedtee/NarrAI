import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm

PLOT_PLAN_PATH = "data/generated/plot_plan.json"
WORLD_STATE_PATH = "data/generated/world_state.json"
CHARACTER_STATE_PATH = "data/generated/character_state.json"
LOCATION_STATE_PATH = "data/generated/location_state.json"
PLOT_THREADS_PATH = "data/generated/plot_threads.json"

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


def build_planner_context() -> dict:
    characters = load_json(CHARACTER_STATE_PATH)
    locations = load_json(LOCATION_STATE_PATH)
    return {
        "plot_threads": load_json(PLOT_THREADS_PATH),
        "world": load_json(WORLD_STATE_PATH),
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
    print("[1/6] Running plot planner...")

    context = build_planner_context()

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
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    result = json.loads(content)

    plan = {"arc_name": result.get("arc_name", ""), "planned": result.get("planned", {})}
    with open(PLOT_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    plot_threads = load_json(PLOT_THREADS_PATH)
    plot_threads["planned"] = plan["planned"]
    with open(PLOT_THREADS_PATH, "w", encoding="utf-8") as f:
        json.dump(plot_threads, f, indent=2, ensure_ascii=False)

    selected_context = dict(state["selected_context"])
    selected_context["plot_threads"] = plot_threads
    return {"selected_context": selected_context}
