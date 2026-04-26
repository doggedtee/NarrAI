import json
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm

SYSTEM_PROMPT = """You are a story analyst. Based on the current story state, predict what will happen next.

Provide 3-5 concise plot predictions — one sentence each. Be brief. You CAN include new characters and locations in your predictions."""


def predictor(state: NarrAIState) -> dict:
    print("[2/5] Running predictor...")

    human_content = f"""Plot threads:
{json.dumps(state["selected_context"].get("plot_threads", {}), indent=2, ensure_ascii=False)}

World:
{json.dumps(state["selected_context"].get("world", {}), indent=2, ensure_ascii=False)}

Active characters and locations:
{json.dumps({"characters": state["selected_context"].get("characters", {}), "locations": state["selected_context"].get("locations", {})}, indent=2, ensure_ascii=False)}

Current scene state:
{json.dumps(state["active_state"], indent=2, ensure_ascii=False)}

Last chapter summary:
{state["chapter_summary"]}"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ])
    return {"predictions": response.content}
