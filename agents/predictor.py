import json
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm

SYSTEM_PROMPT = """You are a story analyst. Based on the current story state, predict what will happen next.

Provide 3-5 detailed plot predictions. For each prediction, describe what happens, who is involved, and what consequences it may have. You CAN include new characters and locations in your predictions."""


def predictor(state: NarrAIState) -> dict:
    if state.get("on_agent"): state["on_agent"]("predictor", "active")
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
    if state.get("on_agent"): state["on_agent"]("predictor", "done")
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    return {"predictions": response.content, "total_tokens": tokens}
