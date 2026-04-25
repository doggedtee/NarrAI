import json
from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def predictor(state: NarrAIState) -> dict:
    print("[2/5] Running predictor...")

    prompt = f"""You are a story analyst. Based on the information below, predict what will happen next in the story.

Plot threads:
{json.dumps(state["selected_context"].get("plot_threads", {}), indent=2)}

World:
{json.dumps(state["selected_context"].get("world", {}), indent=2)}

Active characters and locations:
{json.dumps({"characters": state["selected_context"].get("characters", {}), "locations": state["selected_context"].get("locations", {})}, indent=2)}

Current scene state:
{json.dumps(state["active_state"], indent=2)}

Last chapter summary:
{state["chapter_summary"]}

Provide 3-5 concise plot predictions — one sentence each. Be brief."""

    response = llm.invoke(prompt)
    return {"predictions": response.content}
