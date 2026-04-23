import json
from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def predictor(state: NarrAIState) -> dict:
    print("[2/5] Running predictor...")

    last_chapter = state["chapters"][-1]["text"]

    prompt = f"""You are a story analyst. Based on the information below, predict what will happen next in the story.

Style analysis:
{state["style_analysis"]}

World state:
{json.dumps(state["whole_state"], indent=2)}

Current scene state:
{json.dumps(state["active_state"], indent=2)}

Last chapter:
{last_chapter}

Provide 3-5 concise plot predictions — one sentence each. Be brief."""

    response = llm.invoke(prompt)
    return {"predictions": response.content}
