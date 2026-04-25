import json
from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def writer(state: NarrAIState) -> dict:
    print("[4/5] Running writer...")

    feedback_section = ""
    if state["critic_feedback"]:
        feedback_section = f"""
Previous critic feedback to address:
{"\n".join(f"- {f}" for f in state["critic_feedback"])}
"""

    prompt = f"""You are a ghost writer. Write the next chapter of the story in the exact style of the author.

Style analysis:
{state["style_analysis"]}

Plot threads:
{json.dumps(state["selected_context"].get("plot_threads", {}), indent=2)}

World:
{json.dumps(state["selected_context"].get("world", {}), indent=2)}

Active characters and locations:
{json.dumps({"characters": state["selected_context"].get("characters", {}), "locations": state["selected_context"].get("locations", {})}, indent=2)}

Current scene state:
{json.dumps(state["active_state"], indent=2)}

Plot predictions to follow:
{state["predictions"]}

Last chapter summary:
{state["chapter_summary"]}
{feedback_section}
Write the NEXT chapter as a standalone chapter that begins after the last chapter ends. Do not continue mid-scene — start fresh as a new chapter. Match the author's voice, pacing, and tone precisely. Always write a complete chapter with a proper ending — never cut off mid-sentence or mid-scene."""

    response = llm.invoke(prompt)
    return {"generated_text": response.content}
