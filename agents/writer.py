import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm

SYSTEM_PROMPT = """You are a ghost writer. Write the next chapter of a story in the exact style of the original author.

Match the author's voice, pacing, and tone precisely. Always write a complete chapter with a proper ending — never cut off mid-sentence or mid-scene. Write the NEXT chapter as a standalone chapter that begins after the last chapter ends. Do not continue mid-scene — start fresh as a new chapter. The chapter MUST be more than 2000 words.

Start your response with the chapter title on the first line in this format:
Chapter Title: [title]

Then write the chapter text after a blank line. After the blank line there must be only the chapter text — no headers, labels, or metadata."""


def save_checkpoint(session_dir: str, stage: str):
    path = os.path.join(session_dir, "data", "pipeline_checkpoint.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"resume_from": stage}, f)


def writer(state: NarrAIState) -> dict:
    if state.get("on_agent"): state["on_agent"]("writer", "active")
    print("[4/5] Running writer...")

    static_context = f"""Style analysis:
{state["style_analysis"]}

Plot threads:
{json.dumps(state["selected_context"].get("plot_threads", {}), indent=2, ensure_ascii=False)}

World:
{json.dumps(state["selected_context"].get("world", {}), indent=2, ensure_ascii=False)}

Active characters and locations:
{json.dumps({"characters": state["selected_context"].get("characters", {}), "locations": state["selected_context"].get("locations", {})}, indent=2, ensure_ascii=False)}

Current scene state:
{json.dumps(state["active_state"], indent=2, ensure_ascii=False)}

Plot predictions to follow:
{state["predictions"]}

Last chapter summary:
{state["chapter_summary"]}"""

    feedback_section = ""
    if state["critic_feedback"]:
        feedback_section = f"\n\nPrevious critic feedback to address:\n{chr(10).join(f'- {f}' for f in state['critic_feedback'])}"

    dynamic = f"Write chapter {state['next_chapter_num']}.{feedback_section}"

    try:
        response = llm.invoke([
            SystemMessage(content=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]),
            HumanMessage(content=[
                {"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic}
            ])
        ])
    except Exception as e:
        print(f"  Writer error: {e}")
        save_checkpoint(state["session_dir"], "writer")
        if state.get("on_agent"): state["on_agent"]("writer", f"error:Writer failed — {e}")
        return {"pipeline_error": True}

    if state.get("on_agent"): state["on_agent"]("writer", "done")
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    return {"generated_text": response.content, "total_tokens": tokens}
