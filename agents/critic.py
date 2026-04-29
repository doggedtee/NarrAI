import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.llm import llm

SYSTEM_PROMPT = """You are a literary critic. Review generated story chapter continuations for consistency and quality.

Approve if:
- Lore and world-building is consistent
- Characters behave consistently with their established traits
- Plot is coherent

Only decline for style if it is a MAJOR deviation — minor style differences are acceptable. Creativity is encouraged.

Respond in this exact format:
APPROVED: true or false
FEEDBACK: brief feedback in 2-3 sentences"""


def save_checkpoint(session_dir: str, stage: str):
    path = os.path.join(session_dir, "data", "pipeline_checkpoint.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"resume_from": stage}, f)


def critic(state: NarrAIState) -> dict:
    if state.get("on_agent"): state["on_agent"]("critic", "active")
    print("[5/5] Running critic...")

    static_context = f"""Style analysis:
{state["style_analysis"]}

Plot threads:
{json.dumps(state["selected_context"].get("plot_threads", {}), indent=2, ensure_ascii=False)}

World:
{json.dumps(state["selected_context"].get("world", {}), indent=2, ensure_ascii=False)}

Active characters:
{json.dumps(state["selected_context"].get("characters", {}), indent=2, ensure_ascii=False)}

Current scene state:
{json.dumps(state["active_state"], indent=2, ensure_ascii=False)}"""

    try:
        response = llm.invoke([
            SystemMessage(content=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]),
            HumanMessage(content=[
                {"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"Generated continuation to review:\n{state['generated_text']}"}
            ])
        ])
    except Exception as e:
        print(f"  Critic error: {e}")
        save_checkpoint(state["session_dir"], "writer")
        if state.get("on_agent"): state["on_agent"]("critic", f"error:Critic failed — {e}")
        return {"pipeline_error": True}

    content = response.content
    approved = any(phrase in content.lower() for phrase in ["approved: true", "approved:true", "approved: yes"])
    feedback = content.split("FEEDBACK:")[-1].strip()

    print(f"  Approved: {approved}")
    print(f"  Feedback: {feedback}")

    if state.get("on_agent"): state["on_agent"]("critic", "done")
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    return {
        "approved": approved,
        "critic_feedback": [feedback],
        "iteration": state["iteration"] + 1,
        "total_tokens": tokens
    }
