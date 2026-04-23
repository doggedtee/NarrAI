import json
from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def critic(state: NarrAIState) -> dict:
    print("[5/5] Running critic...")

    prompt = f"""You are a literary critic. Review the generated chapter continuation for consistency and quality.

Style analysis:
{state["style_analysis"]}

World state:
{json.dumps(state["whole_state"], indent=2)}

Current scene state:
{json.dumps(state["active_state"], indent=2)}

Generated continuation to review:
{state["generated_text"]}

Approve if:
- Lore and world-building is consistent
- Characters behave consistently with their established traits
- Plot is coherent

Only decline for style if it is a MAJOR deviation — minor style differences are acceptable. Creativity is encouraged.

Respond in this exact format:
APPROVED: true or false
FEEDBACK: brief feedback in 2-3 sentences"""

    response = llm.invoke(prompt)
    content = response.content

    approved = any(phrase in content.lower() for phrase in ["approved: true", "approved:true", "approved: yes"])
    feedback = content.split("FEEDBACK:")[-1].strip()

    print(f"  Approved: {approved}")
    print(f"  Feedback: {feedback}")

    return {
        "approved": approved,
        "critic_feedback": [feedback],
        "iteration": state["iteration"] + 1
    }
