import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.rag import search
from core.llm import llm


SYSTEM_PROMPT = """You are a literary analyst specializing in writing style analysis.

Analyze the writing style of the author based on the provided excerpts. Focus on:
- Narrative voice and tone
- Pacing and sentence structure
- Descriptive patterns
- Dialogue style
- Recurring themes and motifs

Provide a detailed style analysis."""


def analyzer(state: NarrAIState) -> dict:
    if state.get("on_agent"): state["on_agent"]("analyzer", "active")
    print("[1/5] Running analyzer...")

    gen = os.path.join(state["session_dir"], "data", "generated")
    style_analysis_path = os.path.join(gen, "style_analysis.txt")

    has_predicted = any(c["filename"].startswith("predicted") for c in state["chapters"])
    if has_predicted and os.path.exists(style_analysis_path):
        print("  Loading cached style analysis...")
        if state.get("on_agent"): state["on_agent"]("analyzer", "done")
        with open(style_analysis_path, "r", encoding="utf-8") as f:
            return {"style_analysis": f.read()}

    context = search(state["vectorstore"], "writing style pacing tone narrative voice")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Excerpts:\n\n{chr(10).join(context)}")
    ])

    os.makedirs(gen, exist_ok=True)
    with open(style_analysis_path, "w", encoding="utf-8") as f:
        f.write(response.content)

    if state.get("on_agent"): state["on_agent"]("analyzer", "done")
    tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    return {"style_analysis": response.content, "total_tokens": tokens}
