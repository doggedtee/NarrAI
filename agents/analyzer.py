import os
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import NarrAIState
from core.rag import search
from core.llm import llm

STYLE_ANALYSIS_PATH = "data/generated/style_analysis.txt"

SYSTEM_PROMPT = """You are a literary analyst specializing in writing style analysis.

Analyze the writing style of the author based on the provided excerpts. Focus on:
- Narrative voice and tone
- Pacing and sentence structure
- Descriptive patterns
- Dialogue style
- Recurring themes and motifs

Provide a detailed style analysis."""


def analyzer(state: NarrAIState) -> dict:
    print("[1/5] Running analyzer...")

    has_predicted = any(c["filename"].startswith("predicted") for c in state["chapters"])
    if has_predicted and os.path.exists(STYLE_ANALYSIS_PATH):
        print("  Loading cached style analysis...")
        with open(STYLE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            return {"style_analysis": f.read()}

    context = search(state["vectorstore"], "writing style pacing tone narrative voice")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Excerpts:\n\n{chr(10).join(context)}")
    ])

    os.makedirs("data/generated", exist_ok=True)
    with open(STYLE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write(response.content)

    return {"style_analysis": response.content}
