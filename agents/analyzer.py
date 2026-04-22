from langchain_anthropic import ChatAnthropic
from core.state import NarrAIState
from core.rag import search

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def analyzer(state: NarrAIState) -> dict:
    print("[1/5] Running analyzer...")
    context = search(state["vectorstore"], "writing style pacing tone narrative voice")

    prompt = f"""You are a literary analyst. Analyze the writing style of the author based on the following excerpts.

Focus on:
- Narrative voice and tone
- Pacing and sentence structure
- Descriptive patterns
- Dialogue style
- Recurring themes and motifs

Excerpts:
{"\n\n".join(context)}

Provide a detailed style analysis."""

    response = llm.invoke(prompt)
    return {"style_analysis": response.content}
