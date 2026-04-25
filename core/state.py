import operator
from typing import TypedDict, Optional, Annotated
from langchain_community.vectorstores import FAISS


class NarrAIState(TypedDict):
    chapters: list[dict]
    vectorstore: Optional[FAISS]
    style_analysis: Optional[str]
    selected_context: Optional[dict]
    active_state: Optional[dict]
    chapter_summary: Optional[str]
    predictions: Optional[str]
    generated_text: Optional[str]
    critic_feedback: Annotated[list[str], operator.add]
    approved: bool
    iteration: int
