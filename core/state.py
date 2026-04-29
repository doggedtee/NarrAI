import operator
from typing import TypedDict, Optional, Annotated, Callable
from langchain_community.vectorstores import FAISS


class NarrAIState(TypedDict):
    chapters: list[dict]
    vectorstore: Optional[FAISS]
    on_agent: Optional[Callable]
    session_dir: str
    resume_from: Optional[str]
    style_analysis: Optional[str]
    selected_context: Optional[dict]
    active_state: Optional[dict]
    chapter_summary: Optional[str]
    predictions: Optional[str]
    generated_text: Optional[str]
    next_chapter_num: int
    critic_feedback: Annotated[list[str], operator.add]
    total_tokens: Annotated[int, operator.add]
    approved: bool
    iteration: int
