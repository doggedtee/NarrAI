import operator
from typing import TypedDict, Optional, Annotated


class NarrAIState(TypedDict):
    chapters: list[dict]
    style_analysis: Optional[str]
    whole_state: Optional[dict]
    active_state: Optional[dict]
    predictions: Optional[str]
    generated_text: Optional[str]
    critic_feedback: Annotated[list[str], operator.add]
    approved: bool
    iteration: int
