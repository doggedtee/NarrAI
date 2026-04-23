from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from core.state import NarrAIState
from core.parser import load_chapters
from core.rag import build_vectorstore
from agents.world_builder import world_builder
from agents.analyzer import analyzer
from agents.predictor import predictor
from agents.writer import writer
from agents.critic import critic
from db.database import init_db, save_prediction


MAX_ITERATIONS = 3

def should_rewrite(state: NarrAIState) -> str:
    if state["approved"] or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "writer"


def build_graph() -> StateGraph:
    graph = StateGraph(NarrAIState)

    graph.add_node("world_builder", world_builder)
    graph.add_node("analyzer", analyzer)
    graph.add_node("predictor", predictor)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)

    graph.set_entry_point("world_builder")
    graph.add_edge("world_builder", "analyzer")
    graph.add_edge("analyzer", "predictor")
    graph.add_edge("predictor", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", should_rewrite, {
        "writer": "writer",
        "end": END
    })

    return graph.compile()


def run(chapters_dir: str = "chapters/"):
    init_db()

    chapters = load_chapters(chapters_dir)
    vectorstore = build_vectorstore(chapters)

    initial_state = {
        "chapters": chapters,
        "vectorstore": vectorstore,
        "style_analysis": None,
        "whole_state": None,
        "active_state": None,
        "predictions": None,
        "generated_text": None,
        "critic_feedback": [],
        "approved": False,
        "iteration": 0
    }

    app = build_graph()
    result = app.invoke(initial_state)

    save_prediction(result["predictions"], chapters[-1]["filename"])

    print("=== PREDICTIONS ===")
    print(result["predictions"])
    print("\n=== GENERATED CHAPTER ===")
    print(result["generated_text"])


if __name__ == "__main__":
    run()
