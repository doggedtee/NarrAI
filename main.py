import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from core.state import NarrAIState
from core.parser import load_chapters
from core.rag import build_vectorstore
from agents.world_builder import world_builder
from agents.plot_planner import plot_planner
from agents.cleaner import cleaner
from agents.analyzer import analyzer
from agents.predictor import predictor
from agents.writer import writer
from agents.critic import critic
from db.database import init_db, save_prediction


MAX_ITERATIONS = 3


def should_plan(state: NarrAIState) -> str:
    last_chapter = state["chapters"][-1]
    if not last_chapter["filename"].startswith("predicted"):
        return "cleaner"
    planned = state["selected_context"]["plot_threads"].get("planned", {})
    if not planned:
        return "cleaner"
    return "analyzer"


def should_rewrite(state: NarrAIState) -> str:
    if state["approved"] or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "writer"


def build_graph() -> StateGraph:
    graph = StateGraph(NarrAIState)

    graph.add_node("world_builder", world_builder)
    graph.add_node("cleaner", cleaner)
    graph.add_node("plot_planner", plot_planner)
    graph.add_node("analyzer", analyzer)
    graph.add_node("predictor", predictor)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)

    graph.set_entry_point("world_builder")
    graph.add_conditional_edges("world_builder", should_plan, {
        "cleaner": "cleaner",
        "analyzer": "analyzer"
    })
    graph.add_edge("cleaner", "plot_planner")
    graph.add_edge("plot_planner", "analyzer")
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

    original_count = len(os.listdir("original_chapters")) if os.path.exists("original_chapters") else len(chapters)
    predicted_count = len([f for f in os.listdir("predicted_chapters") if f.endswith(".txt") and "_predictions" not in f]) if os.path.exists("predicted_chapters") else 0
    next_chapter_num = original_count + predicted_count + 1

    initial_state = {
        "chapters": chapters,
        "vectorstore": vectorstore,
        "next_chapter_num": next_chapter_num,
        "style_analysis": None,
        "selected_context": None,
        "active_state": None,
        "chapter_summary": None,
        "predictions": None,
        "generated_text": None,
        "critic_feedback": [],
        "approved": False,
        "iteration": 0
    }

    app = build_graph()
    result = app.invoke(initial_state)

    with open("chapters/predicted.txt", "w", encoding="utf-8") as f:
        f.write(result["generated_text"])

    os.makedirs("predicted_chapters", exist_ok=True)
    existing = [f for f in os.listdir("predicted_chapters") if f.startswith("predicted_") and f.endswith(".txt") and "_predictions" not in f]
    next_num = len(existing) + 1
    with open(f"predicted_chapters/predicted_{next_num}.txt", "w", encoding="utf-8") as f:
        f.write(result["generated_text"])

    with open(f"predicted_chapters/predicted_{next_num}_predictions.txt", "w", encoding="utf-8") as f:
        f.write(result["predictions"])

    save_prediction(result["predictions"], chapters[-1]["filename"])

    print("=== PREDICTIONS ===")
    print(result["predictions"])
    print("\n=== GENERATED CHAPTER ===")
    print(result["generated_text"])


if __name__ == "__main__":
    run()
