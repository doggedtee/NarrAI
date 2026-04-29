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


def run(session_dir: str = ".", on_agent=None):
    init_db()

    chapters_dir = os.path.join(session_dir, "chapters")
    predicted_dir = os.path.join(session_dir, "predicted_chapters")
    original_dir = os.path.join(session_dir, "original_chapters")

    chapters = load_chapters(chapters_dir)
    if not chapters and os.path.exists(original_dir):
        chapters = load_chapters(original_dir)

    order_path = os.path.join(session_dir, "order.json")
    if os.path.exists(order_path):
        import json as _json
        with open(order_path, encoding="utf-8") as f:
            order = _json.load(f)
        order_map = {name: i for i, name in enumerate(order)}
        chapters.sort(key=lambda c: order_map.get(c["filename"], len(order)))

    vectorstore = build_vectorstore(chapters)

    original_count = len(os.listdir(original_dir)) if os.path.exists(original_dir) else len(chapters)
    predicted_count = len([f for f in os.listdir(predicted_dir) if f.endswith(".txt") and "_predictions" not in f]) if os.path.exists(predicted_dir) else 0
    next_chapter_num = original_count + predicted_count + 1

    pipeline_checkpoint_path = os.path.join(session_dir, "data", "pipeline_checkpoint.json")
    resume_from = None
    if os.path.exists(pipeline_checkpoint_path):
        import json as _json2
        with open(pipeline_checkpoint_path, encoding="utf-8") as f:
            resume_from = _json2.load(f).get("resume_from")

    initial_state = {
        "chapters": chapters,
        "vectorstore": vectorstore,
        "next_chapter_num": next_chapter_num,
        "session_dir": session_dir,
        "resume_from": resume_from,
        "on_agent": on_agent,
        "style_analysis": None,
        "selected_context": None,
        "active_state": None,
        "chapter_summary": None,
        "predictions": None,
        "generated_text": None,
        "critic_feedback": [],
        "total_tokens": 0,
        "approved": False,
        "iteration": 0
    }

    app = build_graph()
    result = app.invoke(initial_state)

    generated_text = result["generated_text"]
    chapter_title = None
    if generated_text.startswith("Chapter Title:"):
        lines = generated_text.split("\n", 2)
        chapter_title = lines[0].replace("Chapter Title:", "").strip()
        generated_text = lines[2].strip() if len(lines) > 2 else generated_text
        result["generated_text"] = generated_text
        result["chapter_title"] = chapter_title

    with open(os.path.join(chapters_dir, "predicted.txt"), "w", encoding="utf-8") as f:
        f.write(generated_text)

    os.makedirs(predicted_dir, exist_ok=True)
    if chapter_title:
        import re as _re
        base_name = _re.sub(r'[\\/*?:"<>|]', "", chapter_title).strip()
        if os.path.exists(os.path.join(predicted_dir, f"{base_name}.txt")):
            counter = 2
            while os.path.exists(os.path.join(predicted_dir, f"{base_name} {counter}.txt")):
                counter += 1
            base_name = f"{base_name} {counter}"
    else:
        existing = [f for f in os.listdir(predicted_dir) if f.endswith(".txt") and not f.endswith("_predictions.txt")]
        base_name = f"predicted_{len(existing) + 1}"

    with open(os.path.join(predicted_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
        f.write(result["generated_text"])

    with open(os.path.join(predicted_dir, f"{base_name}_predictions.txt"), "w", encoding="utf-8") as f:
        f.write(result["predictions"])

    save_prediction(result["predictions"], chapters[-1]["filename"])

    print("=== PREDICTIONS ===")
    print(result["predictions"])
    print("\n=== GENERATED CHAPTER ===")
    print(result["generated_text"])

    return result


if __name__ == "__main__":
    run()
