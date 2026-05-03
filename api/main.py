import os
import re
import time
import shutil
import asyncio
import json
import io
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ebooklib import epub
from main import run
from db.database import get_predictions

SESSIONS_DIR = "sessions"
SESSION_TTL = 24 * 60 * 60

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


HOST_SESSION_ID = os.getenv("HOST_SESSION_ID", "damir-main")

def cleanup_sessions():
    if not os.path.exists(SESSIONS_DIR):
        return
    now = time.time()
    for sid in os.listdir(SESSIONS_DIR):
        if sid == HOST_SESSION_ID:
            continue
        path = os.path.join(SESSIONS_DIR, sid)
        if os.path.isdir(path) and now - os.path.getmtime(path) > SESSION_TTL:
            shutil.rmtree(path)


def get_session_id(request: Request) -> str:
    sid = request.headers.get("X-Session-ID", "") or request.query_params.get("session_id", "")
    sid = re.sub(r"[^a-zA-Z0-9-]", "", sid)[:64]
    return sid if sid else "default"


def get_session_dir(request: Request) -> str:
    sid = get_session_id(request)
    path = os.path.join(SESSIONS_DIR, sid)
    os.makedirs(os.path.join(path, "chapters"), exist_ok=True)
    return path


def is_host(request: Request) -> bool:
    return get_session_id(request) == HOST_SESSION_ID


cleanup_sessions()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)):
    session_dir = get_session_dir(request)
    chapters_dir = os.path.join(session_dir, "chapters")

    if not is_host(request):
        existing = [f for f in os.listdir(chapters_dir) if f.endswith(".txt") and not f.startswith("predicted")]
        original_dir = os.path.join(session_dir, "original_chapters")
        if os.path.exists(original_dir):
            existing += os.listdir(original_dir)
        if existing:
            raise HTTPException(status_code=403, detail="Demo limit: only 1 source chapter allowed.")

    saved = []
    for file in files:
        content = await file.read()
        if not is_host(request) and len(content.decode("utf-8", errors="ignore").split()) > 3000:
            raise HTTPException(status_code=403, detail="Demo limit: chapter must be under 3000 words.")
        path = os.path.join(chapters_dir, file.filename)
        with open(path, "wb") as f:
            f.write(content)
        saved.append(file.filename)
    return {"saved": saved}


@app.get("/api/session/info")
async def session_info(request: Request):
    return {"is_host": is_host(request)}


@app.get("/api/generate/check")
async def generate_check(request: Request):
    if not is_host(request):
        session_dir = get_session_dir(request)
        predicted_dir = os.path.join(session_dir, "predicted_chapters")
        if os.path.exists(predicted_dir):
            existing = [f for f in os.listdir(predicted_dir) if f.endswith(".txt") and not f.endswith("_predictions.txt")]
            if existing:
                raise HTTPException(status_code=403, detail="Demo limit: only 1 generated chapter allowed.")
    return {"ok": True}


@app.get("/api/generate")
async def generate(request: Request):
    session_dir = get_session_dir(request)

    if not is_host(request):
        predicted_dir = os.path.join(session_dir, "predicted_chapters")
        if os.path.exists(predicted_dir):
            existing = [f for f in os.listdir(predicted_dir) if f.endswith(".txt") and not f.endswith("_predictions.txt")]
            if existing:
                raise HTTPException(status_code=403, detail="Demo limit: only 1 generated chapter allowed.")

    async def event_stream():
        queue = asyncio.Queue()
        result_holder = {}

        def on_agent(agent: str, status: str):
            queue.put_nowait({"type": "agent", "agent": agent, "status": status})

        def run_pipeline():
            try:
                result = run(session_dir=session_dir, on_agent=on_agent)
                result_holder["result"] = result
                if result.get("pipeline_error"):
                    queue.put_nowait({"type": "pipeline_error"})
                else:
                    queue.put_nowait({"type": "done"})
            except Exception as e:
                queue.put_nowait({"type": "pipeline_error", "message": str(e)})

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, run_pipeline)

        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] == "pipeline_error":
                break
            if event["type"] == "done":
                result = result_holder.get("result", {})
                yield f"data: {json.dumps({'type': 'result', 'text': result.get('generated_text', ''), 'title': result.get('chapter_title', ''), 'tokens': result.get('total_tokens', 0)})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/chapters")
async def list_chapters(request: Request):
    session_dir = get_session_dir(request)
    chapters_dir = os.path.join(session_dir, "chapters")
    original_dir = os.path.join(session_dir, "original_chapters")
    predicted_dir = os.path.join(session_dir, "predicted_chapters")
    order_path = os.path.join(session_dir, "order.json")

    source_map = {}
    for src_dir in [original_dir, chapters_dir]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                if f.endswith(".txt") and not f.startswith("predicted") and f not in source_map:
                    with open(os.path.join(src_dir, f), encoding="utf-8") as fh:
                        source_map[f] = {"name": f, "content": fh.read(), "type": "source"}

    generated_map = {}
    if os.path.exists(predicted_dir):
        for f in os.listdir(predicted_dir):
            if f.endswith(".txt") and not f.endswith("_predictions.txt"):
                with open(os.path.join(predicted_dir, f), encoding="utf-8") as fh:
                    text = fh.read()
                generated_map[f] = {"name": f, "text": text, "words": len(text.split()), "type": "generated"}

    all_map = {**source_map, **generated_map}

    if os.path.exists(order_path):
        with open(order_path, encoding="utf-8") as f:
            order = json.load(f)
        ordered = [all_map[n] for n in order if n in all_map]
        remaining = [v for k, v in all_map.items() if k not in order]
        items = ordered + remaining
    else:
        items = sorted(all_map.values(), key=lambda x: os.path.getmtime(
            os.path.join(src_dir if x["type"] == "source" else predicted_dir, x["name"])
        ))

    source = [i for i in items if i["type"] == "source"]
    generated = [i for i in items if i["type"] == "generated"]
    return {"source": source, "generated": generated}


@app.post("/api/chapters/order")
async def save_order(request: Request):
    session_dir = get_session_dir(request)
    body = await request.json()
    with open(os.path.join(session_dir, "order.json"), "w", encoding="utf-8") as f:
        json.dump(body.get("order", []), f)
    return {"ok": True}


@app.delete("/api/chapters/{filename}")
async def delete_chapter(filename: str, request: Request):
    session_dir = get_session_dir(request)
    for subdir in ("chapters", "original_chapters", "predicted_chapters"):
        path = os.path.join(session_dir, subdir, filename)
        if os.path.exists(path):
            os.remove(path)
            return {"deleted": filename}
    return {"deleted": None}


@app.post("/api/export")
async def export(data: dict):
    title = data.get("title", "Book")
    chapters_data = data.get("chapters", [])

    book = epub.EpubBook()
    book.set_title(title)
    book.set_language("en")

    epub_chapters = []
    for i, chapter in enumerate(chapters_data):
        c = epub.EpubHtml(title=f"Chapter {i + 1}", file_name=f"chap_{i + 1}.xhtml", lang="en")
        text = chapter.get("text", "")
        paragraphs = "".join(f"<p>{line}</p>" for line in text.split("\n") if line.strip())
        c.content = f"<h1>Chapter {i + 1}</h1>{paragraphs}"
        book.add_item(c)
        epub_chapters.append(c)

    book.toc = epub_chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    buf = io.BytesIO()
    epub.write_epub(buf, book)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/epub+zip",
        headers={"Content-Disposition": f'attachment; filename="{title}.epub"'}
    )


@app.get("/predictions")
def predictions():
    return get_predictions()
