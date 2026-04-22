import os

def load_chapters(chapters_dir: str) -> list[dict]:
    chapters = []

    files = sorted(os.listdir(chapters_dir))

    for file in files:
        if not file.endswith(".txt"):
            continue

        path = os.path.join(chapters_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chapters.append({
            "filename": file,
            "text": text
        })

    return chapters
