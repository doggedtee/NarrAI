import os
import sqlite3

os.makedirs("data", exist_ok=True)

def get_connection():
    conn = sqlite3.connect("data/narrai.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            based_on_chapter TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
        )
    """)

    conn.commit()
    conn.close()

def save_prediction(text: str, based_on_chapter: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions (text, based_on_chapter) VALUES (?, ?)",
        (text, based_on_chapter)
    )

    conn.commit()
    conn.close()

def get_predictions() -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, text, based_on_chapter, created_at FROM predictions")
    rows = cursor.fetchall()

    conn.close()

    return [
        {"id": row["id"], "text": row["text"], "based_on_chapter": row["based_on_chapter"], "created_at": row["created_at"]}
        for row in rows
    ]