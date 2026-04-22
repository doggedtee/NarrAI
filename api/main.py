from fastapi import FastAPI
from main import run
from db.database import get_predictions

app = FastAPI()


@app.post("/run")
def run_pipeline():
    run()
    return {"status": "done"}


@app.get("/predictions")
def predictions():
    return get_predictions()
