"""FastAPI server: dataset upload, training control, status polling, before/after generation."""
import re
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import core

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="GPT-2 Fine-tuning Lab")

STATE = {"texts": [], "prog": core.Progress(), "thread": None}


class TrainRequest(BaseModel):
    base_model: str = Field("distilgpt2", max_length=80)
    run_name: str = Field("my_run", pattern=r"^[A-Za-z0-9_-]{1,40}$")
    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=64)
    lr: float = Field(5e-5, gt=0, le=1e-2)
    block_size: int = Field(64, ge=8, le=512)
    max_steps: int = Field(0, ge=0, le=100000)
    seed: int = 42


class TextBody(BaseModel):
    text: str = Field(..., max_length=2_000_000)


class GenRequest(BaseModel):
    run: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")
    prompt: str = Field("", max_length=1000)
    max_new_tokens: int = Field(60, ge=1, le=200)
    temperature: float = Field(0.8, gt=0, le=2)
    top_k: int = Field(50, ge=0, le=500)
    top_p: float = Field(0.95, gt=0, le=1)
    seed: int = 0
    samples: int = Field(2, ge=1, le=5)


def _set_dataset(raw: str) -> dict:
    texts = core.split_examples(core.clean_text(raw))
    if not texts:
        raise HTTPException(400, "No usable text found (each example needs at least 10 characters).")
    STATE["texts"] = texts
    return core.dataset_stats(texts)


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.post("/api/dataset/text")
def dataset_text(body: TextBody):
    return _set_dataset(body.text)


@app.post("/api/dataset/file")
async def dataset_file(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(413, "File too large (max 2 MB).")
    return _set_dataset(raw.decode("utf-8", errors="ignore"))


@app.post("/api/dataset/sample")
def dataset_sample():
    return _set_dataset((ROOT / "data" / "campus_announcements.txt").read_text())


@app.get("/api/dataset")
def dataset_info():
    return core.dataset_stats(STATE["texts"]) if STATE["texts"] else {"examples": 0}


def _run_training(texts, cfg, prog):
    try:
        core.train(texts, cfg, prog)
    except Exception as e:          # surface errors in the UI instead of dying silently
        prog.status, prog.message = "error", f"{type(e).__name__}: {e}"


@app.post("/api/train")
def start_training(req: TrainRequest):
    if not STATE["texts"]:
        raise HTTPException(400, "Upload or load a dataset first.")
    t = STATE["thread"]
    if t and t.is_alive():
        raise HTTPException(409, "A training run is already in progress.")
    prog = core.Progress()
    STATE["prog"] = prog
    cfg = core.TrainConfig(**req.model_dump())
    STATE["thread"] = threading.Thread(target=_run_training, args=(list(STATE["texts"]), cfg, prog), daemon=True)
    STATE["thread"].start()
    return {"started": True}


@app.post("/api/stop")
def stop():
    STATE["prog"].stop = True
    return {"stopping": True}


@app.get("/api/status")
def status():
    p = STATE["prog"]
    d = asdict(p)
    d.pop("stop")
    return d


@app.get("/api/runs")
def runs():
    return core.list_runs()


@app.post("/api/generate")
def gen(req: GenRequest):
    try:
        return core.compare(req.run, req.prompt, n=req.samples, max_new_tokens=req.max_new_tokens,
                            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p, seed=req.seed)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
