"""service/api.py
FastAPI app exposing `/predict` endpoint for SMP2020 emotion model.

Start the server:
uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("checkpoints/bert")  # default path, override via env if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="SMP2020 Emotion API", version="1.0")


class TextIn(BaseModel):
    texts: List[str]

    class Config:
        schema_extra = {"example": {"texts": ["今天阳光真好，我感到很开心！"]}}


class PredOut(BaseModel):
    labels: List[str]
    probs: List[List[float]]


@app.on_event("startup")
def load_model():
    """Load model into global state."""
    global tokenizer, model, id2label

    if not MODEL_DIR.exists():
        raise RuntimeError("Model directory not found. Train the model first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE).eval()

    with open(MODEL_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)["id2label"]
    id2label = {int(k): v for k, v in mapping.items()}


@app.post("/predict", response_model=PredOut)
def predict(payload: TextIn):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    enc = tokenizer(
        payload.texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().tolist()
        probs = probs.cpu().tolist()
    labels = [id2label[p] for p in preds]
    return PredOut(labels=labels, probs=probs)