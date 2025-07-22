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
from torch.nn import functional as F  # kept for softmax
import jieba
import numpy as np
import onnxruntime as ort
from torch.nn.utils.rnn import pad_sequence

MODEL_DIR = Path("checkpoints/rnn")  # default path, override via env if needed
MAX_LEN = 128
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
    """Load ONNX session, vocab, and label mapping into app state."""
    global session, input_name, word2idx, id2label

    if not MODEL_DIR.exists():
        raise RuntimeError("Model directory not found. Train the model first.")

    # ONNX session (CPU/gpu if available)
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"))
    input_name = session.get_inputs()[0].name

    # Vocabulary
    with open(MODEL_DIR / "vocab.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    # Label mapping
    with open(MODEL_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)["id2label"]
    id2label = {int(k): v for k, v in mapping.items()}


def _encode(text: str) -> torch.Tensor:
    ids = [word2idx.get(tok, 1) for tok in jieba.lcut(text)][:MAX_LEN]
    return torch.tensor(ids, dtype=torch.long)


@app.post("/predict", response_model=PredOut)
def predict(payload: TextIn):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    encoded = [_encode(t) for t in payload.texts]
    batch = pad_sequence(encoded, batch_first=True, padding_value=0)
    ort_inputs = {input_name: batch.numpy()}
    logits = session.run(None, ort_inputs)[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=-1)
    preds = probs.argmax(dim=-1).tolist()
    probs = probs.tolist()
    labels = [id2label[p] for p in preds]
    return PredOut(labels=labels, probs=probs)