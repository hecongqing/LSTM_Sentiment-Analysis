"""models/predict_onnx.py
Run inference for the LSTM ONNX model produced by `train_rnn.py`.

Examples
--------
# Single text
python models/predict_onnx.py --model_dir checkpoints/rnn --input "天气真好，我好开心"

# CSV batch
python models/predict_onnx.py --model_dir checkpoints/rnn --input_file data/processed/val.csv --output_file preds.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Union

import jieba
import numpy as np
import onnxruntime as ort
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch

DEVICE = "cuda" if ort.get_device() == "GPU" else "cpu"

# ----------------------- Helpers ----------------------- #

def load_resources(model_dir: Union[str, Path]):
    model_dir = Path(model_dir)

    # Vocabulary
    with open(model_dir / "vocab.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    # Label mapping
    with open(model_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    session = ort.InferenceSession(str(model_dir / "model.onnx"))
    input_name = session.get_inputs()[0].name
    return session, input_name, word2idx, id2label


def encode(text: str, word2idx: dict[str, int], max_len: int = 128) -> torch.Tensor:
    idxs = [word2idx.get(tok, 1) for tok in jieba.lcut(text)][:max_len]
    return torch.tensor(idxs, dtype=torch.long)


def predict(texts: List[str], session, input_name, word2idx, id2label):
    encoded = [encode(t, word2idx) for t in texts]
    batch = pad_sequence(encoded, batch_first=True, padding_value=0)
    ort_inputs = {input_name: batch.cpu().numpy()}
    logits = session.run(None, ort_inputs)[0]  # (B, C)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy().tolist()
    preds = np.argmax(logits, axis=-1).tolist()
    labels = [id2label[p] for p in preds]
    return labels, probs


# ------------------------  CLI  ------------------------ #

def main(model_dir: str, input_text: str | None, input_file: str | None, output_file: str | None):
    session, input_name, word2idx, id2label = load_resources(model_dir)

    if input_text:
        labels, probs = predict([input_text], session, input_name, word2idx, id2label)
        print("Input:", input_text)
        print("Predicted label:", labels[0])
        print("Probabilities:", probs[0])
    elif input_file:
        df = pd.read_csv(input_file)
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column")
        batch_labels, _ = predict(df["text"].astype(str).tolist(), session, input_name, word2idx, id2label)
        df["pred"] = batch_labels
        save_path = output_file or (Path(input_file).with_suffix("_pred.csv"))
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    else:
        raise ValueError("Provide --input or --input_file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/rnn")
    parser.add_argument("--input", dest="input_text", help="Raw sentence input")
    parser.add_argument("--input_file", help="Path to CSV file with a 'text' column")
    parser.add_argument("--output_file", help="Where to write predictions (CSV)")
    args = parser.parse_args()

    main(args.model_dir, args.input_text, args.input_file, args.output_file)