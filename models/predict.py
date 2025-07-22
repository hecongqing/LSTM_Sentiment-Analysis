"""models/predict.py
Utility for batch or single-sentence predictions.

Examples
--------
# Single text
python models/predict.py --model_dir checkpoints/bert --input "天气真好，我好开心"

# CSV file prediction
python models/predict.py --model_dir checkpoints/bert --input_file data/processed/val.csv --output_file preds.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Union[str, Path]):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()

    with open(model_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)["id2label"]
        id2label = {int(k): v for k, v in mapping.items()}
    return tokenizer, model, id2label


def predict(texts: List[str], tokenizer, model, id2label):
    enc = tokenizer(
        texts,
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
    return labels, probs


def main(model_dir: str, input_text: str | None, input_file: str | None, output_file: str | None):
    tokenizer, model, id2label = load_model(model_dir)

    if input_text:
        labels, probs = predict([input_text], tokenizer, model, id2label)
        print("Input:", input_text)
        print("Predicted label:", labels[0])
        print("Probabilities:", probs[0])
    elif input_file:
        df = pd.read_csv(input_file)
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column")
        batch_labels, _ = predict(df["text"].tolist(), tokenizer, model, id2label)
        df["pred"] = batch_labels
        save_path = output_file or (Path(input_file).with_suffix("_pred.csv"))
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    else:
        raise ValueError("Provide --input or --input_file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/bert")
    parser.add_argument("--input", dest="input_text", help="Raw sentence input")
    parser.add_argument("--input_file", help="Path to CSV file with a 'text' column")
    parser.add_argument("--output_file", help="Where to write predictions (CSV)")
    args = parser.parse_args()

    main(args.model_dir, args.input_text, args.input_file, args.output_file)