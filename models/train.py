"""models/train.py
Fine-tune a Chinese BERT (or other transformer) on SMP2020 emotion labels.

The script relies solely on the official 🤗 Transformers Trainer so that the
resulting checkpoints can be reused directly for FastAPI / Streamlit demo.

Example run (GPU):
```
python models/train.py --data_dir data/processed --model_dir checkpoints/bert
```
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def get_label_mapping(data_dir: Path):
    labels_path = data_dir / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError("labels.txt not found, run data/preprocess.py first")
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label


def tokenize_fn(examples, tokenizer, label2id):
    return {
        **tokenizer(
            examples["text"], truncation=True, max_length=128, padding="max_length"
        ),
        "labels": [label2id[l] for l in examples["label"]],
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


def main(data_dir: str, model_dir: str, pretrained_model: str = "bert-base-chinese"):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    labels, label2id, id2label = get_label_mapping(data_dir)
    num_labels = len(labels)

    # Load csv via 🤗 Datasets for convenience
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(data_dir / "train.csv"),
            "validation": str(data_dir / "val.csv"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized = dataset.map(
        lambda example: tokenize_fn(example, tokenizer, label2id),
        batched=True,
        remove_columns=["text", "label"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(model_dir))

    # Save label mapping for inference scripts
    with open(model_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    print("Training complete. Model saved to", model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="checkpoints/bert")
    parser.add_argument("--pretrained_model", default="bert-base-chinese")
    args = parser.parse_args()

    # If GPU available, use it by default
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main(args.data_dir, args.model_dir, args.pretrained_model)