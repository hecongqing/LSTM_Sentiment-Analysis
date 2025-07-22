"""models/train_rnn.py
Train an LSTM-based text classifier on the SMP2020 emotion dataset **without** using
HuggingFace Transformers. After training, the script exports the model to ONNX so
that it can be served by a lightweight runtime (e.g. onnxruntime).

Example
-------
python models/train_rnn.py \
    --data_dir data/processed \
    --model_dir checkpoints/rnn \
    --epochs 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import jieba
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ------------------------------ Vocabulary ------------------------------ #

class Vocab:
    """A minimal word-level vocabulary."""

    def __init__(self, min_freq: int = 2, pad_token: str = "<pad>", unk_token: str = "<unk>"):
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx: dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: dict[int, str] = {0: pad_token, 1: unk_token}

    def build(self, texts: List[str]):
        counter: dict[str, int] = {}
        for sent in texts:
            for tok in jieba.lcut(sent):
                counter[tok] = counter.get(tok, 0) + 1
        for tok, freq in counter.items():
            if freq >= self.min_freq and tok not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[tok] = idx
                self.idx2word[idx] = tok

    def __len__(self) -> int:  # noqa: D401 (simple verb form)
        return len(self.word2idx)

    def encode(self, sentence: str) -> List[int]:
        return [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in jieba.lcut(sentence)]

# -----------------------------  Dataset  ------------------------------ #

class CsvDataset(Dataset):
    def __init__(self, csv_path: Path, vocab: Vocab, label2id: dict[str, int], max_len: int = 128):
        df = pd.read_csv(csv_path)
        if {"text", "label"}.difference(df.columns):
            raise ValueError("CSV must contain 'text' and 'label' columns")
        self.texts = df["text"].astype(str).tolist()
        self.labels = [label2id[l] for l in df["label"].tolist()]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):  # noqa: D401
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.vocab.encode(self.texts[idx])[: self.max_len], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return ids, label


def collate(batch):
    ids, labels = zip(*batch)
    ids_pad = pad_sequence(ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return ids_pad, labels

# ------------------------------- Model ------------------------------- #

class TextRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids):  # input: (B, L)
        emb = self.embedding(input_ids)  # (B, L, D)
        out, _ = self.lstm(emb)  # (B, L, 2H)
        pooled, _ = torch.max(out, dim=1)  # (B, 2H)
        logits = self.fc(pooled)  # (B, num_labels)
        return logits

# ----------------------------- Training ----------------------------- #

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for ids, labels in dataloader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return np.mean(losses), accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro")


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    losses, preds_all, labels_all = [], [], []
    with torch.no_grad():
        for ids, labels in dataloader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    return np.mean(losses), accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro")

# ------------------------------  Main  ------------------------------ #

def main(data_dir: str, model_dir: str, epochs: int = 5, batch_size: int = 64, embed_dim: int = 128, hidden_dim: int = 256):
    data_dir_p = Path(data_dir)
    model_dir_p = Path(model_dir)
    model_dir_p.mkdir(parents=True, exist_ok=True)

    # Label mapping from preprocessing step
    labels = (data_dir_p / "labels.txt").read_text(encoding="utf-8").splitlines()
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Build vocabulary on training texts
    train_df = pd.read_csv(data_dir_p / "train.csv")
    vocab = Vocab()
    vocab.build(train_df["text"].astype(str).tolist())
    print(f"Built vocab with {len(vocab)} tokens")

    # Save vocab
    with open(model_dir_p / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.word2idx, f, ensure_ascii=False, indent=2)

    # Prepare datasets
    train_ds = CsvDataset(data_dir_p / "train.csv", vocab, label2id)
    val_ds = CsvDataset(data_dir_p / "val.csv", vocab, label2id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextRNN(len(vocab), embed_dim, hidden_dim, len(labels)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} f1 {train_f1:.3f} || "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}"
        )

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_dir_p / "model.pt")

    # ------------------ Export to ONNX ------------------ #
    print("Exporting ONNX model …")
    dummy_input = torch.zeros(1, 128, dtype=torch.long)
    model_cpu = model.cpu()
    torch.onnx.export(
        model_cpu,
        dummy_input,
        model_dir_p / "model.onnx",
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Save label mapping for inference
    with open(model_dir_p / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    print("Finished. Artifacts saved to", model_dir_p.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="checkpoints/rnn")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    main(args.data_dir, args.model_dir, args.epochs)