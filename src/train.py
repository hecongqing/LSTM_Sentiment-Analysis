# -*- coding: utf-8 -*-
"""train.py

单文件训练脚本，可通过命令行参数控制训练过程。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from .dataset import SentimentDataset, collate_fn
from .model import LSTMModel
from .utils import compute_metrics, get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="训练 LSTM 情感分类模型")
    parser.add_argument("--train", type=str, default="./data/usual_train.txt", help="训练集路径")
    parser.add_argument("--dev", type=str, default="./data/usual_eval_labeled.txt", help="验证集路径")
    parser.add_argument("--word2id", type=str, default="./data/word2id.json", help="word2id 路径")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM 隐藏层维度")
    parser.add_argument("--embedding_dim", type=int, default=300, help="词向量维度")
    parser.add_argument("--output", type=str, default="./outputs", help="模型与日志输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def build_dataloader(
    data_path: str | Path,
    word2id_path: str | Path,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:  # noqa: D401
    dataset = SentimentDataset(data_path, word2id_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)


def prepare_model(word2id_path: str | Path, args: argparse.Namespace) -> LSTMModel:  # noqa: D401
    vocab_size = len(torch.load(word2id_path)) if str(word2id_path).endswith(".pt") else len(__import__("json").load(open(word2id_path)))
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=1,
        bidirectional=True,
        dropout=0.1,
        pad_idx=0,
        num_labels=6,
    )
    return model


def train_one_epoch(
    model: LSTMModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CrossEntropyLoss,
    device: torch.device,
) -> Tuple[float, list[int], list[int]]:  # noqa: D401
    model.train()
    total_loss = 0.0
    preds: list[int] = []
    labels: list[int] = []
    for sentence_ids, label_ids in dataloader:
        sentence_ids, label_ids = sentence_ids.to(device), label_ids.to(device)
        outputs = model(sentence_ids)
        loss = criterion(outputs, label_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.extend(outputs.argmax(dim=1).cpu().tolist())
        labels.extend(label_ids.cpu().tolist())
    avg_loss = total_loss / len(dataloader)
    return avg_loss, labels, preds


def evaluate(
    model: LSTMModel,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    device: torch.device,
) -> Tuple[float, list[int], list[int]]:  # noqa: D401
    model.eval()
    total_loss = 0.0
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for sentence_ids, label_ids in dataloader:
            sentence_ids, label_ids = sentence_ids.to(device), label_ids.to(device)
            outputs = model(sentence_ids)
            loss = criterion(outputs, label_ids)
            total_loss += loss.item()
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            labels.extend(label_ids.cpu().tolist())
    avg_loss = total_loss / len(dataloader)
    return avg_loss, labels, preds


def main():  # noqa: D401
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    output_dir = Path(args.output)
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    # DataLoader
    train_loader = build_dataloader(args.train, args.word2id, args.batch_size, shuffle=True)
    dev_loader = build_dataloader(args.dev, args.word2id, batch_size=256, shuffle=False)

    # Model
    model = prepare_model(args.word2id, args).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    best_dev_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_labels, train_preds = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = compute_metrics(train_labels, train_preds)

        dev_loss, dev_labels, dev_preds = evaluate(model, dev_loader, criterion, device)
        dev_metrics = compute_metrics(dev_labels, dev_preds)

        # TensorBoard 记录
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/dev", dev_loss, epoch)
        writer.add_scalar("acc/train", train_metrics["acc"], epoch)
        writer.add_scalar("acc/dev", dev_metrics["acc"], epoch)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            f"train_acc={train_metrics['acc']:.4f} dev_acc={dev_metrics['acc']:.4f}"
        )

        # 保存最优模型
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, ckpt_dir / "best.pt")

    print("训练完毕, 最佳 dev_loss = {:.4f}".format(best_dev_loss))


if __name__ == "__main__":
    main()