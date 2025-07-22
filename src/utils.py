# -*- coding: utf-8 -*-
"""utils.py

该模块包含公共的工具函数，例如指标计算、设备选择等。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

__all__ = ["set_seed", "get_device", "compute_metrics", "save_checkpoint"]


def set_seed(seed: int = 42):
    """设置随机种子，保证可复现性"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:  # noqa: D401
    """优先使用 GPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    """计算准确率与宏平均 P/R/F1"""
    acc = accuracy_score(labels, preds)
    mp, mr, mf, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "acc": acc,
        "mp": mp,
        "mr": mr,
        "mf": mf,
    }


def save_checkpoint(model: torch.nn.Module, path: str | Path):
    """保存模型参数"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))