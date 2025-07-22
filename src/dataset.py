# -*- coding: utf-8 -*-
"""dataset.py

包含情感分类任务的数据集类 ``SentimentDataset`` 以及与之配套的 ``collate_fn``。
所有注释采用中文。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import jieba
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

__all__ = ["SentimentDataset", "collate_fn"]


class SentimentDataset(Dataset):
    """自定义情感分类数据集"""

    LABEL2ID = {
        "angry": 0,
        "neutral": 1,
        "happy": 2,
        "sad": 3,
        "surprise": 4,
        "fear": 5,
    }

    def __init__(self, data_path: str | Path, word2id_path: str | Path, max_token_len: int = 100) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.word2id_path = Path(word2id_path)
        self.max_token_len = max_token_len

        # 读取数据与词表
        self._examples = json.load(self.data_path.open("r", encoding="utf-8"))
        self._word2id: dict[str, int] = json.load(self.word2id_path.open("r", encoding="utf-8"))

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _text_to_ids(self, sentence: str) -> List[int]:
        """将句子分词并映射到 id 序列。未知词使用 1 (<unk>) 表示。"""
        ids: List[int] = []
        for word in jieba.lcut(sentence)[: self.max_token_len]:
            ids.append(self._word2id.get(word, 1))
        return ids

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, int]:
        sample = self._examples[index]
        sentence_tensor = torch.LongTensor(self._text_to_ids(sample["content"]))
        label_id = self.LABEL2ID[sample["label"]]
        return sentence_tensor, label_id

    def __len__(self) -> int:  # noqa: D401
        return len(self._examples)


# ----------------------------------------------------------------------
# DataLoader 批处理函数
# ----------------------------------------------------------------------

def collate_fn(batch):
    """根据最大句长填充文本序列，并将 label 列表转为张量。"""
    sentence_ids, label_ids = zip(*batch)
    padded_sentence_ids = pad_sequence(sentence_ids, batch_first=True, padding_value=0)
    label_tensor = torch.LongTensor(label_ids)
    return padded_sentence_ids, label_tensor