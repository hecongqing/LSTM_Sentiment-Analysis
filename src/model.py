# -*- coding: utf-8 -*-
"""model.py

该模块实现基于 LSTM 的中文情感分类模型 ``LSTMModel``。
"""
from __future__ import annotations

import torch.nn as nn

__all__ = ["LSTMModel"]


class LSTMModel(nn.Module):
    """LSTM + Global Max Pooling 用于文本分类"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_idx: int = 0,
        num_labels: int = 6,
    ) -> None:
        super().__init__()

        # 词向量层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM 编码层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # DropOut 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 分类层
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * direction_factor, num_labels)

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self, input_ids):  # noqa: D401
        """前向传播逻辑

        参数
        ------
        input_ids: torch.LongTensor
            shape = ``(batch_size, seq_len)``
        """
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)  # [batch, seq_len, hidden]

        # Global Max Pooling
        pooled_output, _ = lstm_output.max(dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        return logits