# -*- coding: utf-8 -*-
"""convert_to_onnx.py

将训练得到的 PyTorch 模型权重转换为 ONNX 格式，方便部署。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

# 允许直接从项目根目录运行，方便找到 src 包
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import LSTMModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="将 pt 格式模型转换为 onnx")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重 .pt 文件路径")
    parser.add_argument("--output", type=str, default="model_best.onnx", help="输出 onnx 文件路径")
    parser.add_argument("--vocab_size", type=int, required=True, help="词表大小 (需与训练时一致)")
    return parser.parse_args()


def main():  # noqa: D401
    args = parse_args()
    device = torch.device("cpu")

    # 与训练时超参数保持一致
    model = LSTMModel(vocab_size=args.vocab_size)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randint(0, args.vocab_size, (1, 50), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}},
    )
    print(f"已成功导出 ONNX 模型至 {args.output}")


if __name__ == "__main__":
    main()