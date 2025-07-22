# -*- coding: utf-8 -*-
"""onnx_server.py

使用 FastAPI 部署 ONNX 模型，提供 HTTP 推理接口。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

APP = FastAPI(title="中文情感分类 ONNX 服务")

MODEL_PATH = Path(__file__).resolve().parent / "model_best.onnx"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"未找到 ONNX 模型文件: {MODEL_PATH}")

# 创建推理 session
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name
OUTPUT_NAME = SESSION.get_outputs()[0].name


@APP.post("/predict", summary="批量预测", response_model=List[int])
async def predict(batch_inputs: List[List[int]]):
    """接收 batch 的 input_ids，返回预测类别索引列表。"""
    # 使用 0 进行 padding ，以保证输入为矩阵
    max_len = max(len(x) for x in batch_inputs)
    batch_array = np.zeros((len(batch_inputs), max_len), dtype=np.int64)
    for idx, seq in enumerate(batch_inputs):
        batch_array[idx, : len(seq)] = seq

    logits = SESSION.run([OUTPUT_NAME], {INPUT_NAME: batch_array})[0]
    preds = logits.argmax(axis=1).tolist()
    return preds


if __name__ == "__main__":
    uvicorn.run("onnx_server:APP", host="0.0.0.0", port=8000, reload=False)