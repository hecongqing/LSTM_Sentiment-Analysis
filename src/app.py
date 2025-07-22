# -*- coding: utf-8 -*-
"""app.py

Streamlit 可视化界面：
1. 用户输入待分类的中文句子（可多句换行）。
2. 将句子分词并根据 word2id.json 映射为 id 序列。
3. 调用 ONNX 推理服务 (FastAPI) 的 `/predict` 接口。
4. 在页面上展示情感分类结果。

启动方式：
```bash
streamlit run src/app.py
```
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import jieba
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 配置区（可在侧边栏修改）
# -----------------------------------------------------------------------------
DEFAULT_SERVICE_URL = "http://localhost:8000/predict"
DEFAULT_WORD2ID_PATH = "data/word2id.json"
MAX_TOKEN_LEN_DEFAULT = 100

service_url = st.sidebar.text_input("ONNX 服务地址", value=DEFAULT_SERVICE_URL)
word2id_path = st.sidebar.text_input("word2id.json 路径", value=DEFAULT_WORD2ID_PATH)
max_token_len = st.sidebar.slider("最大截断长度", min_value=10, max_value=200, value=MAX_TOKEN_LEN_DEFAULT, step=10)

# -----------------------------------------------------------------------------
# 常量与缓存
# -----------------------------------------------------------------------------
LABELS = {
    0: "愤怒 (angry)",
    1: "中性 (neutral)",
    2: "开心 (happy)",
    3: "伤心 (sad)",
    4: "惊讶 (surprise)",
    5: "害怕 (fear)",
}


@st.cache_data(show_spinner=False)
def load_word2id(path_str: str) -> Dict[str, int]:
    """加载词表映射。若文件不存在则返回空字典。"""
    path = Path(path_str)
    if not path.exists():
        st.error(f"word2id 文件不存在: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data: Dict[str, int] = json.load(f)
        return data
    except Exception as exc:  # noqa: BLE001
        st.error(f"加载 word2id 失败: {exc}")
        return {}


def sentence_to_ids(sentence: str, word2id: Dict[str, int], max_len: int) -> List[int]:
    """将句子转换为 id 序列，未知词 id = 1 (<unk>)。"""
    return [word2id.get(tok, 1) for tok in jieba.lcut(sentence)[:max_len]] or [0]


word2id_mapping = load_word2id(word2id_path)

# -----------------------------------------------------------------------------
# 页面主体
# -----------------------------------------------------------------------------
st.title("中文情感分类在线演示")

st.write("输入一条或多条中文句子，每行一句，点击“预测”按钮查看情感分类结果。")

user_text = st.text_area("请输入句子：", height=200)

if st.button("预测"):
    if not user_text.strip():
        st.warning("请输入待分类的文本。")
    elif not word2id_mapping:
        st.error("无法加载 word2id.json，无法进行预测。")
    else:
        sentences = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        batch_inputs = [sentence_to_ids(s, word2id_mapping, max_token_len) for s in sentences]

        st.info("正在调用 ONNX 服务进行预测……")
        try:
            resp = requests.post(service_url, json=batch_inputs, timeout=15)
            resp.raise_for_status()
            preds: List[int] = resp.json()

            st.success("预测完成！")
            for sent, pred in zip(sentences, preds):
                label = LABELS.get(pred, str(pred))
                st.write(f"**{label}**: {sent}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"调用服务失败: {exc}")