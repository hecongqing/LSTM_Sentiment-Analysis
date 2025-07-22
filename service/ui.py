"""service/ui.py
A lightweight Streamlit front-end that talks to the local FastAPI service.
Run it after starting the API:

```bash
streamlit run service/ui.py
```
"""
import json
from typing import List

import requests
import streamlit as st

# Configuration ----------------------------------------------------------------
API_URL = "http://localhost:8000/predict"  # adjust if API runs elsewhere

st.set_page_config(page_title="SMP2020 Emotion Demo", page_icon="🧠")
st.title("🧠 SMP2020 Weibo Emotion Classification Demo")

# User input --------------------------------------------------------------------
input_text = st.text_area("输入微博文本 (Chinese Weibo sentence)", height=150)

if st.button("Predict") and input_text.strip():
    payload = {"texts": [input_text.strip()]}
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        label = data["labels"][0]
        probs: List[float] = data["probs"][0]

        st.markdown(f"### 预测情绪: **{label}**")

        # Show probability bar chart
        st.write("#### 概率分布：")
        st.bar_chart({"probability": probs})
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {e}")
else:
    st.info("请输入文本并点击 Predict")