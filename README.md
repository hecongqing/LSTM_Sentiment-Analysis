# SMP2020 Weibo Emotion Classification Pipeline

A complete, commented reference implementation for the SMP2020-EWECT dataset covering:

1. Data processing (`data/preprocess.py`)
2. Model training (`models/train.py`)
3. Batch & online prediction (`models/predict.py`)
4. RESTful deployment (`service/api.py`)
5. Visual demo (`service/ui.py`)

> All code is heavily commented so it can be used as teaching material.

## Quick start (GPU recommended)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Prepare dataset
python data/preprocess.py --raw_dir /path/to/xlsx --out_dir data/processed

# 3. Train
python models/train.py --data_dir data/processed --model_dir checkpoints/bert

# 4. Evaluate / predict
python models/predict.py --model_dir checkpoints/bert --input "今天阳光真好，我感到很开心"

# 5. Start REST API
uvicorn service.api:app --reload

# 6. Launch Streamlit UI (in another terminal)
streamlit run service/ui.py
```

## Directory layout

```
├── data
│   ├── preprocess.py       # XLSX -> CSV + tokenized splits
│   └── processed           # Generated after step 2
├── models
│   ├── train.py            # Fine-tune Chinese BERT for 6-way emotion clf
│   └── predict.py          # Helper for offline / online inference
├── service
│   ├── api.py              # FastAPI server exposing `/predict`
│   └── ui.py               # Streamlit front-end calling the API
├── requirements.txt
└── README.md
```

## Dataset

The code expects the original SMP2020 training/validation/test Excel files:

- `SMP2020_微博情绪分类_通用_Train.xlsx`
- `SMP2020_微博情绪分类_疫情_Train.xlsx`
- `...` (plus validation/test)

Place them under any folder and pass `--raw_dir` to the preprocessing script.

## Teaching notes

Each script contains step-by-step comments explaining what is happening and why.
Feel free to open the files while lecturing and walk through the stages.

Happy hacking! ✨