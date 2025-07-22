"""data/preprocess.py
Preprocess the original SMP2020-EWECT Excel files into plain CSV ready for
HuggingFace `datasets` or pandas loading.

The script performs the following steps:
1. Read every *.xlsx file inside `--raw_dir`.
2. Concatenate them and map Chinese emotion labels to English tokens.
3. Split into train/val/test folds if not already separated.
4. Save the cleaned data to `--out_dir` in CSV format: `text,label`.

Usage
-----
python data/preprocess.py \
    --raw_dir ./raw_xlsx \
    --out_dir data/processed

Author: OpenAI ChatGPT (2025-07)
"""
from __future__ import annotations

import argparse
import pathlib
import random
from typing import List

import pandas as pd

# Mapping provided by the competition guideline
CHINESE2EN = {
    "积极": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "fear",
    "惊奇": "surprise",
    "无情绪": "neutral",  # organisers use neural -> here unify as neutral
    "neural": "neutral",  # keep original spelling if any
}

RANDOM_SEED = 42


def read_xlsx(path: pathlib.Path) -> pd.DataFrame:
    """Load a single xlsx file and rename columns to `text` and `label`."""
    df = pd.read_excel(path, engine="openpyxl")

    # Heuristically infer text/label column names
    col_lower = [c.lower() for c in df.columns]
    if {"文本", "情绪标签"}.issubset(df.columns):
        text_col, label_col = "文本", "情绪标签"
    elif {"text", "label"}.issubset(col_lower):
        text_col, label_col = df.columns[col_lower.index("text")], df.columns[
            col_lower.index("label")
        ]
    else:
        # Fallback: assume first two columns after id
        text_col, label_col = df.columns[1], df.columns[2]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map labels to English tokens."""
    df["label"] = (
        df["label"].astype(str).str.strip().map(CHINESE2EN).fillna(df["label"])
    )
    return df


def main(raw_dir: str, out_dir: str, val_ratio: float = 0.1):
    random.seed(RANDOM_SEED)
    raw_dir_path = pathlib.Path(raw_dir)
    out_dir_path = pathlib.Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Collect all xlsx files
    paths: List[pathlib.Path] = list(raw_dir_path.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No xlsx files found in {raw_dir}")

    data_frames: List[pd.DataFrame] = []
    for p in paths:
        df = read_xlsx(p)
        df = clean_labels(df)
        data_frames.append(df)
        print(f"Loaded {len(df):5d} rows from {p.name}")

    all_df = pd.concat(data_frames, ignore_index=True)
    print("Total rows:", len(all_df))

    # Shuffle deterministically
    all_df = all_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Split if there is no pre-separated validation/test
    train_cut = int(len(all_df) * (1 - val_ratio))
    train_df = all_df.iloc[:train_cut]
    val_df = all_df.iloc[train_cut:]

    train_df.to_csv(out_dir_path / "train.csv", index=False)
    val_df.to_csv(out_dir_path / "val.csv", index=False)

    # Save label mapping
    labels = sorted(train_df["label"].unique())
    (out_dir_path / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    print(f"Saved processed data to {out_dir_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Folder containing raw xlsx files")
    parser.add_argument("--out_dir", default="data/processed", help="Output folder")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.val_ratio)