"""Slice main.ipynb into 5 part notebooks per the assignment PDF.

Run from repo root:
    python scripts/split_notebook.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "main.ipynb"

# Cell index ranges from main.ipynb (inclusive on both ends).
# Verify with:
#   python -c "import json; nb=json.load(open('main.ipynb')); [print(i, c['source'][0][:60]) for i,c in enumerate(nb['cells'])]"
PART_RANGES = {
    "part1": (3, 9),    # tokenize/train/eval/plots/threshold
    "part2": (10, 15),  # cohorts/metrics/CMs/markdown
    "part3": (16, 22),  # perturb/evasion/poison/FNR/markdown
    "part4": (23, 34),  # 3 mitigations + pareto + base-rate + save
    "part5": (35, 43),  # regex/calibrator/pipeline/analysis/sensitivity
}

PART_TITLES = {
    "part1": "Part 1 — Baseline DistilBERT Classifier",
    "part2": "Part 2 — Bias Audit",
    "part3": "Part 3 — Adversarial Attacks",
    "part4": "Part 4 — Bias Mitigation",
    "part5": "Part 5 — Guardrail Moderation Pipeline",
}

# What each part needs prepended after the data loader.
# Keys map to extra setup blocks.
PART_NEEDS = {
    "part1": [],                                 # trains baseline from scratch
    "part2": ["load_baseline", "score_eval", "helpers"],
    "part3": ["load_baseline", "score_eval", "helpers"],
    "part4": ["load_baseline", "score_eval", "helpers"],
    "part5": ["load_baseline", "score_eval", "helpers"],
}

NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


# Reusable setup blocks ------------------------------------------------------
PIP_CELL = code(
    "!pip install -q transformers torch scikit-learn fairlearn aif360 pandas matplotlib seaborn"
)

DATA_CELL = code('''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

USE_COLS = ["comment_text", "toxic", "black", "white", "muslim", "jewish"]
IDENTITY_COLS = ["black", "white", "muslim", "jewish"]

raw = pd.read_csv("data/jigsaw-unintended-bias-train.csv", usecols=USE_COLS)
raw["label"] = (raw["toxic"] >= 0.5).astype(int)

sample = raw.sample(n=120_000, random_state=SEED)
train_df, eval_df = train_test_split(
    sample, test_size=20_000, stratify=sample["label"], random_state=SEED,
)
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
print("train:", train_df.shape, "eval:", eval_df.shape)
''')

LOAD_BASELINE_CELL = code('''import os, glob
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128

# Prefer best mitigated checkpoint if present (Part 5 scenario), else baseline,
# else fall back to the pretrained backbone.
_candidates = (sorted(glob.glob("checkpoints/best_mitigated_*"))
               + ["checkpoints/baseline"])
CKPT_DIR = next((c for c in _candidates if os.path.isdir(c)), MODEL_NAME)
print("loading model from", CKPT_DIR)

tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR, num_labels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class ToxicDataset(Dataset):
    def __init__(self, df):
        enc = tokenizer(df["comment_text"].astype(str).tolist(),
                        truncation=True, padding="max_length", max_length=MAX_LEN)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = df["label"].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i]),
            "attention_mask": torch.tensor(self.attention_mask[i]),
            "labels": torch.tensor(self.labels[i]),
        }

train_ds = ToxicDataset(train_df)
eval_ds = ToxicDataset(eval_df)
''')

SCORE_EVAL_CELL = code('''# Score the eval set so downstream cells can reuse predictions.
from transformers import Trainer, TrainingArguments

_args = TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=64,
                          report_to="none")
_trainer = Trainer(model=model, args=_args)
_logits = _trainer.predict(eval_ds).predictions
eval_probs = torch.softmax(torch.tensor(_logits), dim=-1)[:, 1].numpy()
eval_preds = (eval_probs >= 0.5).astype(int)
eval_df = eval_df.assign(prob=eval_probs, pred=eval_preds)
trainer = _trainer  # downstream cells reference `trainer` and `metrics`
metrics = _trainer.evaluate(eval_ds)
print({k: round(v, 4) for k, v in metrics.items() if k.startswith("eval_")})
''')

HELPERS_CELL = code('''# Shared helpers reused across multiple parts (defined once here so each
# notebook is fully standalone).
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_score, recall_score)

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "auc_roc": roc_auc_score(labels, probs),
    }

def per_cohort_metrics(df):
    y, p = df["label"].values, df["pred"].values
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": len(df),
        "TPR": tp / (tp + fn) if (tp + fn) else 0.0,
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (tp + fn) if (tp + fn) else 0.0,
        "Precision": precision_score(y, p, zero_division=0),
    }
''')

SETUP_BLOCKS = {
    "load_baseline": LOAD_BASELINE_CELL,
    "score_eval": SCORE_EVAL_CELL,
    "helpers": HELPERS_CELL,
}


def build_notebook(part_name: str, src_cells: list[dict]) -> dict:
    cells = [
        md(f"# {PART_TITLES[part_name]}\n\nSelf-contained notebook. Re-runs the data load "
           "and (where needed) reloads the saved baseline checkpoint so it can execute "
           "standalone on Colab."),
        PIP_CELL,
        DATA_CELL,
    ]
    for setup in PART_NEEDS[part_name]:
        cells.append(copy.deepcopy(SETUP_BLOCKS[setup]))
    # Strip the merged-notebook section header from the first source cell if it's a markdown
    # header — each part already has its own title above.
    body = list(copy.deepcopy(c) for c in src_cells)
    if body and body[0]["cell_type"] == "markdown":
        first_src = "".join(body[0]["source"]).lstrip()
        if first_src.startswith("## Section"):
            body = body[1:]
    cells.extend(body)
    return {"cells": cells, "metadata": NB_META, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    src = json.loads(SRC.read_text())
    src_cells = src["cells"]
    for part_name, (lo, hi) in PART_RANGES.items():
        nb = build_notebook(part_name, src_cells[lo:hi + 1])
        out = REPO / f"{part_name}.ipynb"
        out.write_text(json.dumps(nb, indent=1))
        print(f"wrote {out.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
