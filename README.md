# Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety

FAST-NUCES Responsible & Explainable AI – Assignment 2.

End-to-end audit and mitigation of a DistilBERT toxicity classifier trained on the
[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
dataset.

## Repository layout

| File | Purpose |
|------|---------|
| `part1.ipynb` | Baseline DistilBERT fine-tune + threshold analysis |
| `part2.ipynb` | Bias audit (high-black vs reference cohort) |
| `part3.ipynb` | Adversarial attacks (evasion + poisoning) |
| `part4.ipynb` | Mitigation (Reweighing, ThresholdOptimizer, Oversampling) + Pareto |
| `part5.ipynb` | Guardrail pipeline demo on 1k samples |
| `pipeline.py` | `ModerationPipeline` class (regex → calibrated model → review queue) |
| `requirements.txt` | Pinned dependencies |

## Environment

- **Python**: 3.10 or 3.11 (Colab default works)
- **GPU**: NVIDIA T4 (Google Colab free tier) used for all training. CPU works but ~20× slower.
- **Disk**: ~1 GB for `data/jigsaw-unintended-bias-train.csv`

## Reproducing

### Local (CPU/GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place the Kaggle CSVs here:
#   data/jigsaw-unintended-bias-train.csv
#   data/validation.csv

jupyter notebook
```

Run notebooks in order: `part1` → `part2` → `part3` → `part4` → `part5`.
Each notebook is **self-contained** (re-loads the data with the same seed, re-loads the
saved baseline checkpoint where needed) but they share artefacts under `checkpoints/`:

| Notebook | Reads | Writes |
|---|---|---|
| `part1.ipynb` | `data/jigsaw-unintended-bias-train.csv` | `checkpoints/baseline/` |
| `part2.ipynb` | `checkpoints/baseline/` | — |
| `part3.ipynb` | `checkpoints/baseline/` | `checkpoints/poisoned/` |
| `part4.ipynb` | `checkpoints/baseline/` | `checkpoints/best_mitigated_*/` |
| `part5.ipynb` | `checkpoints/best_mitigated_*/` (falls back to `baseline/`) | `checkpoints/baseline/isotonic.pkl` |

`pipeline.py` is the standalone module exposed by Part 5 and can be used after
running the notebooks: `python pipeline.py "you should kill yourself"`.

### Google Colab

1. Upload the two CSVs to Colab (or mount Google Drive containing them).
2. Adjust `DATA_DIR` in the first cell of each notebook if needed.
3. Runtime → Change runtime type → **GPU (T4)**.
4. Run all cells.

## Data

The two required files are **not** committed (per assignment rules and GitHub size limits):

- `data/jigsaw-unintended-bias-train.csv` (~860 MB)
- `data/validation.csv` (~3 MB)

Download from Kaggle after accepting the competition terms.

## Notes on submission

- Notebooks must be re-executed on GPU before final submission so output cells are populated
  (the scaffolded notebooks here contain the complete code; outputs will appear after a full
  run on Colab).
- Commit history is preserved per part to demonstrate incremental development.
