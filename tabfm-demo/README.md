# TabFM PyTorch demo

Standalone demo of [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch) — Google's zero-shot tabular foundation model. Unrelated to the rest of this repo.

TabFM treats your training rows as **in-context examples**: `fit()` only preprocesses data (encoding, scaling). Weights are **not** fine-tuned. Predictions come from a single forward pass.

## Requirements

- Python **3.11+**
- ~2 GB disk for PyTorch + model weights (first run downloads from Hugging Face)

## Setup (run on a machine with enough disk/GPU optional)

```bash
cd tabfm-demo
python3.12 -m venv .venv
source .venv/bin/activate

# CPU-only (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tabfm pandas scikit-learn numpy

# Or full PyTorch + CUDA via extras:
# pip install -r requirements.txt
```

## Run

```bash
python demo_classification.py   # wine cultivar (3 classes)
python demo_regression.py       # California housing prices
```

First run downloads model weights and may take several minutes.

## What each script does

| Script | Task | Dataset | TabFM checkpoint |
|--------|------|---------|------------------|
| `demo_classification.py` | Multiclass | UCI Wine (sklearn) | `classification/` |
| `demo_regression.py` | Regression | California housing (sklearn, 400-row sample) | `regression/` |

## Notes

- Classification supports at most **10 classes** (model limit).
- Memory scales with training context size; regression demo caps training rows at 400.
- Weights are under the [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch).

## References

- [Hugging Face model card](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- [Google Research blog](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
- [Source code](https://github.com/google-research/tabfm)
