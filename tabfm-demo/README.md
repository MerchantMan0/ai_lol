# TabFM LoL demo

Zero-shot match outcome prediction with [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch) on features built from the ailol Postgres database.

Each row is **one hero in one ranked match**. Target: `WIN` or `LOSS`.

## Features (24 numeric columns)

| # | Column(s) | Description |
|---|-----------|-------------|
| 1 | `overall_win_rate` | Hero's overall win rate (training-set stats) |
| 2–15 | `wr_duration_*` | Win rate in each of 14 game-duration buckets |
| 16–20 | `adv_vs_enemy_1..5` | Win rate vs that enemy minus overall win rate |
| 21–24 | `syn_with_ally_1..4` | Win rate with that ally minus overall win rate |

Stats are computed from **training matches only**; test matches are held out by `match_id`.

## Requirements

- Python **3.11+**
- GPU recommended (~13 GB VRAM for float32 weights on a T4)
- Populated Postgres DB (see `../db details/`) or a pre-exported CSV

## Setup

```bash
cd tabfm-demo
python3 -m venv .venv
source .venv/bin/activate
pip install 'psycopg[binary]' python-dotenv pandas
# For TabFM demo (GPU machine / Colab):
pip install -r requirements.txt
```

If you see `ImportError: no pq wrapper available`, you installed plain `psycopg` without libpq. Use **`psycopg[binary]`** (above), or on Fedora/Bazzite: `sudo dnf install libpq`.

Postgres credentials are read from `../db details/.env` automatically.

## 1. Export data from the database

```bash
python extract_lol_features.py
# → data/lol_participants.csv
```

Optional: `-o /path/to/out.csv` or set `LOL_PARTICIPANTS_CSV` for the demo to read.

## 2. Run TabFM demo

```bash
python demo_lol_classification.py
```

First run downloads ~6 GB classification weights from Hugging Face.

The demo loads `data/lol_participants.csv` if present; otherwise it queries Postgres directly.

## Colab

Use `demo.ipynb` at the repo root. Set Colab secrets: `HF_TOKEN` (optional), `POSTGRES_*` (or upload `data/lol_participants.csv`).

## Notes

- TabFM `fit()` stores training rows as context; training rows are capped at 800 by default.
- Weights: [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch).
- PyPI `tabfm` expects `pytorch_model.bin` but HF hosts `model.safetensors`; `tabfm_demo_common.py` loads safetensors directly.
