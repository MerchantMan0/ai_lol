"""Shared helpers for TabFM Colab/local demos."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"
CHECKPOINT_GB = 6.1


def log(msg: str) -> None:
    print(msg, flush=True)


def setup_hf_download() -> str | None:
    """Enable HF progress bars and return token if set."""
    try:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()
        log("HF Hub progress bars: enabled")
    except Exception as exc:
        log(f"HF Hub progress bars: unavailable ({exc})")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        log("HF token: set (faster downloads, higher rate limits)")
    else:
        log("HF token: not set — add Colab secret HF_TOKEN to speed up the ~6 GB download")

    cache = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    log(f"HF cache root: {cache}")
    return token


def log_hf_cache_size(model_type: str) -> None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(hub.glob("models--google--tabfm*"))
    if not matches:
        log("HF cache: no TabFM files yet")
        return
    for path in matches:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        log(f"HF cache: {path.name} → {total / 1e9:.2f} GB on disk")
        weights = list(path.rglob(f"{model_type}/model.safetensors"))
        if weights:
            size = weights[0].stat().st_size
            log(f"  {model_type}/model.safetensors: {size / 1e9:.2f} GB")


def predownload_checkpoint(model_type: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    log(f"Downloading checkpoint (~{CHECKPOINT_GB} GB): {HF_REPO_ID}/{model_type}/")
    log("  this is the slow step — expect several minutes on first run")
    log_hf_cache_size(model_type)

    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{model_type}/**"],
        token=token,
    )
    elapsed = time.time() - t0
    checkpoint_dir = Path(local_dir) / model_type
    weights = checkpoint_dir / "model.safetensors"

    if weights.is_file():
        log(f"  download complete in {elapsed:.1f}s ({weights.stat().st_size / 1e9:.2f} GB)")
    else:
        log(f"  snapshot finished in {elapsed:.1f}s but weights file not found at {weights}")

    log_hf_cache_size(model_type)
    return checkpoint_dir


def load_tabfm_model(model_type: str, device: str):
    """Load TabFM from the HF cache (call predownload_checkpoint first)."""
    import torch
    from tabfm.src.pytorch.tabfm_v1_0_0 import HF_REPO_ID, TabFM_HF

    map_location = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    log(f"Loading weights onto {map_location} (bfloat16)...")
    log("  bypassing tabfm.load() — PyPI build reads weights with map_location='cpu'")
    log_torch_device()

    t0 = time.time()
    model = TabFM_HF.from_pretrained(
        HF_REPO_ID,
        subfolder=model_type,
        map_location=map_location,
    )
    model = model.to(torch.bfloat16)
    param_device = next(model.parameters()).device
    if map_location != "cpu" and param_device.type != map_location:
        log(f"  moving model from {param_device} to {map_location}")
        model = model.to(map_location)
    model.eval()

    log(f"  model ready in {time.time() - t0:.1f}s")
    log_torch_device(model)
    return model


def resolve_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_torch_device(model=None) -> None:
    try:
        import torch
    except ImportError:
        log("PyTorch: not installed yet")
        return

    log(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        log(f"GPU memory: {alloc_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")
    else:
        log("CUDA: not available (model will run on CPU — slower)")

    if model is not None:
        try:
            param = next(model.parameters())
            log(f"Model weights: device={param.device}, dtype={param.dtype}")
        except StopIteration:
            pass


def configure_logging() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
