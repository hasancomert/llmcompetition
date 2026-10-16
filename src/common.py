"""Shared paths, class order, data / fold loading, prediction saving and experiment logging.

Prediction files follow the reuse/ev/cv_preds layout (PLAN.md, conflict Ç2):
outputs/preds/oof_<tag>.npy, pred_<tag>.npy, meta_<tag>.json (class order stored and checked on load).
Metrics come from reuse/airline/metrics.py (tested equal to sklearn).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PREDS = OUT / "preds"
FOLDS_CSV = OUT / "folds.csv"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"

for p in (ROOT / "reuse" / "airline", ROOT / "reuse"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ev import cv_preds  # noqa: E402  (reuse/ev, package import)
import metrics  # noqa: E402  (reuse/airline/metrics.py)

# The 18 labels exactly as spelled by the competition; column k of every prediction matrix is CLASSES[k].
CLASSES = sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
])
K = len(CLASSES)
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

N_FOLDS = 5
FOLD_SEED = 42


def lab_of(model: str) -> str:
    for lab in ("gpt", "gemini", "claude", "qwen", "glm", "kimi"):
        if model.startswith(lab):
            return lab
    raise ValueError(model)


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv")
    assert set(df["model"]) == set(CLASSES), sorted(set(df["model"]) ^ set(CLASSES))
    assert df["id"].is_unique
    return df


def load_test(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA / "test.csv")


def encode_y(models) -> np.ndarray:
    return np.array([CLASS_INDEX[m] for m in models], dtype=np.int64)


def load_folds(train: pd.DataFrame | None = None) -> np.ndarray:
    """Fold number per train row, in train.csv row order (checked by id)."""
    train = load_train() if train is None else train
    f = pd.read_csv(FOLDS_CSV)
    assert (f["id"].values == train["id"].values).all(), "folds.csv is not in train.csv row order"
    return f["fold"].to_numpy(np.int64)


def iter_folds(folds: np.ndarray):
    return cv_preds.iter_folds(folds)


def scores(y, P, folds) -> dict:
    """Pooled OOF balanced accuracy / log loss plus fold mean ± std of balanced accuracy."""
    per = [metrics.balanced_accuracy(y[va], P[va]) for _, _, va in iter_folds(folds)]
    return dict(ba=metrics.balanced_accuracy(y, P), logloss=metrics.mlogloss(y, P),
                ba_fold_mean=float(np.mean(per)), ba_fold_std=float(np.std(per)), ba_folds=per)


def src_sha() -> str:
    """Hash of every src/*.py file: which code version produced a saved member."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        h.update(f.name.encode() + b"\x00" + f.read_bytes())
    return h.hexdigest()[:12]


def save_member(tag: str, oof, test, **meta) -> None:
    """Refuses to overwrite an existing tag (set NLLM_FORCE=1 to allow): a silent overwrite would make an
    earlier EXPERIMENTS.md row and every blend that lists the tag point at different numbers."""
    PREDS.mkdir(parents=True, exist_ok=True)
    if (PREDS / f"meta_{tag}.json").exists() and os.environ.get("NLLM_FORCE") != "1":
        raise FileExistsError(f"member {tag!r} exists; use a new tag or NLLM_FORCE=1")
    cv_preds.save_preds(tag, cv_preds.check_proba(oof, K), cv_preds.check_proba(test, K), CLASSES,
                        outdir=str(PREDS), src_sha=src_sha(), **meta)


def load_member(spec: str):
    oof, test, cls = cv_preds.load_preds(spec, outdir=str(PREDS), classes=CLASSES)
    return oof, test


def next_experiment_no() -> int:
    if not EXPERIMENTS_MD.exists():
        return 1
    nums = []
    for line in EXPERIMENTS_MD.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].isdigit():
            nums.append(int(cells[0]))
    return max(nums, default=0) + 1


def log_experiment(tag: str, change: str, s: dict, seconds: float | None = None, note: str = "",
                   lb: float | None = None) -> int:
    """Append one row to EXPERIMENTS.md: | no | tag | change | CV BA | fold BA mean ± std | CV log loss | public LB | time | note |"""
    no = next_experiment_no()
    t = f"{seconds:.0f}s" if seconds is not None else ""
    note = f"{note}; src {src_sha()}" if note else f"src {src_sha()}"
    row = (f"| {no} | {tag} | {change} | {s['ba']:.4f} | {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f} | "
           f"{s['logloss']:.4f} | {'' if lb is None else f'{lb:.3f}'} | {t} | {note} |\n")
    with open(EXPERIMENTS_MD, "a", encoding="utf-8") as f:
        f.write(row)
    return no


def run_member(tag: str, change: str, oof, test, y, folds, t0: float, note: str = "", log: bool = True,
               **meta) -> dict:
    """Score, save the (oof, test) pair and log the experiment row."""
    s = scores(y, oof, folds)
    save_member(tag, oof, test, ba=s["ba"], logloss=s["logloss"], change=change, **meta)
    secs = time.time() - t0
    if log:
        s["no"] = log_experiment(tag, change, s, secs, note)
    print(f"[{tag}] BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
          f"log loss {s['logloss']:.4f}  {secs:.0f}s", flush=True)
    return s


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
