"""End-to-end check of src/submission_pipeline_gpu.py on a held-out fold, with labels (CPU, precomputed LM features).

Fake competition: train = rows of folds 1-4 (folds renumbered 0-3), test = fold 0 rows with only id/text (optionally
with model-specific drops, as in the simulations). LM features are sliced from outputs/emb (the same npy files
the CV members used). The pipeline writes submission.parquet; this script scores it against the true labels.
Everything is written under --work (scratch), nothing under the repo.

Usage: python tests/e2e_gpu_pipeline.py --work /tmp/scratch/e2e [--keep 0.35,1.0] [--seed 0]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES, encode_y, load_folds, load_train, metrics  # noqa: E402
from matching import drop_mask  # noqa: E402
from submission_pipeline_gpu import LM_MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", default="", help="lo,hi model keep rates for the fake test (empty: no drops)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra pipeline args")
    a = ap.parse_args()
    work = Path(a.work)
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "lm").mkdir(exist_ok=True)
    tr = load_train()
    folds = load_folds(tr)
    y = encode_y(tr["model"])
    tr_rows = np.flatnonzero(folds != 0)
    te_rows = np.flatnonzero(folds == 0)
    if a.keep:
        lo, hi = map(float, a.keep.split(","))
        kept = drop_mask(y[te_rows], None, np.random.default_rng(a.seed), lo, hi)
        te_rows = te_rows[kept]
    tr.iloc[tr_rows].to_csv(work / "data" / "train.csv", index=False)
    tr.iloc[te_rows][["id", "text"]].to_csv(work / "data" / "test.csv", index=False)
    pd.DataFrame({"id": tr["id"].values[tr_rows], "fold": folds[tr_rows] - 1}).to_csv(work / "folds.csv", index=False)
    emb = ROOT / "outputs" / "emb"
    for name, _ in LM_MODELS:
        for kind in ("emb", "lp"):
            X = np.load(emb / f"{kind}_{name}_train.npy")
            np.save(work / "lm" / f"{kind}_{name}_train.npy", X[tr_rows])
            np.save(work / "lm" / f"{kind}_{name}_test.npy", X[te_rows])
    out = work / "submission.parquet"
    cmd = [sys.executable, str(ROOT / "src" / "submission_pipeline_gpu.py"), "--data-dir", str(work / "data"),
           "--folds", str(work / "folds.csv"), "--tokenizers-flat",
           str(ROOT / "outputs" / "kaggle_build" / "ds_tokenizers"), "--out", str(out),
           "--precomputed-lm", str(work / "lm"), *a.extra]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    sub = pd.read_parquet(out)
    truth = tr.iloc[te_rows].set_index("id")["model"]
    pred = sub.set_index("id")["model"].reindex(truth.index)
    yt = np.searchsorted(CLASSES, truth.values)
    yp = np.searchsorted(CLASSES, pred.values)
    print(f"E2E fold-0 test ({len(te_rows)} rows{', keep ' + a.keep if a.keep else ''}): "
          f"balanced accuracy {metrics.balanced_accuracy(yt, np.eye(len(CLASSES))[yp]):.4f}", flush=True)


if __name__ == "__main__":
    main()
