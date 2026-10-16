"""End-to-end check of src/replay.py on a held-out fold, with labels (CPU, precomputed LM features).

Fake competition: train = rows of folds 1-4 (folds renumbered 0-3), test = fold 0 rows with only id/text. LM features
are sliced from outputs/emb (with meta json row ids, as gpu_embed writes them). The replay runs from a fresh copy of
src/ and reuse/ under --work (so nothing touches the project's outputs/), writes submission.parquet, and this script
scores it against the true labels. Exercises the large-test code paths (e.g. the Gram cache with 539 test rows).

Usage: python tests/e2e_replay.py --work /tmp/scratch/e2e_replay --recipe v4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES, load_folds, load_train, metrics  # noqa: E402

LM = ["qwen3_0.6b_base", "qwen3_1.7b_base", "qwen3_4b_base", "qwen3_8b_base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--recipe", default="v4")
    a = ap.parse_args()
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    (work / "comp").mkdir(parents=True)
    (work / "lm").mkdir()
    shutil.copytree(ROOT / "src", work / "code" / "src")
    shutil.copytree(ROOT / "reuse", work / "code" / "reuse")
    tr = load_train()
    folds = load_folds(tr)
    tr_rows, te_rows = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    ftr = tr.iloc[tr_rows].reset_index(drop=True)
    fte = tr.iloc[te_rows][["id", "text"]].reset_index(drop=True)
    ftr.to_csv(work / "comp" / "train.csv", index=False)
    fte.to_csv(work / "comp" / "test.csv", index=False)
    pd.DataFrame({"id": ftr["id"], "fold": folds[tr_rows] - 1}).to_csv(work / "folds.csv", index=False)
    emb = ROOT / "outputs" / "emb"
    for name in LM:
        meta = json.loads((emb / f"meta_{name}_train.json").read_text())
        for split, rows, df in (("train", tr_rows, ftr), ("test", te_rows, fte)):
            for kind in ("emb", "lp"):
                np.save(work / "lm" / f"{kind}_{name}_{split}.npy", np.load(emb / f"{kind}_{name}_train.npy")[rows])
            m = dict(meta, ids=pd.read_csv(work / "comp" / f"{split}.csv")["id"].tolist())
            (work / "lm" / f"meta_{name}_{split}.json").write_text(json.dumps(m))
    out = work / "submission.parquet"
    cmd = [sys.executable, str(work / "code" / "src" / "replay.py"), "--recipe", a.recipe, "--data-dir",
           str(work / "comp"), "--folds", str(work / "folds.csv"), "--tokenizers-flat",
           str(ROOT / "outputs" / "kaggle_build" / "ds_tokenizers"), "--out", str(out), "--precomputed-lm",
           str(work / "lm"), "--no-fallback"]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    sub = pd.read_parquet(out)
    truth = tr.iloc[te_rows].set_index("id")["model"]
    pred = sub.assign(id=sub["id"].astype(str)).set_index("id")["model"].reindex(truth.index.astype(str))
    yt, yp = np.searchsorted(CLASSES, truth.values), np.searchsorted(CLASSES, pred.values)
    print(f"E2E replay {a.recipe}, fold-0 test ({len(te_rows)} rows, trained on folds 1-4): balanced accuracy "
          f"{metrics.balanced_accuracy(yt, np.eye(len(CLASSES))[yp]):.4f}", flush=True)


if __name__ == "__main__":
    main()
