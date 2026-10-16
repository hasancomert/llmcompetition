"""Which hidden layers carry the signal? Centered LogReg on single layers / layer subsets of one LM's pooled states.

The gpu_embed matrix is the concatenation of equal-width blocks, one per layer in meta["layers"]. Every subset is
centered with the same all-train clusters as train_emb.py --center and scored with the shared folds (not saved,
not a member; one EXPERIMENTS.md row per subset with --log).
Usage: python layer_probe.py --name qwen3_8b_base --subsets 0 1 2 3 1+2 --C 0.001
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import encode_y, load_folds, load_train, log_experiment, scores
from train_emb import EMB, center_by_clusters, fold_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--subsets", nargs="+", required=True, help="block indices joined by +, e.g. 0 1+2")
    ap.add_argument("--C", type=float, default=0.001)
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    meta = json.loads((EMB / f"meta_{a.name}_train.json").read_text())
    X = np.load(EMB / f"emb_{a.name}_train.npy").astype(np.float32)
    nb = len(meta["layers"])
    w = X.shape[1] // nb
    Xc, _ = center_by_clusters(X, tr["text"].tolist(), a.thr)
    for sub in a.subsets:
        t0 = time.time()
        blocks = [int(b) for b in sub.split("+")]
        cols = np.concatenate([np.arange(b * w, (b + 1) * w) for b in blocks])
        oof, _ = fold_fit("emb", a.C, Xc[:, cols], Xc[:1, cols], y, folds)
        s = scores(y, oof, folds)
        layers = [meta["layers"][b] for b in blocks]
        print(f"{a.name} layers {layers}: BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
              f"log loss {s['logloss']:.4f}  {time.time() - t0:.0f}s", flush=True)
        if a.log:
            log_experiment(f"layers_{a.name}_{'+'.join(map(str, layers))}_ctr_C{a.C:g}",
                           f"layer probe: centered pooled states of {a.name} layers {layers} only ({len(cols)} cols), "
                           f"LogReg C={a.C:g}", s, time.time() - t0, note="probe, not saved as a member")


if __name__ == "__main__":
    main()
