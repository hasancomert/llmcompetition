"""Test-time adaptive prompt-centering for a submission (replay step): centered-twin evidence for clustered test texts.

Same model as adaptive_sim.py (validated there on held-out prompts): per frozen-LM view and per fold, a LogReg head on
the fold's training rows centered by their TRUE prompt_id (exact, in Gram space). At test time the test texts are
clustered by topic (matching.cluster_prompts, no labels); texts in clusters of size min_size..max_size get each view's
twin on features centered by their cluster. The twin probability of a text is the mean over the five fold models
(as for every member's test prediction); the evidence written is the sum over views of its log-probability (0 for
texts without a twin). replay.py adds weight * evidence to the blend's test log-probabilities.

Output: outputs/twins/<tag>.npz with evidence (n_test, K), adapt (n_test,) bool, clusters (n_test,), meta json.
--primal / --ext pub (extdata flags): primal twins (adaptive_sim.TwinPrimal), the external passages with LM features
joining every fold's training set centered within their own prompt blocks (as simulated in adaptive_sim.py).
Usage (replay): python adaptive_twins.py --views qwen3_8b_base_l12,qwen3_4b_base_l12 --C 0.001 --thr 0.85 --tag twins_l12
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import extdata
from adaptive_sim import Twin, primal_twins
from common import K, OUT, encode_y, load_folds, load_test, load_train
from emb_kernel import combined_gram
from emb_raw_explore import parse_blocks
from matching import cluster_prompts

TW = OUT / "twins"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="qwen3_8b_base_l12,qwen3_4b_base_l12")
    ap.add_argument("--C", type=float, default=0.001)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=18)
    ap.add_argument("--tag", default="twins_l12")
    ap.add_argument("--primal", action="store_true", help="primal twins (always with --ext)")
    ap.add_argument("--ext-scale", default="official", choices=["official", "own"])
    extdata.add_args(ap)
    a = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    prompts = tr["prompt_id"].to_numpy()
    n_tr, n_te, n_f = len(tr), len(te), int(folds.max()) + 1
    g = cluster_prompts(te["text"].tolist(), a.thr)        # < 40 texts (the public 3-row test): all singletons
    sizes = np.bincount(g)[g]
    adapt = (sizes >= a.min_size) & (sizes <= a.max_size)
    ev = np.zeros((n_te, K))
    rows = np.flatnonzero(adapt)
    print(f"[twins] test {n_te} texts: {len(np.unique(g))} clusters, {len(rows)} texts in clusters of "
          f"{a.min_size}..{a.max_size} (cut {a.thr})", flush=True)
    ext = extdata.load(a, tr)
    if len(rows):
        for v in a.views.split(","):
            P = np.zeros((len(rows), K))
            if a.primal or ext is not None:
                for k, tw in primal_twins(v, tr, te, y, folds, prompts, [a.C], ext, a.ext_scale).items():
                    P += tw.predict(n_tr + rows, g[rows], a.C) / n_f
            else:
                for k in range(n_f):
                    Kfull = combined_gram(parse_blocks(v), folds, k, n_te)
                    trn = np.flatnonzero(folds != k)
                    tw = Twin(Kfull, trn, y[trn], prompts[trn], [a.C])
                    P += tw.predict(n_tr + rows, g[rows], a.C) / n_f
                    del Kfull, tw
            ev[rows] += np.log(np.clip(P, 1e-9, 1))
            print(f"[twins] view {v}: done ({time.time() - t0:.0f}s)", flush=True)
    TW.mkdir(parents=True, exist_ok=True)
    meta = {"views": a.views, "C": a.C, "thr": a.thr, "min_size": a.min_size, "max_size": a.max_size,
            "primal": bool(a.primal or ext is not None), "ext": a.ext,
            "n_test": n_te, "n_adapted": int(len(rows)), "n_clusters": int(len(np.unique(g)))}
    np.savez(TW / f"{a.tag}.npz", evidence=ev, adapt=adapt, clusters=g, meta=json.dumps(meta))
    print(f"[twins] wrote {TW / (a.tag + '.npz')}: {meta}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
