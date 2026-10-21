"""Test-time shared-prompt Gaussian group evidence for a submission (replay step), as simulated in adaptive_sim.py
--gauss-views (EXPERIMENTS gauss_sim_*: twins + Gaussian w 0.25 at T 3 + Hungarian m 3 beat the twins alone in every
simulated prompt structure).

Per view and per fold, a GaussHead (gauss_head.py) is fitted on the fold's training rows with their true prompt_ids.
The test texts are clustered by topic exactly as for the twins (matching.cluster_prompts, cut --thr); every cluster of
min_size..max_size texts is decoded jointly by each fold's head at temperature T (distinct labels), the probabilities
are averaged over the five folds, and the evidence written is the sum over views of their logs (0 for texts outside
such clusters). replay.py adds weight * evidence, like the twins.

Output: outputs/twins/<tag>.npz with evidence (n_test, K), adapt (n_test,) bool, clusters (n_test,), meta json.
Usage (replay): python gauss_twins.py --views "qwen3_8b_base_l12;qwen3_4b_base_l12" --T 3 --thr 0.85 --tag gauss_l12
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import K, OUT, encode_y, load_folds, load_test, load_train
from emb_primal import raw_blocks
from gauss_head import GaussHead
from matching import cluster_prompts

TW = OUT / "twins"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="qwen3_8b_base_l12;qwen3_4b_base_l12", help="';'-separated block specs")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=18)
    ap.add_argument("--tag", default="gauss_l12")
    a = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    prompts = tr["prompt_id"].to_numpy()
    n_te, n_f = len(te), int(folds.max()) + 1
    g = cluster_prompts(te["text"].tolist(), a.thr)        # < 40 texts (the public 3-row test): all singletons
    sizes = np.bincount(g)[g]
    adapt = (sizes >= a.min_size) & (sizes <= a.max_size)
    rows = np.flatnonzero(adapt)
    ev = np.zeros((n_te, K))
    print(f"[gauss] test {n_te} texts: {len(np.unique(g))} clusters, {len(rows)} texts in clusters of "
          f"{a.min_size}..{a.max_size} (cut {a.thr})", flush=True)
    if len(rows):
        for v in [v for v in a.views.split(";") if v]:
            Xtr, Xte, _ = raw_blocks(v, tr, te)
            P = np.zeros((len(rows), K))
            for k in range(n_f):
                trn = np.flatnonzero(folds != k)
                h = GaussHead(d=a.d).fit(Xtr[trn], y[trn], prompts[trn], K)
                P += np.exp(h.scores(h.transform(Xte[rows]), g[rows], a.T)) / n_f
            ev[rows] += np.log(np.clip(P, 1e-9, 1))
            del Xtr, Xte
            print(f"[gauss] view {v}: done ({time.time() - t0:.0f}s)", flush=True)
    TW.mkdir(parents=True, exist_ok=True)
    meta = {"views": a.views, "d": a.d, "T": a.T, "thr": a.thr, "min_size": a.min_size, "max_size": a.max_size,
            "n_test": n_te, "n_adapted": int(len(rows)), "n_clusters": int(len(np.unique(g)))}
    np.savez(TW / f"{a.tag}.npz", evidence=ev, adapt=adapt, clusters=g, meta=json.dumps(meta))
    print(f"[gauss] wrote {TW / (a.tag + '.npz')}: {meta}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
