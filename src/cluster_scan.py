"""Clustering quality for the adaptive twins: pair precision / recall of topic clusters on held-out-like pools.

Pools as adaptive_sim.py (every prompt keeps a random a..b texts; 150 prompts pooled). Distances: content-word TF-IDF
cosine (matching.topic_matrix, the current clustering) mixed with the cosine distance of one frozen-LM layer
(mean-centered over the pool, L2-normalized): D = (1 - alpha) * D_tfidf + alpha * D_emb, average linkage, cut grid.
Reported per setting: pair precision / recall (same cluster vs same prompt) and the share of texts in clusters of 2..18.
No labels are used by the clustering itself; prompt ids only score it.

Usage: python cluster_scan.py --ranges 1-5,2-4 --sims 3 --layers qwen3_8b_base_l12@0,...
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from adaptive_sim import pair_stats
from common import encode_y, load_train
from matching import topic_matrix
from structure_sim import keep_rows


def emb_layer(name, li):
    from train_emb import load_feats
    X, meta = load_feats("emb", [name], "train")[0]
    w = meta["dim"] // len(meta["layers"])
    return X[:, li * w:(li + 1) * w].astype(np.float64), meta["layers"][li]


def cos_dist(E):
    Ec = E - E.mean(0)
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-12
    return 1.0 - Ec @ Ec.T


def cut_scores(D, prompts, cuts, lo=2, hi=18):
    D = np.clip((D + D.T) / 2, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    out = []
    for c in cuts:
        g = np.unique(fcluster(Z, c, criterion="distance"), return_inverse=True)[1]
        sizes = np.bincount(g)[g]
        pp, pr = pair_stats(g, prompts)
        out.append((c, pp, pr, float(((sizes >= lo) & (sizes <= hi)).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", default="1-5,2-4")
    ap.add_argument("--sims", type=int, default=3)
    ap.add_argument("--layers", default="qwen3_8b_base_l12@0,qwen3_8b_base_l12@1,qwen3_8b_base_l12@3,"
                                        "qwen3_8b_base_l12@5,qwen3_8b_base_l12@7,qwen3_8b_base_l12@11,"
                                        "qwen3_4b_base_l12@1,qwen3_4b_base_l12@5")
    ap.add_argument("--alphas", default="0.25,0.5")
    ap.add_argument("--cuts", default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, prompts, texts = encode_y(tr["model"]), tr["prompt_id"].to_numpy(), tr["text"].tolist()
    embs = {}
    for spec in a.layers.split(","):
        nm, li = spec.split("@")
        embs[spec] = emb_layer(nm, int(li))
    cuts = [float(c) for c in a.cuts.split(",")]
    rng = np.random.default_rng(a.seed)
    rows = []
    for rg in a.ranges.split(","):
        lo, hi = (int(x) for x in rg.split("-"))
        for sim in range(a.sims):
            keep = keep_rows("perprompt", y, prompts, rng, argparse.Namespace(a=lo, b=hi))
            X = topic_matrix([texts[i] for i in keep], 40)
            Dt = 1.0 - (X @ X.T).toarray()
            for c, pp, pr, sh in cut_scores(Dt.copy(), prompts[keep], cuts):
                rows.append(dict(range=rg, sim=sim, view="tfidf", alpha=0.0, cut=c, prec=pp, rec=pr, share=sh))
            for spec, (E, layer) in embs.items():
                De = cos_dist(E[keep])
                for al in (float(x) for x in a.alphas.split(",")):
                    for c, pp, pr, sh in cut_scores((1 - al) * Dt + al * De, prompts[keep], cuts):
                        rows.append(dict(range=rg, sim=sim, view=f"{spec}(L{layer})", alpha=al, cut=c, prec=pp,
                                         rec=pr, share=sh))
        print(f"range {rg}: {a.sims} sims ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).groupby(["range", "view", "alpha", "cut"])[["prec", "rec", "share"]].mean().reset_index()
    pd.set_option("display.width", 200)
    for rg, d in df.groupby("range"):
        print(f"== {rg}: best recall at precision >= 0.97 / >= 0.95, per view and alpha")
        for thr in (0.97, 0.95):
            best = d[d["prec"] >= thr].sort_values("rec", ascending=False).groupby(["view", "alpha"]).head(1)
            print(f"-- precision >= {thr}")
            print(best.sort_values("rec", ascending=False).head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
