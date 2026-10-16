"""Test-like simulation of a full submission pipeline: members -> blend -> prompt clustering -> soft matching.

Training side exactly as at submission time: every row of the training folds; centered members are centered with
clusters of the training-fold texts. Test side: the held-out fold's rows are dropped with model-specific keep
rates (unequal test counts); the kept texts are clustered on their own and centered members are centered with
those clusters (their predictions depend on the test composition, so they are recomputed per simulation). Raw
members' OOF rows do not depend on the composition and are taken from outputs/preds. Matching then runs on the
pooled kept rows, as on the test. Nothing uses labels except fitting on training folds and scoring.

Usage: python pipeline_sim.py --raw svc=tfidf_char_svc_C0.3 --raw lp=<tag> --ctr emb4c=qwen3_4b_base:0.001 \
           --ctr emb8c=qwen3_8b_base:0.001 --weights svc=1,lp=1,emb4c=1,emb8c=1 --sims 3
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, encode_y, load_folds, load_member, load_train, log_experiment, metrics
from matching import cluster_prompts, drop_mask, match
from train_emb import center_by_clusters, load_feats, make_model


def blend(members, weights):
    z = sum(w * np.log(np.clip(members[k], 1e-9, 1)) for k, w in weights.items()) / sum(weights.values())
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", default=[], help="name=tag of a saved member (composition-free)")
    ap.add_argument("--ctr", action="append", default=[], help="name=feature:C, a centered embedding member")
    ap.add_argument("--weights", required=True, help="name=w,... (log-probability blend weights)")
    ap.add_argument("--sims", type=int, default=3)
    ap.add_argument("--keep", default="0.35,1.0")
    ap.add_argument("--lams", default="0,2,4,8")
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--log", default="", help="EXPERIMENTS.md tag prefix (empty: do not log)")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    texts = tr["text"].tolist()
    n_f = int(folds.max()) + 1
    weights = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.weights.split(",")}
    raw = {kv.split("=")[0]: load_member(kv.split("=")[1])[0] for kv in a.raw}
    ctr_specs = {kv.split("=")[0]: (kv.split("=")[1].split(":")[0], float(kv.split("=")[1].split(":")[1])) for kv in a.ctr}
    feats = {nm: load_feats("emb", [feat], "train")[0][0] for nm, (feat, _) in ctr_specs.items()}
    models = {}
    for nm, (feat, C) in ctr_specs.items():                # training side: independent of the simulation
        for k in range(n_f):
            trn = np.flatnonzero(folds != k)
            Xc, _ = center_by_clusters(feats[nm][trn], [texts[i] for i in trn], a.thr)
            models[nm, k] = make_model("emb", C).fit(Xc, y[trn])
        print(f"fitted centered member {nm} ({feat}, C={C}) on 5 training folds  ({time.time() - t0:.0f}s)", flush=True)
    lo, hi = map(float, a.keep.split(","))
    lams = [float(x) for x in a.lams.split(",")]
    rng = np.random.default_rng(0)
    rows = []
    for sim in range(a.sims):
        keep = np.flatnonzero(drop_mask(y, None, rng, lo, hi))
        fk = folds[keep]
        P = {nm: raw[nm][keep] for nm in raw}
        for nm in ctr_specs:
            Pn = np.zeros((len(keep), K))
            for k in range(n_f):
                loc = np.flatnonzero(fk == k)
                rws = keep[loc]
                Xc, _ = center_by_clusters(feats[nm][rws], [texts[i] for i in rws], a.thr)
                m = models[nm, k]
                Pn[np.ix_(loc, m.classes_)] = m.predict_proba(Xc)
            P[nm] = Pn
        Pb = blend(P, weights)
        groups = cluster_prompts([texts[i] for i in keep], a.thr)
        res = {"sim": sim, "n": len(keep),
               **{f"member_{nm}": metrics.balanced_accuracy(y[keep], P[nm]) for nm in P},
               "blend": metrics.balanced_accuracy(y[keep], Pb), "blend_ll": metrics.mlogloss(y[keep], Pb)}
        for lam in lams:
            res[f"match_{lam:g}"] = metrics.balanced_accuracy(y[keep], np.eye(K)[match(Pb, groups, lam)])
        rows.append(res)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    mean = df.drop(columns=["sim"]).mean()
    print("MEAN " + " ".join(f"{k}={v:.4f}" for k, v in mean.items()), flush=True)
    if a.log:
        best = max(lams, key=lambda l: mean[f"match_{l:g}"])
        s = dict(ba=float(mean[f"match_{best:g}"]), ba_fold_mean=float(mean[f"match_{best:g}"]),
                 ba_fold_std=float(df[f"match_{best:g}"].std(ddof=0)), logloss=float("nan"))
        log_experiment(a.log, f"pipeline sim (keep {lo}-{hi}, {a.sims} sims): raw {list(raw)} + centered "
                       f"{list(ctr_specs)}, weights {weights}, clustering thr {a.thr} + matching", s, time.time() - t0,
                       note=f"argmax blend BA {mean['blend']:.4f}; matched BA by lambda "
                            + ", ".join(f"{l:g}:{mean[f'match_{l:g}']:.4f}" for l in lams)
                            + f" (best lambda {best:g} picked on these same sims: optimistic by selection)")


if __name__ == "__main__":
    main()
