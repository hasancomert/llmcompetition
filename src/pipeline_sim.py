"""Test-like simulation of a full submission pipeline: members -> blend -> prompt clustering -> soft matching.

Training side exactly as at submission time: every row of the training folds; centered members are centered with
clusters of the training-fold texts. Test side: the held-out fold's rows are dropped with model-specific keep
rates (unequal test counts); the kept texts are clustered on their own and centered members are centered with
those clusters (their predictions depend on the test composition, so they are recomputed per simulation). Raw
members' OOF rows do not depend on the composition and are taken from outputs/preds. Matching then runs on the
pooled kept rows, as on the test. Nothing uses labels except fitting on training folds and scoring.

Usage: python pipeline_sim.py --raw svc=tfidf_char_svc_C0.3 --raw lp=<tag> --ctr emb4c=qwen3_4b_base:0.001 \
           --ctr emb8c=qwen3_8b_base:0.001 --ctr lpc=lps:qwen3_0.6b_base+qwen3_8b_base:0.1 \
           --weights svc=1,lp=1,emb4c=1,emb8c=1,lpc=1 --sims 3
--ctr spec: name=[kind:]feat1+feat2:C with kind emb (default; StandardScaler + LogReg), lp (log-prob stats),
lps (log-prob stats + style+tok v2, all centered), style (style+tok v2 only; feats empty), tfidf / tfidfsvc (feats
= train_tfidf analyzers, e.g. char; train_tfidf_ctr.fit_member with LogReg / LinearSVC + inner temperature,
--tfidf-max-features columns per analyzer). lp/lps/style use the QuantileTransformer + scaler + LogReg pipeline, as train_emb.py.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, encode_y, load_folds, load_member, load_test, load_train, log_experiment, metrics
from matching import cluster_prompts, drop_mask, match
from train_emb import center_by_clusters, load_feats, make_model


def parse_ctr(spec):
    """'name=[kind:]f1+f2:C' -> (name, kind, [features], C)."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    kind = parts[0] if len(parts) == 3 else "emb"
    feats = [f for f in parts[-2].split("+") if f]
    assert kind in ("emb", "lp", "lps", "style", "tfidf", "tfidfsvc"), kind
    return name, kind, feats, float(parts[-1])


def ctr_matrix(kind, feats, tr):
    """Raw (uncentered) training matrix of a centered member, in train.csv row order (None for tfidf: per fold)."""
    if kind.startswith("tfidf"):
        return None
    mats = [load_feats("emb" if kind == "emb" else "lp", [f], "train")[0][0] for f in feats]
    if kind in ("lps", "style"):
        from train_gbdt import build
        mats.append(build("style+tok", tr, load_test())[0])
    return np.hstack(mats).astype(np.float32)


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
    ap.add_argument("--tfidf-max-features", type=int, default=50_000)
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    texts = tr["text"].tolist()
    prompts = tr["prompt_id"].to_numpy()
    n_f = int(folds.max()) + 1
    weights = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.weights.split(",")}
    raw = {kv.split("=")[0]: load_member(kv.split("=")[1])[0] for kv in a.raw}
    ctr_specs = {nm: (kind, fl, C) for nm, kind, fl, C in map(parse_ctr, a.ctr)}
    feats = {nm: ctr_matrix(kind, fl, tr) for nm, (kind, fl, _) in ctr_specs.items()}
    models = {}
    for nm, (kind, fl, C) in ctr_specs.items():            # training side: independent of the simulation
        for k in range(n_f):
            trn = np.flatnonzero(folds != k)
            if kind.startswith("tfidf"):                  # a predict(texts) closure that centers its own input
                import train_tfidf_ctr
                targs = argparse.Namespace(feats="+".join(fl), max_features=a.tfidf_max_features, thr=a.thr, C=C,
                                           model="svc" if kind == "tfidfsvc" else "lr")
                models[nm, k] = train_tfidf_ctr.fit_member(targs, [texts[i] for i in trn], y[trn], prompts[trn])
                continue
            Xc, _ = center_by_clusters(feats[nm][trn], [texts[i] for i in trn], a.thr)
            models[nm, k] = make_model("emb" if kind == "emb" else "lp", C).fit(Xc, y[trn])
        ncol = feats[nm].shape[1] if feats[nm] is not None else f"<= {a.tfidf_max_features} per analyzer"
        print(f"fitted centered member {nm} ({kind}: {'+'.join(fl)}, {ncol} cols, C={C}) on "
              f"{n_f} training folds  ({time.time() - t0:.0f}s)", flush=True)
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
                if ctr_specs[nm][0].startswith("tfidf"):
                    Pn[loc] = models[nm, k]([texts[i] for i in rws])
                    continue
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
