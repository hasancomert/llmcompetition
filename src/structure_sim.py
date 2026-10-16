"""Which hidden-test structure explains the public LB? Score whole pipelines on simulated test sets.

The held-out folds play the hidden test. Every simulation keeps a subset of their rows (structure set by --mode),
clusters the kept texts ONCE as one pooled set (the hidden test is clustered as one set of ~4,000 texts), centers
the centered members with those pooled clusters, predicts each row with the model of its own fold, blends, and
scores argmax and soft matching (lambda grid) for every pipeline given with --pipe.

Modes:
  grid       model-specific keep rates q_m ~ U(lo, hi) (as pipeline_sim.py / matching.drop_mask)
  perprompt  every prompt keeps a random number of texts in [a, b] (distinct random models): few texts per prompt

Members: --raw name=tag (saved OOF, composition-free) and --ctr name=[kind:]feats:C (as pipeline_sim.py; training
side = training folds centered with their own clusters). Pipelines: --pipe name=member:w,member:w,...

Usage: python structure_sim.py --raw svc=tfidf_char_svc_C0.3 --ctr emb8c=qwen3_8b_base:0.001 \
           --pipe a=svc:1 --pipe b=emb8c:1 --mode perprompt --a 1 --b 3 --sims 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import K, encode_y, load_folds, load_member, load_train, metrics
from matching import cluster_prompts, drop_mask, match
from pipeline_sim import blend, ctr_matrix, parse_ctr
from train_emb import center_by_clusters, center_groups, make_model


def keep_rows(mode, y, prompts, rng, a):
    if mode == "grid":
        return np.flatnonzero(drop_mask(y, None, rng, a.lo, a.hi))
    keep = []
    for p in np.unique(prompts):
        ix = np.flatnonzero(prompts == p)
        m = min(len(ix), int(rng.integers(a.a, a.b + 1)))
        keep.extend(rng.choice(ix, m, replace=False))
    return np.sort(np.array(keep))


def purity(g, prompts):
    """Share of kept texts whose cluster's majority prompt is their own prompt; mean cluster size."""
    df = pd.DataFrame({"g": g, "p": prompts})
    maj = df.groupby("g")["p"].agg(lambda s: s.value_counts().iloc[0])
    return float(maj.sum() / len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", default=[])
    ap.add_argument("--ctr", action="append", default=[])
    ap.add_argument("--pipe", action="append", required=True)
    ap.add_argument("--mode", default="grid", choices=["grid", "perprompt"])
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--a", type=int, default=1)
    ap.add_argument("--b", type=int, default=3)
    ap.add_argument("--sims", type=int, default=10)
    ap.add_argument("--lams", default="0,1,2,4")
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--tfidf-max-features", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    texts = tr["text"].tolist()
    prompts = tr["prompt_id"].to_numpy()
    n_f = int(folds.max()) + 1
    raw = {kv.split("=")[0]: load_member(kv.split("=")[1])[0] for kv in a.raw}
    ctr_specs = {nm: (kind, fl, C) for nm, kind, fl, C in map(parse_ctr, a.ctr)}
    feats = {nm: ctr_matrix(kind, fl, tr) for nm, (kind, fl, _) in ctr_specs.items()}
    pipes = {}
    for spec in a.pipe:
        name, ws = spec.split("=", 1)
        pipes[name] = {m.split(":")[0]: float(m.split(":")[1]) for m in ws.split(",")}
    models = {}
    for nm, (kind, fl, C) in ctr_specs.items():            # training side: independent of the simulation
        for k in range(n_f):
            trn = np.flatnonzero(folds != k)
            if kind.startswith("tfidf"):
                import train_tfidf_ctr
                targs = argparse.Namespace(feats="+".join(fl), max_features=a.tfidf_max_features, thr=a.thr, C=C,
                                           model="svc" if kind == "tfidfsvc" else "lr")
                models[nm, k] = train_tfidf_ctr.fit_member(targs, [texts[i] for i in trn], y[trn], prompts[trn])
                continue
            Xc, _ = center_by_clusters(feats[nm][trn], [texts[i] for i in trn], a.thr)
            models[nm, k] = make_model("emb" if kind == "emb" else "lp", C).fit(Xc, y[trn])
        print(f"fitted centered member {nm} ({kind}) on {n_f} training folds  ({time.time() - t0:.0f}s)", flush=True)
    lams = [float(x) for x in a.lams.split(",")]
    rng = np.random.default_rng(a.seed)
    rows = []
    for sim in range(a.sims):
        keep = keep_rows(a.mode, y, prompts, rng, a)
        kt = [texts[i] for i in keep]
        g = cluster_prompts(kt, a.thr) if len(keep) >= 40 else np.arange(len(keep))
        sizes = np.bincount(g)
        fk = folds[keep]
        P = {nm: raw[nm][keep] for nm in raw}
        for nm, (kind, _, _) in ctr_specs.items():
            Pn = np.zeros((len(keep), K))
            Xc = None if kind.startswith("tfidf") else center_groups(feats[nm][keep], g)
            for k in range(n_f):
                loc = np.flatnonzero(fk == k)
                if not len(loc):
                    continue
                if kind.startswith("tfidf"):
                    Pn[loc] = models[nm, k](kt, g)[loc]
                else:
                    m = models[nm, k]
                    Pn[np.ix_(loc, m.classes_)] = m.predict_proba(Xc[loc])
            P[nm] = Pn
        yk = y[keep]
        res = {"sim": sim, "n": len(keep), "clusters": len(sizes), "mean_size": float(len(keep) / len(sizes)),
               "frac_small": float((sizes[g] < 3).mean()), "purity": purity(g, prompts[keep])}
        for nm in P:
            res[f"m_{nm}"] = metrics.balanced_accuracy(yk, P[nm])
        for pn, w in pipes.items():
            Pb = blend({m: P[m] for m in w}, w)
            for lam in lams:
                pred = Pb.argmax(1) if lam == 0 else match(Pb, g, lam)
                res[f"{pn}_l{lam:g}"] = metrics.balanced_accuracy(yk, np.eye(K)[pred])
        rows.append(res)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    df = pd.DataFrame(rows)
    mean = df.drop(columns=["sim"]).mean()
    se = df.drop(columns=["sim"]).std(ddof=1) / np.sqrt(len(df))
    print(f"MEAN over {len(df)} sims (mode {a.mode}):", flush=True)
    for k in mean.index:
        print(f"  {k}: {mean[k]:.4f} ± {se[k]:.4f}", flush=True)


if __name__ == "__main__":
    main()
