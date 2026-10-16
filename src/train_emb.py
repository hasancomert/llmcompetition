"""Members from frozen-LM features (PLAN.md steps A, B, I): fitted per fold on the shared folds.

--kind emb : mean-pooled hidden states of a few layers -> StandardScaler + LogReg (step A).
             --center: prompt centering (step B). Texts are clustered into prompts WITHOUT labels or prompt_id
             (matching.cluster_prompts, as at test time) and each embedding has its cluster mean subtracted, for
             training and validation rows alike, so that topic is removed the same way it would be on test.
--kind lp  : the model's token log-probability statistics (step I), alone or added to style+tok v2 features
             (--with-style) -> QuantileTransformer + LogReg, like train_gbdt.py --model lr.
Features come from outputs/emb/{emb,lp}_<name>_train.npy (src/gpu_embed.py on Kaggle).

Examples:
  python train_emb.py --kind emb --names qwen3_0.6b_base --C 0.05
  python train_emb.py --kind emb --names qwen3_0.6b_base --C 0.05 --center
  python train_emb.py --kind lp --names qwen3_0.6b_base,qwen3_1.7b_base --with-style
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import K, OUT, encode_y, load_folds, load_test, load_train, run_member

EMB = OUT / "emb"


def make_model(kind, C):
    """emb: StandardScaler + LogReg; lp (optionally with style+tok columns): QuantileTransformer + scaler + LogReg."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    if kind == "emb":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                         StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fold_fit(kind, C, Xtr, Xte, y, folds):
    """Fit one model per shared fold; OOF rows from the held-out fold model, test = mean of the fold models."""
    oof, test = np.zeros((len(y), K)), np.zeros((len(Xte), K))
    n_f = int(folds.max()) + 1
    for k in range(n_f):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        m = make_model(kind, C).fit(Xtr[trn], y[trn])
        oof[np.ix_(val, m.classes_)] = m.predict_proba(Xtr[val])
        test[:, m.classes_] += m.predict_proba(Xte) / n_f
    return oof, test


def load_feats(kind, names, split):
    mats = []
    for nm in names:
        meta = json.loads((EMB / f"meta_{nm}_{split}.json").read_text())
        mats.append(np.load(EMB / f"{kind}_{nm}_{split}.npy").astype(np.float32))
        mats[-1] = (mats[-1], meta)
    return mats


def center_by_clusters(X, texts, thr, min_size=3):
    """Subtract each unsupervised cluster's mean; clusters smaller than min_size are left as they are (a singleton
    would become an all-zero row)."""
    from matching import cluster_prompts
    g = cluster_prompts(texts, thr)
    Xc = X.copy()
    for c in np.unique(g):
        ix = g == c
        if ix.sum() >= min_size:
            Xc[ix] -= X[ix].mean(0)
    return Xc, g


def center_sim(Xtr_raw, y, folds, texts, thr, C, sims, seed=0):
    """Test-like check of prompt centering (audit finding: on full train grids a cluster mean is pure topic, on the
    test it also carries the styles of whichever models are present). Per simulation: drop rows with model-specific
    keep rates, cluster only the kept texts, center kept rows with those clusters, fit/score per fold on kept rows;
    the same kept rows are scored without centering for a paired comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from matching import drop_mask
    from common import metrics
    rng = np.random.default_rng(seed)
    res = []
    for sim in range(sims):
        keep = np.flatnonzero(drop_mask(y, None, rng, 0.35, 1.0))
        Xk = Xtr_raw[keep]
        Xc, _ = center_by_clusters(Xk, [texts[i] for i in keep], thr)
        yk, fk = y[keep], folds[keep]
        out = {}
        for name, X in (("raw", Xk), ("centered", Xc)):
            P = np.zeros((len(keep), K))
            for k in np.unique(fk):
                trn, val = np.flatnonzero(fk != k), np.flatnonzero(fk == k)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[trn], yk[trn])
                P[np.ix_(val, m.classes_)] = m.predict_proba(X[val])
            out[name] = metrics.balanced_accuracy(yk, P)
        res.append(out)
        print(f"sim {sim}: kept {len(keep)} rows, raw BA {out['raw']:.4f}, centered BA {out['centered']:.4f}", flush=True)
    d = np.array([r["centered"] - r["raw"] for r in res])
    print(f"centering under test-like drops: mean diff {d.mean():+.4f} (per sim {np.round(d, 4).tolist()})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["emb", "lp"], required=True)
    ap.add_argument("--names", required=True, help="comma-separated feature names (gpu_embed --name)")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--with-style", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--center-sim", type=int, default=0, help="only run the test-like centering check (N sims)")
    args = ap.parse_args()

    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    names = args.names.split(",")
    tr_m, te_m = load_feats(args.kind, names, "train"), load_feats(args.kind, names, "test")
    for (X, meta) in tr_m:
        assert meta["ids"] == tr["id"].tolist(), "feature rows are not in train.csv order"
    for (X, meta) in te_m:
        assert meta["ids"] == te["id"].tolist()
    Xtr = np.hstack([m[0] for m in tr_m])
    Xte = np.hstack([m[0] for m in te_m])
    if args.center_sim:
        return center_sim(Xtr, y, folds, tr["text"].tolist(), args.thr, args.C, args.center_sim)
    note = ""
    if args.center:
        Xtr, g = center_by_clusters(Xtr, tr["text"].tolist(), args.thr)
        Xte, _ = center_by_clusters(Xte, te["text"].tolist(), args.thr) if len(te) >= 40 else (Xte - Xte.mean(0), None)
        note = f"centered by {len(np.unique(g))} unsupervised clusters (thr {args.thr})"
    if args.with_style:
        from train_gbdt import build
        Str, Ste, _ = build("style+tok", tr, te)
        Xtr, Xte = np.hstack([Xtr, Str]), np.hstack([Xte, Ste])

    oof, test = fold_fit(args.kind, args.C, Xtr, Xte, y, folds)
    tag = args.tag or (f"{args.kind}_{'+'.join(names)}{'_ctr' if args.center else ''}"
                       f"{'_style' if args.with_style else ''}_C{args.C:g}")
    change = (f"{'frozen LM embeddings' if args.kind == 'emb' else 'LM token log-prob stats'} ({', '.join(names)}, "
              f"{Xtr.shape[1]} cols){' + prompt centering' if args.center else ''}"
              f"{' + style+tok v2' if args.with_style else ''}, LogReg C={args.C:g}")
    run_member(tag, change, oof, test, y, folds, t0, note=note, names=names, C=args.C, center=args.center,
               with_style=args.with_style)


if __name__ == "__main__":
    main()
