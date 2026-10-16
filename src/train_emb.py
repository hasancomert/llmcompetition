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
    return center_groups(X, g, min_size), g


def center_groups(X, g, min_size=3):
    """Subtract each group's mean (groups of < min_size rows unchanged)."""
    Xc = X.copy()
    for c in np.unique(g):
        ix = g == c
        if ix.sum() >= min_size:
            Xc[ix] -= X[ix].mean(0)
    return Xc


def center_sim(Xtr_raw, y, folds, texts, thr, C, sims, seed=0):
    """Test-like check of prompt centering. Training side as at submission time: every row of the training folds,
    centered with clusters of the training-fold texts. Test side: rows of the held-out fold are dropped with
    model-specific keep rates (unequal test counts), the kept texts are clustered ON THEIR OWN (as the test is) and
    centered with those clusters. Raw and centered features are scored on the same kept rows (paired).
    (A first version also dropped training rows; that made the training classes unbalanced and confounded the
    comparison.)"""
    from matching import drop_mask
    from common import metrics
    rng = np.random.default_rng(seed)
    n_f = int(folds.max()) + 1
    fold_models = {}
    for k in range(n_f):                                     # training side does not depend on the simulation
        trn = np.flatnonzero(folds != k)
        Xc_tr, _ = center_by_clusters(Xtr_raw[trn], [texts[i] for i in trn], thr)
        fold_models[k] = (make_model("emb", C).fit(Xtr_raw[trn], y[trn]), make_model("emb", C).fit(Xc_tr, y[trn]))
    res = []
    for sim in range(sims):
        keep = np.flatnonzero(drop_mask(y, None, rng, 0.35, 1.0))
        P_raw, P_ctr = np.zeros((len(keep), K)), np.zeros((len(keep), K))
        fk = folds[keep]
        for k in range(n_f):
            loc = np.flatnonzero(fk == k)
            rows = keep[loc]
            Xc_va, _ = center_by_clusters(Xtr_raw[rows], [texts[i] for i in rows], thr)
            m_raw, m_ctr = fold_models[k]
            P_raw[loc] = m_raw.predict_proba(Xtr_raw[rows])
            P_ctr[loc] = m_ctr.predict_proba(Xc_va)
        out = {"raw": metrics.balanced_accuracy(y[keep], P_raw), "centered": metrics.balanced_accuracy(y[keep], P_ctr),
               "ll_raw": metrics.mlogloss(y[keep], P_raw), "ll_centered": metrics.mlogloss(y[keep], P_ctr)}
        res.append(out)
        print(f"sim {sim}: kept {len(keep)} rows, raw BA {out['raw']:.4f} / ll {out['ll_raw']:.4f}, "
              f"centered BA {out['centered']:.4f} / ll {out['ll_centered']:.4f}", flush=True)
    d = np.array([r["centered"] - r["raw"] for r in res])
    print(f"centering under test-like drops (full training folds): mean BA diff {d.mean():+.4f} "
          f"(per sim {np.round(d, 4).tolist()})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["emb", "lp", "style"], required=True, help="style: only the style+tok v2 columns")
    ap.add_argument("--names", default="", help="comma-separated feature names (gpu_embed --name)")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--with-style", action="store_true")
    ap.add_argument("--center-style", action="store_true", help="also center the style+tok columns (default: raw)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--center-sim", type=int, default=0, help="only run the test-like centering check (N sims)")
    args = ap.parse_args()

    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    names = args.names.split(",") if args.names else []
    if args.kind == "style":
        names, args.with_style, args.center_style = [], True, args.center
    tr_m, te_m = load_feats(args.kind, names, "train"), load_feats(args.kind, names, "test")
    for (X, meta) in tr_m:
        assert meta["ids"] == tr["id"].tolist(), "feature rows are not in train.csv order"
    for (X, meta) in te_m:
        assert meta["ids"] == te["id"].tolist()
    Xtr = np.hstack([m[0] for m in tr_m]) if tr_m else np.zeros((len(tr), 0), np.float32)
    Xte = np.hstack([m[0] for m in te_m]) if te_m else np.zeros((len(te), 0), np.float32)
    if args.with_style and args.center_style:              # style+tok columns join before centering
        from train_gbdt import build
        Str, Ste, _ = build("style+tok", tr, te)
        Xtr, Xte = np.hstack([Xtr, Str]).astype(np.float32), np.hstack([Xte, Ste]).astype(np.float32)
    if args.center_sim:
        return center_sim(Xtr, y, folds, tr["text"].tolist(), args.thr, args.C, args.center_sim)
    note = ""
    if args.center:
        Xtr, g = center_by_clusters(Xtr, tr["text"].tolist(), args.thr)
        Xte, _ = center_by_clusters(Xte, te["text"].tolist(), args.thr) if len(te) >= 40 else (Xte - Xte.mean(0), None)
        note = f"centered by {len(np.unique(g))} unsupervised clusters (thr {args.thr})"
    if args.with_style and not args.center_style:
        from train_gbdt import build
        Str, Ste, _ = build("style+tok", tr, te)
        Xtr, Xte = np.hstack([Xtr, Str]), np.hstack([Xte, Ste])

    oof, test = fold_fit(args.kind, args.C, Xtr, Xte, y, folds)
    sty = "_style" + ("c" if args.center_style else "") if args.with_style else ""
    tag = args.tag or (f"{args.kind}_{'+'.join(names)}{'_ctr' if args.center else ''}{sty}_C{args.C:g}"
                       if names else f"style+tok{'_ctr' if args.center else ''}_C{args.C:g}")
    family = {"emb": "frozen LM embeddings", "lp": "LM token log-prob stats", "style": "style+tok v2 features"}[args.kind]
    change = (f"{family} ({', '.join(names) + ', ' if names else ''}{Xtr.shape[1]} cols)"
              f"{' + prompt centering' if args.center else ''}"
              f"{' + style+tok v2' + (' (centered too)' if args.center_style else ' (not centered)') if names and args.with_style else ''}"
              f", LogReg C={args.C:g}")
    run_member(tag, change, oof, test, y, folds, t0, note=note, names=names, C=args.C, center=args.center,
               with_style=args.with_style, center_style=args.center_style)


if __name__ == "__main__":
    main()
