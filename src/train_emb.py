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


def load_feats(kind, names, split):
    mats = []
    for nm in names:
        meta = json.loads((EMB / f"meta_{nm}_{split}.json").read_text())
        mats.append(np.load(EMB / f"{kind}_{nm}_{split}.npy").astype(np.float32))
        mats[-1] = (mats[-1], meta)
    return mats


def center_by_clusters(X, texts, thr):
    from matching import cluster_prompts
    g = cluster_prompts(texts, thr)
    Xc = X.copy()
    for c in np.unique(g):
        ix = g == c
        Xc[ix] -= X[ix].mean(0)
    return Xc, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["emb", "lp"], required=True)
    ap.add_argument("--names", required=True, help="comma-separated feature names (gpu_embed --name)")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--with-style", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

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
    note = ""
    if args.center:
        Xtr, g = center_by_clusters(Xtr, tr["text"].tolist(), args.thr)
        Xte, _ = center_by_clusters(Xte, te["text"].tolist(), args.thr) if len(te) >= 40 else (Xte - Xte.mean(0), None)
        note = f"centered by {len(np.unique(g))} unsupervised clusters (thr {args.thr})"
    if args.with_style:
        from train_gbdt import build
        Str, Ste, _ = build("style+tok", tr, te)
        Xtr, Xte = np.hstack([Xtr, Str]), np.hstack([Xte, Ste])

    def model():
        if args.kind == "emb":
            return make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000))
        return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                             StandardScaler(), LogisticRegression(C=args.C, max_iter=5000))

    oof, test = np.zeros((len(y), K)), np.zeros((len(Xte), K))
    for k in range(int(folds.max()) + 1):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        m = model().fit(Xtr[trn], y[trn])
        oof[val] = m.predict_proba(Xtr[val])
        test += m.predict_proba(Xte) / (folds.max() + 1)
    tag = args.tag or (f"{args.kind}_{'+'.join(names)}{'_ctr' if args.center else ''}"
                       f"{'_style' if args.with_style else ''}_C{args.C:g}")
    change = (f"{'frozen LM embeddings' if args.kind == 'emb' else 'LM token log-prob stats'} ({', '.join(names)}, "
              f"{Xtr.shape[1]} cols){' + prompt centering' if args.center else ''}"
              f"{' + style+tok v2' if args.with_style else ''}, LogReg C={args.C:g}")
    run_member(tag, change, oof, test, y, folds, t0, note=note, names=names, C=args.C, center=args.center,
               with_style=args.with_style)


if __name__ == "__main__":
    main()
