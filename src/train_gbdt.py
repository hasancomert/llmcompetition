"""Baselines 2 and 3: LightGBM on style features, tokenizer features, or both, on the shared folds.

No early stopping on the validation fold (PLAN.md U4, conflict Ç5): inside every outer training fold an inner
StratifiedGroupKFold(4) by prompt_id picks the number of rounds (mean best iteration x 1.1), then one model is
refitted on the whole outer training fold with that many rounds and predicts the validation fold and the test.
Features have no fitted state (pure functions of one text), so they are computed once for all rows.

Examples:
  python train_gbdt.py --feats style
  python train_gbdt.py --feats style+tok
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import K, OUT, encode_y, load_folds, load_test, load_train, run_member
from style_features import style_frame
from tokenizer_features import TOKENIZER_DIR, tokenizer_frame

FEAT_DIR = OUT / "features"
PARAMS = dict(objective="multiclass", num_class=K, learning_rate=0.03, num_leaves=15, min_data_in_leaf=15,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              num_threads=4)


def feature_key(texts: pd.Series) -> str:
    """Cache key: feature code + tokenizer file hashes + the texts themselves (audit finding: a cache keyed by
    kind only lets CV use stale train features while the test features come from the edited code)."""
    h = hashlib.sha256()
    for f in ("style_features.py", "tokenizer_features.py"):
        h.update((Path(__file__).with_name(f)).read_bytes())
    for src in sorted(TOKENIZER_DIR.glob("*/SOURCE.json")):
        h.update(src.read_bytes())
    h.update("\x00".join(texts).encode("utf-8"))
    return h.hexdigest()[:12]


def features(kind: str, texts: pd.Series, split: str) -> pd.DataFrame:
    path = FEAT_DIR / f"{kind}_{split}_{feature_key(texts)}.parquet"
    if split == "train" and path.exists():
        F = pd.read_parquet(path)
        assert len(F) == len(texts)
        return F
    F = style_frame(texts) if kind == "style" else tokenizer_frame(texts)
    if split == "train":
        FEAT_DIR.mkdir(parents=True, exist_ok=True)
        F.to_parquet(path)
    return F


def build(feats: str, tr: pd.DataFrame, te: pd.DataFrame):
    Xtr, Xte = [], []
    for kind in feats.split("+"):
        Xtr.append(features(kind, tr["text"], "train"))
        Xte.append(features(kind, te["text"], "test"))
    Xtr, Xte = pd.concat(Xtr, axis=1), pd.concat(Xte, axis=1)
    keep = [c for c in Xtr.columns if Xtr[c].nunique() > 1]
    return Xtr[keep].to_numpy(np.float32), Xte[keep].to_numpy(np.float32), keep


def fit_fold(X_tr, y_tr, g_tr, X_apply, params, seed, max_rounds=3000):
    best = []
    for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=100 + seed).split(X_tr, y_tr, g_tr):
        p = dict(params, seed=seed)
        b = lgb.train(p, lgb.Dataset(X_tr[itr], y_tr[itr]), max_rounds,
                      valid_sets=[lgb.Dataset(X_tr[iva], y_tr[iva])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        best.append(b.best_iteration)
    rounds = max(20, int(np.mean(best) * 1.1))
    b = lgb.train(dict(params, seed=seed), lgb.Dataset(X_tr, y_tr), rounds)
    return [b.predict(X) for X in X_apply], rounds, b


def fit_fold_lr(X_tr, y_tr, X_apply, C):
    """Linear family for the same features: scaler + multinomial LR fitted on the training fold only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                      StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    m.fit(X_tr, y_tr)
    return [m.predict_proba(X) for X in X_apply]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="style")
    ap.add_argument("--model", default="lgb", choices=["lgb", "lr"])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    g = tr["prompt_id"].to_numpy()
    folds = load_folds(tr)
    Xtr, Xte, names = build(args.feats, tr, te)
    oof = np.zeros((len(y), K))
    test = np.zeros((len(Xte), K))
    rounds, gain = [], np.zeros(len(names))
    for k in range(int(folds.max()) + 1):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        if args.model == "lr":
            oof[val], p_te = fit_fold_lr(Xtr[trn], y[trn], [Xtr[val], Xte], args.C)
            test += p_te / (folds.max() + 1)
            continue
        (p_va, p_te), r, b = fit_fold(Xtr[trn], y[trn], g[trn], [Xtr[val], Xte], PARAMS, args.seed)
        oof[val], rounds = p_va, rounds + [r]
        test += p_te / (folds.max() + 1)
        gain += b.feature_importance("gain")
        print(f"  fold {k}: rounds {r}", flush=True)
    if args.model == "lr":
        tag = args.tag or f"lr_{args.feats.replace('+', '_')}_C{args.C:g}"
        run_member(tag, f"QuantileTransformer + LogReg C={args.C:g} on {args.feats} features ({len(names)} cols)",
                   oof, test, y, folds, t0, feats=args.feats, n_features=len(names), C=args.C)
        return
    tag = args.tag or f"lgb_{args.feats.replace('+', '_')}_s{args.seed}"
    top = ", ".join(names[i] for i in np.argsort(-gain)[:8])
    run_member(tag, f"LightGBM on {args.feats} features ({len(names)} cols), rounds from inner grouped CV",
               oof, test, y, folds, t0, note=f"rounds {rounds}", feats=args.feats, n_features=len(names),
               rounds=rounds, top_gain=top, params=PARAMS, seed=args.seed)
    print("top gain:", top)


if __name__ == "__main__":
    main()
