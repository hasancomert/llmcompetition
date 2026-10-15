"""Adversarial validation: can a classifier tell set A (e.g. train) from set B (e.g. test)?

Adapted from S6E10 src/adversarial.py (LightGBM 300 trees, lr 0.05, 63 leaves, colsample 0.8, 5-fold
StratifiedKFold seed 0, ROC AUC of the OOF "is B" score, gain importances). Measured there: train vs test 0.500
(same distribution), train vs original data 0.834 (delays and Flight Distance drove it; the original rows were then
added with an is_orig flag).
Changes: any numeric feature matrices (e.g. text embeddings or style statistics); optional groups so that rows
of one group (one prompt) never sit on both sides of a fold, otherwise the classifier can learn the topic of a
prompt from its other rows; the OOF score is returned so that the most B-like A rows can be inspected.
Not measured: anything on text.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def adversarial_auc(A, B, groups=None, names=None, k=5, seed=0, params=None, top=6):
    """Returns dict(auc, oof, top) where oof is the out-of-fold P(row is from B) for the rows of [A; B]."""
    import lightgbm as lgb
    X = np.vstack([np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32)])
    t = np.r_[np.zeros(len(A)), np.ones(len(B))].astype(int)
    if groups is None:
        sp = StratifiedKFold(k, shuffle=True, random_state=seed).split(X, t)
    else:
        sp = StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, t, np.asarray(groups))
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8, verbose=-1)
    p.update(params or {})
    oof, imp = np.zeros(len(t)), np.zeros(X.shape[1])
    for tr, va in sp:
        m = lgb.LGBMClassifier(**p).fit(X[tr], t[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        imp += m.booster_.feature_importance("gain")
    names = names or [f"f{i}" for i in range(X.shape[1])]
    order = np.argsort(-imp)[:top]
    return dict(auc=float(roc_auc_score(t, oof)), oof=oof,
                top={names[i]: round(float(imp[i] / max(imp.sum(), 1e-12)), 3) for i in order})


def _selftest():
    rng = np.random.default_rng(6)
    n, d = 800, 6
    A, B = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    same = adversarial_auc(A, B)
    B2 = B.copy()
    B2[:, 2] += 1.5                                     # shift one feature of B (best possible AUC 0.86)
    shifted = adversarial_auc(A, B2, groups=np.r_[rng.integers(0, 50, n), 50 + rng.integers(0, 50, n)])
    assert abs(same["auc"] - 0.5) < 0.06, same["auc"]
    assert shifted["auc"] > 0.75 and next(iter(shifted["top"])) == "f2", shifted["auc"]
    print(f"adversarial selftest OK: same distribution AUC {same['auc']:.3f}, one shifted feature AUC "
          f"{shifted['auc']:.3f} (top feature {next(iter(shifted['top']))})")


if __name__ == "__main__":
    _selftest()
