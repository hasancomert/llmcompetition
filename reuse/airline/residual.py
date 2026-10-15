"""Residual boosting: a GBDT that starts from another model's margins (logits) instead of from a constant;
multiclass N x K version.

Source: the S6E9 recipe in LESSONS.md ("artık model"): XGBoost base_margin / LightGBM init_score / CatBoost
baseline start from the GLR logit; the training rows' logits come from a GLR re-fitted with an inner 5-fold
split inside each outer fold's training rows (so no training row sees a margin fitted on its own label), the
validation and test rows' logits from the outer-fold GLR. Measured there (binary AUC, 10 folds): XGBoost
0.94652 -> 0.94658, LightGBM 0.94650 -> 0.94656, CatBoost 0.94648 -> 0.94655.
The S6E9 code itself (train.py margin=1) is not in this repository; this module re-implements the recipe from
that description. S6E10 never used base margins. CatBoost (Pool(baseline=...)) is not implemented here.
Changes for the new task: margins are (N, K) (e.g. an LLM classifier's logits); LightGBM gets a 2-D
init_score, XGBoost a 2-D base_margin with multi:softprob; predictions add the margins back before softmax.
Not measured: any of this on text / 18 classes.
"""
import numpy as np

from oof_io import make_folds


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nested_margins(fit_base, X, y, outer_folds, X_test=None, groups=None, inner_k=5, seed=0):
    """Leak-free base margins per outer fold.
    fit_base(X_train, y_train) must return a function X -> (n, K) margins (logits / log-probabilities).
    Returns a list with one dict per outer fold: tr_margin (inner-fold OOF of the training rows),
    va_margin and te_margin (base fitted on all training rows of the fold)."""
    y = np.asarray(y)
    out = []
    for tr, va in outer_folds:
        g_tr = None if groups is None else np.asarray(groups)[tr]
        k = None
        tr_m = None
        for itr, iva in make_folds(y[tr], g_tr, k=inner_k, seed=seed):
            f = fit_base(X[tr][itr], y[tr][itr])
            mv = f(X[tr][iva])
            if tr_m is None:
                k = mv.shape[1]
                tr_m = np.zeros((len(tr), k))
            tr_m[iva] = mv
        f = fit_base(X[tr], y[tr])
        out.append(dict(tr_margin=tr_m, va_margin=f(X[va]),
                        te_margin=None if X_test is None else f(X_test)))
    return out


def lgb_residual(Xtr, ytr, mtr, Xva, mva, Xte=None, mte=None, yva=None, params=None, rounds=2000,
                 early_stopping=100):
    """LightGBM multiclass from init_score = margins. Early stopping on (Xva, yva) if yva is given (the
    usual practice in both projects: it makes the OOF slightly optimistic). Returns (p_va, p_te, iterations)."""
    import lightgbm as lgb
    k = mtr.shape[1]
    p = dict(objective="multiclass", num_class=k, learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    p.update(params or {})
    dtr = lgb.Dataset(Xtr, ytr, init_score=mtr)
    kw = {}
    if yva is not None:
        kw = dict(valid_sets=[lgb.Dataset(Xva, yva, init_score=mva, reference=dtr)],
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    b = lgb.train(p, dtr, rounds, **kw)
    it = b.best_iteration or rounds
    p_va = softmax(b.predict(Xva, raw_score=True, num_iteration=it) + mva)   # predict() ignores init_score
    p_te = None if Xte is None else softmax(b.predict(Xte, raw_score=True, num_iteration=it) + mte)
    return p_va, p_te, it


def xgb_residual(Xtr, ytr, mtr, Xva, mva, Xte=None, mte=None, yva=None, params=None, rounds=2000,
                 early_stopping=100):
    """XGBoost multi:softprob from base_margin = margins (the margin must also be set on every DMatrix that
    is predicted, XGBoost then adds it). Returns (p_va, p_te, iterations)."""
    import xgboost as xgb
    k = mtr.shape[1]
    p = dict(objective="multi:softprob", num_class=k, eta=0.05, max_depth=4, subsample=0.8,
             colsample_bytree=0.8, eval_metric="mlogloss", tree_method="hist", seed=0)
    p.update(params or {})
    dtr = xgb.DMatrix(Xtr, label=ytr, base_margin=mtr)
    dva = xgb.DMatrix(Xva, label=yva, base_margin=mva) if yva is not None else xgb.DMatrix(Xva, base_margin=mva)
    kw = dict(evals=[(dva, "valid")], early_stopping_rounds=early_stopping, verbose_eval=False) \
        if yva is not None else {}
    b = xgb.train(p, dtr, rounds, **kw)
    rng = (0, b.best_iteration + 1) if yva is not None else (0, rounds)
    p_va = b.predict(dva, iteration_range=rng)
    p_te = None if Xte is None else b.predict(xgb.DMatrix(Xte, base_margin=mte), iteration_range=rng)
    return p_va, p_te, rng[1]


def _selftest():
    from sklearn.linear_model import LogisticRegression
    from metrics import balanced_accuracy
    rng = np.random.default_rng(4)
    n, t, d, k = 1500, 300, 8, 6
    X = rng.normal(size=(n + t, d))
    lin = X[:, :3] @ rng.normal(size=(3, k))                     # part a linear base model can learn
    nonlin = np.stack([np.sin(2 * X[:, 3 + (c % 5)]) * 2 for c in range(k)], 1)   # part it cannot
    yall = (lin + nonlin + rng.gumbel(size=(n + t, k))).argmax(1)
    X, Xt, y, yt = X[:n], X[n:], yall[:n], yall[n:]
    groups = rng.integers(0, 100, n)

    def fit_base(Xa, ya):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xa, ya)
        return lambda Xb: m.decision_function(Xb)
    folds = make_folds(y, groups, k=5, seed=42)
    marg = nested_margins(fit_base, X, y, folds, X_test=Xt, groups=groups, inner_k=5)
    base_oof, res_lgb, res_xgb = np.zeros((n, k)), np.zeros((n, k)), np.zeros((n, k))
    for (tr, va), m in zip(folds, marg):
        assert m["tr_margin"].shape == (len(tr), k) and m["te_margin"].shape == (t, k)
        base_oof[va] = softmax(m["va_margin"])
        res_lgb[va], pt, _ = lgb_residual(X[tr], y[tr], m["tr_margin"], X[va], m["va_margin"], Xt,
                                          m["te_margin"], yva=y[va], params=dict(num_leaves=7))
        res_xgb[va], _, _ = xgb_residual(X[tr], y[tr], m["tr_margin"], X[va], m["va_margin"], Xt,
                                         m["te_margin"], yva=y[va], params=dict(max_depth=3))
        assert np.allclose(pt.sum(1), 1) and np.allclose(res_xgb[va].sum(1), 1, atol=1e-5)
    print(f"residual selftest OK (synthetic data, linear base + non-linear part): OOF balanced accuracy "
          f"base {balanced_accuracy(y, base_oof):.4f}, LightGBM residual {balanced_accuracy(y, res_lgb):.4f}, "
          f"XGBoost residual {balanced_accuracy(y, res_xgb):.4f}")


if __name__ == "__main__":
    _selftest()
