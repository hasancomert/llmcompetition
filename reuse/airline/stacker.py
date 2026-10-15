"""L2 logistic-regression stack over many members, nested meta-CV, paired meta-seed comparison, source
ablation and family-wise L2; multiclass N x K version.

Adapted from S6E10 src/stack.py (binary, logit(p) inputs, C = 0.001, nested meta folds
StratifiedKFold(5, seed 7), ablate=1, mseed=, fam=tabpfn:3,tfm:1.5) and the decision rule written in
experiments.md: a change is real only if its paired nested-CV difference is positive and large enough on
several meta-fold seeds (noise of a paired difference there: about ±0.000004 AUC).
Changes for the new task:
  - inputs: each member contributes its K log-probabilities, centred per row (log p - mean_k log p, the
    multiclass analogue of logit(p); a constant per row and member carries no information);
  - stacker: multinomial LogisticRegression (one weight per member column and class);
  - meta folds can be grouped (StratifiedGroupKFold on prompt_id);
  - score: balanced accuracy of the argmax (log loss also reported; the LR itself minimises log loss).
Not measured for the multiclass case: the right C, whether centring helps, the size of the noise.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from metrics import balanced_accuracy, mlogloss
from oof_io import make_folds


def member_features(P):
    """(N, K) probabilities -> (N, K) row-centred log-probabilities."""
    L = np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-9, 1))
    return L - L.mean(1, keepdims=True)


def design(members, names=None, family=None, factors=None):
    """Stack the member features side by side. family: {member name: family}; factors: {family: s}.
    Multiplying a member's columns by s divides their L2 penalty by s^2 (family-wise L2)."""
    names = list(members) if names is None else names
    cols = []
    for nm in names:
        s = float((factors or {}).get((family or {}).get(nm), 1.0))
        cols.append(member_features(members[nm]) * s)
    return np.hstack(cols)


def fit_lr(X, y, C=0.001, tol=1e-4, max_iter=3000):
    return LogisticRegression(C=C, tol=tol, max_iter=max_iter).fit(X, y)


def nested(X, y, meta_folds, C=0.001, **kw):
    """Nested meta-CV: each row is predicted by a stacker fitted on the other meta folds."""
    y = np.asarray(y)
    k = len(np.unique(y))
    out = np.zeros((len(y), k))
    for tr, va in meta_folds:
        m = fit_lr(X[tr], y[tr], C=C, **kw)
        out[np.ix_(va, m.classes_)] = m.predict_proba(X[va])
    return dict(pred=out, ba=balanced_accuracy(y, out), logloss=mlogloss(y, out))


def paired_compare(XA, XB, y, groups=None, seeds=(7, 1, 2, 3), C=0.001, metric="ba", **kw):
    """Score stacks A and B on the same meta folds for several meta-fold seeds; return per-seed scores and
    paired differences (B - A). Decide on the paired differences, not on single numbers."""
    rows = []
    for s in seeds:
        mf = make_folds(y, groups, k=5, seed=s)
        a, b = nested(XA, y, mf, C=C, **kw)[metric], nested(XB, y, mf, C=C, **kw)[metric]
        rows.append((s, a, b, b - a))
    d = np.array([r[3] for r in rows])
    return dict(rows=rows, mean_diff=float(d.mean()), n_positive=int((d > 0).sum()), n=len(d))


def ablate(members, source_of, y, meta_folds, C=0.001, metric="ba", **kw):
    """Leave-one-source-out: change of the nested score when all members of one source are removed."""
    names = list(members)
    full = nested(design(members, names), y, meta_folds, C=C, **kw)[metric]
    out = {}
    for src in sorted(set(source_of.values())):
        keep = [nm for nm in names if source_of[nm] != src]
        if keep:
            out[src] = nested(design(members, keep), y, meta_folds, C=C, **kw)[metric] - full
    return full, out


def fit_predict(members, test_members, y, names=None, family=None, factors=None, C=0.001, **kw):
    """Final stacker on all OOF rows, applied to the test matrices of the same members."""
    names = list(members) if names is None else names
    m = fit_lr(design(members, names, family, factors), y, C=C, **kw)
    P = np.zeros((len(next(iter(test_members.values()))), len(np.unique(y))))
    P[:, m.classes_] = m.predict_proba(design(test_members, names, family, factors))
    return P, m


def _selftest():
    rng = np.random.default_rng(3)
    n, t, k = 1200, 300, 18
    y = rng.integers(0, k, n)
    yt = rng.integers(0, k, t)
    groups = rng.integers(0, 80, n)

    def member(strength, seed):
        r = np.random.default_rng(seed)
        out = []
        for lab, nn in ((y, n), (yt, t)):
            z = r.normal(size=(nn, k))
            z[np.arange(nn), lab] += strength
            e = np.exp(z)
            out.append(e / e.sum(1, keepdims=True))
        return out
    pairs = {"gbdt_a": member(1.0, 20), "gbdt_b": member(0.9, 21), "llm_emb": member(1.3, 22),
             "weak": member(0.3, 23)}
    members = {nm: p[0] for nm, p in pairs.items()}
    tests = {nm: p[1] for nm, p in pairs.items()}
    mf = make_folds(y, groups, k=5, seed=7)
    base = nested(design(members), y, mf, C=0.01)
    fam = {"llm_emb": "llm"}
    res = paired_compare(design(members), design(members, family=fam, factors={"llm": 3.0}), y, groups,
                         seeds=(7, 1), C=0.01)
    full, abl = ablate(members, {"gbdt_a": "gbdt", "gbdt_b": "gbdt", "llm_emb": "llm", "weak": "weak"},
                       y, mf, C=0.01)
    P, _ = fit_predict(members, tests, y, C=0.01)
    assert P.shape == (t, k) and np.allclose(P.sum(1), 1)
    assert base["pred"].shape == (n, k) and len(res["rows"]) == 2 and set(abl) == {"gbdt", "llm", "weak"}
    print(f"stacker selftest OK: nested BA {base['ba']:.4f} / log loss {base['logloss']:.4f}; family L2 "
          f"(llm x3) paired diff {res['mean_diff']:+.4f} ({res['n_positive']}/{res['n']} positive); ablation "
          f"{ {k_: round(v, 4) for k_, v in abl.items()} }; test BA of the final fit {balanced_accuracy(yt, P):.4f}")


if __name__ == "__main__":
    _selftest()
