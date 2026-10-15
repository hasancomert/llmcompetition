"""Greedy hill-climbing blend with repeated selection, scored nested; multiclass N x K version.

Adapted from S6E10 src/blend.py (fit_hill + nested comparison) and the S6E9 blend_ext.py method described in
LESSONS.md (repeated-selection greedy hill climbing from the best single model, stop when the gain is below
1e-7, weights = selection counts / total; nested: weights chosen on 4/5 of the OOF rows and scored on the other
1/5; AVGW = final weights are the mean of the 5 nested weight vectors).
Changes: members are (N, K) probability matrices; the blend averages probabilities ("prob") or
log-probabilities ("logprob", a geometric mean, soft-maxed back); the default score is balanced accuracy.
Not carried over: rank space. In the binary AUC projects every member was rank-transformed first; with an
argmax over 18 classes per-class ranks are not comparable across classes, so there is no rank space here.
"""
import numpy as np

from metrics import balanced_accuracy


def _space(Ps, space):
    if space == "prob":
        return [np.asarray(P, dtype=np.float64) for P in Ps]
    if space == "logprob":
        return [np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-9, 1)) for P in Ps]
    raise ValueError(space)


def _to_probs(S, space):
    if space == "prob":
        return S / S.sum(1, keepdims=True)
    z = S - S.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def blend(Ps, w, space="prob"):
    """Weighted blend of member matrices, returned as probabilities."""
    T = _space(Ps, space)
    w = np.asarray(w, dtype=np.float64) / np.sum(w)
    return _to_probs(sum(wi * t for wi, t in zip(w, T)), space)


def hill_climb(Ps, y, idx=None, metric=balanced_accuracy, space="prob", max_steps=200, tol=1e-7,
               higher_is_better=True):
    """Selection counts and weights fitted on rows idx (all rows if None)."""
    y = np.asarray(y)
    idx = np.arange(len(y)) if idx is None else idx
    T = [t[idx] for t in _space(Ps, space)]
    sgn = 1 if higher_is_better else -1
    score = lambda S, c: sgn * metric(y[idx], _to_probs(S / c, space))
    single = [score(t, 1) for t in T]
    cnt = np.zeros(len(T))
    j0 = int(np.argmax(single))
    cnt[j0], S, best = 1, T[j0].copy(), single[j0]
    for _ in range(max_steps):
        cand = [score(S + T[j], cnt.sum() + 1) for j in range(len(T))]
        j = int(np.argmax(cand))
        if cand[j] <= best + tol:
            break
        best, cnt[j], S = cand[j], cnt[j] + 1, S + T[j]
    return cnt, cnt / cnt.sum()


def nested_hill(Ps, y, meta_folds, avg_weights=True, **kw):
    """Honest blend score: for each meta fold, weights from hill_climb on the other folds, applied to this one.
    Returns dict(nested_pred, nested_score, fold_weights, weights) where weights is the mean of the fold
    weight vectors (avg_weights=True, S6E9 AVGW) or a refit on all rows (False)."""
    y = np.asarray(y)
    metric = kw.get("metric", balanced_accuracy)
    space = kw.get("space", "prob")
    k = np.asarray(Ps[0]).shape[1]
    pred = np.zeros((len(y), k))
    fw = []
    for tr, va in meta_folds:
        _, w = hill_climb(Ps, y, idx=tr, **kw)
        fw.append(w)
        pred[va] = blend([np.asarray(P)[va] for P in Ps], w, space)
    w_final = np.mean(fw, axis=0) if avg_weights else hill_climb(Ps, y, **kw)[1]
    return dict(nested_pred=pred, nested_score=metric(y, pred), fold_weights=fw, weights=w_final)


def _selftest():
    from metrics import balanced_accuracy, mlogloss
    from oof_io import make_folds
    rng = np.random.default_rng(2)
    n, k = 900, 18
    y = rng.integers(0, k, n)
    groups = rng.integers(0, 60, n)

    def member(strength, noise_seed):
        r = np.random.default_rng(noise_seed)
        z = r.normal(size=(n, k))
        z[np.arange(n), y] += strength
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    Ps = [member(1.2, 10), member(1.0, 11), member(0.2, 12), member(1.1, 13)]
    singles = [balanced_accuracy(y, P) for P in Ps]
    cnt, w = hill_climb(Ps, y)
    assert abs(w.sum() - 1) < 1e-12 and cnt[2] == 0, (cnt, singles)   # the weak member is not picked
    folds = make_folds(y, groups, k=5, seed=7)
    r = nested_hill(Ps, y, folds)
    eq = balanced_accuracy(y, blend(Ps, np.ones(4)))
    r_ll = nested_hill(Ps, y, folds, metric=mlogloss, higher_is_better=False, space="logprob")
    assert r["nested_pred"].shape == (n, k) and len(r["fold_weights"]) == 5
    print(f"hill_climb selftest OK: singles {np.round(singles, 4).tolist()}, weights {np.round(w, 3).tolist()}, "
          f"nested BA {r['nested_score']:.4f} (equal-weight mean {eq:.4f}), "
          f"nested log-loss fit in logprob space: log loss {r_ll['nested_score']:.4f}")


if __name__ == "__main__":
    _selftest()
