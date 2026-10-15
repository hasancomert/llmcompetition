"""Metrics for an N x K class-probability matrix (multiclass), balanced accuracy first.

New for the multiclass setting: the S6E9 / S6E10 projects only used ROC AUC. Balanced accuracy is the mean
per-class recall of argmax predictions, i.e. sklearn.metrics.balanced_accuracy_score on P.argmax(1).
Classes that never occur in y are left out of the mean (as sklearn does).
"""
import numpy as np


def predict_labels(P):
    """argmax over classes; P is (N, K) probabilities, log-probabilities or any per-class scores."""
    return np.asarray(P).argmax(axis=1)


def per_class_recall(y, P, n_classes=None):
    """Recall of every class present in y (NaN for classes absent from y)."""
    y = np.asarray(y)
    k = n_classes or np.asarray(P).shape[1]
    pred = predict_labels(P)
    rec = np.full(k, np.nan)
    for c in range(k):
        m = y == c
        if m.any():
            rec[c] = (pred[m] == c).mean()
    return rec


def balanced_accuracy(y, P):
    """Mean recall over the classes present in y."""
    return float(np.nanmean(per_class_recall(y, P)))


def mlogloss(y, P, eps=1e-15):
    """Multiclass log loss; rows are renormalized after clipping (smooth objective for weight fitting)."""
    P = np.clip(np.asarray(P, dtype=np.float64), eps, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.log(P[np.arange(len(y)), np.asarray(y)]).mean())


def fold_scores(y, P, folds, metric=balanced_accuracy):
    """(mean, std, list) of the metric on each validation fold; folds = [(train_idx, valid_idx), ...]."""
    s = [metric(np.asarray(y)[va], np.asarray(P)[va]) for _, va in folds]
    return float(np.mean(s)), float(np.std(s)), s


def _selftest():
    import warnings
    from sklearn.metrics import balanced_accuracy_score, log_loss
    rng = np.random.default_rng(0)
    n, k = 600, 18
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 1.5                       # some signal
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ba = balanced_accuracy(y, P)
    assert abs(ba - balanced_accuracy_score(y, P.argmax(1))) < 1e-12, ba
    assert abs(mlogloss(y, P) - log_loss(y, P, labels=np.arange(k))) < 1e-9
    rec = per_class_recall(y, P)
    assert rec.shape == (k,) and np.all((rec >= 0) & (rec <= 1))
    y2 = y.copy()
    y2[y2 == 3] = 4                                      # class 3 absent: left out of the mean
    with warnings.catch_warnings():                      # sklearn warns that class 3 is predicted but absent
        warnings.simplefilter("ignore")
        assert abs(balanced_accuracy(y2, P) - balanced_accuracy_score(y2, P.argmax(1))) < 1e-12
    folds = [(np.arange(n)[np.arange(n) % 5 != f], np.arange(n)[np.arange(n) % 5 == f]) for f in range(5)]
    m, s, per = fold_scores(y, P, folds)
    assert len(per) == 5 and 0 < m < 1
    print(f"metrics selftest OK: balanced accuracy {ba:.4f} (= sklearn), log loss {mlogloss(y, P):.4f}, "
          f"fold mean {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    _selftest()
