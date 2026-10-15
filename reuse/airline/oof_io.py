"""OOF / test prediction bookkeeping for an N x K multiclass problem.

Adapted from the S6E10 project (binary, ROC AUC):
  - src/common.py      folds() and save(): one fixed split for every model; oof/<tag>.npy + preds/<tag>.npy
  - src/train.py       per-fold checkpoint (a restarted run resumes), skip a tag that is already done,
                       one experiments.md row per run (fold mean ± std and the pooled OOF score)
  - src/blend.py, stack.py   "a,b,c" = average several tags (seeds) before blending
  - src/ext_pub.py     add(): shape / finiteness / scale checks of a foreign prediction pair and a
                       plausibility window on its OOF score (too good = suspected leak, rejected)
  - pool hygiene       logit correlation >= 0.99999 = a copy of an existing member (S6E10 refreshes)
Changes for the new task: predictions are (N, K) matrices; the split is a stratified *group* split
(StratifiedGroupKFold on prompt_id, so that no prompt is in two folds); the score is balanced accuracy.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_folds(y, groups=None, k=5, seed=42):
    """[(train_idx, valid_idx), ...]: StratifiedGroupKFold when groups are given (no group in two folds),
    StratifiedKFold otherwise. Keep the same (k, seed) for every model so that OOFs can be stacked."""
    y = np.asarray(y)
    if groups is None:
        sp = StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        sp = StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)
    folds = [(tr, va) for tr, va in sp]
    if groups is not None:
        g = np.asarray(groups)
        for tr, va in folds:
            assert not set(g[tr]) & set(g[va]), "a group is on both sides of a fold"
    return folds


def fold_ids(folds, n):
    """Fold number of every row (handy to save next to the OOF file)."""
    f = np.full(n, -1)
    for i, (_, va) in enumerate(folds):
        f[va] = i
    assert (f >= 0).all()
    return f


def _check_matrix(a, n, k, name):
    a = np.asarray(a, dtype=np.float64)
    assert a.shape == (n, k), f"{name}: shape {a.shape}, expected {(n, k)}"
    assert np.isfinite(a).all(), f"{name}: non-finite values"
    return a


def save_pred(root, tag, oof, test, n_classes=None):
    """Write oof/<tag>.npy and preds/<tag>.npy (float32) under root."""
    root = Path(root)
    k = n_classes or np.asarray(oof).shape[1]
    oof = _check_matrix(oof, len(oof), k, "oof")
    test = _check_matrix(test, len(test), k, "test")
    for d, a in (("oof", oof), ("preds", test)):
        (root / d).mkdir(parents=True, exist_ok=True)
        np.save(root / d / f"{tag}.npy", a.astype(np.float32))


def load_pred(root, tag):
    """(oof, test) of a tag; "a,b,c" averages several tags (e.g. seeds) first."""
    root = Path(root)
    tags = tag.split(",")
    oof = np.mean([np.load(root / "oof" / f"{t}.npy").astype(np.float64) for t in tags], axis=0)
    test = np.mean([np.load(root / "preds" / f"{t}.npy").astype(np.float64) for t in tags], axis=0)
    return oof, test


def is_done(root, tag):
    return (Path(root) / "oof" / f"{tag}.npy").exists()


class FoldCheckpoint:
    """Per-fold checkpoint: cache/partial/<tag>__f<i>.npz; a restarted run resumes at the first missing fold."""

    def __init__(self, root, tag):
        self.dir = Path(root) / "cache" / "partial"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag

    def path(self, i):
        return self.dir / f"{self.tag}__f{i}.npz"

    def load(self, i):
        p = self.path(i)
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=False)
        return {k: d[k] for k in d.files}

    def save(self, i, **arrays):
        np.savez(self.path(i), **arrays)

    def clear(self, k):
        for i in range(k):
            self.path(i).unlink(missing_ok=True)


def log_row(path, tag, note, fold_mean, fold_std, oof_score, extra="", seconds=None):
    """Append one markdown row: | tag | note | fold mean ± std | pooled OOF | extra | time |."""
    t = f"{seconds:.0f}s" if seconds is not None else ""
    with open(path, "a") as f:
        f.write(f"| {tag} | {note} | {fold_mean:.5f} ± {fold_std:.5f} | {oof_score:.5f} | {extra} | {t} |\n")


def to_probs(a):
    """Rows that are not probabilities (negative values, or rows not summing to 1) are treated as logits and
    soft-maxed. S6E10 lesson: one public source stored scores in [-2, 2.4] instead of probabilities and the
    stacker silently clipped them; check the scale of every foreign file."""
    a = np.asarray(a, dtype=np.float64)
    if a.min() < 0 or not np.allclose(a.sum(1), 1, atol=1e-3):
        z = a - a.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True), "softmax"
    return a, "probabilities"


def validate_member(name, oof, test, y, metric, plausible=(0.0, 1.0)):
    """Checks of a foreign OOF/test pair (from ext_pub.add): shapes, finiteness, scale, OOF score inside a
    plausibility window. Returns (ok, reason, oof, test) with oof/test as probabilities."""
    y = np.asarray(y)
    oof, test = np.asarray(oof, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if oof.ndim != 2 or test.ndim != 2 or oof.shape[0] != len(y) or oof.shape[1] != test.shape[1]:
        return False, f"{name}: shape {oof.shape} / {test.shape}", None, None
    if not (np.isfinite(oof).all() and np.isfinite(test).all()):
        return False, f"{name}: non-finite", None, None
    oof, s1 = to_probs(oof)
    test, s2 = to_probs(test)
    if s1 != s2:
        return False, f"{name}: OOF stored as {s1}, test as {s2}", None, None
    sc = metric(y, oof)
    if not plausible[0] < sc < plausible[1]:
        return False, f"{name}: OOF score {sc:.5f} outside {plausible}", None, None
    return True, f"{name}: ok ({s1}, OOF {sc:.5f})", oof, test


def near_duplicates(members, thr=0.99999):
    """Pairs of members whose flattened log-probability OOFs correlate >= thr (copies of each other).
    members: {name: (N, K) OOF probabilities}."""
    names = list(members)
    Z = []
    for nm in names:
        p = np.clip(np.asarray(members[nm], dtype=np.float64), 1e-9, 1)
        z = np.log(p).ravel()
        Z.append((z - z.mean()) / z.std())
    Z = np.array(Z)
    C = Z @ Z.T / Z.shape[1]
    return [(names[i], names[j], float(C[i, j])) for i in range(len(names)) for j in range(i + 1, len(names))
            if C[i, j] >= thr]


def _selftest():
    import tempfile
    from metrics import balanced_accuracy
    rng = np.random.default_rng(1)
    n, m, k = 400, 150, 18
    y = rng.integers(0, k, n)
    groups = rng.integers(0, 40, n)                       # 40 "prompts"
    folds = make_folds(y, groups, k=5, seed=42)
    f = fold_ids(folds, n)
    assert len(folds) == 5 and all(len(set(groups[f == i]) & set(groups[f != i])) == 0 for i in range(5))
    oof = rng.dirichlet(np.ones(k), n)
    test = rng.dirichlet(np.ones(k), m)
    with tempfile.TemporaryDirectory() as root:
        save_pred(root, "a_s0", oof, test)
        save_pred(root, "a_s1", oof[::-1], test[::-1])
        o, t = load_pred(root, "a_s0,a_s1")
        assert o.shape == (n, k) and np.allclose(o, (oof + oof[::-1]) / 2, atol=1e-6)
        assert is_done(root, "a_s0") and not is_done(root, "b")
        ck = FoldCheckpoint(root, "a")
        ck.save(0, oof=oof[:5], pred=test[:5])
        assert np.allclose(ck.load(0)["oof"], oof[:5]) and ck.load(1) is None
        ck.clear(5)
        assert ck.load(0) is None
        log_row(Path(root) / "experiments.md", "a", "selftest", 0.5, 0.01, 0.5, seconds=1)
        assert (Path(root) / "experiments.md").read_text().startswith("| a | selftest |")
    logits = rng.normal(size=(n, k))
    ok, why, po, pt = validate_member("logits", logits, rng.normal(size=(m, k)), y, balanced_accuracy)
    assert ok and "softmax" in why and np.allclose(po.sum(1), 1)
    ok, why, _, _ = validate_member("leaky", np.eye(k)[y], test, y, balanced_accuracy, plausible=(0, 0.99))
    assert not ok and "outside" in why
    ok, why, _, _ = validate_member("short", oof[:10], test, y, balanced_accuracy)
    assert not ok and "shape" in why
    d = near_duplicates({"a": oof, "a_copy": oof * 1.0, "b": rng.dirichlet(np.ones(k), n)})
    assert [(a, b) for a, b, _ in d] == [("a", "a_copy")]
    print(f"oof_io selftest OK: 5 group folds without overlap, save/load/seed average, checkpoint, log row, "
          f"member validation ({why}), duplicate check")


if __name__ == "__main__":
    _selftest()
