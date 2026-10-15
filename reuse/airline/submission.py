"""Write a submission in the sample file's id order and check its format; class labels instead of probabilities.

Adapted from S6E10 src/submit_check.py (CSV, sample_submission order, ids equal, values finite and in [0, 1],
"suspiciously few distinct values" guard, a printed summary). Changes for the new task: the prediction is a class
label per row (argmax of an N x K matrix mapped through the class names); the file can be parquet (the new task's
submission.parquet); labels must come from the allowed set; the predicted class counts are printed, because the
class mix of the test set differs from train and balanced accuracy weights every class equally.
The column names (id / model) are parameters: the new competition's sample file was not available here.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def labels_from_probs(P, classes):
    """argmax of an (N, K) matrix -> class names (classes[k] is the name of column k)."""
    return np.asarray(classes)[np.asarray(P).argmax(1)]


def _read(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write_submission(path, ids, labels, sample=None, id_col="id", target_col="model", allowed=None,
                     min_classes=2):
    """Write ids + labels (in the sample's id order when a sample DataFrame/path is given) and check the file."""
    sub = pd.DataFrame({id_col: np.asarray(ids), target_col: np.asarray(labels)})
    assert sub[id_col].is_unique, "duplicate ids"
    if sample is not None:
        ss = _read(sample) if isinstance(sample, (str, Path)) else sample
        assert set(ss[id_col]) == set(sub[id_col]), "ids differ from the sample file"
        sub = sub.set_index(id_col).loc[ss[id_col]].reset_index()[list(ss.columns)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(path, index=False) if path.suffix == ".parquet" else sub.to_csv(path, index=False)
    return check(path, sample=sample, id_col=id_col, target_col=target_col, allowed=allowed,
                 min_classes=min_classes)


def check(path, sample=None, id_col="id", target_col="model", allowed=None, min_classes=2):
    sub = _read(path)
    if sample is not None:
        ss = _read(sample) if isinstance(sample, (str, Path)) else sample
        assert list(sub.columns) == list(ss.columns), list(sub.columns)
        assert len(sub) == len(ss) and (sub[id_col].values == ss[id_col].values).all(), "id order mismatch"
    assert sub[target_col].notna().all(), "missing labels"
    if allowed is not None:
        bad = set(sub[target_col]) - set(allowed)
        assert not bad, f"labels outside the allowed set: {sorted(bad)[:5]}"
    counts = sub[target_col].value_counts()
    assert len(counts) >= min_classes, f"only {len(counts)} distinct labels predicted"
    print(f"{Path(path).name}: format OK ({len(sub)} rows, {len(counts)} distinct labels; most / least predicted "
          f"{counts.index[0]} {counts.iloc[0]} / {counts.index[-1]} {counts.iloc[-1]})")
    return counts


def _selftest():
    import tempfile
    rng = np.random.default_rng(8)
    classes = [f"llm_{i:02d}" for i in range(18)]
    n = 300
    ids = np.arange(1000, 1000 + n)
    sample = pd.DataFrame({"id": rng.permutation(ids), "model": classes[0]})
    P = rng.dirichlet(np.ones(18), n)
    with tempfile.TemporaryDirectory() as tmp:
        counts = write_submission(Path(tmp) / "submission.parquet", ids, labels_from_probs(P, classes),
                                  sample=sample, allowed=classes, min_classes=10)
        back = pd.read_parquet(Path(tmp) / "submission.parquet")
        assert (back["id"].values == sample["id"].values).all() and counts.sum() == n
        lab = dict(zip(ids, labels_from_probs(P, classes)))
        assert all(lab[i] == m for i, m in zip(back["id"], back["model"]))
        for bad_labels, msg in ((np.array(["gpt-x"] * n, dtype=object), "allowed"),
                                (np.array([classes[3]] * n, dtype=object), "distinct")):
            try:
                write_submission(Path(tmp) / "bad.parquet", ids, bad_labels, allowed=classes, min_classes=10)
                raise RuntimeError("bad submission accepted")
            except AssertionError as e:
                assert msg in str(e), e
    print("submission selftest OK: parquet in sample order, label mapping kept, bad label sets rejected")


if __name__ == "__main__":
    _selftest()
