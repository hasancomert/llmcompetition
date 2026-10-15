"""Write outputs/folds.csv once: StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by prompt_id.

Every model reads this file (PLAN.md T1); re-running must reproduce it exactly (checked against the existing file).
"""
import hashlib

import numpy as np
import pandas as pd

from common import FOLD_SEED, FOLDS_CSV, N_FOLDS, cv_preds, encode_y, load_train

import oof_io  # reuse/airline: same split, (tr, va) form, asserts no group overlap


def main():
    tr = load_train()
    y = encode_y(tr["model"])
    g = tr["prompt_id"].to_numpy()
    folds = cv_preds.make_folds(y, groups=g, n_splits=N_FOLDS, seed=FOLD_SEED)
    # cross-check with the airline implementation (same sklearn call; must be identical)
    alt = oof_io.fold_ids(oof_io.make_folds(y, g, k=N_FOLDS, seed=FOLD_SEED), len(y))
    assert (alt == folds).all()
    for k in range(N_FOLDS):
        assert not set(g[folds == k]) & set(g[folds != k]), "prompt on both sides"
        assert len(np.unique(y[folds != k])) == len(np.unique(y)), "a class is missing from a training fold"
    out = pd.DataFrame({"id": tr["id"], "prompt_id": g, "model": tr["model"], "fold": folds})
    digest = hashlib.sha256(out.to_csv(index=False).encode()).hexdigest()[:16]
    if FOLDS_CSV.exists():
        old = pd.read_csv(FOLDS_CSV)
        assert old.equals(out), "existing folds.csv differs: refusing to overwrite"
        print(f"folds.csv unchanged (sha256 {digest})")
    else:
        FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(FOLDS_CSV, index=False)
        print(f"wrote {FOLDS_CSV} (sha256 {digest})")
    print(out.groupby("fold").agg(rows=("id", "size"), prompts=("prompt_id", "nunique")).to_string())
    print("class counts per validation fold (min, max):",
          [(int(np.bincount(y[folds == k], minlength=18).min()), int(np.bincount(y[folds == k], minlength=18).max()))
           for k in range(N_FOLDS)])


if __name__ == "__main__":
    main()
