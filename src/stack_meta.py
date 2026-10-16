"""Stacking check: multinomial LogReg on the members' OOF log-probabilities (nested by the shared folds).

Level-1 OOF rows of fold k come from models that never saw fold k; the meta model for fold k is fitted on the
other folds' OOF rows only, so the stacked OOF is honest in the same sense as the nested hill-climb.
Usage: python stack_meta.py --tags a b c --C 0.01,0.1,1 [--log --tag name]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, encode_y, iter_folds, load_folds, load_member, load_train, metrics, run_member


def stack(oofs, tests, y, folds, C):
    from sklearn.linear_model import LogisticRegression
    Z = np.hstack([np.log(np.clip(o, 1e-9, 1)) for o in oofs])
    Zt = np.hstack([np.log(np.clip(t, 1e-9, 1)) for t in tests])
    oof, test = np.zeros((len(y), K)), np.zeros((len(Zt), K))
    splits = list(iter_folds(folds))
    for _, trn, val in splits:
        m = LogisticRegression(C=C, max_iter=5000).fit(Z[trn], y[trn])
        oof[val] = m.predict_proba(Z[val])
        test += m.predict_proba(Zt) / len(splits)
    return oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--C", default="0.003,0.01,0.03,0.1,1")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    members = [load_member(t) for t in a.tags]
    oofs, tests = [m[0] for m in members], [m[1] for m in members]
    best = None
    for C in map(float, a.C.split(",")):
        oof, test = stack(oofs, tests, y, folds, C)
        ba, ll = metrics.balanced_accuracy(y, oof), metrics.mlogloss(y, oof)
        print(f"C={C:g}: stacked BA {ba:.4f}, log loss {ll:.4f}", flush=True)
        if best is None or ll < best[1]:
            best = (C, ll, oof, test)
    if a.log:
        C, _, oof, test = best
        run_member(a.tag or "stack_" + "+".join(a.tags), f"stacking: LogReg C={C:g} on OOF log-probs of "
                   f"{', '.join(a.tags)} (meta folds = shared folds)", oof, test, y, folds, t0,
                   note="C picked by OOF log loss over the grid (mild selection optimism)", members=a.tags, C=C)


if __name__ == "__main__":
    main()
