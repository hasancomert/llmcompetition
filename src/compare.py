"""Paired comparison of two OOF prediction matrices on the same folds (PLAN.md U3).

Balanced accuracy differences are judged by a paired bootstrap over prompts (the CV groups): resample the
150 prompts with replacement, recompute BA of A and B on the same rows, report the mean difference and a
95% interval, plus how many rows only A / only B gets right.
Usage: python compare.py <tagA> <tagB>
"""
from __future__ import annotations

import sys

import numpy as np

from common import encode_y, load_member, load_train, metrics


def paired_bootstrap(y, PA, PB, groups, n_boot=2000, seed=0) -> dict:
    y = np.asarray(y)
    pa, pb = np.asarray(PA).argmax(1), np.asarray(PB).argmax(1)
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    k = int(y.max()) + 1

    def ba(idx, pred):
        yy, pp = y[idx], pred[idx]
        rec = [np.mean(pp[yy == c] == c) for c in range(k) if np.any(yy == c)]
        return float(np.mean(rec))

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(ug), len(ug))])
        diffs[b] = ba(idx, pb) - ba(idx, pa)
    allr = np.arange(len(y))
    return dict(ba_a=ba(allr, pa), ba_b=ba(allr, pb), diff=ba(allr, pb) - ba(allr, pa),
                ci_lo=float(np.quantile(diffs, 0.025)), ci_hi=float(np.quantile(diffs, 0.975)),
                p_le0=float((diffs <= 0).mean()),
                only_a=int(((pa == y) & (pb != y)).sum()), only_b=int(((pb == y) & (pa != y)).sum()),
                ll_a=metrics.mlogloss(y, PA), ll_b=metrics.mlogloss(y, PB))


def fmt(r: dict) -> str:
    return (f"BA {r['ba_a']:.4f} -> {r['ba_b']:.4f} (diff {r['diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
            f"P(diff<=0) {r['p_le0']:.3f}); only A right {r['only_a']}, only B right {r['only_b']}; "
            f"log loss {r['ll_a']:.4f} -> {r['ll_b']:.4f}")


if __name__ == "__main__":
    tr = load_train()
    y = encode_y(tr["model"])
    A, _ = load_member(sys.argv[1])
    B, _ = load_member(sys.argv[2])
    print(f"{sys.argv[1]} vs {sys.argv[2]}: " + fmt(paired_bootstrap(y, A, B, tr["prompt_id"].values)))
