"""Primal counterparts of the per-fold Gram machinery (emb_kernel.py) for training sets too large for Gram matrices:
official rows + thousands of external rows (the public passages). The same standardized layer blocks, as explicit
feature matrices: a block spec 'name[@layers][*weight]' (emb_raw_explore.parse_blocks) becomes the concatenation of the
selected layer slices, each multiplied by sqrt(weight), so that Z Z^T equals emb_kernel.combined_gram.

Per fold, the statistics come from the fold's official training rows (as emb_kernel.grams); held-out / test rows may be
standardized with their own statistics (test_std, emb_kernel.own_stats); external rows with the official training
statistics ('official') or their own ('own').
"""
from __future__ import annotations

import numpy as np

from emb_kernel import layer_info, own_stats, raw_matrix


def raw_blocks(spec, tr, te, ext=None):
    """(Xtr, Xte, Xe) float32 raw features of a block spec; Xe (external rows in ext.df order) is None without ext."""
    from emb_raw_explore import parse_blocks
    from train_emb import ext_feats
    parts_tr, parts_te, parts_e = [], [], []
    for name, ls, wt in parse_blocks(spec):
        layers, w = layer_info(name)
        cols = np.concatenate([np.arange(li * w, (li + 1) * w) for li in (ls if ls is not None else range(len(layers)))])
        X = raw_matrix(name, tr, te)
        f = np.float32(np.sqrt(wt))
        parts_tr.append(X[:len(tr), cols] * f)
        parts_te.append(X[len(tr):, cols] * f)
        if ext is not None:
            parts_e.append(ext_feats("emb", [name], ext)[:, cols] * f)
    cat = lambda p: np.hstack(p) if len(p) > 1 else p[0]
    return cat(parts_tr), cat(parts_te), (cat(parts_e) if ext is not None else None)


def train_stats(X):
    mu, sd = X.mean(0, dtype=np.float64), X.std(0, dtype=np.float64)
    sd[sd == 0] = 1.0
    return mu, sd


def scale(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def fold_matrices(Xtr, Xte, Xe, trn, val, test_std=False, ext_rows=None, ext_scale="official"):
    """Standardized (Z_trn, Z_val, Z_te, Z_e) of one fold: statistics of the official training rows trn; val / test
    with their own statistics under test_std; external rows ext_rows with the training statistics or their own."""
    mu, sd = train_stats(Xtr[trn])
    own = (lambda X: own_stats(X, mu, sd)) if test_std else (lambda X: (mu, sd))
    Z_val = scale(Xtr[val], *own(Xtr[val]))
    Z_te = scale(Xte, *own(Xte))
    Z_e = None
    if Xe is not None and ext_rows is not None:
        Z_e = scale(Xe[ext_rows], *(train_stats(Xe[ext_rows]) if ext_scale == "own" else (mu, sd)))
    return scale(Xtr[trn], mu, sd), Z_val, Z_te, Z_e


def group_center(Z, groups):
    """Each row minus the mean of its group's rows."""
    Zc = Z.copy()
    for g in np.unique(groups):
        ix = groups == g
        Zc[ix] -= Zc[ix].mean(0)
    return Zc
