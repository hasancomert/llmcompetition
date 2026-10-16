"""Topic removal that does not need test clusters: nuisance attribute projection (NAP).

Prompt centering helped in CV (+0.15) but HURT on the hidden test (LB ablation, EXPERIMENTS.md): the test has few
texts per prompt, so its clusters cannot supply prompt means. NAP learns the between-prompt ("topic") subspace from
the TRAINING prompts only (top-k right singular vectors of the training-fold prompt-mean matrix, true prompt_id) and
projects it out of every text on its own. Held-out rows are transformed one by one, so the CV score does not depend
on how the hidden test is grouped.

Usage: python nap_probe.py --kind emb --names qwen3_8b_base --C 0.001 --k 16,64,119 [--log]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, encode_y, load_folds, load_train, log_experiment, metrics, scores
from train_emb import load_feats, make_model


def nap_basis(Z, prompts, k):
    """Top-k directions of the between-prompt scatter (rows of the returned (k, d) matrix are orthonormal)."""
    ps = np.unique(prompts)
    M = np.stack([Z[prompts == p].mean(0) for p in ps])
    M = M - M.mean(0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:min(k, len(ps) - 1)]


def nap_fold_fit(kind, C, X, y, prompts, folds, k, train_prompt_center=False):
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros((len(y), K))
    for f in range(int(folds.max()) + 1):
        trn, val = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        sc = StandardScaler().fit(X[trn])
        Z = sc.transform(X).astype(np.float32)
        V = nap_basis(Z[trn], prompts[trn], k) if k > 0 else np.zeros((0, Z.shape[1]), np.float32)
        Zp = Z - (Z @ V.T) @ V
        Ztr = Zp[trn]
        if train_prompt_center:                      # training rows: exact within-prompt deviations
            for p in np.unique(prompts[trn]):
                ix = prompts[trn] == p
                Ztr[ix] -= Ztr[ix].mean(0)
        m = make_model(kind, C).fit(Ztr, y[trn])
        oof[np.ix_(val, m.classes_)] = m.predict_proba(Zp[val])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="emb", choices=["emb", "lp"])
    ap.add_argument("--names", required=True)
    ap.add_argument("--C", type=float, default=0.001)
    ap.add_argument("--k", default="0,16,64,119")
    ap.add_argument("--train-prompt-center", action="store_true")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    prompts = tr["prompt_id"].to_numpy()
    names = a.names.split(",")
    X = np.hstack([m[0] for m in load_feats(a.kind, names, "train")])
    for k in map(int, a.k.split(",")):
        t0 = time.time()
        oof = nap_fold_fit(a.kind, a.C, X, y, prompts, folds, k, a.train_prompt_center)
        s = scores(y, oof, folds)
        print(f"{'+'.join(names)} NAP k={k}{' +train prompt-centering' if a.train_prompt_center else ''}: "
              f"BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) log loss {s['logloss']:.4f}"
              f"  {time.time() - t0:.0f}s", flush=True)
        if a.log:
            log_experiment(f"nap_{a.kind}_{'+'.join(names)}_k{k}{'_tpc' if a.train_prompt_center else ''}_C{a.C:g}",
                           f"NAP probe: standardize, project out top-{k} between-prompt directions (training-fold prompt "
                           f"means), LogReg C={a.C:g}; held-out rows transformed one by one (no test clustering)"
                           + ("; training rows also prompt-centered (true prompt_id)" if a.train_prompt_center else ""),
                           s, time.time() - t0, note="probe, not saved as a member")


if __name__ == "__main__":
    main()
