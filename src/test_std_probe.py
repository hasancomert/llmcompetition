"""Test-set standardization of frozen-LM features, probed on single LR members (CV, no labels used at test time).

Public SOTA notebooks standardize the LM features of the test set with the TEST set's own mean / std instead of the
training statistics (transductive; reported "never negative, +0.002..+0.01 depending on test size"). Here, per fold,
the held-out fold plays the test set: the training rows are standardized with their own statistics (as always), the
held-out rows either with the training statistics (baseline = train_emb / emb_kernel) or with statistics of the
held-out set itself. Variants:
  train : (x - mu_tr) / sd_tr                      (baseline)
  test  : (x - mu_te) / sd_te                      (public recipe)
  mean  : (x - mu_te) / sd_tr                      (only the mean shift)
Test-set statistics come from the evaluated set only: the whole held-out fold (432 texts, 24 prompts), or, with
--ranges, a random a..b texts per held-out prompt (smaller test sets, as in the structure sims).
Exact in Gram space (as emb_kernel; per-dimension standardization), LogReg C as the members (train_emb --C).

Usage: python test_std_probe.py --views qwen3_8b_base_l12,qwen3_4b_base_l12 --C 0.0003 [--ranges 1-2,3-8 --sims 5]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, encode_y, load_folds, load_train, metrics
from emb_kernel import raw_matrix
from structure_sim import keep_rows

VARIANTS = ("train", "test", "mean")


def standardize(X, rows):
    mu, sd = X[rows].mean(0), X[rows].std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def eval_grams(X, ev, mu, sd, Ztr):
    """K_ev,tr for every variant. Per-dimension standardization, so the sum of the per-layer block Grams of emb_kernel
    is the Gram of the whole standardized matrix."""
    mu_e, sd_e = standardize(X, ev)
    B = X[ev]
    return {"train": ((B - mu) / sd) @ Ztr.T, "test": ((B - mu_e) / sd_e) @ Ztr.T, "mean": ((B - mu_e) / sd) @ Ztr.T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="qwen3_8b_base_l12,qwen3_4b_base_l12")
    ap.add_argument("--C", type=float, default=0.0003)
    ap.add_argument("--ranges", default="", help="also test on a..b texts per held-out prompt (comma-separated a-b)")
    ap.add_argument("--sims", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    tr = load_train()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    prompts = tr["prompt_id"].to_numpy()
    n_f = int(folds.max()) + 1
    for name in a.views.split(","):
        X = raw_matrix(name)[:len(y)]
        rng = np.random.default_rng(a.seed)
        sets = [("full", None, 0)] + [(rg, tuple(int(x) for x in rg.split("-")), s)
                                      for rg in a.ranges.split(",") if rg for s in range(a.sims)]
        res = {}                                            # (set, sim) -> {variant: (y_true, pred)}
        for k in range(n_f):
            trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            mu, sd = standardize(X, trn)
            Ztr = (X[trn] - mu) / sd
            Ktr = (Ztr @ Ztr.T).astype(np.float64)
            lam, U = np.linalg.eigh((Ktr + Ktr.T) / 2)
            lam, U = lam[::-1], U[:, ::-1]
            keep = lam > 1e-10 * lam[0]
            lam, U = lam[keep], U[:, keep]
            m = LogisticRegression(C=a.C, max_iter=3000).fit(U * np.sqrt(lam), y[trn])
            for sname, ab, s in sets:
                ev = val if ab is None else val[keep_rows("perprompt", y[val], prompts[val], rng,
                                                          argparse.Namespace(a=ab[0], b=ab[1]))]
                Kev = eval_grams(X, ev, mu, sd, Ztr)
                for v in VARIANTS:
                    P = np.zeros((len(ev), K))
                    P[:, m.classes_] = m.predict_proba(Kev[v] @ (U / np.sqrt(lam)))
                    yt, pr = res.setdefault((sname, s), {}).setdefault(v, ([], []))
                    yt.append(y[ev])
                    pr.append(P)
            print(f"[{name}] fold {k} done ({time.time() - t0:.0f}s)", flush=True)
        out = {}
        for (sname, s), d in res.items():
            ba = {v: metrics.balanced_accuracy(np.concatenate(d[v][0]), np.vstack(d[v][1])) for v in VARIANTS}
            out.setdefault(sname, []).append(ba)
        for sname, lst in out.items():
            arr = {v: np.array([b[v] for b in lst]) for v in VARIANTS}
            msg = " ".join(f"{v} {arr[v].mean():.4f}" for v in VARIANTS)
            dif = " ".join(f"d_{v} {np.mean(arr[v] - arr['train']):+.4f}"
                           + (f" ± {np.std(arr[v] - arr['train'], ddof=1) / np.sqrt(len(lst)):.4f}"
                              if len(lst) > 1 else "") for v in VARIANTS[1:])
            print(f"[{name}] C {a.C:g} test set {sname} ({len(lst)} sims): {msg} | {dif}", flush=True)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
