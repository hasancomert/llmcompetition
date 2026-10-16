"""Learning curve of the strongest single CPU member (ng_char-word-fw_lr_C100_T) on the official training data.

Per outer fold the member (train_ngram.run_fold, same settings as the saved member) is trained on a random fraction of
the training-fold prompts (whole prompts: all their answers stay together; seed per fold) and scored on the complete
held-out fold. fraction 1.0 reproduces the saved member.

Why: the fit test v2 CV check adds 414 generated rows (about 330 per training fold, ~15% of it); its confidence
intervals (+-0.01 BA) cannot resolve the gain such a small addition could bring. The slope of BA in the amount of
official data bounds what more data of official quality is worth for this member, i.e. what scaling the generated
data could bring at best.

Usage: python learning_curve.py --fracs 0.5,0.75,0.88 [--seeds 0] [--log]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from joblib import Parallel, delayed

from common import K, encode_y, load_folds, load_member, load_train, log_experiment, scores
from compare import fmt, paired_bootstrap
import fx_predict
import train_ngram

FEATS, C, SAVED = "char+word+fw", 100.0, "ng_char-word-fw_lr_C100_T"


def one_fold(args, T_all, y, g, folds, k, frac, seed):
    trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    if frac < 1.0:
        pids = np.unique(g[trn])
        keep = np.random.default_rng(seed * 100 + k).choice(pids, size=int(round(frac * len(pids))), replace=False)
        trn = trn[np.isin(g[trn], keep)]
    take = lambda idx: {t: [v[i] for i in idx] for t, v in T_all.items()}
    p_va, _, _ = train_ngram.run_fold(args, take(trn), y[trn], g[trn], take(val), take(val[:1]))
    return k, val, p_va, len(trn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.5,0.75,0.88")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].to_numpy()
    args = fx_predict.ng_args(FEATS, C)
    T_all = train_ngram.transformed_texts(FEATS, tr["text"].tolist())
    ref = load_member(SAVED)[0]
    s_ref = scores(y, ref, folds)
    print(f"saved member {SAVED}: BA {s_ref['ba']:.4f} log loss {s_ref['logloss']:.4f}", flush=True)
    for frac in [float(f) for f in a.fracs.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            oof, n_tr = np.zeros((len(y), K)), []
            for k, val, p, n in Parallel(n_jobs=a.n_jobs)(delayed(one_fold)(args, T_all, y, g, folds, k, frac, seed)
                                                          for k in range(int(folds.max()) + 1)):
                oof[val] = p
                n_tr.append(n)
            s = scores(y, oof, folds)
            r = paired_bootstrap(y, ref, oof, g)
            print(f"frac {frac:g} seed {seed}: {np.mean(n_tr):.0f} training rows per fold, BA {s['ba']:.4f} "
                  f"(folds {s['ba_fold_mean']:.4f} +- {s['ba_fold_std']:.4f}) log loss {s['logloss']:.4f}; "
                  f"vs full data: {fmt(r)}; {time.time() - t0:.0f}s", flush=True)
            if a.log:
                log_experiment(f"lcurve_ng_fw_f{frac:g}_s{seed}",
                               f"LEARNING CURVE (diagnostic, not a model): {SAVED} trained on {frac:g} of the "
                               f"training-fold prompts (whole prompts, seed {seed}; mean {np.mean(n_tr):.0f} rows per "
                               f"training fold), scored on the full held-out fold (learning_curve.py)",
                               s, time.time() - t0, note=f"vs the same member on all training rows: {fmt(r)}; not saved")


if __name__ == "__main__":
    main()
