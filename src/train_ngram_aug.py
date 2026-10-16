"""Per-text n-gram LR member trained on the training fold + sentence-window crops of the training-fold texts.

Same streams / vectorizers / LogisticRegression / inner temperature as train_ngram.py (imported, unchanged), plus
training-set augmentation (augment.py): for every training-fold text up to --crops distinct contiguous sentence
windows (fraction --lo..--hi of its sentences, >= --min-words words, --mode window|prefix), labelled with the
source model and weighted --crop-weight in the LR loss. Validation and test rows are never augmented or changed.

Leakage rules: crops are made only from rows of the training fold (their source row decides membership); the
vectorizers are fitted on the original training-fold texts only (--vec-fit orig, default) or on originals + crops
(--vec-fit all); the inner temperature split is the same StratifiedGroupKFold(4, random_state=7) over the original
training rows as train_ngram.py, its inner-train part gets the crops of its own rows only and the inner held-out
part is original texts only. Crops are seeded per text id (deterministic for replay, any test size).

Examples:
  python train_ngram_aug.py --feats char+word+fw --C 100 --crops 0            # = ng_char-word-fw_lr_C100 (check)
  python train_ngram_aug.py --feats char+word+fw --C 100 --crops 1 --crop-weight 0.5
  python train_ngram_aug.py --feats char+word+fw --C 100 --crops 2 --crop-weight 0.5 --temp --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from augment import crop_stats, make_crops
from common import (K, OUT, encode_y, iter_folds, load_folds, load_member, load_test, load_train, log_experiment,
                    run_member, scores)
from compare import fmt, paired_bootstrap
from train_ngram import STREAMS, make_vectorizer, transformed_texts
from train_tfidf import fit_temperature, softmax

SCRATCH = OUT / "aug_scratch"
REF_TAG = "ng_char-word-fw_lr_C100_T"


def fit_lr(args, A, y, sw):
    m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-4).fit(A, y, sample_weight=sw)
    return lambda X: m.predict_log_proba(X)


def run_fold(args, T_tr, y_tr, g_tr, T_cr, y_cr, src_cr, T_va, T_te):
    """T_*: dict transform -> texts. src_cr: position (in the training fold) of each crop's source row."""
    b_tr, b_cr, b_va, b_te = [], [], [], []
    for f in args.feats.split("+"):
        tname = STREAMS[f][0]
        fit_texts = T_tr[tname] + (T_cr[tname] if args.vec_fit == "all" else [])
        v = make_vectorizer(f, args).fit(fit_texts)
        b_tr.append(v.transform(T_tr[tname]))
        b_cr.append(v.transform(T_cr[tname]) if len(y_cr) else None)
        b_va.append(v.transform(T_va[tname]))
        b_te.append(v.transform(T_te[tname]))
    A_tr, A_va, A_te = (sp.hstack(b).tocsr() for b in (b_tr, b_va, b_te))
    A_cr = sp.hstack(b_cr).tocsr() if len(y_cr) else None

    def train_set(rows):
        """original rows `rows` (positions in the training fold) + the crops of those rows."""
        if A_cr is None:
            return A_tr[rows], y_tr[rows], np.ones(len(rows))
        keep = np.flatnonzero(np.isin(src_cr, rows))
        A = sp.vstack([A_tr[rows], A_cr[keep]]).tocsr()
        yy = np.concatenate([y_tr[rows], y_cr[keep]])
        sw = np.concatenate([np.ones(len(rows)), np.full(len(keep), args.crop_weight)])
        if args.balance == "text":              # every source text (original + its crops) has total weight 1
            m = np.bincount(src_cr[keep], minlength=len(y_tr)).astype(float)
            d = 1.0 + args.crop_weight * m
            sw = np.concatenate([1.0 / d[rows], args.crop_weight / d[src_cr[keep]]])
        if args.renorm:                         # keep the total weight of the loss = number of original rows
            sw *= len(rows) / sw.sum()
        return A, yy, sw

    T = 1.0
    if args.temp:
        dec_in = np.zeros((len(y_tr), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A_tr, y_tr, g_tr):
            dec_in[iva] = fit_lr(args, *train_set(itr))(A_tr[iva])
        T = fit_temperature(dec_in, y_tr)
    f = fit_lr(args, *train_set(np.arange(len(y_tr))))
    return softmax(f(A_va) * T), softmax(f(A_te) * T), T


def default_tag(args) -> str:
    t = f"ngaug_{args.feats.replace('+', '-')}_lr_C{args.C:g}_k{args.crops}"
    if args.crops:
        t += f"_w{args.crop_weight:g}_{args.mode}"
        if (args.lo, args.hi) != (0.5, 0.9):
            t += f"_r{args.lo:g}-{args.hi:g}"
        if args.min_words != 40:
            t += f"_mw{args.min_words}"
        if args.vec_fit != "orig":
            t += "_vall"
        if args.renorm:
            t += "_rn"
        if args.balance != "none":
            t += f"_b{args.balance}"
        if args.seed:
            t += f"_s{args.seed}"
    return t + ("_T" if args.temp else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="char+word+fw")
    ap.add_argument("--C", type=float, default=100.0)
    ap.add_argument("--temp", action="store_true", help="inner-split temperature on the log-probabilities")
    ap.add_argument("--crops", type=int, default=1, help="crops per training text (0 = no augmentation)")
    ap.add_argument("--crop-weight", type=float, default=1.0)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--mode", default="window", choices=["window", "prefix"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vec-fit", default="orig", choices=["orig", "all"])
    ap.add_argument("--renorm", action="store_true")
    ap.add_argument("--balance", default="none", choices=["none", "text"],
                    help="text: original weight 1/(1+w*m), each of its m crops w/(1+w*m) (total 1 per source text)")
    # vectorizer options used by train_ngram.make_vectorizer (defaults = the reference member)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--no-sublinear", action="store_true")
    ap.add_argument("--no-idf", action="store_true")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    groups = tr["prompt_id"].to_numpy()
    tag = args.tag or default_tag(args)
    src, crops = make_crops(tr["id"].tolist(), tr["text"].tolist(), args.crops, args.lo, args.hi, args.min_words,
                            args.mode, args.seed)
    stats = crop_stats(tr["text"], src, crops)
    print(f"[{tag}] {stats}", flush=True)
    T_all = transformed_texts(args.feats, tr["text"].tolist())
    T_crop = transformed_texts(args.feats, crops)
    T_te = transformed_texts(args.feats, te["text"].tolist())
    y_crop = y[src] if len(src) else np.zeros(0, dtype=np.int64)
    splits = list(iter_folds(folds))

    def take(d, idx):
        return {k: [v[i] for i in idx] for k, v in d.items()}

    jobs = []
    for _, trn, val in splits:
        ci = np.flatnonzero(np.isin(src, trn))            # crops whose source row is in the training fold
        pos = np.searchsorted(trn, src[ci])                # trn is sorted (np.flatnonzero in iter_folds)
        assert (trn[pos] == src[ci]).all()
        jobs.append(delayed(run_fold)(args, take(T_all, trn), y[trn], groups[trn], take(T_crop, ci), y_crop[ci],
                                      pos, take(T_all, val), T_te))
    res = Parallel(n_jobs=args.n_jobs)(jobs)
    oof = np.zeros((len(y), K))
    test = np.zeros((len(te), K))
    for (_, trn, val), (p_va, p_te, T) in zip(splits, res):
        oof[val] = p_va
        test += p_te / len(splits)
    temps = [round(r[2], 3) for r in res]
    s = scores(y, oof, folds)
    try:                                      # reference only for the log; absent in a fresh replay (replay.py)
        ref, _ = load_member(REF_TAG)
        cmp_txt = f"paired vs {REF_TAG}: " + fmt(paired_bootstrap(y, ref, oof, groups))
    except FileNotFoundError:
        cmp_txt = f"paired vs {REF_TAG}: n/a (reference member absent)"
    print(f"[{tag}] BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
          f"log loss {s['logloss']:.4f}; temps {temps}; {time.time() - t0:.0f}s\n   {cmp_txt}", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(SCRATCH / f"{tag}.npz", oof=oof, test=test, ba=s["ba"], logloss=s["logloss"])
    change = (f"per-text n-gram streams {args.feats} (min_df {args.min_df}) + LogReg C={args.C:g}"
              f"{' + inner temperature' if args.temp else ''}; training-fold augmentation: {args.crops} sentence-window "
              f"crop(s)/text ({args.mode}, {args.lo:g}-{args.hi:g} of sentences, >= {args.min_words} words, "
              f"weight {args.crop_weight:g}{', renormalised' if args.renorm else ''}"
              f"{', per-text total weight 1' if args.balance == 'text' else ''}, vectorizer fit on "
              f"{args.vec_fit}, seed {args.seed}) (train_ngram_aug.py)")
    note = "; ".join(x for x in [args.note, stats + " (all train rows)", cmp_txt,
                                 f"inner temps {temps}" if args.temp else ""] if x)
    if args.save:
        run_member(tag, change, oof, test, y, folds, t0, note=note, feats=args.feats, C=args.C, crops=args.crops,
                   crop_weight=args.crop_weight, lo=args.lo, hi=args.hi, min_words=args.min_words, mode=args.mode,
                   seed=args.seed, vec_fit=args.vec_fit, renorm=args.renorm, balance=args.balance, temps=temps,
                   script="train_ngram_aug.py")
    else:
        log_experiment(tag, change, s, time.time() - t0, note=note + "; not saved as a member")


if __name__ == "__main__":
    main()
