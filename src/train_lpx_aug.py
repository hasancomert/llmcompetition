"""Per-text stylometric LR member (train_lpx.py style+style2 blocks, no LM) trained on the training fold +
sentence-window crops of the training-fold texts.

Features, column filtering (train_lpx.build_blocks on the original train/test rows), --drop, the head
(QuantileTransformer + StandardScaler + LogisticRegression) and the inner grouped temperature are those of
train_lpx.py (imported, unchanged). Augmentation (augment.py): for every training-fold text up to --crops distinct
contiguous sentence windows (--lo..--hi of its sentences, >= --min-words words, --mode window|prefix), labelled with
the source model and weighted --crop-weight in the LR loss (the quantile transform / scaler are fitted on originals
+ crops, or on originals only with --prep-fit orig). Crop features are the same pure per-text functions
(style_features / tokenizer_features / style2_features) applied to the crop text. Validation and test rows are
never augmented or changed; the inner temperature split is the StratifiedGroupKFold(4, random_state=7) of the
original training rows (inner-train gets only its own rows' crops, inner held-out = originals only). Crops are
seeded per text id (deterministic for replay, any test size).

Examples:
  python train_lpx_aug.py --crops 0 --temp                      # = lpx_style2_nolm_C0.003_T (check)
  python train_lpx_aug.py --crops 2 --crop-weight 0.5 --temp
  python train_lpx_aug.py --crops 2 --crop-weight 0.5 --temp --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from augment import crop_stats, make_crops
from common import K, OUT, encode_y, load_folds, load_member, load_test, load_train, log_experiment, run_member, \
    scores
from compare import fmt, paired_bootstrap
from train_lpx import apply_temp, build_blocks, fit_temperature, make_head, style2_cached

SCRATCH = OUT / "aug_scratch"
REF_TAG = "lpx_style2_nolm_C0.003_T"


def crop_features(crops: list[str], names: list[str], blocks: list[str]) -> np.ndarray:
    """Same columns as build_blocks (names) for the crop texts."""
    from train_gbdt import features
    s = pd.Series(crops)
    parts = []
    if "style" in blocks:
        F = pd.concat([features("style", s, "train"), features("tok", s, "train")], axis=1)   # cached by text hash
        parts.append(pd.DataFrame(F.to_numpy(np.float32), columns=["s_" + c for c in F.columns]))
    if "style2" in blocks:
        parts.append(style2_cached(s, "aug").add_prefix("s2_"))
    X = pd.concat(parts, axis=1)
    return X[names].to_numpy(np.float64)


def fit_apply(args, Xo, yo, Xc, yc, pos, X_apply):
    """Fit the head on originals Xo (+ crops Xc, weight args.crop_weight; pos[j] = row of Xo that crop j comes
    from); returns predicted probs for X_apply."""
    m = make_head("lr", args.C)
    if len(yc):
        X = np.vstack([Xo, Xc])
        y = np.concatenate([yo, yc])
        sw = np.concatenate([np.ones(len(yo)), np.full(len(yc), args.crop_weight)])
        if args.balance == "text":          # every source text (original + its crops) has total weight 1
            d = 1.0 + args.crop_weight * np.bincount(pos, minlength=len(yo))
            sw = np.concatenate([1.0 / d, args.crop_weight / d[pos]])
        if args.renorm:
            sw *= len(yo) / sw.sum()
    else:
        X, y, sw = Xo, yo, np.ones(len(yo))
    if args.prep_fit == "orig" and len(yc):
        prep = m[:-1].fit(Xo)                      # quantile transform + scaler on originals only
        lr = m[-1].fit(prep.transform(X), y, sample_weight=sw)
        out = []
        for A in X_apply:
            P = np.zeros((len(A), K))
            P[:, lr.classes_] = lr.predict_proba(prep.transform(A))
            out.append(P)
        return out
    m.fit(X, y, logisticregression__sample_weight=sw)
    out = []
    for A in X_apply:
        P = np.zeros((len(A), K))
        P[:, m.classes_] = m.predict_proba(A)
        out.append(P)
    return out


def default_tag(args) -> str:
    t = f"lpxaug_{args.blocks}_C{args.C:g}_k{args.crops}"
    if args.crops:
        t += f"_w{args.crop_weight:g}_{args.mode}"
        if (args.lo, args.hi) != (0.5, 0.9):
            t += f"_r{args.lo:g}-{args.hi:g}"
        if args.min_words != 40:
            t += f"_mw{args.min_words}"
        if args.prep_fit != "all":
            t += "_porig"
        if args.renorm:
            t += "_rn"
        if args.balance != "none":
            t += f"_b{args.balance}"
        if args.seed:
            t += f"_s{args.seed}"
    return t + ("_T" if args.temp else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="style+style2")
    ap.add_argument("--drop", default="s2_pb_")
    ap.add_argument("--C", type=float, default=0.003)
    ap.add_argument("--temp", action="store_true")
    ap.add_argument("--crops", type=int, default=1)
    ap.add_argument("--crop-weight", type=float, default=1.0)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--mode", default="window", choices=["window", "prefix"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prep-fit", default="all", choices=["all", "orig"])
    ap.add_argument("--renorm", action="store_true")
    ap.add_argument("--balance", default="none", choices=["none", "text"],
                    help="text: original weight 1/(1+w*m), each of its m crops w/(1+w*m) (total 1 per source text)")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    g = tr["prompt_id"].to_numpy()
    folds = load_folds(tr)
    blocks = args.blocks.split("+")
    assert not any(b in ("lp4", "glm", "cross") for b in blocks), "LM blocks have no features for crops"
    Xtr, Xte, names = build_blocks(blocks, tr, te)
    if args.drop:
        keep = [i for i, n in enumerate(names) if not n.startswith(tuple(args.drop.split(",")))]
        Xtr, Xte, names = Xtr[:, keep], Xte[:, keep], [names[i] for i in keep]
    tag = args.tag or default_tag(args)
    src, crops = make_crops(tr["id"].tolist(), tr["text"].tolist(), args.crops, args.lo, args.hi, args.min_words,
                            args.mode, args.seed)
    stats = crop_stats(tr["text"], src, crops)
    Xc = crop_features(crops, names, blocks) if len(crops) else np.zeros((0, len(names)))
    yc_all = y[src] if len(src) else np.zeros(0, dtype=np.int64)
    print(f"[{tag}] {len(names)} cols; {stats}; features {time.time() - t0:.0f}s", flush=True)
    oof, test = np.zeros((len(y), K)), np.zeros((len(Xte), K))
    n_f = int(folds.max()) + 1
    temps = []
    for k in range(n_f):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        ci = np.flatnonzero(np.isin(src, trn))          # crops whose source row is in the training fold
        src_loc = np.searchsorted(trn, src[ci])          # their source position inside the training fold
        Xo, yo, go = Xtr[trn], y[trn], g[trn]
        pv, pt = fit_apply(args, Xo, yo, Xc[ci], yc_all[ci], src_loc, [Xtr[val], Xte])
        T = 1.0
        if args.temp:
            inner = np.zeros((len(yo), K))
            for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(Xo, yo, go):
                sel = np.isin(src_loc, itr)
                cj = ci[sel]
                inner[iva] = fit_apply(args, Xo[itr], yo[itr], Xc[cj], yc_all[cj],
                                       np.searchsorted(itr, src_loc[sel]), [Xo[iva]])[0]
            T = fit_temperature(inner, yo)
            pv, pt = apply_temp(pv, T), apply_temp(pt, T)
        oof[val] = pv
        test += pt / n_f
        temps.append(round(T, 3))
    s = scores(y, oof, folds)
    try:                                      # reference only for the log; absent in a fresh replay (replay.py)
        ref, _ = load_member(REF_TAG)
        cmp_txt = f"paired vs {REF_TAG}: " + fmt(paired_bootstrap(y, ref, oof, g))
    except FileNotFoundError:
        cmp_txt = f"paired vs {REF_TAG}: n/a (reference member absent)"
    print(f"[{tag}] BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
          f"log loss {s['logloss']:.4f}; temps {temps}; {time.time() - t0:.0f}s\n   {cmp_txt}", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(SCRATCH / f"{tag}.npz", oof=oof, test=test, ba=s["ba"], logloss=s["logloss"])
    change = (f"per-text lr on {args.blocks} ({len(names)} cols, dropped {args.drop or 'nothing'}), "
              f"QuantileTransformer+scaler (fit on {args.prep_fit}), C={args.C:g}"
              f"{', inner grouped temperature' if args.temp else ''}; training-fold augmentation: {args.crops} "
              f"sentence-window crop(s)/text ({args.mode}, {args.lo:g}-{args.hi:g} of sentences, >= {args.min_words} "
              f"words, weight {args.crop_weight:g}{', renormalised' if args.renorm else ''}"
              f"{', per-text total weight 1' if args.balance == 'text' else ''}, seed {args.seed}) "
              f"(train_lpx_aug.py)")
    note = "; ".join(x for x in [args.note, stats + " (all train rows)", cmp_txt,
                                 f"temps {temps}" if args.temp else ""] if x)
    if args.save:
        run_member(tag, change, oof, test, y, folds, t0, note=note, blocks=args.blocks, drop=args.drop, C=args.C,
                   temp=args.temp, temps=temps, crops=args.crops, crop_weight=args.crop_weight, lo=args.lo,
                   hi=args.hi, min_words=args.min_words, mode=args.mode, seed=args.seed, prep_fit=args.prep_fit,
                   renorm=args.renorm, balance=args.balance, n_features=len(names), script="train_lpx_aug.py")
    else:
        log_experiment(tag, change, s, time.time() - t0, note=note + "; not saved")


if __name__ == "__main__":
    main()
