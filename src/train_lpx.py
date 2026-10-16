"""Per-text stylometric + LM log-prob statistic members with cross-model features and new style features.

Every feature is a pure function of one text (LM statistics come from outputs/emb/lp_<name>_*.npy); the held-out
and test rows are classified one by one, no test-time structure. All fitted steps (quantile transform, scaler,
classifier, temperature) are fit on the training folds only.

Blocks (--blocks, '+'-separated):
  lp4     13 token log-prob statistics x 4 Qwen3 base models (n_tok kept once: same tokenizer)
  glm     the 13 statistics of glm4_9b
  cross   cross-model features: differences / ratios of the same statistic between models (8B-0.6B, 8B-1.7B,
          4B-0.6B, 8B-4B, 1.7B-0.6B, glm-8B), Binoculars-style ratios (log-ppl of one model / entropy of another),
          slope and curvature of each statistic over log model size, glm/qwen token-count ratio
  style   style+tok v2 columns (train_gbdt.build('style+tok'))
  style2  new hand-crafted features (style2_features.py)
Heads (--head): lr (QuantileTransformer + StandardScaler + LogReg C), et (ExtraTrees), svc (RBF SVC, Platt),
lgb (strongly regularised LightGBM, fixed rounds), each optionally followed by an inner grouped temperature (--temp).

Usage:
  python train_lpx.py --blocks lp4+style+cross --head lr --C 0.1            # scored + logged, not saved
  python train_lpx.py --blocks lp4+style+cross --head lr --C 0.1 --save     # saved as a member
  python train_lpx.py --blocks lp4+style+cross --head lr --Cs 0.03,0.1,0.3  # C grid (each one logged)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import K, OUT, encode_y, load_folds, load_member, load_test, load_train, log_experiment, metrics, \
    run_member, scores
from compare import fmt, paired_bootstrap

EMB = OUT / "emb"
SCRATCH = OUT / "lpx_scratch"
REF = "lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1"
QWEN = ["qwen3_0.6b_base", "qwen3_1.7b_base", "qwen3_4b_base", "qwen3_8b_base"]
SIZES = {"qwen3_0.6b_base": 0.6, "qwen3_1.7b_base": 1.7, "qwen3_4b_base": 4.0, "qwen3_8b_base": 8.0}
SHORT = {"qwen3_0.6b_base": "q06", "qwen3_1.7b_base": "q17", "qwen3_4b_base": "q4", "qwen3_8b_base": "q8",
         "glm4_9b": "glm"}


def lp_frame(name, split, ids):
    meta = json.loads((EMB / f"meta_{name}_{split}.json").read_text())
    assert meta["ids"] == list(ids), f"{name} {split}: rows not in csv order"
    X = np.load(EMB / f"lp_{name}_{split}.npy").astype(np.float64)
    return pd.DataFrame(X, columns=[f"{SHORT[name]}_{c}" for c in meta["lp_names"]])


STATS = ["lp_mean", "lp_std", "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "logrank_mean", "top1_rate",
         "ent_mean", "ent_std", "lp_minus_negent"]
PAIRS = [("q8", "q06"), ("q8", "q17"), ("q4", "q06"), ("q8", "q4"), ("q17", "q06"), ("glm", "q8"), ("glm", "q06")]


def cross_frame(L: pd.DataFrame) -> pd.DataFrame:
    """L holds the lp columns of the 4 Qwen models and glm (prefix_stat)."""
    out = {}
    for a, b in PAIRS:
        for s in STATS:
            out[f"x_{a}-{b}_{s}"] = L[f"{a}_{s}"] - L[f"{b}_{s}"]
        # log-perplexity ratio and Binoculars-style ratios (observed log-ppl of one model / expected of another)
        out[f"x_{a}/{b}_ppl"] = L[f"{a}_lp_mean"] / L[f"{b}_lp_mean"]
        out[f"x_{a}/{b}_bino"] = -L[f"{a}_lp_mean"] / L[f"{b}_ent_mean"]
        out[f"x_{b}/{a}_bino"] = -L[f"{b}_lp_mean"] / L[f"{a}_ent_mean"]
        out[f"x_{a}/{b}_rank"] = L[f"{a}_logrank_mean"] / L[f"{b}_logrank_mean"]
    for m in ["q06", "q17", "q4", "q8", "glm"]:
        out[f"x_{m}_bino_self"] = -L[f"{m}_lp_mean"] / L[f"{m}_ent_mean"]
        out[f"x_{m}_lrr"] = -L[f"{m}_lp_mean"] / L[f"{m}_logrank_mean"].clip(lower=1e-3)   # log-likelihood/log-rank
    # slope / curvature over log model size (Qwen family)
    ls = np.log([0.6, 1.7, 4.0, 8.0])
    ls = ls - ls.mean()
    q2 = ls ** 2 - (ls ** 2).mean()
    for s in STATS:
        M = np.stack([L[f"{m}_{s}"].to_numpy() for m in ["q06", "q17", "q4", "q8"]], 1)
        out[f"x_slope_{s}"] = M @ ls / (ls @ ls)
        out[f"x_curv_{s}"] = M @ q2 / (q2 @ q2)
    out["x_glm/qwen_ntok"] = np.log(L["glm_n_tok"] / L["q06_n_tok"])
    return pd.DataFrame(out).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def style2_cached(texts: pd.Series, split: str) -> pd.DataFrame:
    from style2_features import style2_frame
    h = hashlib.sha256(Path(__file__).with_name("style2_features.py").read_bytes())
    h.update("\x00".join(texts).encode("utf-8"))
    p = SCRATCH / f"style2_{split}_{h.hexdigest()[:12]}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    F = style2_frame(texts)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    F.to_parquet(p)
    return F


def build_blocks(blocks, tr, te):
    Xs = {"train": [], "test": []}
    for split, df in (("train", tr), ("test", te)):
        needs_lm = any(b in ("lp4", "glm", "cross") for b in blocks)      # style-only members need no LM files
        L = pd.concat([lp_frame(n, split, df["id"]) for n in QWEN + ["glm4_9b"]], axis=1) if needs_lm else None
        for b in blocks:
            if b == "lp4":
                cols = [c for c in L.columns if not c.startswith("glm") and (c.startswith("q06") or not c.endswith("n_tok"))]
                Xs[split].append(L[cols])
            elif b == "glm":
                Xs[split].append(L[[c for c in L.columns if c.startswith("glm")]])
            elif b == "cross":
                Xs[split].append(cross_frame(L))
            elif b == "style":
                from train_gbdt import build
                Str, Ste, names = build("style+tok", tr, te)
                Xs[split].append(pd.DataFrame(Str if split == "train" else Ste, columns=["s_" + n for n in names]))
            elif b == "style2":
                Xs[split].append(style2_cached(df["text"], split).add_prefix("s2_"))
            else:
                raise ValueError(b)
    Xtr, Xte = pd.concat(Xs["train"], axis=1), pd.concat(Xs["test"], axis=1)
    keep = [c for c in Xtr.columns if Xtr[c].nunique() > 1]
    return Xtr[keep].to_numpy(np.float64), Xte[keep].to_numpy(np.float64), keep


COLW = None     # optional per-column multipliers applied after standardisation (block weights, --bw)
PREFIX = {"lp4": ("q06_", "q17_", "q4_", "q8_"), "glm": ("glm_",), "cross": ("x_",), "style": ("s_",),
          "style2": ("s2_",)}


def block_weights(names, spec):
    """spec 'lp4=2,style2=0.5' -> multiplier per column (1 for unnamed blocks); a larger multiplier = weaker
    effective L2 penalty on that block."""
    w = np.ones(len(names))
    for item in spec.split(","):
        b, v = item.split("=")
        ix = [i for i, n in enumerate(names) if n.startswith(PREFIX[b]) and not (b == "style" and n.startswith("s2_"))]
        w[ix] = float(v)
    return w


def make_head(head, C, seed=0, n_jobs=2):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0)
    if head == "lr":
        from sklearn.linear_model import LogisticRegression
        if COLW is not None:
            from sklearn.preprocessing import FunctionTransformer
            w = COLW.copy()
            return make_pipeline(qt, StandardScaler(), FunctionTransformer(lambda X: X * w),
                                 LogisticRegression(C=C, max_iter=5000))
        return make_pipeline(qt, StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    if head == "gam":                       # additive per-feature splines on the quantile-normal scale + LogReg
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import SplineTransformer
        return make_pipeline(qt, SplineTransformer(n_knots=4, degree=2, knots="quantile", extrapolation="constant"),
                             StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    if head == "svc":
        from sklearn.svm import SVC
        return make_pipeline(qt, StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale", probability=True,
                                                       random_state=seed))
    if head == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=int(C), max_features="sqrt", n_jobs=n_jobs,
                                    random_state=seed, class_weight="balanced")
    if head == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="multiclass", learning_rate=0.03, n_estimators=int(C), num_leaves=7,
                                  min_child_samples=30, colsample_bytree=0.3, subsample=0.7, subsample_freq=1,
                                  reg_lambda=10.0, verbose=-1, n_jobs=n_jobs, random_state=seed)
    raise ValueError(head)


def fit_temperature(P, y):
    """Scalar temperature on log-probabilities minimising log loss (inner held-out rows only)."""
    from scipy.optimize import minimize_scalar
    L = np.log(np.clip(P, 1e-12, 1))

    def nll(lt):
        Z = L / np.exp(lt)
        Z = Z - Z.max(1, keepdims=True)
        Q = np.exp(Z)
        Q /= Q.sum(1, keepdims=True)
        return metrics.mlogloss(y, Q)
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temp(P, T):
    Z = np.log(np.clip(P, 1e-12, 1)) / T
    Z = Z - Z.max(1, keepdims=True)
    Q = np.exp(Z)
    return Q / Q.sum(1, keepdims=True)


def fit_predict(head, C, Xtr, ytr, gtr, X_apply, temp):
    m = make_head(head, C).fit(Xtr, ytr)
    Ps = []
    for X in X_apply:
        P = np.zeros((len(X), K))
        P[:, m.classes_] = m.predict_proba(X)
        Ps.append(P)
    T = 1.0
    if temp:
        from sklearn.model_selection import StratifiedGroupKFold
        inner = np.zeros((len(ytr), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(Xtr, ytr, gtr):
            mi = make_head(head, C).fit(Xtr[itr], ytr[itr])
            inner[np.ix_(iva, mi.classes_)] = mi.predict_proba(Xtr[iva])
        T = fit_temperature(inner, ytr)
        Ps = [apply_temp(P, T) for P in Ps]
    return Ps, T


def run_cv(head, C, Xtr, Xte, y, g, folds, temp=False):
    oof, test = np.zeros((len(y), K)), np.zeros((len(Xte), K))
    n_f = int(folds.max()) + 1
    temps = []
    for k in range(n_f):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        (pv, pt), T = fit_predict(head, C, Xtr[trn], y[trn], g[trn], [Xtr[val], Xte], temp)
        oof[val] = pv
        test += pt / n_f
        temps.append(T)
    return oof, test, temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="lp4+style")
    ap.add_argument("--head", default="lr", choices=["lr", "gam", "svc", "et", "lgb"])
    ap.add_argument("--C", type=float, default=0.1, help="LR/SVC C; et: min_samples_leaf; lgb: n_estimators")
    ap.add_argument("--Cs", default="", help="comma-separated grid (each value logged)")
    ap.add_argument("--temp", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--drop", default="", help="comma-separated column-name prefixes to drop, e.g. s2_pb_,s2_fw_")
    ap.add_argument("--bw", default="", help="block weights after standardisation, e.g. lp4=2,style2=0.5 (lr only)")
    args = ap.parse_args()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    g = tr["prompt_id"].to_numpy()
    folds = load_folds(tr)
    blocks = args.blocks.split("+")
    Xtr, Xte, names = build_blocks(blocks, tr, te)
    if args.drop:
        keep = [i for i, n in enumerate(names) if not n.startswith(tuple(args.drop.split(",")))]
        Xtr, Xte, names = Xtr[:, keep], Xte[:, keep], [names[i] for i in keep]
    if args.bw:
        global COLW
        COLW = block_weights(names, args.bw)
    print(f"blocks {args.blocks}: {Xtr.shape[1]} cols", flush=True)
    try:                                      # reference only for the log; absent in a fresh replay (replay.py)
        ref_oof, _ = load_member(REF)
    except FileNotFoundError:
        ref_oof = None
    Cs = [float(c) for c in args.Cs.split(",")] if args.Cs else [args.C]
    for C in Cs:
        t0 = time.time()
        oof, test, temps = run_cv(args.head, C, Xtr, Xte, y, g, folds, args.temp)
        tag = args.tag or (f"lpx_{args.blocks}_{args.head}_C{C:g}{'_T' if args.temp else ''}"
                           + (f"_bw{args.bw.replace('=', '').replace(',', '_')}" if args.bw else "")
                           + (f"_drop{args.drop.replace(',', '_')}" if args.drop else ""))
        r = paired_bootstrap(y, ref_oof, oof, g) if ref_oof is not None else None
        change = (f"per-text {args.head} on {args.blocks} ({Xtr.shape[1]} cols)"
                  f"{', QuantileTransformer+scaler' if args.head in ('lr', 'gam', 'svc') else ''}, C/param={C:g}"
                  f"{', inner grouped temperature' if args.temp else ''}"
                  f"{', block weights ' + args.bw if args.bw else ''}"
                  f"{', dropped column prefixes ' + args.drop if args.drop else ''} (train_lpx.py)")
        note = (f"paired vs {REF}: {fmt(r) if r is not None else 'n/a (reference member absent)'}"
                + (f"; temps {np.round(temps, 3).tolist()}" if args.temp else "")
                + (f"; {args.note}" if args.note else ""))
        if args.save:
            run_member(tag, change, oof, test, y, folds, t0, note=note, blocks=args.blocks, head=args.head, C=C,
                       temp=args.temp, temps=temps, n_features=len(names))
        else:
            s = scores(y, oof, folds)
            log_experiment(tag, change, s, time.time() - t0, note=note + "; not saved")
            SCRATCH.mkdir(parents=True, exist_ok=True)
            np.savez(SCRATCH / f"{tag}.npz", oof=oof, test=test)
            print(f"[{tag}] BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
                  f"log loss {s['logloss']:.4f}  {time.time() - t0:.0f}s", flush=True)
        print("   " + (fmt(r) if r is not None else "paired: n/a (reference member absent)"), flush=True)


if __name__ == "__main__":
    main()
