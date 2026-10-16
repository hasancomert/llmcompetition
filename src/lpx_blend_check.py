"""Does a train_lpx.py candidate help the per-text blends? Nested hill-climb (shared folds as meta folds, weights
chosen on the other 4 folds) of a base set vs (a) base with the lp+style reference member replaced by the candidate
and (b) base + candidate; paired bootstrap over prompts of the nested predictions. No test-time structure is used.

Candidates: saved member tags or outputs/lpx_scratch/<name>.npz.
Usage: python lpx_blend_check.py --base gpu --cand lpx_... [--log]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, scores
from compare import paired_bootstrap
from ngram_blend_check import BASES, nested
from train_lpx import REF, SCRATCH

BASES = dict(BASES)
# the gpu set plus the best saved per-text n-gram member (row 179 summary)
BASES["gpu+ng"] = BASES["gpu"] + ["ng_char-word-fw_lr_C100_T"]


def load_any(name):
    p = SCRATCH / f"{name}.npz"
    if p.exists():
        return np.load(p)["oof"]
    return load_member(name)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpu", choices=list(BASES))
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--metric", default="logloss", choices=["ba", "logloss"])
    ap.add_argument("--replace", default=REF, help="base member the candidate replaces (default: lp+style reference)")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    g = tr["prompt_id"].values
    meta = [(trn, val) for _, trn, val in iter_folds(folds)]
    base = BASES[args.base]
    oof_base = [load_any(t) for t in base]
    cands = [load_any(c) for c in args.cand]
    P0, w0 = nested(oof_base, y, meta, args.metric)
    print(f"base {args.base} ({args.metric}): BA {metrics.balanced_accuracy(y, P0):.4f} "
          f"ll {metrics.mlogloss(y, P0):.4f} w {np.round(w0, 2).tolist()}")
    variants = []
    if args.replace in base:
        i = base.index(args.replace)
        variants.append(("replace", oof_base[:i] + cands + oof_base[i + 1:], base[:i] + args.cand + base[i + 1:]))
    variants.append(("add", oof_base + cands, base + args.cand))
    out = []
    for name, oofs, tags in variants:
        P1, w1 = nested(oofs, y, meta, args.metric)
        r = paired_bootstrap(y, P0, P1, g)
        txt = (f"{name}: BA {r['ba_a']:.4f} -> {r['ba_b']:.4f} (diff {r['diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, "
               f"{r['ci_hi']:+.4f}]), log loss {r['ll_a']:.4f} -> {r['ll_b']:.4f}, weights "
               + ", ".join(f"{t.split('_')[0] if t in base else 'cand'}={w:.2f}" for t, w in zip(tags, w1)))
        print(txt, flush=True)
        out.append((name, P1, txt))
    if args.log:
        name, P1, _ = out[0]
        s = scores(y, P1, folds)
        log_experiment(f"lpxblend_{args.base}_{name}_{'+'.join(args.cand)}",
                       f"nested hill-climb ({args.metric}) of the per-text '{args.base}' base set, {args.replace} "
                       f"replaced by / added to {', '.join(args.cand)} (lpx_blend_check.py)", s, time.time() - t0,
                       note=" | ".join(o[2] for o in out) + "; blend check only, not saved")


if __name__ == "__main__":
    main()
