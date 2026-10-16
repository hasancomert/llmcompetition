"""Does a new per-text n-gram member help a per-text blend? Nested hill-climb (shared folds as meta folds, weights
chosen on the other 4 folds) of a base set vs base + candidate(s), and base with tfidf_charword_lr_C30 replaced by the
candidate; paired bootstrap over prompts of the nested predictions. No test-time structure is used.

Candidates are saved member tags or names of outputs/ngram_scratch/<name>.npz (train_ngram.py runs).
Usage: python ngram_blend_check.py --base gpu --cand ng_char-word-fw_lr_C30 [--metric logloss] [--log]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, scores
from compare import paired_bootstrap
from train_ngram import REF_TAG, SCRATCH

import hill_climb  # noqa: E402  (reuse/airline, path set by common)

BASES = {
    # the per-text members of the GPU submission blend (blend_v3_emb_logloss, row 55)
    "gpu": ["tfidf_charword_lr_C30", "tfidf_char_svc_C0.3",
            "lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1",
            "emb_qwen3_4b_base_C0.003", "emb_qwen3_8b_base_C0.003"],
    # the CPU submission blend (blend_v2_tok_ba, row 38)
    "cpu": ["tfidf_charword_lr_C30", "tfidf_char_svc_C0.3", "tfidf_word_lr_C10", "lr_style_tok_v2",
            "lgb_style_tok_v2_s0", "lgb_style_tok_v2_s1", "lgb_style_tok_v2_s2"],
    "text": ["tfidf_charword_lr_C30", "tfidf_char_svc_C0.3", "tfidf_word_lr_C10"],
}


def load_any(name):
    p = SCRATCH / f"{name}.npz"
    if p.exists():
        return np.load(p)["oof"]
    return load_member(name)[0]


def nested(oofs, y, meta, metric):
    kw = dict(space="logprob")
    if metric == "logloss":
        kw.update(metric=metrics.mlogloss, higher_is_better=False)
    r = hill_climb.nested_hill(oofs, y, meta, avg_weights=True, **kw)
    return r["nested_pred"], r["weights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpu", choices=list(BASES))
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--metric", default="logloss", choices=["ba", "logloss"])
    ap.add_argument("--no-replace", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--label", default="")
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
    print(f"base {args.base} ({args.metric}): BA {metrics.balanced_accuracy(y, P0):.4f} ll {metrics.mlogloss(y, P0):.4f} "
          f"w {np.round(w0, 2).tolist()}")
    variants = [("add", oof_base + cands, base + args.cand)]
    if not args.no_replace and REF_TAG in base:
        i = base.index(REF_TAG)
        variants.append(("replace", oof_base[:i] + cands + oof_base[i + 1:], base[:i] + args.cand + base[i + 1:]))
    out = []
    for name, oofs, tags in variants:
        P1, w1 = nested(oofs, y, meta, args.metric)
        r = paired_bootstrap(y, P0, P1, g)
        txt = (f"{name}: BA {r['ba_a']:.4f} -> {r['ba_b']:.4f} (diff {r['diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, "
               f"{r['ci_hi']:+.4f}]), log loss {r['ll_a']:.4f} -> {r['ll_b']:.4f}, weights "
               + ", ".join(f"{t.split('_')[0] if t in base else t}={w:.2f}" for t, w in zip(tags, w1)))
        print(txt, flush=True)
        out.append((name, P1, txt))
    if args.log:
        name, P1, _ = out[0]
        s = scores(y, P1, folds)
        log_experiment(f"blendchk_{args.base}+{'+'.join(args.cand)}{args.label}",
                       f"nested hill-climb ({args.metric}) of the per-text '{args.base}' base set + candidate(s) "
                       f"{', '.join(args.cand)} (ngram_blend_check.py)", s, time.time() - t0,
                       note=" | ".join(o[2] for o in out) + "; blend check only, not saved")


if __name__ == "__main__":
    main()
