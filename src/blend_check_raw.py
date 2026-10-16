"""Does a candidate raw (per-text) member add to the raw GPU blend? Nested hill-climb (logprob, weights by log loss,
weights chosen on 4 folds and scored on the 5th, as blend_baselines.py) of the base raw members, with and without
the candidate(s); paired bootstrap over prompts. Candidates are saved member tags or OOF .npy files
(outputs/kgram/cand/*.npy written by emb_raw_explore.py). OOF only: nothing is saved.

Usage: python blend_check_raw.py --cands emb_x outputs/kgram/cand/embraw_y_C0.001.npy [--replace-emb] [--log]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, scores
from compare import fmt, paired_bootstrap

import hill_climb  # noqa: E402  (reuse/airline)

BASE = ["tfidf_charword_lr_C30", "tfidf_char_svc_C0.3",
        "lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1",
        "emb_qwen3_4b_base_C0.003", "emb_qwen3_8b_base_C0.003"]      # blend_v3_emb_logloss (0.4430 / 1.5775)


def load_oof(c):
    return np.load(c) if c.endswith(".npy") else load_member(c)[0]


def nested(oofs, y, folds):
    meta = [(trn, val) for _, trn, val in iter_folds(folds)]
    r = hill_climb.nested_hill(oofs, y, meta, avg_weights=True, space="logprob", metric=metrics.mlogloss,
                               higher_is_better=False)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", default=BASE)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--name", default="blendcheck_raw")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    base = [load_oof(t) for t in a.base]
    rb = nested(base, y, folds)
    cands = [load_oof(c) for c in a.cands]
    rc = nested(base + cands, y, folds)
    Pb, Pc = rb["nested_pred"], rc["nested_pred"]
    sb, sc = scores(y, Pb, folds), scores(y, Pc, folds)
    cmp = fmt(paired_bootstrap(y, Pb, Pc, g))
    names = [Path(c).stem if c.endswith(".npy") else c for c in a.cands]
    print(f"base blend: BA {sb['ba']:.4f} ll {sb['logloss']:.4f}; + {names}: BA {sc['ba']:.4f} ll {sc['logloss']:.4f}"
          f"; weights {np.round(rc['weights'], 3).tolist()}\n  paired: {cmp}", flush=True)
    if a.log:
        log_experiment(a.name, f"nested raw blend (blend_v3_emb_logloss members) + {', '.join(names)}", sc,
                       time.time() - t0, note=f"base {sb['ba']:.4f} / {sb['logloss']:.4f}; weights "
                       f"{np.round(rc['weights'], 3).tolist()}; paired vs base: {cmp}")


if __name__ == "__main__":
    main()
