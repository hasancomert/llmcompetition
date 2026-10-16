"""Hard-group (sub-family) combiner on top of a saved per-text output, fully nested.

Base: a saved member's OOF / test predictions (default blend_pt_gpu_ba, itself a nested OOF). Groups G of
confusable models (subfam_spec.GROUPS) are processed in the order given. For every outer fold k (shared folds),
using ONLY the rows of the other four folds whose TRUE model is in G:
    z_m = a_G * log P(m) + sum_v w_vG * log S_v(m) [+ c_m]     q(m | G) = softmax over G of z
    P(m) <- P(G) * q(m | G)          (the group mass of P is kept; other columns unchanged)
with S_v = group specialists (subfam_spec.py grp_* files, or family_spec.py lab / super files: only the
log-ratios inside G matter) and optional per-class offsets c (L2 penalty). a, w, c minimise the conditional NLL of
the training rows of G (family_combine.fit_family). The fitted map is applied to the held-out fold row by row;
test uses the mean of the five fold parameters. Nothing clusters, centers or matches held-out / test texts.

Usage:
  python subfam_combine.py --group gem78=grp_gem78_emb_qwen3_8b_base_C0.001,grp_gem78_ng_char+word+fw_C30 \
                           --group glm=grp_glm_emb_qwen3_8b_base_C0.0003 [--offsets 1.0] [--base blend_pt_gpu_ba]
                           [--log NAME | --save TAG]
  (a group with no specialists, e.g. --group gem78=, only re-weights / offsets the base inside G)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, run_member, scores
from compare import fmt, paired_bootstrap
from family_combine import fit_family, lg, load_spec
from subfam_spec import group_cols, in_group


def apply_group(P, LS, cols, th):
    V = len(LS)
    a, w = th[0], th[1:1 + V]
    c = th[1 + V:] if len(th) > 1 + V else 0.0
    LB = lg(P[:, cols])
    z = a * LB + sum(wv * s[:, cols] for wv, s in zip(w, LS)) + c
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(1, keepdims=True)
    out = P.copy()
    out[:, cols] = P[:, cols].sum(1, keepdims=True) * q
    return out


def parse_groups(items):
    out = []
    for it in items:
        name, _, specs = it.partition("=")
        out.append((name, [s for s in specs.split(",") if s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="name=spec1,spec2 (repeatable, applied in order)")
    ap.add_argument("--base", default="blend_pt_gpu_ba")
    ap.add_argument("--offsets", type=float, default=None, help="L2 weight of per-class offsets (None: none)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--compare", default=None, help="also print the paired comparison with this member")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    B_oof, B_test = load_member(a.base)
    groups = parse_groups(a.group)
    specs = {name: [load_spec(s) for s in ss] for name, ss in groups}
    LS_oof = {name: [lg(s[0]) for s in v] for name, v in specs.items()}
    LS_test = {name: [lg(s[1]) for s in v] for name, v in specs.items()}
    out = np.zeros_like(B_oof)
    fold_params = {name: [] for name, _ in groups}
    for _, trn, val in iter_folds(folds):
        P = B_oof.copy()
        for name, _ in groups:
            cols = group_cols(name)
            r = trn[np.isin(y[trn], cols)]
            yloc = np.searchsorted(cols, y[r])
            th = fit_family(lg(P[np.ix_(r, cols)]), [s[np.ix_(r, cols)] for s in LS_oof[name]], yloc, len(cols),
                            a.offsets)
            fold_params[name].append(th)
            P = apply_group(P, LS_oof[name], cols, th)
        out[val] = P[val]
    Pt = B_test.copy()
    for name, _ in groups:
        Pt = apply_group(Pt, LS_test[name], group_cols(name), np.mean(fold_params[name], 0))
    s, sb = scores(y, out, folds), scores(y, B_oof, folds)
    cmp = fmt(paired_bootstrap(y, B_oof, out, g))
    print(f"result: BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) ll {s['logloss']:.4f}"
          f"   base {a.base}: BA {sb['ba']:.4f} ll {sb['logloss']:.4f}")
    print(f"  paired vs {a.base}: {cmp}")
    if a.compare:
        print(f"  paired vs {a.compare}: {fmt(paired_bootstrap(y, load_member(a.compare)[0], out, g))}")
    ig = []
    for name, ss in groups:
        cols = group_cols(name)
        b0, b1 = in_group(y, B_oof, cols)[0], in_group(y, out, cols)[0]
        ig.append(f"{name} {b0:.3f}->{b1:.3f}")
        if not a.quiet:
            print(f"  {name}: in-group BA {b0:.4f} -> {b1:.4f}; params (mean of folds) "
                  f"{np.round(np.mean(fold_params[name], 0), 3).tolist()}  specs {ss}")
    desc = (f"hard-group combiner on {a.base} (subfam_combine.py): "
            + "; ".join(f"{n}: {'+'.join(ss) if ss else 'no specialist'}" for n, ss in groups)
            + (f"; offsets L2 {a.offsets:g}" if a.offsets is not None else ""))
    note = f"{a.note}; in-group BA {', '.join(ig)}; paired vs {a.base}: {cmp}".lstrip("; ")
    if a.save:
        run_member(a.save, desc, out, Pt, y, folds, t0, note=note, base=a.base,
                   groups={n: ss for n, ss in groups}, offsets=a.offsets)
    elif a.log:
        log_experiment(a.log, desc, s, time.time() - t0, note=note + "; not saved")


if __name__ == "__main__":
    main()
