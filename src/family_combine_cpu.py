"""Family combine (family_combine.py) on an arbitrary base member set, e.g. a CPU-only base with CPU-only (TF-IDF)
family specialists. Same nested procedure as family_combine.py: for every outer fold k, using ONLY the other four
folds: hill-climb blend weights (logprob, log loss) of the base members -> B_k; per-family (or shared) combiner
of log B_k with the family specialists' conditional log-probs; applied to the held-out fold one row at a time.
Test: mean fold weights / mean combiner parameters. Nothing clusters, centers or matches held-out / test texts.

FAMSPEC_GROUPS=super selects super-family specialists (sup_*), as in family_spec.py.
Usage: python family_combine_cpu.py --base tfidf_char_svc_C0.3 ... --specs ng_char+word+fw_C30 [--per-family]
                                    [--ref blend_v2_tok_ba] [--log NAME] [--save TAG] [--note TEXT]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import encode_y, iter_folds, load_folds, load_member, load_train, log_experiment, metrics, run_member, scores
from compare import fmt, paired_bootstrap
from family_combine import apply_family, fit_combiner, lg, load_spec
from family_spec import FAMS, GROUPING, MULTI

import hill_climb  # noqa: E402  (reuse/airline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--per-family", action="store_true")
    ap.add_argument("--ref", default="blend_v2_tok_ba")
    ap.add_argument("--log", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    members = [load_member(t) for t in a.base]
    m_oof, m_test = [m[0] for m in members], [m[1] for m in members]
    specs = [load_spec(s) for s in a.specs]
    LS_oof, LS_test = [lg(s[0]) for s in specs], [lg(s[1]) for s in specs]
    n, k = m_oof[0].shape
    out, base_nested = np.zeros((n, k)), np.zeros((n, k))
    fold_w, fold_params = [], []
    for _, trn, val in iter_folds(folds):
        _, w = hill_climb.hill_climb(m_oof, y, idx=trn, space="logprob", metric=metrics.mlogloss,
                                     higher_is_better=False)
        fold_w.append(w)
        B = hill_climb.blend(m_oof, w, "logprob")
        base_nested[val] = B[val]
        params = fit_combiner(B, LS_oof, y, trn, a.per_family, None)
        fold_params.append(params)
        out[val] = apply_family(B, LS_oof, params)[val]
    Bt = hill_climb.blend(m_test, np.mean(fold_w, 0), "logprob")
    avg = {f: np.mean([p[f] for p in fold_params], 0) for f in MULTI}
    Pt = apply_family(Bt, LS_test, avg)
    sb, s = scores(y, base_nested, folds), scores(y, out, folds)
    ref, _ = load_member(a.ref)
    cmp_ref = fmt(paired_bootstrap(y, ref, out, g))
    cmp_base = fmt(paired_bootstrap(y, base_nested, out, g))
    w_mean = {t: round(float(x), 3) for t, x in zip(a.base, np.mean(fold_w, 0)) if x > 0}
    print(f"base nested blend: BA {sb['ba']:.4f} ll {sb['logloss']:.4f}; weights {w_mean}")
    print(f"result: BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) ll {s['logloss']:.4f}")
    print(f"  paired vs {a.ref}: {cmp_ref}\n  paired vs nested base blend: {cmp_base}")
    for f in MULTI:
        print(f"  {FAMS[f]} params (mean of folds): {np.round(avg[f], 3).tolist()}")
    desc = (f"family combine{' (super-families)' if GROUPING != 'lab' else ''} on base {', '.join(a.base)}; "
            f"specialists {', '.join(a.specs)}{'; per-family' if a.per_family else ''} (family_combine_cpu.py)")
    note = (f"{a.note}; nested base blend BA {sb['ba']:.4f} ll {sb['logloss']:.4f}; paired vs {a.ref}: {cmp_ref}; "
            f"vs nested base blend: {cmp_base}").lstrip("; ")
    if a.save:
        run_member(a.save, desc, out, Pt, y, folds, t0, note=note, members=a.base, specs=a.specs,
                   per_family=a.per_family, grouping=GROUPING, blend_weights=[float(v) for v in np.mean(fold_w, 0)])
    elif a.log:
        log_experiment(a.log, desc, s, time.time() - t0, note=note + "; not saved")


if __name__ == "__main__":
    main()
