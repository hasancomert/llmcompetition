"""Two-stage hierarchical combination (per-text, fully nested): super-family specialists, then lab specialists.

Per outer fold k, on the rows of the other four folds only: hill-climb blend weights (as family_combine.py) -> B;
stage 1: super-family specialists ({claude, glm}, {gpt, kimi, qwen}, {gemini}; sup_* outputs of family_spec.py),
per-family log-linear weights -> P1 (super-family masses of B kept); stage 2: lab specialists (claude, gemini, glm,
gpt) on top of P1, per-family weights -> P (lab masses of P1 kept). Applied to fold k row by row; test uses the
mean of the fold parameters. Nothing clusters, centers or matches held-out / test texts.

Usage: python family_chain.py --specs1 sup_a sup_b --specs2 lab_a lab_b [--extra-members ng_...] [--log N | --save T]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import family_combine as fc
from common import CLASSES, encode_y, iter_folds, lab_of, load_folds, load_member, load_train, log_experiment, \
    metrics, run_member, scores
from compare import fmt, paired_bootstrap
from family_spec import SUPER

import hill_climb  # noqa: E402  (reuse/airline)


def grouping(kind):
    labs = np.array([lab_of(c) if kind == "lab" else SUPER[lab_of(c)] for c in CLASSES])
    fams = sorted(set(labs))
    foc = np.array([fams.index(l) for l in labs])
    multi = [f for f in range(len(fams)) if (foc == f).sum() > 1]
    return fams, foc, multi


def use(kind):
    fc.FAMS, fc.FAM_OF_CLASS, fc.MULTI = grouping(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs1", nargs="+", required=True)
    ap.add_argument("--specs2", nargs="+", required=True)
    ap.add_argument("--extra-members", nargs="*", default=[])
    ap.add_argument("--log", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    members = [load_member(t) for t in fc.BASE + a.extra_members]
    m_oof, m_test = [m[0] for m in members], [m[1] for m in members]
    s1 = [fc.load_spec(s) for s in a.specs1]
    s2 = [fc.load_spec(s) for s in a.specs2]
    L1o, L1t = [fc.lg(s[0]) for s in s1], [fc.lg(s[1]) for s in s1]
    L2o, L2t = [fc.lg(s[0]) for s in s2], [fc.lg(s[1]) for s in s2]
    n, K = len(y), len(CLASSES)
    out, base_nested = np.zeros((n, K)), np.zeros((n, K))
    fw, p1s, p2s = [], [], []
    for _, trn, val in iter_folds(folds):
        _, w = hill_climb.hill_climb(m_oof, y, idx=trn, space="logprob", metric=metrics.mlogloss,
                                     higher_is_better=False)
        fw.append(w)
        B = hill_climb.blend(m_oof, w, "logprob")
        base_nested[val] = B[val]
        use("super")
        p1 = fc.fit_combiner(B, L1o, y, trn, True, None)
        P1 = fc.apply_family(B, L1o, p1)
        use("lab")
        p2 = fc.fit_combiner(P1, L2o, y, trn, True, None)
        P = fc.apply_family(P1, L2o, p2)
        p1s.append(p1)
        p2s.append(p2)
        out[val] = P[val]
    Bt = hill_climb.blend(m_test, np.mean(fw, 0), "logprob")
    use("super")
    Pt = fc.apply_family(Bt, L1t, {f: np.mean([p[f] for p in p1s], 0) for f in fc.MULTI})
    use("lab")
    Pt = fc.apply_family(Pt, L2t, {f: np.mean([p[f] for p in p2s], 0) for f in fc.MULTI})
    ref, _ = load_member(fc.REF)
    s = scores(y, out, folds)
    cmp = fmt(paired_bootstrap(y, ref, out, g))
    cmp_base = fmt(paired_bootstrap(y, base_nested, out, g))
    print(f"result: BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) ll {s['logloss']:.4f}")
    print(f"  paired vs {fc.REF}: {cmp}\n  paired vs recomputed base blend: {cmp_base}")
    desc = (f"two-stage family combine (super-family then lab specialists, per-family weights): base "
            f"{'+'.join(['blend_v3 members'] + a.extra_members)}; stage 1 {', '.join(a.specs1)}; stage 2 "
            f"{', '.join(a.specs2)}")
    note = f"{a.note}; paired vs {fc.REF}: {cmp}; vs nested base blend: {cmp_base}".lstrip("; ")
    if a.save:
        run_member(a.save, desc, out, Pt, y, folds, t0, note=note, members=fc.BASE + a.extra_members,
                   specs1=a.specs1, specs2=a.specs2, blend_weights=[float(v) for v in np.mean(fw, 0)])
    elif a.log:
        log_experiment(a.log, desc, s, time.time() - t0, note=note + "; not saved")


if __name__ == "__main__":
    main()
