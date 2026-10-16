"""Family combiner (family_combine.py procedure) with a selectable grouping, plus its effect inside the per-text
blend, fully nested.

Groupings: lab and super (as family_spec.py / family_combine.py) and finer partitions built from subfam_spec.GROUPS
(FINE below: listed groups are families, every other model is a family of its own). The combiner is exactly
family_combine.py's (per outer fold: log-loss hill-climb of the base members on the other four folds -> B_k; per
family log-linear weights of log B_k and the log specialists fit on that family's training rows; family masses of
B_k kept; applied to the held-out fold row by row; test = mean fold parameters), with the grouping swapped in the
family_combine module as family_chain.py does. Specialist files: family_spec.py (lab / super) or subfam_spec.py
(grp_*; any family that contains the group).

--blend-check: nested hill-climb (logprob, weights by BA, as blend_pt_gpu_ba) of the blend_pt gpu_noglm candidate
set plus this combiner's OOF (computed in-process, not saved), paired bootstrap vs blend_pt_gpu_ba.
--save TAG saves the combiner as a member (run_member), --log NAME logs it without saving.

Usage: python subfam_famc.py --grouping fine1 --specs grp_c48sf_emb_qwen3_8b_base_C0.001 ... --per-family \
                             --extra-members ng_char-word-fw_lr_C100_T [--blend-check] [--save TAG | --log NAME]
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
from subfam_spec import GROUPS

import hill_climb  # noqa: E402  (reuse/airline)

FINE = {
    "fine1": ["c48sf", "gem678", "glm", "gpt4"],
    "fine2": ["fabglm", "gem78", "gptlst"],
    "fine3": ["claude", "gem678", "glm", "gpt4"],
}
REF = "blend_pt_gpu_ba"


def grouping(kind):
    if kind in ("lab", "super"):
        labs = [lab_of(c) if kind == "lab" else SUPER[lab_of(c)] for c in CLASSES]
    else:
        labs = list(CLASSES)
        for gname in FINE[kind]:
            for m in GROUPS[gname]:
                labs[CLASSES.index(m)] = gname
    fams = sorted(set(labs))
    foc = np.array([fams.index(l) for l in labs])
    multi = [f for f in range(len(fams)) if (foc == f).sum() > 1]
    return fams, foc, multi


def famc(kind, specs, per_family, extra_members, offsets=None):
    fc.FAMS, fc.FAM_OF_CLASS, fc.MULTI = grouping(kind)
    tr = load_train()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    members = [load_member(t) for t in fc.BASE + extra_members]
    m_oof, m_test = [m[0] for m in members], [m[1] for m in members]
    sp = [fc.load_spec(s) for s in specs]
    LS_oof, LS_test = [fc.lg(s[0]) for s in sp], [fc.lg(s[1]) for s in sp]
    out = np.zeros_like(m_oof[0])
    fold_w, fold_params = [], []
    for _, trn, val in iter_folds(folds):
        _, w = hill_climb.hill_climb(m_oof, y, idx=trn, space="logprob", metric=metrics.mlogloss,
                                     higher_is_better=False)
        fold_w.append(w)
        B = hill_climb.blend(m_oof, w, "logprob")
        params = fc.fit_combiner(B, LS_oof, y, trn, per_family, offsets)
        fold_params.append(params)
        out[val] = fc.apply_family(B, LS_oof, params)[val]
    Bt = hill_climb.blend(m_test, np.mean(fold_w, 0), "logprob")
    avg = {f: np.mean([p[f] for p in fold_params], 0) for f in fc.MULTI}
    Pt = fc.apply_family(Bt, LS_test, avg)
    return out, Pt, fold_w, fold_params


def blend_check(extra_oofs, y, folds, g, drop=()):
    """Nested BA hill-climb of blend_pt's gpu_noglm set (minus `drop`) + extra OOFs; returns scores, OOF, weights."""
    from blend_pt import SETS
    tags = [t for t in SETS["gpu_noglm"] if t not in drop]
    oofs = [load_member(t)[0] for t in tags] + list(extra_oofs)
    meta = [(trn, val) for _, trn, val in iter_folds(folds)]
    r = hill_climb.nested_hill(oofs, y, meta, avg_weights=True, space="logprob")
    names = tags + [f"new{i}" for i in range(len(extra_oofs))]
    used = {t: round(float(x), 3) for t, x in zip(names, r["weights"]) if x > 0}
    return scores(y, r["nested_pred"], folds), r["nested_pred"], used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouping", default="lab", choices=["lab", "super"] + list(FINE))
    ap.add_argument("--specs", nargs="*", default=[])
    ap.add_argument("--per-family", action="store_true")
    ap.add_argument("--offsets", type=float, default=None)
    ap.add_argument("--extra-members", nargs="*", default=[])
    ap.add_argument("--compare", default=None, help="member to compare the combiner with (paired)")
    ap.add_argument("--blend-check", action="store_true")
    ap.add_argument("--drop", nargs="*", default=[], help="gpu_noglm members left out of the blend check")
    ap.add_argument("--log", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    out, Pt, fold_w, fold_params = famc(a.grouping, a.specs, a.per_family, a.extra_members, a.offsets)
    s = scores(y, out, folds)
    print(f"[{a.grouping}] combiner: BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
          f"ll {s['logloss']:.4f}  families {[f for f in fc.FAMS]}")
    note = a.note
    if a.compare:
        c = fmt(paired_bootstrap(y, load_member(a.compare)[0], out, g))
        print(f"  paired vs {a.compare}: {c}")
        note += f"; combiner paired vs {a.compare}: {c}"
    if a.blend_check:
        sb, nb, used = blend_check([out], y, folds, g, a.drop)
        c = fmt(paired_bootstrap(y, load_member(REF)[0], nb, g))
        print(f"  blend check (gpu_noglm{' - ' + '/'.join(a.drop) if a.drop else ''} + this): nested BA "
              f"{sb['ba']:.4f} ll {sb['logloss']:.4f} weights {used}\n  blend paired vs {REF}: {c}")
        note += f"; in nested BA blend (gpu_noglm + this): BA {sb['ba']:.4f} ll {sb['logloss']:.4f}, weights {used}, paired vs {REF}: {c}"
    desc = (f"family combiner, grouping {a.grouping} ({'/'.join(fc.FAMS[f] for f in fc.MULTI)}) (subfam_famc.py): "
            f"base blend_v3 members+{'+'.join(a.extra_members)}; specialists {', '.join(a.specs)}"
            f"{'; per-family' if a.per_family else ''}{f'; offsets L2 {a.offsets:g}' if a.offsets is not None else ''}")
    note = note.lstrip("; ")
    if a.save:
        run_member(a.save, desc, out, Pt, y, folds, t0, note=note, members=fc.BASE + a.extra_members,
                   specs=a.specs, per_family=a.per_family, grouping=a.grouping, offsets=a.offsets,
                   blend_weights=[float(v) for v in np.mean(fold_w, 0)])
    elif a.log:
        log_experiment(a.log, desc, s, time.time() - t0, note=note + "; not saved")


if __name__ == "__main__":
    main()
