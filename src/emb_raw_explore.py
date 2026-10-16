"""Raw (uncentered, per-text) frozen-LM embedding heads, fit exactly in Gram space (emb_kernel.py).

Every held-out / test text is classified from its own embedding; nothing clusters, centers or matches held-out
texts. Training-side tricks may use the true prompt_id of the TRAINING rows (--aug-center).

Block spec (--blocks, comma separated): name[@l0+l1...][*weight], e.g. qwen3_8b_base, qwen3_8b_base@1+2,
qwen3_4b_base*0.5. Layers are block indices into meta['layers'].
Modes:
  default  : one head on the (weighted) concatenation of all blocks (sum of Gram matrices)
  --late   : one head per block (or per layer with --per-layer), log-probabilities averaged (geometric mean)
Heads: lr (LogReg), svc (LinearSVC + inner grouped temperature), ridge (RidgeClassifier + inner temperature).
Transforms (fit on the training fold): --power p (eigen-spectrum power, 1 = plain, 0 = PCA whitening),
--drop-top k, --kernel cos|rbf, --aug-center w (add prompt-centered copies of the training rows, weight w).

Examples:
  python emb_raw_explore.py --blocks qwen3_8b_base --C 0.001              # = emb_qwen3_8b_base_C0.001
  python emb_raw_explore.py --blocks qwen3_4b_base,qwen3_8b_base --C 0.0005,0.001 --log
  python emb_raw_explore.py --blocks qwen3_8b_base --C 0.001 --save emb8_xyz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, OUT, encode_y, load_folds, load_member, load_test, load_train, log_experiment, run_member, scores
from compare import fmt, paired_bootstrap
from emb_kernel import combined_gram, cosine_kernel, fit_head, kcoords, layer_info, rbf_kernel

REF = "emb_qwen3_8b_base_C0.001"
CAND = OUT / "kgram" / "cand"


def parse_blocks(spec):
    out = []
    for item in spec.split(","):
        w = 1.0
        if "*" in item:
            item, ws = item.split("*")
            w = float(ws)
        ls = None
        if "@" in item:
            item, lss = item.split("@")
            ls = [int(v) for v in lss.split("+")]
        out.append((item, ls, w))
    return out


def group_center(P, groups, w_mean=None):
    """Rows minus their prompt mean plus the global mean (training rows only, true prompt_id)."""
    Pc = P.copy()
    for g in np.unique(groups):
        ix = groups == g
        Pc[ix] -= P[ix].mean(0)
    return Pc + P.mean(0)


def run_cv(args, blocks, y, folds, groups, n_test, Cs):
    n = len(y)
    oofs = {C: np.zeros((n, K)) for C in Cs}
    tests = {C: np.zeros((n_test, K)) for C in Cs}
    n_f = int(folds.max()) + 1
    for f in range(n_f):
        trn, val = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        ev = np.concatenate([val, n + np.arange(n_test)])
        if args.late:
            units = []
            for (nm, ls, w) in blocks:
                lays = ls if ls is not None else list(range(len(layer_info(nm)[0])))
                units += [[(nm, [li], w)] for li in lays] if args.per_layer else [[(nm, ls, w)]]
        else:
            units = [blocks]
        logp = {C: np.zeros((len(ev), K)) for C in Cs}
        wsum = 0.0
        for unit in units:
            Kf = combined_gram(unit, folds, f, n_test, equalize=args.equalize)
            centered = False
            if args.kernel == "cos":
                Kf = cosine_kernel(Kf)
                centered = True
            elif args.kernel == "rbf":
                Kf = rbf_kernel(Kf, trn, args.gamma)
                centered = True
            Ptr, Pev, lam = kcoords(Kf, trn, ev, power=args.power, drop_top=args.drop_top, center=centered)
            if args.rowl2:
                Ptr = Ptr / np.linalg.norm(Ptr, axis=1, keepdims=True) * np.sqrt(Ptr.shape[1])
                Pev = Pev / np.linalg.norm(Pev, axis=1, keepdims=True) * np.sqrt(Ptr.shape[1])
            ytr, gtr, sw = y[trn], groups[trn], None
            if args.aug_center > 0:
                Pc = group_center(Ptr, gtr)
                Ptr = np.vstack([Ptr, Pc])
                ytr, gtr = np.concatenate([ytr, ytr]), np.concatenate([gtr, gtr])
                sw = np.concatenate([np.ones(len(trn)), np.full(len(trn), args.aug_center)])
            uw = unit[0][2] if args.late else 1.0
            for C in Cs:
                P = fit_head(args.head, C, Ptr, ytr, Pev, groups_tr=gtr, sample_weight=sw)
                logp[C] += uw * np.log(np.clip(P, 1e-12, 1))
            wsum += uw
        for C in Cs:
            lp = logp[C] / wsum
            P = np.exp(lp - lp.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            oofs[C][val] = P[:len(val)]
            tests[C] += P[len(val):] / n_f
        print(f"  fold {f} done ({time.time() - T0:.0f}s)", flush=True)
    return oofs, tests


def describe(args, blocks):
    parts = []
    for nm, ls, w in blocks:
        layers, _ = layer_info(nm)
        lstr = "all layers" if ls is None else "layers " + "+".join(str(layers[i]) for i in ls)
        parts.append(f"{nm} ({lstr}{'' if w == 1 else f', w {w:g}'})")
    head = {"lr": "LogReg", "svc": "LinearSVC + inner temperature", "ridge": "RidgeClassifier + inner temperature"}
    s = (f"RAW frozen LM emb {' + '.join(parts)}; "
         f"{'late fusion (mean log-prob of one head per ' + ('layer' if args.per_layer else 'block') + ')' if args.late else 'one head on the concatenation'}"
         f"{', blocks equalized' if args.equalize else ''}; StandardScaler (train fold) + {head[args.head]}")
    if args.kernel != "linear":
        s += f", {args.kernel} kernel" + (f" gamma {args.gamma:g}/median d2" if args.kernel == "rbf" else "")
    if args.power != 1:
        s += f", eigen-spectrum power {args.power:g} (train-fold PCA)"
    if args.drop_top:
        s += f", top {args.drop_top} train-fold PCs removed"
    if args.rowl2:
        s += ", row L2 norm of PCA coords"
    if args.aug_center:
        s += f", + prompt-centered copies of the training rows (true prompt_id, weight {args.aug_center:g})"
    return s


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--C", default="0.001")
    ap.add_argument("--head", default="lr", choices=["lr", "svc", "ridge"])
    ap.add_argument("--late", action="store_true")
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--equalize", action="store_true")
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--kernel", default="linear", choices=["linear", "cos", "rbf"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--rowl2", action="store_true")
    ap.add_argument("--aug-center", type=float, default=0.0)
    ap.add_argument("--log", action="store_true", help="log every C as an EXPERIMENTS.md row (no member saved)")
    ap.add_argument("--save", default=None, help="save the run (single C) as a member with this tag")
    ap.add_argument("--name", default=None, help="short experiment name for logged rows")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    T0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    groups = tr["prompt_id"].to_numpy()
    blocks = parse_blocks(args.blocks)
    Cs = [float(c) for c in args.C.split(",")]
    if args.save:
        assert len(Cs) == 1, "--save takes one C"
    oofs, tests = run_cv(args, blocks, y, folds, groups, len(te), Cs)
    ref_oof, _ = load_member(REF)
    desc = describe(args, blocks)
    for C in Cs:
        s = scores(y, oofs[C], folds)
        cmp = fmt(paired_bootstrap(y, ref_oof, oofs[C], groups))
        print(f"C={C:g}: BA {s['ba']:.4f} ({s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) ll {s['logloss']:.4f} | "
              f"vs {REF}: {cmp}", flush=True)
        change = f"{desc}, C={C:g}"
        CAND.mkdir(parents=True, exist_ok=True)                   # OOF of every run, for blend_check_raw.py
        np.save(CAND / f"{args.save or args.name or 'embraw'}_C{C:g}.npy", oofs[C])
        np.save(CAND / f"{args.save or args.name or 'embraw'}_C{C:g}.test.npy", tests[C])
        note = (args.note + "; " if args.note else "") + f"paired vs {REF}: {cmp}"
        if args.save:
            run_member(args.save, change, oofs[C], tests[C], y, folds, T0, note=note, blocks=args.blocks, C=C,
                       head=args.head, late=args.late, per_layer=args.per_layer, equalize=args.equalize,
                       power=args.power, drop_top=args.drop_top, kernel=args.kernel, gamma=args.gamma,
                       rowl2=args.rowl2, aug_center=args.aug_center, script="emb_raw_explore.py")
        elif args.log:
            log_experiment(f"{args.name or 'embraw'}_C{C:g}", change, s, time.time() - T0, note)


if __name__ == "__main__":
    main()
