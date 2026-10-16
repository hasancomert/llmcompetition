"""Hierarchical family combination + decision-level adjustments on top of a per-text blend, fully nested.

For every outer fold k (the shared folds), using ONLY the rows of the other four folds:
  1. blend weights: hill-climb (logprob space, log loss) of the base members' OOF -> B_k (= blend_v3_emb_logloss
     when the base members are its members);
  2. optional family specialists (family_spec.py outputs, conditional p(model | family)): inside each family F
        z_m = a_F * log B_k(m) + sum_v w_vF * log S_v(m) [+ c_m]      q(m | F) = softmax over F of z
        p(m) = B_k(F) * q(m | F)                                        (the family mass of the blend is kept)
     a, w (shared across families, or per family with --per-family) and optional per-class offsets c (L2 penalty)
     are fit by minimising the conditional NLL of the training rows of F;
  3. optional per-class log-prob biases for balanced accuracy (coordinate ascent on a grid, training rows only);
then everything is applied to the held-out fold k, one row at a time. Test: blend weights, combiner parameters and
biases are the means of the five fold fits. Nothing clusters, centers or matches held-out / test texts.

Usage:
  python family_combine.py --specs emb_qwen3_8b_base_C0.001 lpstyle_C0.03 [--per-family] [--offsets 1.0]
                           [--bias] [--extra-members ng_char-word-fw_lr_C100_T] [--log NAME] [--save TAG]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize

from common import (CLASSES, K, OUT, encode_y, iter_folds, load_folds, load_member, load_train, log_experiment,
                    metrics, run_member, scores)
from compare import fmt, paired_bootstrap
from family_spec import FAM_OF_CLASS, FAMS, GROUPING, MULTI, SPEC

import hill_climb  # noqa: E402  (reuse/airline)

REF = "blend_v3_emb_logloss"
BASE = ["tfidf_charword_lr_C30", "tfidf_char_svc_C0.3",
        "lp_qwen3_0.6b_base+qwen3_1.7b_base+qwen3_4b_base+qwen3_8b_base_style_C0.1",
        "emb_qwen3_4b_base_C0.003", "emb_qwen3_8b_base_C0.003"]
EPS = 1e-9


def lg(P):
    return np.log(np.clip(P, EPS, 1))


def fam_mass(P):
    return np.stack([P[:, FAM_OF_CLASS == f].sum(1) for f in range(len(FAMS))], 1)


def fit_family(LB, LS, yloc, n_cls, offsets_l2, init_a=1.0):
    """Parameters [a, w_1..w_V, c_1..c_n (if offsets_l2 is not None)] for one family's block (rows of that family).
    LB: (n, n_cls) log blend; LS: list of (n, n_cls) log specialist; yloc: position of the true model in the block."""
    V = len(LS)
    use_c = offsets_l2 is not None
    n = len(yloc)
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), yloc] = 1

    def unpack(th):
        return th[0], th[1:1 + V], (th[1 + V:] if use_c else np.zeros(n_cls))

    def f(th):
        a, w, c = unpack(th)
        z = a * LB + sum(wv * s for wv, s in zip(w, LS)) + c
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        q = np.exp(z - lse[:, None])
        loss = -(z[np.arange(n), yloc] - lse).mean()
        G = (q - Y) / n
        g = [np.sum(G * LB)] + [np.sum(G * s) for s in LS]
        if use_c:
            loss += offsets_l2 * np.sum(c ** 2)
            g += list(G.sum(0) + 2 * offsets_l2 * c)
        return loss, np.array(g)

    th0 = np.concatenate([[init_a], np.zeros(V), np.zeros(n_cls) if use_c else []])
    r = minimize(f, th0, jac=True, method="L-BFGS-B")
    return r.x


def apply_family(B, LSall, params):
    """params: dict family -> theta. Returns the combined (n, K) probabilities."""
    P = B.copy()
    LB = lg(B)
    FM = fam_mass(B)
    for f in MULTI:
        cols = np.flatnonzero(FAM_OF_CLASS == f)
        th = params[f]
        V = len(LSall)
        a, w = th[0], th[1:1 + V]
        c = th[1 + V:] if len(th) > 1 + V else 0.0
        z = a * LB[:, cols] + sum(wv * s[:, cols] for wv, s in zip(w, LSall)) + c
        z = z - z.max(1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(1, keepdims=True)
        P[:, cols] = FM[:, [f]] * q
    return P


def fit_combiner(B, LSall, y, rows, per_family, offsets_l2):
    """per_family: a, w (and offsets) fit separately for every family; otherwise a, w shared by all families (one
    joint fit, block softmax per row; offsets, if any, still per class)."""
    if not per_family:
        return fit_shared_with_offsets(B, LSall, y, rows, offsets_l2)
    params = {}
    for f in MULTI:
        cols = np.flatnonzero(FAM_OF_CLASS == f)
        r = rows[np.isin(y[rows], cols)]
        yloc = np.searchsorted(cols, y[r])
        params[f] = fit_family(lg(B[np.ix_(r, cols)]), [s[np.ix_(r, cols)] for s in LSall], yloc, len(cols),
                               offsets_l2)
    return params


def fit_shared_with_offsets(B, LSall, y, rows, offsets_l2):
    V = len(LSall)
    fam_rows, fam_cols = [], []
    for f in MULTI:
        cols = np.flatnonzero(FAM_OF_CLASS == f)
        fam_rows.append(rows[np.isin(y[rows], cols)])
        fam_cols.append(cols)
    ntot = sum(len(r) for r in fam_rows)
    data = []
    for r, cols in zip(fam_rows, fam_cols):
        yloc = np.searchsorted(cols, y[r])
        Y = np.zeros((len(r), len(cols)))
        Y[np.arange(len(r)), yloc] = 1
        data.append((lg(B[np.ix_(r, cols)]), [s[np.ix_(r, cols)] for s in LSall], yloc, Y, cols))
    use_c = offsets_l2 is not None

    def f(th):
        a, w = th[0], th[1:1 + V]
        c = th[1 + V:] if use_c else np.zeros(K)
        loss, g = 0.0, np.zeros(len(th))
        for LB, LS, yloc, Y, cols in data:
            z = a * LB + sum(wv * s for wv, s in zip(w, LS)) + c[cols]
            z = z - z.max(1, keepdims=True)
            lse = np.log(np.exp(z).sum(1))
            q = np.exp(z - lse[:, None])
            loss -= (z[np.arange(len(yloc)), yloc] - lse).sum() / ntot
            G = (q - Y) / ntot
            g[0] += np.sum(G * LB)
            for v in range(V):
                g[1 + v] += np.sum(G * LS[v])
            if use_c:
                g[1 + V + cols] += G.sum(0)
        if use_c:
            loss += offsets_l2 * np.sum(c ** 2)
            g[1 + V:] += 2 * offsets_l2 * c
        return loss, g

    th0 = np.concatenate([[1.0], np.zeros(V), np.zeros(K) if use_c else []])
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    params = {}
    for fi, f_ in enumerate(MULTI):
        cols = fam_cols[fi]
        params[f_] = np.concatenate([th[:1 + V], th[1 + V:][cols] if use_c else []])
    return params


def ba_from_pred(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(K) if np.any(y == c)]))


def fit_bias(P, y, grid=np.arange(-1.0, 1.0001, 0.05), passes=3):
    """Per-class additive log-prob biases maximising balanced accuracy on (P, y) by coordinate ascent; ties are
    broken towards the bias with the smallest |value|."""
    L = lg(P)
    b = np.zeros(K)
    best = ba_from_pred(y, (L + b).argmax(1))
    for _ in range(passes):
        changed = False
        for c in range(K):
            vals = []
            for v in grid:
                bb = b.copy()
                bb[c] = v
                vals.append((ba_from_pred(y, (L + bb).argmax(1)), -abs(v), v))
            s, _, v = max(vals)
            if s > best + 1e-12 and v != b[c]:
                b[c], best, changed = v, s, True
        if not changed:
            break
    return b


def apply_bias(P, b):
    z = lg(P) + b
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_spec(name):
    d = np.load(SPEC / f"{name}.npz")
    return d["oof"], d["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="*", default=[])
    ap.add_argument("--per-family", action="store_true")
    ap.add_argument("--offsets", type=float, default=None, help="L2 weight of per-class offsets (None: no offsets)")
    ap.add_argument("--bias", action="store_true", help="nested per-class BA biases on the final probabilities")
    ap.add_argument("--famfirst", action="store_true", help="decision: family by mass, then model inside it")
    ap.add_argument("--extra-members", nargs="*", default=[])
    ap.add_argument("--log", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_train()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    members = [load_member(t) for t in BASE + a.extra_members]
    m_oof, m_test = [m[0] for m in members], [m[1] for m in members]
    specs = [load_spec(s) for s in a.specs]
    LS_oof = [lg(s[0]) for s in specs]
    LS_test = [lg(s[1]) for s in specs]
    n = len(y)
    out = np.zeros((n, K))
    base_nested = np.zeros((n, K))
    fold_w, fold_params, fold_bias = [], [], []
    for _, trn, val in iter_folds(folds):
        _, w = hill_climb.hill_climb(m_oof, y, idx=trn, space="logprob", metric=metrics.mlogloss,
                                     higher_is_better=False)
        fold_w.append(w)
        B = hill_climb.blend(m_oof, w, "logprob")
        base_nested[val] = B[val]
        P = B
        if specs:
            params = fit_combiner(B, LS_oof, y, trn, a.per_family, a.offsets)
            fold_params.append(params)
            P = apply_family(B, LS_oof, params)
        if a.bias:
            b = fit_bias(P[trn], y[trn])
            fold_bias.append(b)
            P = apply_bias(P, b)
        out[val] = P[val]
    # test
    Bt = hill_climb.blend(m_test, np.mean(fold_w, 0), "logprob")
    Pt = Bt
    if specs:
        avg = {f: np.mean([p[f] for p in fold_params], 0) for f in MULTI}
        Pt = apply_family(Bt, LS_test, avg)
    if a.bias:
        Pt = apply_bias(Pt, np.mean(fold_bias, 0))
    if a.famfirst:
        FM = fam_mass(out)
        fsel = FM.argmax(1)
        mask = FAM_OF_CLASS[None, :] == fsel[:, None]
        out = np.where(mask, out, out * 1e-3)
        out /= out.sum(1, keepdims=True)
    try:                                      # reference only for the log; absent in a fresh replay (replay.py)
        ref, _ = load_member(REF)
    except FileNotFoundError:
        ref = None
    sb, s = scores(y, base_nested, folds), scores(y, out, folds)
    print(f"base nested blend (recomputed): BA {sb['ba']:.4f} ll {sb['logloss']:.4f}  "
          + (f"(max |diff| vs saved {REF}: {np.abs(base_nested - ref).max():.2e})" if ref is not None else ""))
    cmp = fmt(paired_bootstrap(y, ref, out, g)) if ref is not None else "n/a (reference member absent)"
    if a.extra_members:
        cmp_base = fmt(paired_bootstrap(y, base_nested, out, g))
        print(f"  paired vs recomputed base blend (+{'+'.join(a.extra_members)}): {cmp_base}")
        cmp += f"; vs nested base blend + {'+'.join(a.extra_members)}: {cmp_base}"
    print(f"result: BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) ll {s['logloss']:.4f}")
    print(f"  paired vs {REF}: {cmp}")
    if fold_params:
        for f in MULTI:
            print(f"  {FAMS[f]} params (mean of folds): {np.round(np.mean([p[f] for p in fold_params], 0), 3).tolist()}")
    if fold_bias:
        print(f"  BA biases (mean of folds): {dict(zip(CLASSES, np.round(np.mean(fold_bias, 0), 2)))}")
    if GROUPING != "lab":
        print(f"  grouping {GROUPING}: {FAMS}")
    desc = (f"family combine{' (super-families ' + '/'.join(FAMS) + ')' if GROUPING != 'lab' else ''}: base {'+'.join(['blend_v3 members'] + a.extra_members)}"
            f"{'; specialists ' + ', '.join(a.specs) if a.specs else ''}"
            f"{'; per-family' if a.per_family else ''}{f'; offsets L2 {a.offsets:g}' if a.offsets is not None else ''}"
            f"{'; nested BA biases' if a.bias else ''}{'; family-first decision' if a.famfirst else ''}")
    note = f"{a.note}; paired vs {REF}: {cmp}".lstrip("; ")
    if a.save:
        run_member(a.save, desc, out, Pt, y, folds, t0, note=note, members=BASE + a.extra_members,
                   specs=a.specs, per_family=a.per_family, offsets=a.offsets, bias=a.bias,
                   blend_weights=[float(v) for v in np.mean(fold_w, 0)])
    elif a.log:
        log_experiment(a.log, desc, s, time.time() - t0, note=note + "; not saved")


if __name__ == "__main__":
    main()
