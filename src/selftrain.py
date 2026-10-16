"""Transductive self-training of per-text members; the unlabeled texts are a SET OF INDEPENDENT TEXTS.

Nothing clusters, groups, centers or matches the unlabeled texts: every unlabeled text is pseudo-labeled from its
own prediction, the confident ones join the training rows (predicted label, down-weighted), the head is refit and
the unlabeled texts are predicted again, each from its own content.

CV simulation (always run): for outer fold k the held-out fold plays the unlabeled test set.
  stage 0  head fit on the training folds, held-out rows predicted -> P0 (= the plain member's OOF)
  select   per PREDICTED class, the top --q fraction of the rows predicted as that class, by max probability
           (--sel frac; rare predicted classes keep their share), or the same number round(q*|U|/K) of rows for
           every predicted class (--sel equal); optional floor --thr on the max probability
  refit    training folds + selected held-out rows with their PREDICTED labels (hard: argmax with weight --w;
           soft: top-3 classes, weight w*p renormalised), then the held-out rows are predicted again -> P1.
           --crossfit S >= 2: the unlabeled rows are split at random (per text, seeded) into S parts; part s is
           predicted by a head whose pseudo rows come from the other parts only, so no text is ever predicted by a
           head that was trained on its own pseudo label. S <= 1: one refit, all unlabeled rows predicted by it.
  Held-out labels are used only for scoring (and for the reported pseudo-label accuracy).
Skewed check (--skew N): N seeded draws of model-specific keep rates U(0.2, 1.0) (matching.drop_mask); the kept
held-out rows of each fold are the unlabeled set (self-training sees only those); before / after are scored on the
kept rows. Seeds are fixed (1000 + s), so every member / config sees the same skewed sets.

Members (--member):
  emb8  raw qwen3_8b_base embeddings (all 4 layers), per-fold StandardScaler + LogReg C=--C (default 0.001), fit
        exactly through the per-fold Gram matrices of emb_kernel.py (= emb_qwen3_8b_base_C0.001 up to solver tol).
  ng    train_ngram.py char+word+fw TF-IDF streams (vectorizers fit on the training folds only) + LogReg C=--C
        (default 100) with the stage-0 inner grouped temperature (= ng_char-word-fw_lr_C100_T). Stage-0 OOF and
        temperatures come from that saved member when it exists (same code), else they are computed here.

Submission (--save TAG, first grid config only): the CV version above (balanced held-out fold as the unlabeled set)
is saved as the OOF of member TAG; the test prediction uses the load_test() texts as the unlabeled set: for every
fold model, stage 0 predicts the test texts, the confident ones are pseudo-labeled, the head is refit and the test
texts are predicted again; the member's test prediction is the mean over the five fold models (as for every member).
Works for any test size (with very few test texts nothing may be selected: then P1 = P0).

Examples:
  python selftrain.py --member emb8 --q 0.3 0.6 --w 0.5 1 --crossfit 0 2 --skew 5 --log
  python selftrain.py --member ng --q 0.5 --w 0.5 --crossfit 2 --skew 3 --log --n-jobs 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from argparse import Namespace

import numpy as np

from common import (CLASSES, K, OUT, PREDS, encode_y, load_folds, load_member, load_test, load_train,
                    log_experiment, metrics, run_member, scores)
from compare import fmt, paired_bootstrap

SCRATCH = OUT / "aug_scratch"            # read by aug_blend_check.load_any (oof only)
NG_REF = "ng_char-word-fw_lr_C100_T"


# ---------------------------------------------------------------- pseudo-label selection (per text)

def select_pseudo(P, q, sel="frac", thr=None, labels="hard", w=0.5):
    """Positions (into the unlabeled rows), pseudo labels and sample weights of the selected rows.
    Each row's selection depends on its own prediction and on the per-predicted-class confidence ranking."""
    pred, conf = P.argmax(1), P.max(1)
    n = len(P)
    chosen = []
    for c in range(K):
        ix = np.flatnonzero(pred == c)
        m = int(round(q * len(ix))) if sel == "frac" else int(round(q * n / K))
        if thr is not None:
            ix = ix[conf[ix] >= thr]
        m = min(m, len(ix))
        if m > 0:
            chosen.append(ix[np.argsort(-conf[ix], kind="stable")[:m]])
    pos = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)
    if labels == "hard":
        return pos, pred[pos], np.full(len(pos), float(w))
    top = np.argsort(-P[pos], 1)[:, :3]
    pt = np.take_along_axis(P[pos], top, 1)
    pt = pt / pt.sum(1, keepdims=True)
    return np.repeat(pos, 3), top.ravel(), (w * pt).ravel()


def self_train(fit_predict, L, yL, U, P0U, cfg, rng):
    """One or more rounds of self-training; returns (P1 for the U rows, list of selected positions per round)."""
    P = P0U
    picks = []
    for _ in range(cfg["rounds"]):
        pos, lab, wt = select_pseudo(P, cfg["q"], cfg["sel"], cfg["thr"], cfg["labels"], cfg["w"])
        picks.append((pos, lab))
        S = cfg["crossfit"]
        ones = np.ones(len(L))
        if len(pos) == 0:
            Pn = fit_predict(L, yL, ones, U)
        elif S <= 1:
            Pn = fit_predict(np.r_[L, U[pos]], np.r_[yL, lab], np.r_[ones, wt], U)
        else:
            Pn = np.zeros_like(P)
            part = rng.permutation(len(U)) % S
            for s in range(S):
                use = part[pos] != s
                ev = np.flatnonzero(part == s)
                if len(ev):
                    Pn[ev] = fit_predict(np.r_[L, U[pos[use]]], np.r_[yL, lab[use]], np.r_[ones, wt[use]], U[ev])
        P = Pn
    return P, picks


# ---------------------------------------------------------------- members (backends)

class EmbBackend:
    """Global row index space = [train rows; test rows] of the per-fold Gram matrix (emb_kernel.grams cache)."""

    def __init__(self, a, tr, te, y, folds, groups):
        self.C, self.name = a.C, a.emb_name
        self.folds, self.n_test, self.y = folds, len(te), y

    def prepare(self, k):
        from emb_kernel import combined_gram
        return dict(Kf=combined_gram([(self.name, None, 1.0)], self.folds, k, self.n_test), T=1.0)

    def fit_predict(self, ctx, rows, yrows, wrows, ev):
        from emb_kernel import fit_head, kcoords
        uniq, inv = np.unique(rows, return_inverse=True)
        Ptr_u, Pev, _ = kcoords(ctx["Kf"], uniq, np.asarray(ev))
        return fit_head("lr", self.C, Ptr_u[inv], yrows, Pev, sample_weight=wrows)

    def stage0(self, ctx, k, trn, val):
        return self.fit_predict(ctx, trn, self.y[trn], np.ones(len(trn)), val)


class NgBackend:
    """Global row index space = [train rows; test rows] of the per-fold TF-IDF matrix (vectorizers fit on the
    training folds of that fold only)."""

    def __init__(self, a, tr, te, y, folds, groups):
        from train_ngram import transformed_texts
        self.args = Namespace(feats=a.feats, model="lr", C=a.C, temp=True, binary=False, no_sublinear=False,
                              no_idf=False, min_df=2, max_features=0, stream_weights="")
        self.texts = transformed_texts(a.feats, tr["text"].tolist() + te["text"].tolist())
        self.folds, self.y, self.groups, self.n_tr = folds, y, groups, len(tr)
        self.C = a.C
        self.ref_oof, self.ref_T = None, None
        if a.feats == "char+word+fw" and a.C == 100 and not a.recompute_p0:
            try:
                self.ref_oof = load_member(NG_REF)[0]
                self.ref_T = json.loads((PREDS / f"meta_{NG_REF}.json").read_text())["temps"]
            except FileNotFoundError:
                self.ref_oof = None

    def prepare(self, k):
        import scipy.sparse as sp
        from train_ngram import STREAMS, make_vectorizer
        trn = np.flatnonzero(self.folds != k)
        blocks = []
        for f in self.args.feats.split("+"):
            tname = STREAMS[f][0]
            v = make_vectorizer(f, self.args).fit([self.texts[tname][i] for i in trn])
            blocks.append(v.transform(self.texts[tname]))
        return dict(A=sp.hstack(blocks).tocsr(), T=None)

    def _fit(self, A, rows, yrows, wrows):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=self.C, max_iter=3000, tol=1e-4).fit(A[rows], yrows, sample_weight=wrows)

    def fit_predict(self, ctx, rows, yrows, wrows, ev):
        from train_tfidf import softmax
        m = self._fit(ctx["A"], rows, yrows, wrows)
        return softmax(m.predict_log_proba(ctx["A"][np.asarray(ev)]) * ctx["T"])

    def stage0(self, ctx, k, trn, val):
        """Stage-0 predictions of the val rows + the fold temperature (saved member when present)."""
        if self.ref_oof is not None and np.all(val < self.n_tr):
            ctx["T"] = float(self.ref_T[k])
            return self.ref_oof[val]
        from sklearn.model_selection import StratifiedGroupKFold
        from train_tfidf import fit_temperature
        A, yt = ctx["A"], self.y[trn]
        dec = np.zeros((len(trn), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A[trn], yt, self.groups[trn]):
            dec[iva] = self._fit(A, trn[itr], yt[itr], None).predict_log_proba(A[trn[iva]])
        ctx["T"] = fit_temperature(dec, yt)
        ctx["T_fresh"] = True
        return self.fit_predict(ctx, trn, yt, np.ones(len(trn)), val)


BACKENDS = {"emb8": EmbBackend, "ng": NgBackend}


# ---------------------------------------------------------------- per-fold work

def fold_job(be, k, y, folds, keeps, cfgs, seed, with_test, n_tr, n_te):
    trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    ctx = be.prepare(k)
    p0_val = be.stage0(ctx, k, trn, val)
    fp = lambda rows, yr, wr, ev: be.fit_predict(ctx, rows, yr, wr, ev)  # noqa: E731
    out = {"P0": p0_val, "T": ctx["T"], "res": []}
    for ci, cfg in enumerate(cfgs):
        r = {}
        P1, picks = self_train(fp, trn, y[trn], val, p0_val, cfg, np.random.default_rng([seed, k, ci, 0]))
        pos, lab = picks[0]
        r["bal"] = (P1, len(pos), float(np.mean(lab == y[val][pos])) if len(pos) else float("nan"))
        r["skew"] = []
        for s, keep in enumerate(keeps):
            loc = np.flatnonzero(keep[val])
            U = val[loc]
            P1s, picks_s = self_train(fp, trn, y[trn], U, p0_val[loc], cfg, np.random.default_rng([seed, k, ci, s + 1]))
            r["skew"].append((loc, P1s))
        if with_test and ci == 0:
            te_rows = n_tr + np.arange(n_te)
            P0t = be.fit_predict(ctx, trn, y[trn], np.ones(len(trn)), te_rows)
            P1t, pt = self_train(fp, trn, y[trn], te_rows, P0t, cfg, np.random.default_rng([seed, k, ci, 999]))
            r["test"] = (P0t, P1t, len(pt[0][0]))
        out["res"].append(r)
    return out


def cfg_name(c):
    s = f"q{c['q']:g}_w{c['w']:g}_{c['sel']}_{c['labels']}_cf{c['crossfit']}"
    if c["thr"] is not None:
        s += f"_thr{c['thr']:g}"
    if c["rounds"] > 1:
        s += f"_r{c['rounds']}"
    return s


def ba_on(yv, P):
    return metrics.balanced_accuracy(yv, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=list(BACKENDS), required=True)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--emb-name", default="qwen3_8b_base")
    ap.add_argument("--feats", default="char+word+fw")
    ap.add_argument("--q", type=float, nargs="+", default=[0.5])
    ap.add_argument("--w", type=float, nargs="+", default=[0.5])
    ap.add_argument("--sel", nargs="+", default=["frac"], choices=["frac", "equal"])
    ap.add_argument("--labels", nargs="+", default=["hard"], choices=["hard", "soft"])
    ap.add_argument("--crossfit", type=int, nargs="+", default=[2])
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--skew", type=int, default=3, help="number of seeded skewed held-out sets (keep U(0.2,1))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--recompute-p0", action="store_true", help="ng: compute stage 0 here even if the member exists")
    ap.add_argument("--scratch", action="store_true", help="write outputs/aug_scratch/<tag>.npz (oof) per config")
    ap.add_argument("--log", action="store_true", help="one EXPERIMENTS.md row per config (not saved)")
    ap.add_argument("--save", default=None, help="save the FIRST config as member TAG (test = load_test() as U)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.C is None:
        a.C = 0.001 if a.member == "emb8" else 100.0
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds, groups = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].to_numpy()
    n_tr, n_te = len(tr), len(te)
    from matching import drop_mask
    keeps = [drop_mask(y, None, np.random.default_rng(1000 + s), 0.2, 1.0) for s in range(a.skew)]
    cfgs = [dict(q=q, w=w, sel=s, labels=lb, crossfit=cf, thr=a.thr, rounds=a.rounds)
            for q, w, s, lb, cf in itertools.product(a.q, a.w, a.sel, a.labels, a.crossfit)]
    be = BACKENDS[a.member](a, tr, te, y, folds, groups)
    n_f = int(folds.max()) + 1
    if a.n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=a.n_jobs)(delayed(fold_job)(be, k, y, folds, keeps, cfgs, a.seed, bool(a.save), n_tr, n_te)
                                         for k in range(n_f))
    else:
        outs = [fold_job(be, k, y, folds, keeps, cfgs, a.seed, bool(a.save), n_tr, n_te) for k in range(n_f)]
    print(f"folds done {time.time() - t0:.0f}s; temps {[round(o['T'], 3) for o in outs]}", flush=True)

    P0 = np.zeros((n_tr, K))
    for k, o in enumerate(outs):
        P0[folds == k] = o["P0"]
    s0 = scores(y, P0, folds)
    base = f"{a.member}" + (f"_{a.emb_name}" if a.member == "emb8" else f"_{a.feats.replace('+', '-')}") + f"_C{a.C:g}"
    ref_note = ""
    ref_tag = NG_REF if a.member == "ng" else f"emb_{a.emb_name}_C{a.C:g}"
    try:
        ref_saved = load_member(ref_tag)[0]
        ref_note = f"stage 0 vs saved {ref_tag}: max|dP| {np.abs(ref_saved - P0).max():.1e}, " \
                   f"BA {metrics.balanced_accuracy(y, ref_saved):.4f}"
    except FileNotFoundError:
        pass
    print(f"stage 0 ({base}): BA {s0['ba']:.4f} ll {s0['logloss']:.4f} folds {np.round(s0['ba_folds'], 4).tolist()}; "
          f"{ref_note}", flush=True)
    for ci, cfg in enumerate(cfgs):
        name = f"st_{base}_{cfg_name(cfg)}"
        P1 = np.zeros((n_tr, K))
        nsel, pacc = [], []
        for k, o in enumerate(outs):
            Pk, n_s, acc = o["res"][ci]["bal"]
            P1[folds == k] = Pk
            nsel.append(n_s)
            pacc.append(acc)
        s1 = scores(y, P1, folds)
        cmp_b = paired_bootstrap(y, P0, P1, groups)
        per_fold = "; ".join(f"f{k} {b0:.3f}->{b1:.3f}" for k, (b0, b1) in enumerate(zip(s0["ba_folds"], s1["ba_folds"])))
        sk_txt, sk_d = [], []
        for s in range(a.skew):
            keep = np.zeros(n_tr, bool)
            Ps = np.zeros((n_tr, K))
            for k, o in enumerate(outs):
                val = np.flatnonzero(folds == k)
                loc, Pk = o["res"][ci]["skew"][s]
                keep[val[loc]] = True
                Ps[val[loc]] = Pk
            kk = np.flatnonzero(keep)
            r = paired_bootstrap(y[kk], P0[kk], Ps[kk], groups[kk])
            fb = []
            for k in range(n_f):
                ix = kk[folds[kk] == k]
                fb.append(ba_on(y[ix], Ps[ix]) - ba_on(y[ix], P0[ix]))
            sk_d.append(r["diff"])
            sk_txt.append(f"s{s} n{len(kk)} BA {r['ba_a']:.4f}->{r['ba_b']:.4f} ({r['diff']:+.4f} [{r['ci_lo']:+.4f}, "
                          f"{r['ci_hi']:+.4f}]) ll {r['ll_a']:.4f}->{r['ll_b']:.4f}, folds {np.round(fb, 3).tolist()}")
        print(f"[{name}] balanced: BA {s0['ba']:.4f} -> {s1['ba']:.4f} ll {s0['logloss']:.4f} -> {s1['logloss']:.4f}; "
              f"paired {fmt(cmp_b)}\n  per fold {per_fold}\n  pseudo rows/fold {nsel}, pseudo acc "
              f"{np.round(pacc, 3).tolist()}", flush=True)
        if a.skew:
            print(f"  skewed: mean diff {np.mean(sk_d):+.4f} (per sim {np.round(sk_d, 4).tolist()})\n    " +
                  "\n    ".join(sk_txt), flush=True)
        desc = (f"self-training ({a.member}: {base}; held-out fold = unlabeled set of independent texts): select "
                f"{'top ' + format(cfg['q'], 'g') + ' per predicted class' if cfg['sel'] == 'frac' else format(cfg['q'], 'g') + '*|U|/K rows per predicted class'}"
                f"{', thr ' + format(cfg['thr'], 'g') if cfg['thr'] is not None else ''}, {cfg['labels']} pseudo labels "
                f"weight {cfg['w']:g}, crossfit {cfg['crossfit']}, rounds {cfg['rounds']} (selftrain.py)")
        note = (f"balanced paired vs stage 0 ({base} recomputed, BA {s0['ba']:.4f} ll {s0['logloss']:.4f}): {fmt(cmp_b)}; "
                f"per fold {per_fold}; pseudo rows/fold {nsel}, pseudo-label acc {np.round(pacc, 3).tolist()}; "
                + (f"skewed U(0.2,1) x{a.skew}: mean diff {np.mean(sk_d):+.4f}: " + " | ".join(sk_txt) if a.skew else "")
                + (f"; {ref_note}" if ref_note else ""))
        if a.note:
            note = a.note + "; " + note
        if a.scratch:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            np.savez(SCRATCH / f"{name}.npz", oof=P1, test=np.zeros((n_te, K)), ba=s1["ba"], logloss=s1["logloss"])
            np.savez(SCRATCH / f"st_{base}_stage0.npz", oof=P0, test=np.zeros((n_te, K)), ba=s0["ba"],
                     logloss=s0["logloss"])
        if a.save and ci == 0:
            test = np.mean([o["res"][0]["test"][1] for o in outs], 0)
            n_pt = [o["res"][0]["test"][2] for o in outs]
            run_member(a.save, desc, P1, test, y, folds, t0, note=note + f"; test pseudo rows/fold {n_pt}",
                       member=a.member, C=a.C, cfg=cfg, script="selftrain.py")
        elif a.log:
            log_experiment(name, desc, s1, time.time() - t0, note=note + "; not saved")
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
