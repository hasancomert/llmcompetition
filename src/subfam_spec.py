"""Sub-family / hard-group specialist heads (per-text): one small classifier per GROUP of confusable models, trained
ONLY on the training-fold rows whose TRUE model is in the group (family_spec.py, but for arbitrary groups).

Groups (GROUPS below) are hard pairs / sub-families inside a lab (gemini 3.7 vs 3.8, glm vs glm-flash, gpt-5.6
luna/sol/terra, the five Claude models, ...) or across labs (claude-fable-5 vs the two glm models). For every outer
fold k (shared folds) and every group G, a head is fit on the training-fold rows of G and applied to EVERY held-out
row and every test row, one row at a time. Nothing clusters, centers or matches held-out / test texts.

Output format (drop-in for family_combine.py --specs, any grouping that contains G inside one family):
  columns of G      p(m | G)
  all other columns geometric mean of p(. | G) over G (row-wise)
so that inside any family block the log values are the centered log-ratios of the group head on G and a constant
elsewhere (softmax inside a block is invariant to that constant); every value is <= 1 (family_combine clips logs
at 1). File: outputs/famspec/grp_<group>_<view>_C<C>.npz (oof, test).

--inner adds one more output per view, grp_<group>_<view>_Cin: the C of every outer fold chosen by a grouped inner
CV (StratifiedGroupKFold(4) by prompt_id, conditional log loss) on that fold's training rows of G only.

Views (as family_spec.py):
  emb:<blocks>   frozen-LM mean-pooled hidden states through the per-fold Gram cache (emb_kernel.py), exact LogReg
  ng:<streams>   TF-IDF streams of train_ngram.py (vectorizers fit on the training fold) + LogReg
  lpx:<blocks>   train_lpx.build_blocks tables (style, style2, lp4; s2_pb_ dropped) -> Quantile + scaler + LogReg
  lpstyle        family_spec.lpstyle_matrix (4 Qwen3 lp stats + style+tok v2) -> Quantile + scaler + LogReg

--rbf G (emb views only): RBF kernel exp(-G d^2 / median training d^2) on the standardized embeddings instead of the
linear kernel (double-centered on the group's training rows; exact kernel LogReg in its eigen-coordinates).

Usage: python subfam_spec.py --groups gem78,glm,gptlst --view emb:qwen3_8b_base --Cs 0.0003,0.001,0.003 [--inner]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import CLASS_INDEX, CLASSES, K, encode_y, load_folds, load_test, load_train
from family_spec import SPEC

GROUPS = {
    "gem78": ["gemini-3.7-flash", "gemini-3.8-flash"],
    "gem678": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "glm": ["glm-5.3", "glm-5.3-flash"],
    "gptlst": ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
    "gpt4": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
    "claude": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
    "c48sf": ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-5"],
    "fabglm": ["claude-fable-5", "glm-5.3", "glm-5.3-flash"],
}


def group_cols(name):
    return np.array(sorted(CLASS_INDEX[m] for m in GROUPS[name]))


def to_spec(Pg, cols):
    """(n, |G|) conditional probabilities -> (n, K) file format (see module doc)."""
    Pg = np.clip(Pg, 1e-9, 1.0)
    Pg = Pg / Pg.sum(1, keepdims=True)
    gm = np.exp(np.log(Pg).mean(1, keepdims=True))
    out = np.repeat(gm, K, axis=1)
    out[:, cols] = Pg
    return out


def lr_fit_predict(A, ya, Bs, C):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=5000).fit(A, ya)
    return [m.predict_proba(B) for B in Bs]


def inner_pick(A, ya, ga, Cs, seed=7):
    """C with the lowest grouped inner-CV conditional log loss on (A, ya); groups = prompt ids."""
    from sklearn.model_selection import StratifiedGroupKFold
    losses = np.zeros(len(Cs))
    for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(A, ya, ga):
        for ci, C in enumerate(Cs):
            p = lr_fit_predict(A[itr], ya[itr], [A[iva]], C)[0]
            losses[ci] -= np.log(np.clip(p[np.arange(len(iva)), ya[iva]], 1e-12, 1)).sum()
    return int(np.argmin(losses)), losses / len(ya)


class EmbDesign:
    def __init__(self, spec, folds, n, n_test, rbf=None):
        from emb_raw_explore import parse_blocks
        self.blocks, self.folds, self.n, self.n_test = parse_blocks(spec), folds, n, n_test
        self.rbf = rbf

    def fold(self, k):
        from emb_kernel import combined_gram
        self.Kfull = combined_gram(self.blocks, self.folds, k, self.n_test)
        self.val = np.flatnonzero(self.folds == k)

    def design(self, trn):
        """(train rows of the group, held-out rows, test rows) coordinates; the inner subsets of `trn` use the same
        eigen-coordinates (fit on all of trn: unsupervised, label-free)."""
        from emb_kernel import kcoords, rbf_kernel
        ev = np.concatenate([self.val, self.n + np.arange(self.n_test)])
        if self.rbf:      # RBF kernel, gamma = rbf / median training squared distance (group training rows)
            Ptr, Pev, _ = kcoords(rbf_kernel(self.Kfull, trn, self.rbf), trn, ev, center=True)
        else:
            Ptr, Pev, _ = kcoords(self.Kfull, trn, ev)
        return Ptr, Pev[:len(self.val)], Pev[len(self.val):]


class NgDesign:
    def __init__(self, feats, tr, te, folds):
        from train_ngram import transformed_texts
        self.feats, self.folds = feats, folds
        self.T_all = transformed_texts(feats, tr["text"].tolist())
        self.T_te = transformed_texts(feats, te["text"].tolist())

    def fold(self, k):
        import scipy.sparse as sp
        from train_ngram import STREAMS, make_vectorizer

        class A:  # train_ngram defaults
            no_sublinear, min_df, binary, no_idf, max_features = False, 2, False, False, 0
        trn_all = np.flatnonzero(self.folds != k)
        self.val = np.flatnonzero(self.folds == k)
        btr, bte = [], []
        for f in self.feats.split("+"):
            t = STREAMS[f][0]
            v = make_vectorizer(f, A).fit([self.T_all[t][i] for i in trn_all])
            btr.append(v.transform(self.T_all[t]))
            bte.append(v.transform(self.T_te[t]))
        self.X, self.Xt = sp.hstack(btr).tocsr(), sp.hstack(bte).tocsr()

    def design(self, trn):
        return self.X[trn], self.X[self.val], self.Xt


class TableDesign:
    def __init__(self, Xtr, Xte, folds):
        self.Xtr, self.Xte, self.folds = Xtr, Xte, folds

    def fold(self, k):
        self.val = np.flatnonzero(self.folds == k)

    def design(self, trn):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer, StandardScaler
        pre = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                            StandardScaler()).fit(self.Xtr[trn])
        return pre.transform(self.Xtr[trn]), pre.transform(self.Xtr[self.val]), pre.transform(self.Xte)


def make_design(view, tr, te, folds, rbf=None):
    if view.startswith("emb:"):
        return EmbDesign(view[4:], folds, len(tr), len(te), rbf)
    if view.startswith("ng:"):
        return NgDesign(view[3:], tr, te, folds)
    if view.startswith("lpx:"):
        from train_lpx import build_blocks
        Xtr, Xte, names = build_blocks(view[4:].split("+"), tr, te)
        keep = [i for i, nm in enumerate(names) if not nm.startswith("s2_pb_")]
        return TableDesign(Xtr[:, keep], Xte[:, keep], folds)
    if view == "lpstyle":
        from family_spec import lpstyle_matrix
        Xtr, Xte = lpstyle_matrix(tr, te)
        return TableDesign(Xtr, Xte, folds)
    raise SystemExit(f"unknown view {view}")


def in_group(y, S, cols):
    rows = np.flatnonzero(np.isin(y, cols))
    B = S[np.ix_(rows, cols)]
    pos = np.searchsorted(cols, y[rows])
    ba = np.mean([(B.argmax(1)[pos == j] == j).mean() for j in range(len(cols))])
    B = B / B.sum(1, keepdims=True)
    nll = -np.log(np.clip(B[np.arange(len(rows)), pos], 1e-12, 1)).mean()
    return float(ba), float(nll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True, help="comma-separated names of GROUPS")
    ap.add_argument("--view", required=True)
    ap.add_argument("--Cs", required=True)
    ap.add_argument("--inner", action="store_true", help="also save the per-fold inner-CV choice of C (_Cin)")
    ap.add_argument("--rbf", type=float, default=None, help="emb views: RBF kernel with gamma = rbf / median sq. dist.")
    a = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds, g = encode_y(tr["model"]), load_folds(tr), tr["prompt_id"].values
    Cs = [float(c) for c in a.Cs.split(",")]
    gnames = a.groups.split(",")
    n, n_test = len(y), len(te)
    keys = Cs + (["in"] if a.inner else [])
    res = {(gn, c): (np.zeros((n, len(GROUPS[gn]))), np.zeros((n_test, len(GROUPS[gn])))) for gn in gnames
           for c in keys}
    picks = {gn: [] for gn in gnames}
    D = make_design(a.view, tr, te, folds, a.rbf)
    n_f = int(folds.max()) + 1
    for k in range(n_f):
        D.fold(k)
        val = np.flatnonzero(folds == k)
        for gn in gnames:
            cols = group_cols(gn)
            trn = np.flatnonzero((folds != k) & np.isin(y, cols))
            A, V, T = D.design(trn)
            ya = np.searchsorted(cols, y[trn])
            outs = {}
            for C in Cs:
                pv, pt = lr_fit_predict(A, ya, [V, T], C)
                outs[C] = (pv, pt)
                res[(gn, C)][0][val] = pv
                res[(gn, C)][1][:] += pt / n_f
            if a.inner:
                ci, _ = inner_pick(A, ya, g[trn], Cs)
                picks[gn].append(Cs[ci])
                res[(gn, "in")][0][val] = outs[Cs[ci]][0]
                res[(gn, "in")][1][:] += outs[Cs[ci]][1] / n_f
        print(f"  fold {k} done {time.time() - t0:.0f}s", flush=True)
    SPEC.mkdir(parents=True, exist_ok=True)
    vname = a.view.replace(":", "_").replace(",", "+") + (f"_rbf{a.rbf:g}" if a.rbf else "")
    for (gn, c), (oof, test) in res.items():
        cols = group_cols(gn)
        O, T = to_spec(oof, cols), to_spec(test, cols)
        name = f"grp_{gn}_{vname}_C{c:g}" if c != "in" else f"grp_{gn}_{vname}_Cin"
        np.savez(SPEC / f"{name}.npz", oof=O, test=T)
        ba, nll = in_group(y, O, cols)
        extra = f" (inner picks {picks[gn]})" if c == "in" else ""
        print(f"{name}: in-group BA {ba:.4f} cond NLL {nll:.4f}{extra}", flush=True)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
