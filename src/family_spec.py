"""Family-specialist heads (per-text): one classifier per model family, trained ONLY on that family's training rows.

For every outer fold k (the shared folds) and every family F with more than one model (claude, gemini, glm, gpt),
a head is fit on the training-fold rows whose TRUE model is in F and then applied to EVERY held-out row (and the
test rows), one row at a time. The output is an (N, K) matrix of conditional probabilities p(model | family F),
one block per family (each block sums to 1; kimi / qwen blocks are 1). It is combined later with the family
probabilities of a full blend (family_combine.py). Nothing clusters, centers or matches held-out / test texts.

Views:
  emb:<blocks>  frozen-LM mean-pooled hidden states through the per-fold Gram cache (emb_kernel.py; the
                standardization is fit on the training fold), exact linear LogReg in eigen-coordinates of the
                family's training rows. <blocks> as emb_raw_explore.py, e.g. qwen3_8b_base or qwen3_4b_base,qwen3_8b_base
                --test-std: held-out fold and test rows standardized with their own statistics (emb_kernel test_std;
                transductive, no labels), outputs suffixed _ts
  ng:<streams>  TF-IDF streams of train_ngram.py (e.g. char+word+fw; vectorizers fit on the training fold) + LogReg;
                optional external training rows (extdata.py flags, e.g. --ext pub --ext-balance): the family's external
                rows join its training set with their weights (vectorizers still fit on the official training texts;
                rows topic-matched to a held-out prompt left out), outputs suffixed with the extdata tag
  lpstyle       token log-prob statistics of the 4 Qwen3 base LMs + style+tok v2 columns -> QuantileTransformer +
                StandardScaler + LogReg (as the lp member of blend_v3_emb_logloss)
Output: outputs/famspec/<view>_C<C>.npz (oof, test) for every C of the grid; within-family accuracy printed.

FAMSPEC_GROUPS=super switches the families to super-families {claude, glm}, {gpt, kimi, qwen}, {gemini} (outputs
prefixed sup_; family_combine.py must then run with the same setting).

Usage: python family_spec.py --view emb:qwen3_8b_base --Cs 0.0003,0.001,0.003,0.01
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import CLASSES, K, OUT, encode_y, lab_of, load_folds, load_test, load_train
import extdata

SPEC = OUT / "famspec"
GROUPING = os.environ.get("FAMSPEC_GROUPS", "lab")      # lab | super ({claude, glm}, {gpt, kimi, qwen}, {gemini})
SUPER = {"claude": "claude+glm", "glm": "claude+glm", "gpt": "gpt+kimi+qwen", "kimi": "gpt+kimi+qwen",
         "qwen": "gpt+kimi+qwen", "gemini": "gemini"}
LABS = np.array([lab_of(c) if GROUPING == "lab" else SUPER[lab_of(c)] for c in CLASSES])
FAMS = sorted(set(LABS))
FAM_OF_CLASS = np.array([FAMS.index(l) for l in LABS])
MULTI = [f for f in range(len(FAMS)) if (FAM_OF_CLASS == f).sum() > 1]


def single_blocks(P):
    """Families with one model: conditional probability 1."""
    for f in range(len(FAMS)):
        cols = np.flatnonzero(FAM_OF_CLASS == f)
        if len(cols) == 1:
            P[:, cols[0]] = 1.0
    return P


def within_family_report(y, S, name=""):
    """Accuracy / NLL of the conditional block of the TRUE family (rows of multi-model families only)."""
    fy = FAM_OF_CLASS[y]
    out = []
    accs, nlls = [], []
    for f in MULTI:
        rows = np.flatnonzero(fy == f)
        cols = np.flatnonzero(FAM_OF_CLASS == f)
        B = S[np.ix_(rows, cols)]
        B = B / B.sum(1, keepdims=True)
        pos = np.searchsorted(cols, y[rows])
        acc = (B.argmax(1) == pos).mean()
        # balanced (mean per-class recall) inside the family
        rec = np.mean([(B.argmax(1)[pos == j] == j).mean() for j in range(len(cols))])
        nll = -np.log(np.clip(B[np.arange(len(rows)), pos], 1e-12, 1)).mean()
        accs.append(rec)
        nlls.append(nll * len(rows))
        out.append(f"{FAMS[f]} {rec:.3f}/{nll:.3f}")
    rows_all = np.isin(fy, MULTI).sum()
    return float(np.mean(accs)), float(np.sum(nlls) / rows_all), f"{name} " + " ".join(out)


def emb_view(spec, y, folds, n_test, Cs, test_std=False):
    from emb_kernel import combined_gram, kcoords
    from emb_raw_explore import parse_blocks
    from sklearn.linear_model import LogisticRegression
    blocks = parse_blocks(spec)
    n = len(y)
    res = {C: (np.zeros((n, K)), np.zeros((n_test, K))) for C in Cs}
    n_f = int(folds.max()) + 1
    for k in range(n_f):
        Kfull = combined_gram(blocks, folds, k, n_test, test_std=test_std)
        val = np.flatnonzero(folds == k)
        ev = np.concatenate([val, n + np.arange(n_test)])
        for f in MULTI:
            cols = np.flatnonzero(FAM_OF_CLASS == f)
            trn = np.flatnonzero((folds != k) & np.isin(y, cols))
            Ptr, Pev, _ = kcoords(Kfull, trn, ev)
            for C in Cs:
                m = LogisticRegression(C=C, max_iter=5000).fit(Ptr, y[trn])
                pr = m.predict_proba(Pev)
                oof, te = res[C]
                oof[np.ix_(val, m.classes_)] = pr[:len(val)]
                te[:, m.classes_] += pr[len(val):] / n_f
        print(f"  fold {k} done", flush=True)
    return {C: (single_blocks(o), single_blocks(t)) for C, (o, t) in res.items()}


def lpstyle_matrix(tr, te):
    from train_emb import load_feats
    from train_gbdt import build
    names = ["qwen3_0.6b_base", "qwen3_1.7b_base", "qwen3_4b_base", "qwen3_8b_base"]
    a = np.hstack([m[0] for m in load_feats("lp", names, "train")])
    b = np.hstack([m[0] for m in load_feats("lp", names, "test")])
    Str, Ste, _ = build("style+tok", tr, te)
    return np.hstack([a, Str]).astype(np.float32), np.hstack([b, Ste]).astype(np.float32)


def table_view(Xtr, Xte, y, folds, Cs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    n, n_test = len(y), len(Xte)
    res = {C: (np.zeros((n, K)), np.zeros((n_test, K))) for C in Cs}
    n_f = int(folds.max()) + 1
    for k in range(n_f):
        val = np.flatnonzero(folds == k)
        for f in MULTI:
            cols = np.flatnonzero(FAM_OF_CLASS == f)
            trn = np.flatnonzero((folds != k) & np.isin(y, cols))
            pre = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                                StandardScaler()).fit(Xtr[trn])
            A, V, T = pre.transform(Xtr[trn]), pre.transform(Xtr[val]), pre.transform(Xte)
            for C in Cs:
                m = LogisticRegression(C=C, max_iter=5000).fit(A, y[trn])
                oof, te = res[C]
                oof[np.ix_(val, m.classes_)] = m.predict_proba(V)
                te[:, m.classes_] += m.predict_proba(T) / n_f
        print(f"  fold {k} done", flush=True)
    return {C: (single_blocks(o), single_blocks(t)) for C, (o, t) in res.items()}


def ng_view(feats, tr, te, y, folds, Cs, ext=None):
    """TF-IDF n-gram streams of train_ngram.py (vectorizers fit on the training-fold texts) + LogReg per family;
    ext: extdata.Ext rows added to the training sets only (weights; rows topic-matched to the held-out prompts out)."""
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from train_ngram import STREAMS, make_vectorizer, transformed_texts

    class A:  # train_ngram defaults
        no_sublinear, min_df, binary, no_idf, max_features = False, 2, False, False, 0
    T_all = transformed_texts(feats, tr["text"].tolist())
    T_te = transformed_texts(feats, te["text"].tolist())
    T_ext = transformed_texts(feats, ext.texts) if ext is not None else None
    pids = tr["prompt_id"].to_numpy()
    n, n_test = len(y), len(te)
    res = {C: (np.zeros((n, K)), np.zeros((n_test, K))) for C in Cs}
    n_f = int(folds.max()) + 1
    for k in range(n_f):
        trn_all, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        ok = np.flatnonzero(ext.allowed(set(pids[val]))) if ext is not None else np.zeros(0, int)
        btr, bte, be = [], [], []
        for f in feats.split("+"):
            t = STREAMS[f][0]
            v = make_vectorizer(f, A).fit([T_all[t][i] for i in trn_all])
            btr.append(v.transform(T_all[t]))
            bte.append(v.transform(T_te[t]))
            if ext is not None:
                be.append(v.transform([T_ext[t][i] for i in ok]))
        X = sp.hstack(btr).tocsr()
        Xt = sp.hstack(bte).tocsr()
        Xe = sp.hstack(be).tocsr() if ext is not None else None
        for f in MULTI:
            cols = np.flatnonzero(FAM_OF_CLASS == f)
            trn = np.flatnonzero((folds != k) & np.isin(y, cols))
            Xf, yf, wf = X[trn], y[trn], None
            if ext is not None:
                e = np.flatnonzero(np.isin(ext.y[ok], cols))
                Xf = sp.vstack([Xf, Xe[e]]).tocsr()
                yf = np.concatenate([yf, ext.y[ok][e]])
                wf = np.concatenate([np.ones(len(trn)), ext.w[ok][e]])
            for C in Cs:
                m = LogisticRegression(C=C, max_iter=3000, tol=1e-4).fit(Xf, yf, sample_weight=wf)
                oof, tst = res[C]
                oof[np.ix_(val, m.classes_)] = m.predict_proba(X[val])
                tst[:, m.classes_] += m.predict_proba(Xt) / n_f
        print(f"  fold {k} done", flush=True)
    return {C: (single_blocks(o), single_blocks(t)) for C, (o, t) in res.items()}


def save(view, C, oof, test):
    SPEC.mkdir(parents=True, exist_ok=True)
    name = ("sup_" if GROUPING == "super" else "") + f"{view.replace(':', '_').replace(',', '+')}_C{C:g}"
    np.savez(SPEC / f"{name}.npz", oof=oof, test=test)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True)
    ap.add_argument("--test-std", action="store_true", help="emb views: held-out / test rows standardized by own stats")
    ap.add_argument("--Cs", required=True)
    extdata.add_args(ap)
    a = ap.parse_args()
    if a.ext != "none" and not a.view.startswith("ng:"):
        ap.error("--ext is wired for the ng: view only")
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    Cs = [float(c) for c in a.Cs.split(",")]
    if a.view.startswith("emb:"):
        res = emb_view(a.view[4:], y, folds, len(te), Cs, a.test_std)
    elif a.view.startswith("ng:"):
        res = ng_view(a.view[3:], tr, te, y, folds, Cs, extdata.load(a, tr))
    elif a.view == "lpstyle":
        Xtr, Xte = lpstyle_matrix(tr, te)
        res = table_view(Xtr, Xte, y, folds, Cs)
    else:
        raise SystemExit(f"unknown view {a.view}")
    for C, (oof, test) in res.items():
        name = save(a.view + extdata.tag_suffix(a) + ("_ts" if a.test_std else ""), C, oof, test)
        ba, nll, txt = within_family_report(y, oof, name)
        print(f"{txt} | mean in-family BA {ba:.4f} cond NLL {nll:.4f}", flush=True)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
