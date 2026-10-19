"""Adaptive prompt-centering ("centered twins" in add mode), simulated on held-out prompts.

Recipe (public SOTA, Lucas Boesen's "LoRA x Adaptive Centering", Apache 2.0; re-implemented here, not copied):
  - plain pipeline: unchanged for every text (here: a saved OOF member, default famcsup6_l12 = v5)
  - a centered twin per frozen-LM view: a LogReg head trained on TRAINING rows centered by their TRUE prompt_id
  - test time: texts are clustered by topic (no labels); only texts in clusters of size min_size..max_size get the twin,
    on features centered by their cluster; its log-probabilities are ADDED to the plain ones with weight w
  - singletons are never centered (our 16 Oct centering failed exactly there: raw rows went into a centered model)
Exact and fast in Gram space (emb_kernel): with standardized features Z (statistics of the fold's training rows),
centering is linear, Z_c = (I - M) Z for a group-averaging matrix M, so K_c = (I - M_a) K (I - M_b)^T. The training
side (true prompt groups) is fixed per fold: one eigendecomposition and one LogReg per fold, view and C. Each simulated
test only needs K[test rows, train rows], centered by the test clusters on the left and the train prompts on the right.

Simulation (as the public pooled sim): the held-out folds play the hidden test; every prompt keeps a random number of
texts in [a, b] (distinct models); the kept texts of all 150 prompts are clustered ONCE as one pooled set; each kept
text is predicted by the models of its own fold. Reported per setting: BA of the plain pipeline, BA with twins for a
grid of weights, cluster purity, share of texts that got a twin. --balance adds Sinkhorn marginal balancing
(common.balance, no labels) of the final posterior over the whole pooled set, as on the hidden test.
--ng-views adds n-gram twins (TF-IDF streams of train_ngram.py, linear-kernel Gram; vectorizers fit on the fold's
training texts) with their own C / weight grids, on top of the LM twins (columns C<C>_w<w>_n<nC>x<wn>; C0_... = n-gram
twins alone).

Usage: python adaptive_sim.py --views qwen3_8b_base_l12,qwen3_4b_base_l12 --C 0.0003,0.001 --ranges 1-2,1-5,2-4,3-8
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import K, balance, encode_y, load_folds, load_member, load_test, load_train, log_experiment, metrics
from emb_kernel import combined_gram
from emb_raw_explore import parse_blocks
from matching import cluster_prompts, topic_matrix
from structure_sim import keep_rows, purity


def mixed_clusters(texts, E, thr, max_df_docs=40):
    """Average-linkage clusters on the mean of two cosine distances: content-word TF-IDF (matching.topic_matrix) and a
    frozen-LM embedding E (mean-centered over the set, L2-normalized). No labels."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    X = topic_matrix(texts, max_df_docs)
    Dt = 1.0 - (X @ X.T).toarray() if X is not None else np.ones((len(texts), len(texts)))
    Ec = E - E.mean(0)
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-12
    D = np.clip((Dt + (1.0 - Ec @ Ec.T)) / 2, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    return np.unique(fcluster(linkage(squareform(D, checks=False), method="average"), thr, criterion="distance"),
                     return_inverse=True)[1]


def pair_stats(g, prompts):
    """Pair precision (same cluster -> same prompt) and recall (same prompt -> same cluster)."""
    same_c = g[:, None] == g[None, :]
    same_p = prompts[:, None] == prompts[None, :]
    iu = np.triu_indices(len(g), 1)
    sc, sp = same_c[iu], same_p[iu]
    return float((sc & sp).sum() / max(sc.sum(), 1)), float((sc & sp).sum() / max(sp.sum(), 1))


def group_mean_rows(A, groups):
    """Row i of the result = mean of the rows of A in i's group."""
    out = np.empty_like(A)
    for g in np.unique(groups):
        ix = groups == g
        out[ix] = A[ix].mean(0)
    return out


def center_cols(A, groups):
    """Subtract, for every column j, the mean of the columns in j's group (right-multiplication by (I - M)^T)."""
    return A - group_mean_rows(A.T, groups).T


def ngram_gram(feats, texts, folds, k):
    """(N, N) linear-kernel Gram of the TF-IDF streams `feats` (train_ngram.py STREAMS, e.g. char+word+fw+shapew) over
    `texts` (train rows first, then any test rows); vectorizers fit on the training texts of fold k only."""
    import scipy.sparse as sp
    from train_ngram import STREAMS, make_vectorizer, transformed_texts

    class A:  # train_ngram defaults
        no_sublinear, min_df, binary, no_idf, max_features = False, 2, False, False, 0
    T = transformed_texts(feats, texts)
    trn = np.flatnonzero(folds != k)
    blocks = []
    for f in feats.split("+"):
        t = STREAMS[f][0]
        blocks.append(make_vectorizer(f, A).fit([T[t][i] for i in trn]).transform(T[t]))
    X = sp.hstack(blocks).tocsr()
    return (X @ X.T).toarray().astype(np.float64)


class Twin:
    """Centered twin of one view for one fold: trained on the fold's training rows centered by true prompt_id."""

    def __init__(self, Kfull, trn, ytr, gtr, Cs, eps=1e-10):
        from sklearn.linear_model import LogisticRegression
        self.K, self.trn, self.gtr = Kfull, trn, gtr
        Ktr = Kfull[np.ix_(trn, trn)]
        Ktr = center_cols(Ktr - group_mean_rows(Ktr, gtr), gtr)
        lam, U = np.linalg.eigh((Ktr + Ktr.T) / 2)
        lam, U = lam[::-1], U[:, ::-1]
        keep = lam > eps * lam[0]
        self.lam, self.U = lam[keep], U[:, keep]
        Ptr = self.U * np.sqrt(self.lam)
        self.models = {C: LogisticRegression(C=C, max_iter=3000).fit(Ptr, ytr) for C in Cs}

    def predict(self, rows, groups, C):
        """Twin probabilities of `rows` (indices into Kfull), each centered by its cluster in `groups`."""
        Kev = self.K[np.ix_(rows, self.trn)]
        Kev = center_cols(Kev - group_mean_rows(Kev, groups), self.gtr)
        Pev = Kev @ (self.U / np.sqrt(self.lam))
        m = self.models[C]
        P = np.full((len(rows), K), 1e-9)
        P[:, m.classes_] = m.predict_proba(Pev)
        return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="famcsup6_l12", help="saved OOF member = the plain pipeline")
    ap.add_argument("--views", default="qwen3_8b_base_l12,qwen3_4b_base_l12", help="';'-separated block specs")
    ap.add_argument("--C", default="0.0003,0.001")
    ap.add_argument("--weights", default="0.25,0.5,1.0", help="twin evidence weights (per twin)")
    ap.add_argument("--ranges", default="1-2,1-5,2-4,3-8,18-18", help="texts kept per prompt, a-b")
    ap.add_argument("--sims", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.92, help="cluster cut (tfidf: matching.cluster_prompts; mix: mixed)")
    ap.add_argument("--cluster", default="tfidf", choices=["tfidf", "mix"])
    ap.add_argument("--mix-emb", default="qwen3_8b_base_l12@7", help="--cluster mix: name@layer-index of the embedding")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=18)
    ap.add_argument("--balance", default="", help="Sinkhorn balancing strengths alpha (comma-separated; empty: none)")
    ap.add_argument("--ng-views", default="", help="n-gram twin views, ';'-separated stream sets (e.g. char+word+fw+shapew)")
    ap.add_argument("--ng-C", default="10,30", help="C grid of the n-gram twins")
    ap.add_argument("--ng-weights", default="0.25,0.5", help="weight grid of the n-gram twin evidence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="", help="EXPERIMENTS.md tag prefix (empty: no rows)")
    a = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = encode_y(tr["model"]), load_folds(tr)
    prompts, texts = tr["prompt_id"].to_numpy(), tr["text"].tolist()
    n_f, n_test = int(folds.max()) + 1, len(te)
    plain = load_member(a.plain)[0]
    Cs = [float(c) for c in a.C.split(",")]
    ws = [float(w) for w in a.weights.split(",")]
    alphas = [float(b) for b in a.balance.split(",")] if a.balance else []
    views = a.views.replace(",", ";").split(";") if ";" in a.views else a.views.split(",")
    Emix = None
    if a.cluster == "mix":
        from train_emb import load_feats
        nm, li = a.mix_emb.split("@")
        X, meta = load_feats("emb", [nm], "train")[0]
        w = meta["dim"] // len(meta["layers"])
        Emix = X[:, int(li) * w:(int(li) + 1) * w].astype(np.float64)
        print(f"mixed clustering: TF-IDF + {nm} layer {meta['layers'][int(li)]} (cut {a.thr})", flush=True)
    twins = {}
    for v in views:
        blocks = parse_blocks(v)
        for k in range(n_f):
            Kfull = combined_gram(blocks, folds, k, n_test)
            trn = np.flatnonzero(folds != k)
            twins[v, k] = Twin(Kfull, trn, y[trn], prompts[trn], Cs)
        print(f"twin {v}: 5 folds ({time.time() - t0:.0f}s)", flush=True)
    ng_views = [v for v in a.ng_views.split(";") if v]
    ng_Cs = [float(c) for c in a.ng_C.split(",")]
    ng_ws = [float(w) for w in a.ng_weights.split(",")]
    for v in ng_views:
        for k in range(n_f):
            Kfull = ngram_gram(v, texts + te["text"].tolist(), folds, k)
            trn = np.flatnonzero(folds != k)
            twins[v, k] = Twin(Kfull, trn, y[trn], prompts[trn], ng_Cs)
        print(f"n-gram twin {v}: 5 folds ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for rg in a.ranges.split(","):
        lo, hi = (int(x) for x in rg.split("-"))
        sa = argparse.Namespace(a=lo, b=hi)
        for sim in range(a.sims):
            keep = keep_rows("perprompt", y, prompts, rng, sa)
            kt = [texts[i] for i in keep]
            g = mixed_clusters(kt, Emix[keep], a.thr) if Emix is not None else cluster_prompts(kt, a.thr)
            sizes = np.bincount(g)[g]
            adapt = (sizes >= a.min_size) & (sizes <= a.max_size)
            yk, Pp = y[keep], plain[keep]
            pp, pr = pair_stats(g, prompts[keep])
            res = {"range": rg, "sim": sim, "n": len(keep), "purity": purity(g, prompts[keep]), "pair_prec": pp,
                   "pair_rec": pr, "adapt_share": float(adapt.mean()), "plain": metrics.balanced_accuracy(yk, Pp)}
            fk = folds[keep]
            lp = np.log(np.clip(Pp, 1e-9, 1))
            ba = lambda z: metrics.balanced_accuracy(yk, np.eye(K)[z.argmax(1)])
            for b in alphas:
                res[f"plain_b{b:g}"] = ba(balance(lp, b))
            def evidence(vs, C):
                L = np.zeros((len(keep), K))                 # summed twin log-probabilities (adapted rows only)
                for v in vs:
                    for k in range(n_f):
                        loc = np.flatnonzero((fk == k) & adapt)
                        if len(loc):
                            # the cluster centering uses every kept text of the cluster (any fold), as on the test
                            members = np.flatnonzero(np.isin(g, np.unique(g[loc])))
                            P = twins[v, k].predict(keep[members], g[members], C)
                            pos = {r: i for i, r in enumerate(members)}
                            L[loc] += np.log(np.clip(P[[pos[r] for r in loc]], 1e-9, 1))
                return L
            L_ng = {nC: evidence(ng_views, nC) for nC in ng_Cs} if ng_views else {}
            for C in Cs:
                L = evidence(views, C)
                for w in ws:
                    z = lp + w * L
                    res[f"C{C:g}_w{w:g}"] = ba(z)
                    for b in alphas:
                        res[f"C{C:g}_w{w:g}_b{b:g}"] = ba(balance(z, b))
                    for nC, Ln in L_ng.items():
                        for wn in ng_ws:
                            res[f"C{C:g}_w{w:g}_n{nC:g}x{wn:g}"] = ba(z + wn * Ln)
            for nC, Ln in L_ng.items():
                for wn in ng_ws:
                    res[f"C0_n{nC:g}x{wn:g}"] = ba(lp + wn * Ln)
            rows.append(res)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c not in ("range", "sim", "n")]
    var = [c for c in cols if c.startswith(("C", "plain_b"))]          # variants, compared with the plain pipeline
    summ = df.groupby("range")[cols].mean()
    for c in var:
        d = df[c] - df["plain"]
        summ[f"d_{c}"] = d.groupby(df["range"]).mean()
        summ[f"se_{c}"] = d.groupby(df["range"]).std(ddof=1) / np.sqrt(df.groupby("range").size())
        if "_b" in c:                                                  # balancing alone: vs the same unbalanced variant
            d = df[c] - df[c.rsplit("_b", 1)[0]]
            summ[f"dB_{c}"] = d.groupby(df["range"]).mean()
            summ[f"seB_{c}"] = d.groupby(df["range"]).std(ddof=1) / np.sqrt(df.groupby("range").size())
    pd.set_option("display.width", 250)
    print(summ.round(4).T.to_string(), flush=True)
    if a.log:
        for rg, r in summ.iterrows():
            best = max(var, key=lambda c: r[f"d_{c}"])
            s = {"ba": r[best], "ba_fold_mean": r[best], "ba_fold_std": 0.0, "logloss": float("nan")}
            log_experiment(f"{a.log}_{rg}", f"ADAPTIVE CENTERING SIM (diagnostic): plain {a.plain} + centered twins "
                           f"({', '.join(views)}; true-prompt-centered training, clusters {a.min_size}..{a.max_size} "
                           f"at thr {a.thr}, add mode) on held-out prompts keeping {rg} texts per prompt, "
                           f"{a.sims} sims (adaptive_sim.py)", s, time.time() - t0,
                           note=f"plain {r['plain']:.4f}; " + "; ".join(
                               f"{c}: {r[c]:.4f} ({r[f'd_{c}']:+.4f} ± {r[f'se_{c}']:.4f})"
                               for c in var) +
                           f"; purity {r['purity']:.3f}, adapted share {r['adapt_share']:.3f}; best {best} picked "
                           f"on the same sims")


if __name__ == "__main__":
    main()
