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
--hungarian m1,m2 adds a hard one-to-one assignment (scipy linear_sum_assignment on the final log-posteriors) inside
every cluster of m..max-size texts, after the twins: each text of the cluster gets a different model (columns
plain_h<m>, C<C>_w<w>_h<m>; h<m>_share = share of kept texts in such clusters, h<m>_pure = share of those whose cluster
holds one prompt only).
--gauss-views v1;v2 adds the shared-prompt Gaussian group head (gauss_head.py) of each view: fitted per fold on the
training rows (true prompt ids), it decodes every adapted cluster jointly; its log-posteriors (temperature T) are added
with weight wg on the adapted rows, alone (columns G<T>x<wg>) and on top of the twins (C<C>_w<w>_G<T>x<wg>).
--soft-sink s:rho,... and --typo add the post-steps of the public 0.66 notebooks (post_steps.py) to the plain pipeline
and to every twins + Gaussian variant: damped balancing over the pooled simulated test (suffix _ss<s>r<rho>) and the
gated Beta-Bernoulli typography evidence (suffix _ty; per fold fitted on the training folds), alone and combined.

Usage: python adaptive_sim.py --views qwen3_8b_base_l12,qwen3_4b_base_l12 --C 0.0003,0.001 --ranges 1-2,1-5,2-4,3-8
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import extdata
from common import K, balance, encode_y, load_folds, load_member, load_test, load_train, log_experiment, metrics
from emb_kernel import combined_gram
from emb_raw_explore import parse_blocks
from matching import cluster_prompts, topic_matrix
from scipy.optimize import linear_sum_assignment
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


def hungarian(z, g, lo, hi):
    """argmax predictions, except inside clusters of lo..hi texts: a one-to-one assignment of models to texts
    maximizing the summed log-posteriors (each text of the cluster gets a different model)."""
    pred = z.argmax(1)
    lz = z - z.max(1, keepdims=True)
    lz -= np.log(np.exp(lz).sum(1, keepdims=True))
    for c in np.unique(g):
        ix = np.flatnonzero(g == c)
        if lo <= len(ix) <= min(hi, K):
            r, col = linear_sum_assignment(-lz[ix])
            pred[ix[r]] = col
    return pred


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


class TwinPrimal:
    """Twin with an explicit primal LogReg on centered standardized features (emb_primal.py; same model as Twin), so
    that external training rows can join, centered within their own prompt blocks, with their weights. The model is
    linear, so only the scores S = Z W^T of every row are kept: the centered logits of a cluster are S minus the
    cluster's mean score, plus the intercept (no feature matrix in memory)."""

    def __init__(self, Z_all, trn, ytr, gtr, Cs, Ze=None, ye=None, ge=None, we=None):
        from emb_primal import group_center as gc
        from sklearn.linear_model import LogisticRegression
        Ztr, y, w = gc(Z_all[trn], gtr), ytr, np.ones(len(trn))
        if Ze is not None:
            Ztr, y, w = np.vstack([Ztr, gc(Ze, ge)]), np.concatenate([ytr, ye]), np.concatenate([w, we])
        self.S, self.b, self.cls = {}, {}, {}
        for C in Cs:
            m = LogisticRegression(C=C, max_iter=3000).fit(Ztr, y, sample_weight=w)
            self.S[C], self.b[C], self.cls[C] = Z_all @ m.coef_.T, m.intercept_, m.classes_

    def predict(self, rows, groups, C):
        lg = group_mean_rows(self.S[C][rows], groups)
        lg = self.S[C][rows] - lg + self.b[C]
        lg -= lg.max(1, keepdims=True)
        P = np.full((len(rows), K), 1e-9)
        e = np.exp(lg)
        P[:, self.cls[C]] = e / e.sum(1, keepdims=True)
        return P


def primal_twins(view, tr, te, y, folds, prompts, Cs, ext=None, ext_scale="official"):
    """{fold: TwinPrimal} of one LM view; per fold the official training statistics standardize every row
    (as emb_kernel.grams), external rows allowed for the fold (leak filter) join the training set."""
    from emb_primal import raw_blocks, scale, train_stats
    Xtr, Xte, Xe = raw_blocks(view, tr, te, ext)
    X_all = np.vstack([Xtr, Xte])
    out = {}
    for k in range(int(folds.max()) + 1):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        mu, sd = train_stats(Xtr[trn])
        kw = {}
        if ext is not None:
            ok = np.flatnonzero(ext.allowed(set(prompts[val])))
            Ze = scale(Xe[ok], *(train_stats(Xe[ok]) if ext_scale == "own" else (mu, sd)))
            kw = dict(Ze=Ze, ye=ext.y[ok], ge=ext.df["prompt_group_id"].to_numpy()[ok], we=ext.w[ok])
        out[k] = TwinPrimal(scale(X_all, mu, sd), trn, y[trn], prompts[trn], Cs, **kw)
    return out


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
    ap.add_argument("--linkage", default="average", choices=["average", "complete", "weighted"],
                    help="tfidf clustering: linkage method")
    ap.add_argument("--max-df", type=int, default=40, help="tfidf clustering: absolute max document count")
    ap.add_argument("--ngram", type=int, default=1, help="tfidf clustering: 2 = content-word unigrams + bigrams")
    ap.add_argument("--mix-emb", default="qwen3_8b_base_l12@7", help="--cluster mix: name@layer-index of the embedding")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=18)
    ap.add_argument("--balance", default="", help="Sinkhorn balancing strengths alpha (comma-separated; empty: none)")
    ap.add_argument("--primal", action="store_true", help="LM twins as explicit primal LR (TwinPrimal)")
    ap.add_argument("--ext-scale", default="official", choices=["official", "own"])
    ap.add_argument("--bucket-w", default="", help="';'-separated twin weights by cluster size, 'w2/w3-4/w5+' (e.g. "
                    "0.25/0.5/0.75); columns C<C>_s<w2>-<w34>-<w5>")
    ap.add_argument("--ng-views", default="", help="n-gram twin views, ';'-separated stream sets (e.g. char+word+fw+shapew)")
    ap.add_argument("--ng-C", default="10,30", help="C grid of the n-gram twins")
    ap.add_argument("--ng-weights", default="0.25,0.5", help="weight grid of the n-gram twin evidence")
    ap.add_argument("--gauss-views", default="", help="';'-separated block specs of Gaussian group heads")
    ap.add_argument("--gauss-d", type=int, default=256)
    ap.add_argument("--gauss-T", default="1,3", help="temperature grid of the Gaussian head")
    ap.add_argument("--gauss-w", default="0.1,0.25,0.5", help="weight grid of the Gaussian evidence")
    ap.add_argument("--lora-jobs", default="", help="LoRA24 twins: comma-separated jobs dirs (f0..f4 with held-out "
                    "features and readouts); centered readout on features centered within cluster and fold")
    ap.add_argument("--lora-w", default="0.25,0.5,1", help="weight grid of the LoRA twin evidence")
    ap.add_argument("--soft-sink", default="", help="damped balancing settings strength:rho, comma-separated")
    ap.add_argument("--typo", action="store_true", help="add the gated typography evidence (post_steps.Typography)")
    ap.add_argument("--hungarian", default="", help="one-to-one assignment inside clusters of m..max-size texts after "
                    "the twins; comma-separated m values (empty: none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="", help="EXPERIMENTS.md tag prefix (empty: no rows)")
    extdata.add_args(ap)
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
    hms = [int(m) for m in a.hungarian.split(",")] if a.hungarian else []
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
    ext = extdata.load(a, tr)
    for v in views:
        if a.primal or ext is not None:              # external rows: primal twins (the Gram would be too large)
            for k, tw in primal_twins(v, tr, te, y, folds, prompts, Cs, ext, a.ext_scale).items():
                twins[v, k] = tw
        else:
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
    gviews = [v for v in a.gauss_views.split(";") if v]
    gTs = [float(t) for t in a.gauss_T.split(",")]
    gws = [float(w) for w in a.gauss_w.split(",")]
    gU = {}                                              # (view, fold) -> (head, all train rows transformed)
    if gviews:
        from emb_primal import raw_blocks
        from gauss_head import GaussHead
        for v in gviews:
            Xtr = raw_blocks(v, tr, te)[0]
            for k in range(n_f):
                trn = np.flatnonzero(folds != k)
                h = GaussHead(d=a.gauss_d).fit(Xtr[trn], y[trn], prompts[trn], K)
                gU[v, k] = (h, h.transform(Xtr))
            del Xtr
            print(f"gauss {v}: 5 folds ({time.time() - t0:.0f}s)", flush=True)
    lora = []                                            # per jobs dir: (held-out features of every train row, readouts)
    for d in [d for d in a.lora_jobs.split(",") if d]:
        Z, ros = None, []
        for k in range(n_f):
            idx, V = np.load(f"{d}/f{k}/valid_idx.npy"), np.load(f"{d}/f{k}/valid_features.npy")
            if Z is None:
                Z = np.zeros((len(y), V.shape[1]), np.float32)
            Z[idx] = V
            ros.append(np.load(f"{d}/f{k}/readouts.npz"))
        lora.append((Z, ros))
        print(f"lora twins {d}: features {Z.shape} ({time.time() - t0:.0f}s)", flush=True)
    lws = [float(w) for w in a.lora_w.split(",")] if lora else []
    sinks = [tuple(float(x) for x in t.split(":")) for t in a.soft_sink.split(",") if t]
    prior = (np.bincount(y, minlength=K) + 1.0) / (len(y) + K)
    typo_ev = None
    if a.typo:                                           # out-of-fold typography evidence of every train row
        from post_steps import Typography
        typo_ev, typo_words, typo_active = np.zeros((len(y), K)), np.zeros(len(y)), np.zeros(len(y))
        for k in range(n_f):
            trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            e, w_, act = Typography().fit([texts[i] for i in trn], y[trn], K).evidence([texts[i] for i in val])
            typo_ev[val], typo_words[val], typo_active[val] = e, w_, act

    def post_variants(res, name, z, keep, ba):
        """Soft-Sinkhorn / typography columns of one final score matrix."""
        from post_steps import soft_sinkhorn, typography_apply
        outs = [("", z)] + [(f"_ss{s_:g}r{r_:g}", soft_sinkhorn(z, s_, r_, prior)) for s_, r_ in sinks]
        for sfx, zz in outs:
            if sfx:
                res[f"{name}{sfx}"] = ba(zz)
            if typo_ev is not None:
                res[f"{name}{sfx}_ty"] = ba(typography_apply(zz, typo_ev[keep], typo_words[keep], typo_active[keep]))

    rng = np.random.default_rng(a.seed)
    rows = []
    for rg in a.ranges.split(","):
        lo, hi = (int(x) for x in rg.split("-"))
        sa = argparse.Namespace(a=lo, b=hi)
        for sim in range(a.sims):
            keep = keep_rows("perprompt", y, prompts, rng, sa)
            kt = [texts[i] for i in keep]
            g = mixed_clusters(kt, Emix[keep], a.thr) if Emix is not None else \
                cluster_prompts(kt, a.thr, a.max_df, a.linkage, a.ngram)
            sizes = np.bincount(g)[g]
            adapt = (sizes >= a.min_size) & (sizes <= a.max_size)
            yk, Pp = y[keep], plain[keep]
            pp, pr = pair_stats(g, prompts[keep])
            res = {"range": rg, "sim": sim, "n": len(keep), "purity": purity(g, prompts[keep]), "pair_prec": pp,
                   "pair_rec": pr, "adapt_share": float(adapt.mean()), "plain": metrics.balanced_accuracy(yk, Pp)}
            fk = folds[keep]
            lp = np.log(np.clip(Pp, 1e-9, 1))
            ba = lambda z: metrics.balanced_accuracy(yk, np.eye(K)[z.argmax(1)])
            post_variants(res, "plain", lp, keep, ba)
            for b in alphas:
                res[f"plain_b{b:g}"] = ba(balance(lp, b))
            ba_pred = lambda p: metrics.balanced_accuracy(yk, np.eye(K)[p])
            for m in hms:
                big = (sizes >= m) & (sizes <= a.max_size)
                res[f"h{m}_share"] = float(big.mean())
                res[f"h{m}_pure"] = float(np.mean([len(np.unique(prompts[keep][g == c])) == 1 for c in g[big]])) \
                    if big.any() else float("nan")
                res[f"plain_h{m}"] = ba_pred(hungarian(lp, g, m, a.max_size))
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
            L_g = {}
            for T in (gTs if gviews else []):                # Gaussian group heads: clusters decoded jointly
                Lg = np.zeros((len(keep), K))
                for v in gviews:
                    for k in range(n_f):
                        loc = np.flatnonzero((fk == k) & adapt)
                        if len(loc):
                            members = np.flatnonzero(np.isin(g, np.unique(g[loc])))
                            h, U = gU[v, k]
                            S = h.scores(U[keep[members]], g[members], T)
                            pos = {r: i for i, r in enumerate(members)}
                            Lg[loc] += S[[pos[r] for r in loc]]
                L_g[T] = Lg
            Ll = None
            if lora:                                     # LoRA twins: centered head on cluster-centered features
                from lora24_member import readout
                Ll = np.zeros((len(keep), K))
                for Z, ros in lora:
                    for k in range(n_f):
                        loc = np.flatnonzero((fk == k) & adapt)
                        if len(loc):
                            Zl, gl = Z[keep[loc]], g[loc]
                            Zc = Zl.copy()
                            for c in np.unique(gl):
                                Zc[gl == c] -= Zl[gl == c].mean(0)
                            Ll[loc] += np.log(np.clip(readout(Zc, ros[k], "center"), 1e-9, 1))
                for wl in lws:
                    res[f"L{wl:g}"] = ba(lp + wl * Ll)
            for T, Lg in L_g.items():
                for wg in gws:
                    zg = lp + wg * Lg
                    res[f"G{T:g}x{wg:g}"] = ba(zg)
                    for m in hms:
                        res[f"G{T:g}x{wg:g}_h{m}"] = ba_pred(hungarian(zg, g, m, a.max_size))
            buckets = [tuple(float(x) for x in t.split("/")) for t in a.bucket_w.split(";") if t]
            for C in Cs:
                L = evidence(views, C)
                for bw in buckets:                           # weight by the size of the text's cluster
                    wrow = np.where(sizes == 2, bw[0], np.where(sizes <= 4, bw[1], bw[2]))
                    res[f"C{C:g}_s{bw[0]:g}-{bw[1]:g}-{bw[2]:g}"] = ba(lp + wrow[:, None] * L)
                for w in ws:
                    z = lp + w * L
                    res[f"C{C:g}_w{w:g}"] = ba(z)
                    for m in hms:
                        res[f"C{C:g}_w{w:g}_h{m}"] = ba_pred(hungarian(z, g, m, a.max_size))
                    post_variants(res, f"C{C:g}_w{w:g}", z, keep, ba)
                    for T, Lg in L_g.items():
                        for wg in gws:
                            res[f"C{C:g}_w{w:g}_G{T:g}x{wg:g}"] = ba(z + wg * Lg)
                            for wl in lws:
                                res[f"C{C:g}_w{w:g}_G{T:g}x{wg:g}_L{wl:g}"] = ba(z + wg * Lg + wl * Ll)
                            post_variants(res, f"C{C:g}_w{w:g}_G{T:g}x{wg:g}", z + wg * Lg, keep, ba)
                            for m in hms:
                                res[f"C{C:g}_w{w:g}_G{T:g}x{wg:g}_h{m}"] = ba_pred(hungarian(z + wg * Lg, g, m,
                                                                                             a.max_size))
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
    var = [c for c in cols if c.startswith(("C", "G", "L", "plain_b", "plain_h", "plain_ss", "plain_ty"))]   # vs plain
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
