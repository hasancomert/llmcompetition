"""Prompt clustering + soft one-model-per-prompt assignment (PLAN.md step G), simulated on OOF predictions.

Test texts carry no prompt_id, so prompts are recovered by clustering the texts on topic words. Inside each
cluster the texts are assigned to models with the Hungarian algorithm: assigning the j-th text of a cluster to
the same model costs an extra lam * j (soft constraint; lam = 0 is the plain argmax, lam = inf a strict
one-per-model assignment). In train every prompt has each model exactly once, but test model counts are unequal,
so the simulation drops texts with model-specific keep rates before matching.

Usage: python matching.py --tag blend_v2_tok_ba
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import normalize

from common import K, OUT, encode_y, iter_folds, load_folds, load_member, load_train, metrics


MIN_CLUSTER_ROWS = 40   # below this the test is too small to cluster (the interactive 1-3 row test)


def topic_matrix(texts, max_df_docs=40):
    """Unit-norm TF-IDF rows of content words (topic, not style). max_df is an ABSOLUTE document count: a word
    shared by one prompt's texts appears in <= ~18 documents whatever the test size, so the representation does not
    change with n (audit finding: a relative max_df and a fixed SVD size tied the threshold to 150 prompts)."""
    tv = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, min_df=2,
                         max_df=max(int(max_df_docs), 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        X = tv.fit_transform(texts)
    except ValueError:            # no term survives pruning (tiny or odd input): every text its own topic
        return None
    return normalize(X)


def cluster_tree(texts, max_df_docs=40):
    """Average-linkage tree (scipy) over cosine distances of topic_matrix rows. Returns (Z, nz, n): Z is None
    when fewer than two texts have a surviving content word."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    n = len(texts)
    X = topic_matrix(texts, max_df_docs)
    if X is None:
        return None, np.array([], dtype=np.int64), n
    nz = np.flatnonzero(X.getnnz(axis=1) > 0)
    if len(nz) < 2:
        return None, nz, n
    D = 1.0 - (X[nz] @ X[nz].T).toarray()
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    return linkage(squareform(D, checks=False), method="average"), nz, n


def cut_tree(tree, threshold):
    """Cluster labels for one distance threshold; texts without content words are singletons."""
    from scipy.cluster.hierarchy import fcluster
    Z, nz, n = tree
    labels = np.arange(n) + n
    if Z is not None:
        labels[nz] = fcluster(Z, t=threshold, criterion="distance")
    return np.unique(labels, return_inverse=True)[1]


def cluster_prompts(texts, threshold=0.92, max_df_docs=40):
    """Prompt clusters of a set of texts; the number of clusters follows from the threshold (not given).
    Texts with no surviving content word become singletons instead of crashing the cosine metric."""
    if len(texts) < MIN_CLUSTER_ROWS:
        return np.arange(len(texts))
    return cut_tree(cluster_tree(texts, max_df_docs), threshold)


def scale_study(P, y, prompts, texts, reps=3, seed=0):
    """Cluster quality and matching gain vs the number of prompts, keep scenario, threshold and max_df."""
    rng = np.random.default_rng(seed)
    up = np.unique(prompts)
    rows = []
    thrs = (0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96)
    for n_pr in (30, 60, 100, 150):
        for rep in range(reps if n_pr < 150 else 1):
            chosen = set(rng.choice(up, n_pr, replace=False)) if n_pr < 150 else set(up)
            base_ix = np.flatnonzero([p in chosen for p in prompts])
            for scen, keep in (("full", None), ("drop.35-1", (0.35, 1.0)), ("drop.2-.6", (0.2, 0.6))):
                ix = base_ix if keep is None else base_ix[drop_mask(y[base_ix], None, rng, *keep)]
                for mdf in (25, 40, 80):
                    tree = cluster_tree([texts[i] for i in ix], mdf)
                    base = ba_labels(y[ix], P[ix].argmax(1))
                    for thr in thrs:
                        c = cut_tree(tree, thr)
                        rows.append(dict(n_prompts=n_pr, rep=rep, scen=scen, max_df=mdf, thr=thr, n=len(ix),
                                         ratio=len(np.unique(c)) / n_pr, ari=adjusted_rand_score(prompts[ix], c),
                                         gain=ba_labels(y[ix], match(P[ix], c, 2.0)) - base))
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    g = df.groupby(["max_df", "thr", "scen", "n_prompts"])[["ratio", "ari", "gain"]].mean().round(3)
    print(g.unstack("n_prompts").to_string())
    return df


def match_cluster(L, lam, big=1e6):
    """L: (n, K) log-probabilities of one cluster. The j-th text given to the same model costs lam * j extra
    (lam = inf -> big * j, the large-lambda limit). Copies are not capped for clusters up to 100 texts, so the
    soft penalty is the whole objective. Returns the assigned class per row."""
    n = L.shape[0]
    if n == 1 or lam == 0:
        return L.argmax(1)
    copies = n if n <= 100 else int(np.ceil(n / K)) + 3
    step = big if not np.isfinite(lam) else lam
    cost = np.concatenate([-L + step * j for j in range(copies)], axis=1)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(n, dtype=np.int64)
    out[rows] = cols % K
    return out


def match(P, groups, lam):
    L = np.log(np.clip(P, 1e-12, 1))
    pred = P.argmax(1).copy()
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        pred[ix] = match_cluster(L[ix], lam)
    return pred


def ba_labels(y, pred):
    return metrics.balanced_accuracy(y, np.eye(K)[pred])


def drop_mask(y, prompts, rng, keep_lo=0.35, keep_hi=1.0):
    """Model-specific keep rates (unequal test counts): model m keeps each of its texts with prob q_m."""
    q = rng.uniform(keep_lo, keep_hi, K)
    return rng.random(len(y)) < q[y]


def nested_eval(args):
    """lambda chosen on the other 4 folds' rows, scored on the held-out fold (clusters are unsupervised and
    computed once per scenario on all kept texts, as at test time). Scenarios: full OOF, mild and harsh drops."""
    import time
    from common import log_experiment, scores
    t0 = time.time()
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    P, _ = load_member(args.tag)
    prompts = tr["prompt_id"].to_numpy()
    texts = tr["text"].tolist()
    lams = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
    scen = [("full", None)] + [(f"drop {lo}-{hi}", (lo, hi)) for lo, hi in ((0.35, 1.0), (0.2, 0.6))]
    rng = np.random.default_rng(1)
    for name, keep in scen:
        res_base, res_nested, chosen = [], [], []
        for sim in range(1 if keep is None else args.sims):
            ix = np.arange(len(y)) if keep is None else np.flatnonzero(drop_mask(y, prompts, rng, *keep))
            yy, ff, PP = y[ix], folds[ix], P[ix]
            g = cluster_prompts([texts[i] for i in ix], args.thr)
            preds = {lam: match(PP, g, lam) for lam in lams}
            out = np.empty(len(ix), dtype=np.int64)
            for k in np.unique(ff):
                inn, held = ff != k, ff == k
                best = max(lams, key=lambda lam: ba_labels(yy[inn], preds[lam][inn]))
                chosen.append(best)
                out[held] = preds[best][held]
            res_base.append(ba_labels(yy, PP.argmax(1)))
            res_nested.append(ba_labels(yy, out))
        print(f"{name:14s} sims {len(res_base)}: argmax BA {np.mean(res_base):.4f} -> nested matched BA "
              f"{np.mean(res_nested):.4f} (+{np.mean(res_nested) - np.mean(res_base):.4f}); lambdas chosen "
              f"{sorted(set(chosen), key=float)} counts {[chosen.count(l) for l in sorted(set(chosen), key=float)]}")
        s_ = dict(ba=float(np.mean(res_nested)), ba_fold_mean=float(np.mean(res_nested)), ba_fold_std=float(np.std(res_nested)),
                  logloss=float("nan"))
        log_experiment(f"match_{args.tag}_{name.replace(' ', '').replace('.', '')}",
                       f"prompt clustering (thr {args.thr}) + soft Hungarian on {args.tag}, scenario {name}; lambda nested",
                       s_, time.time() - t0,
                       note=f"argmax BA {np.mean(res_base):.4f} on the same rows; ± = std over {len(res_base)} drop simulations; log loss n/a (labels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="blend_v2_tok_ba")
    ap.add_argument("--sims", type=int, default=5)
    ap.add_argument("--nested", action="store_true", help="nested lambda selection + EXPERIMENTS.md rows")
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--scale", action="store_true", help="cluster/matching scale study (30-150 prompts)")
    args = ap.parse_args()
    if args.nested:
        return nested_eval(args)
    if args.scale:
        tr = load_train()
        P, _ = load_member(args.tag)
        df = scale_study(P, encode_y(tr["model"]), tr["prompt_id"].to_numpy(), tr["text"].tolist())
        df.to_csv(OUT / "matching_scale_study.csv", index=False)
        return
    tr = load_train()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    P, _ = load_member(args.tag)
    prompts = tr["prompt_id"].to_numpy()
    texts = tr["text"].tolist()
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]

    print(f"member {args.tag}: OOF BA {ba_labels(y, P.argmax(1)):.4f}")
    # 1) clustering quality on the pooled 2,699 OOF texts (150 prompts, k not given)
    clus = {}
    for thr in (0.4, 0.5, 0.55, 0.6, 0.7):
        c = cluster_prompts(texts, thr)
        clus[thr] = c
        h, cpl, v = homogeneity_completeness_v_measure(prompts, c)
        print(f"cluster thr {thr}: clusters {len(np.unique(c))} (true 150), ARI {adjusted_rand_score(prompts, c):.3f}, "
              f"homogeneity {h:.3f}, completeness {cpl:.3f}")

    # 2) matching on the full OOF (every prompt has all 18 models: optimistic) and with test-like drops
    rng = np.random.default_rng(0)
    masks = [drop_mask(y, prompts, rng) for _ in range(args.sims)]
    settings = [("true prompt_id", prompts)] + [(f"clusters thr {t}", clus[t]) for t in (0.5, 0.6)]
    for name, groups in settings:
        row_full = [ba_labels(y, match(P, groups, lam)) for lam in lams]
        row_drop = []
        for lam in lams:
            vals = []
            for m in masks:
                ix = np.flatnonzero(m)
                g = groups[ix] if name == "true prompt_id" else cluster_prompts([texts[i] for i in ix], float(name.split()[-1]))
                vals.append(ba_labels(y[ix], match(P[ix], g, lam)))
            row_drop.append(np.mean(vals))
        print(f"{name:18s} full : " + "  ".join(f"lam {l:g}: {v:.4f}" for l, v in zip(lams, row_full)))
        print(f"{name:18s} drop : " + "  ".join(f"lam {l:g}: {v:.4f}" for l, v in zip(lams, row_drop)))


if __name__ == "__main__":
    main()
