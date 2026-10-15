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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import normalize

from common import K, encode_y, iter_folds, load_folds, load_member, load_train, metrics


def topic_vectors(texts, n_components=128, seed=0):
    """Unit-norm LSA vectors of content words (topic, not style): stop words removed, lowercase, sublinear tf."""
    tv = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = tv.fit_transform(texts)
    k = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    Z = TruncatedSVD(k, random_state=seed).fit_transform(X) if k >= 2 else X.toarray()
    return normalize(Z)


def cluster_prompts(texts, threshold=0.75, n_components=128):
    """Average-linkage agglomerative clustering on cosine distance; the number of clusters is not given, it
    follows from the distance threshold. Returns integer cluster labels."""
    n = len(texts)
    if n < 3:
        return np.arange(n)
    Z = topic_vectors(texts, n_components)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=threshold).fit_predict(Z)


def match_cluster(L, lam):
    """L: (n, K) log-probabilities of one cluster. Returns the assigned class per row."""
    n = L.shape[0]
    if n == 1 or lam == 0:
        return L.argmax(1)
    copies = int(np.ceil(n / K)) + 1
    cost = np.concatenate([-L + (lam * j if np.isfinite(lam) else (0 if j == 0 else 1e6)) for j in range(copies)], axis=1)
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
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    if args.nested:
        return nested_eval(args)
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
