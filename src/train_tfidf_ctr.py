"""TF-IDF member with prompt centering (step B applied to n-gram features).

Per fold: the vectorizer is fitted on the training-fold texts (vocabulary capped at --max-features so the centered
matrix can be dense), then every text's TF-IDF row has its unsupervised cluster's mean subtracted. Training-fold
texts are clustered together, held-out texts on their own (as the hidden test is), with matching.cluster_prompts
(threshold 0.92; clusters of < 3 texts stay as they are). No labels and no prompt_id are used for centering.
Model: LogReg on the centered rows (or LinearSVC + inner-fold temperature, as train_tfidf.py).

Usage: python train_tfidf_ctr.py --feats char --max-features 50000 --model lr --C 10
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import K, cv_preds, encode_y, load_folds, load_test, load_train, run_member
from train_emb import center_by_clusters, center_groups
import train_tfidf


def center_rows(X, texts, thr):
    """Cluster-mean centering of one set of texts (tiny sets, which cannot be clustered: global mean)."""
    if len(texts) >= 40:
        return center_by_clusters(X, list(texts), thr)[0]
    return X - X.mean(0)


def fit_member(args, tr_texts, y_tr, g_tr):
    """Fit vectorizers + centered model on the training texts; returns predict(texts, groups=None) -> probabilities.
    predict() centers the given texts with clusters of those texts only (as the hidden test is centered), or with the
    given cluster labels (simulations that cluster a pooled set once).
    g_tr (prompt ids of the training texts) is used only for the SVC's inner temperature split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.svm import LinearSVC
    tr_texts = list(tr_texts)
    vecs = []
    for f in args.feats.split("+"):
        v = train_tfidf.make_vec(f)
        v.set_params(max_features=args.max_features)
        vecs.append(v.fit(tr_texts))

    def dense(texts):
        return np.hstack([v.transform(texts).toarray() for v in vecs]).astype(np.float32)

    def centered(texts, groups):
        texts = list(texts)
        return center_rows(dense(texts), texts, args.thr) if groups is None else center_groups(dense(texts), groups)

    A_tr = center_rows(dense(tr_texts), tr_texts, args.thr)
    if args.model == "lr":
        m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-4).fit(A_tr, y_tr)
        return lambda texts, groups=None: m.predict_proba(centered(texts, groups))
    dec_in = np.zeros((len(y_tr), K))
    for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A_tr, y_tr, g_tr):
        # inner validation rows were centered with the whole training fold's clusters (prompts are grouped,
        # so their clusters hold only inner-validation texts)
        dec_in[iva] = LinearSVC(C=args.C, max_iter=5000).fit(A_tr[itr], y_tr[itr]).decision_function(A_tr[iva])
    T = train_tfidf.fit_temperature(dec_in, y_tr)
    svc = LinearSVC(C=args.C, max_iter=5000).fit(A_tr, y_tr)
    return lambda texts, groups=None: train_tfidf.softmax(svc.decision_function(centered(texts, groups)) * T)


def make_fit_predict(args, groups_all):
    def fit_predict(X_tr, y_tr, X_va, X_te):
        predict = fit_member(args, list(X_tr), y_tr, groups_all[X_tr.index.to_numpy()])
        return predict(list(X_va)), predict(list(X_te))
    return fit_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="char")
    ap.add_argument("--max-features", type=int, default=50_000)
    ap.add_argument("--model", default="lr", choices=["lr", "svc"])
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    tag = args.tag or (f"tfidf_{args.feats.replace('+', '')}_ctr_mf{args.max_features // 1000}k_{args.model}"
                       f"_C{args.C:g}")
    oof, test, _ = cv_preds.cv_predict(make_fit_predict(args, tr["prompt_id"].to_numpy()), tr["text"], y,
                                       te["text"], folds, K, verbose=True)
    change = (f"TF-IDF {args.feats} (max_features {args.max_features}) + prompt centering (train-fold texts and "
              f"held-out texts clustered separately, thr {args.thr}) + "
              f"{'LogReg' if args.model == 'lr' else 'LinearSVC (+inner temperature)'} C={args.C:g}")
    run_member(tag, change, oof, test, y, folds, t0, feats=args.feats, model=args.model, C=args.C,
               max_features=args.max_features, thr=args.thr)


if __name__ == "__main__":
    main()
