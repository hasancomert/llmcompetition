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
from train_emb import center_by_clusters
import train_tfidf


def centered_dense(vecs, fit_texts, groups_of_texts, thr):
    """Fit the vectorizers on fit_texts; return one centered dense matrix per text list in groups_of_texts."""
    out = []
    mats = [[] for _ in groups_of_texts]
    for v in vecs:
        v.fit(fit_texts)
        for i, t in enumerate(groups_of_texts):
            mats[i].append(v.transform(t).toarray())
    for i, t in enumerate(groups_of_texts):
        X = np.hstack(mats[i]).astype(np.float32)
        if len(t) >= 40:
            X, _ = center_by_clusters(X, list(t), thr)
        else:                                                # tiny sets cannot be clustered
            X = X - X.mean(0)
        out.append(X)
    return out


def make_fit_predict(args, groups_all):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.svm import LinearSVC

    def vecs():
        res = []
        for f in args.feats.split("+"):
            v = train_tfidf.make_vec(f)
            v.set_params(max_features=args.max_features)
            res.append(v)
        return res

    def fit_predict(X_tr, y_tr, X_va, X_te):
        tr_texts, va_texts, te_texts = list(X_tr), list(X_va), list(X_te)
        A_tr, A_va, A_te = centered_dense(vecs(), tr_texts, [tr_texts, va_texts, te_texts], args.thr)
        if args.model == "lr":
            m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-4).fit(A_tr, y_tr)
            return m.predict_proba(A_va), m.predict_proba(A_te)
        g_tr = groups_all[X_tr.index.to_numpy()]
        dec_in = np.zeros((len(y_tr), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A_tr, y_tr, g_tr):
            # inner validation rows were centered with the whole training fold's clusters (prompts are grouped,
            # so their clusters hold only inner-validation texts)
            dec_in[iva] = LinearSVC(C=args.C, max_iter=5000).fit(A_tr[itr], y_tr[itr]).decision_function(A_tr[iva])
        T = train_tfidf.fit_temperature(dec_in, y_tr)
        svc = LinearSVC(C=args.C, max_iter=5000).fit(A_tr, y_tr)
        return (train_tfidf.softmax(svc.decision_function(A_va) * T),
                train_tfidf.softmax(svc.decision_function(A_te) * T))
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
