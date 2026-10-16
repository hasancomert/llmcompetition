"""Baseline 1: character / word n-gram TF-IDF + LogisticRegression or LinearSVC, on the shared folds.

The vectorizer is fitted inside each fold on the training texts only (default). --transductive also adds the
validation and test texts (no labels) to the vectorizer fit: at submission time the test texts are available,
so this mirrors what the final notebook could do; it is measured separately (PLAN.md Adım 2.1).
LinearSVC has no probabilities: its decision values are turned into probabilities with a softmax temperature
fitted on an inner grouped split of the training fold (never on the validation fold).

Examples:
  python train_tfidf.py --feats char --model lr --C 10
  python train_tfidf.py --feats char+word --model svc --C 0.3
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from common import K, cv_preds, encode_y, load_folds, load_member, load_test, load_train, run_member
from compare import fmt, paired_bootstrap
import extdata

VECS = {
    "char": lambda: TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, sublinear_tf=True,
                                    min_df=2, max_features=400_000, dtype=np.float32),
    "charwb": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False, sublinear_tf=True,
                                      min_df=2, max_features=400_000, dtype=np.float32),
    "word": lambda: TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, sublinear_tf=True,
                                    min_df=2, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32),
}


def token_id_analyzer(name):
    """Unigrams + bigrams of one lab tokenizer's token ids (tokenizer_features.TOKENIZERS name)."""
    from tokenizer_features import load
    enc = load(name)

    def analyze(text):
        ids = [str(i) for i in enc(text)]
        return ids + [a + "_" + b for a, b in zip(ids, ids[1:])]
    return analyze


def make_vec(f):
    if f.startswith("tok_"):
        return TfidfVectorizer(analyzer=token_id_analyzer(f[4:]), sublinear_tf=True, min_df=2, dtype=np.float32)
    return VECS[f]()


def vectorize(feats, fit_texts, apply_texts):
    mats = [[] for _ in apply_texts]
    for f in feats.split("+"):
        v = make_vec(f).fit(fit_texts)
        for i, t in enumerate(apply_texts):
            mats[i].append(v.transform(t))
    return [sp.hstack(m).tocsr() if len(m) > 1 else m[0] for m in mats]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(dec, y):
    def nll(logt):
        p = softmax(dec * np.exp(logt))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))


def make_fit_predict(args, groups_all, ext=None):
    """ext (extdata.Ext): external rows join the training fold only (minus rows topic-matched to its validated
    prompts); the vectorizers are fitted on the official training texts (--ext-vocab official, default) or on those +
    the external rows (all); validation stays official."""
    if ext is not None:
        return _make_fit_predict_ext(args, groups_all, ext)

    def fit_predict(X_tr, y_tr, X_va, X_te):
        tr_texts, va_texts, te_texts = list(X_tr), list(X_va), list(X_te)
        fit_texts = tr_texts + (va_texts + te_texts if args.transductive else [])
        A_tr, A_va, A_te = vectorize(args.feats, fit_texts, [tr_texts, va_texts, te_texts])
        if args.model == "lr":
            m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-4).fit(A_tr, y_tr)
            return m.predict_proba(A_va), m.predict_proba(A_te)
        # LinearSVC + temperature from an inner grouped split of the training fold
        g_tr = groups_all[X_tr.index.to_numpy()]
        dec_in = np.zeros((len(y_tr), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A_tr, y_tr, g_tr):
            svc = LinearSVC(C=args.C, max_iter=5000).fit(A_tr[itr], y_tr[itr])
            dec_in[iva] = svc.decision_function(A_tr[iva])
        T = fit_temperature(dec_in, y_tr)
        svc = LinearSVC(C=args.C, max_iter=5000).fit(A_tr, y_tr)
        return softmax(svc.decision_function(A_va) * T), softmax(svc.decision_function(A_te) * T)
    return fit_predict


def _make_fit_predict_ext(args, groups_all, ext):
    def fit_predict(X_tr, y_tr, X_va, X_te):
        g_tr = groups_all[X_tr.index.to_numpy()]
        ok = np.flatnonzero(ext.allowed(set(groups_all[X_va.index.to_numpy()])))
        e_texts, y_e, w_e, leak = [ext.texts[i] for i in ok], ext.y[ok], ext.w[ok], [ext.leak[i] for i in ok]
        tr_texts = list(X_tr)
        fit_texts = tr_texts + (e_texts if args.ext_vocab == "all" else [])
        A_tr, A_e, A_va, A_te = vectorize(args.feats, fit_texts, [tr_texts, e_texts, list(X_va), list(X_te)])

        def train_set(rows, e_rows):
            return (sp.vstack([A_tr[rows], A_e[e_rows]]).tocsr(), np.concatenate([y_tr[rows], y_e[e_rows]]),
                    np.concatenate([np.ones(len(rows)), w_e[e_rows]]))

        if args.model == "lr":
            A, yy, sw = train_set(np.arange(len(y_tr)), np.arange(len(y_e)))
            m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-4).fit(A, yy, sample_weight=sw)
            return m.predict_proba(A_va), m.predict_proba(A_te)
        dec_in = np.zeros((len(y_tr), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A_tr, y_tr, g_tr):
            held = set(g_tr[iva])
            e_ok = np.array([i for i, s in enumerate(leak) if not (s & held)], dtype=int)
            A, yy, sw = train_set(itr, e_ok)
            dec_in[iva] = LinearSVC(C=args.C, max_iter=5000).fit(A, yy, sample_weight=sw).decision_function(A_tr[iva])
        T = fit_temperature(dec_in, y_tr)
        A, yy, sw = train_set(np.arange(len(y_tr)), np.arange(len(y_e)))
        svc = LinearSVC(C=args.C, max_iter=5000).fit(A, yy, sample_weight=sw)
        return softmax(svc.decision_function(A_va) * T), softmax(svc.decision_function(A_te) * T)
    return fit_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="char")
    ap.add_argument("--model", default="lr", choices=["lr", "svc"])
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--transductive", action="store_true")
    ap.add_argument("--tag", default=None)
    extdata.add_args(ap)
    ap.add_argument("--ext-vocab", default="official", choices=["official", "all"])
    args = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    ext = extdata.load(args, tr)
    if ext is not None and args.transductive:
        ap.error("--ext and --transductive are exclusive")
    tag0 = args.tag or f"tfidf_{args.feats.replace('+', '')}_{args.model}_C{args.C:g}{'_td' if args.transductive else ''}"
    tag = tag0 + extdata.tag_suffix(args) + ("-vall" if ext is not None and args.ext_vocab == "all" else "")
    oof, test, _ = cv_preds.cv_predict(make_fit_predict(args, tr["prompt_id"].to_numpy(), ext), tr["text"], y,
                                       te["text"], folds, K, verbose=True)
    change = (f"TF-IDF {args.feats} + {'LogReg' if args.model == 'lr' else 'LinearSVC (+inner temperature)'} "
              f"C={args.C:g}{', vectorizer fit incl. val/test texts' if args.transductive else ''}{extdata.describe(args)}")
    note = ""
    if ext is not None:                       # external-data runs: compare with the same member without them
        try:
            note = f"paired vs {tag0}: " + fmt(paired_bootstrap(y, load_member(tag0)[0], oof, tr["prompt_id"].to_numpy()))
        except FileNotFoundError:
            note = f"paired vs {tag0}: n/a (reference member absent)"
        print(note, flush=True)
    vec_params = {f: (repr(make_vec(f).get_params()) if not f.startswith("tok_") else "token-id uni+bigram")
                  for f in args.feats.split("+")}
    run_member(tag, change, oof, test, y, folds, t0, note=note, feats=args.feats, model=args.model, C=args.C,
               transductive=args.transductive, vec_params=vec_params, ext=args.ext,
               ext_weight=args.ext_weight if ext is not None else None, ext_rows=len(ext) if ext else 0)


if __name__ == "__main__":
    main()
