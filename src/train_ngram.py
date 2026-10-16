"""Per-text n-gram members: topic-free text streams, NB-SVM, naive Bayes and temperature-calibrated heads.

Every held-out / test text is classified from its own content only (no test-time clustering, centering or matching).
Per fold, vectorizers are fitted on the training-fold texts only; every supervised step (classifier, NB log-count
ratios, temperature) uses training-fold rows only. Temperatures come from an inner StratifiedGroupKFold(4) split of
the training fold (grouped by prompt_id), never from the held-out fold.

Text streams (each is a transform of the raw text + a TF-IDF vectorizer; several are hstacked with --feats a+b+c):
  char / char16 / char17   raw char n-grams 1-5 / 1-6 / 1-7 (case kept, cross-word punctuation patterns kept)
  word / word13            word uni+bigrams / 1-3 grams (lowercased, punctuation marks are tokens)
  shapec                   char-level shape: letters -> x / X, digits -> 9, punctuation and spaces kept; char 1-7
  shapew                   word-level shape: each word -> x (lower) / X (Capitalized) / A (ALLCAPS) / 9 (number),
                           punctuation and spaces kept; char n-grams 1-8 (punctuation context over ~4 words)
  fw / fwx                 function-word stream: function words kept (lowercased, apostrophe type kept), every
                           content word -> W (fwx: W + coarse suffix class, e.g. Wing / Wly / Wed / Ws / Wtion);
                           word n-grams 1-3 over that token sequence
Models:
  lr      multinomial LogisticRegression (optionally --temp: inner temperature on its log-probabilities)
  svc     LinearSVC (crammer-free one-vs-rest) + inner temperature
  nbsvm   one-vs-rest NB-SVM: per class the features are scaled by the NB log-count ratio r_c (training rows only),
          a binary LinearSVC (or --nb-base lr) is fitted per class; decision values -> softmax with inner temperature
  mnb/cnb MultinomialNB / ComplementNB (alpha=--alpha) on the TF-IDF rows; log-probs -> inner temperature

Examples:
  python train_ngram.py --feats shapew --model lr --C 10
  python train_ngram.py --feats char+word+shapew+fw --model lr --C 30 --save
  python train_ngram.py --feats char+word --model nbsvm --C 0.1 --binary
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from common import (K, OUT, encode_y, iter_folds, load_folds, load_member, load_test, load_train, log_experiment,
                    metrics, run_member, scores)
from compare import paired_bootstrap
from train_tfidf import fit_temperature, softmax

SCRATCH = OUT / "ngram_scratch"  # untracked outputs/ (OOF + test preds of every run, for blend checks)
REF_TAG = "tfidf_charword_lr_C30"

# ---------------------------------------------------------------- text transforms

FUNCTION_WORDS = set("""
a an the this that these those such what which whose whatever whichever
i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her hers
herself it its itself they them their theirs themselves one ones oneself someone something anyone anything everyone
everything nobody nothing none somebody anybody everybody who whom whoever where when why how wherever whenever
however
about above across after against along amid among amongst around as at before behind below beneath beside besides
between beyond by despite down during except for from in inside into like near of off on onto out outside over past
per since through throughout till to toward towards under underneath unlike until up upon via with within without
and but or nor so yet for both either neither not only also whether if unless because although though while whereas
than then once lest
be am is are was were been being have has had having do does did doing done will would shall should can could may
might must ought need dare
not no nor very too quite rather really just even still already almost nearly perhaps maybe often always never
sometimes usually ever again further furthermore moreover therefore thus hence indeed instead otherwise meanwhile
nevertheless nonetheless accordingly consequently likewise similarly namely especially particularly simply merely
here there now today yet soon later more most less least much many few fewer fewest little lot lots several each every
all any some other others another same own enough plenty more so as well
ll ve re s d t m
let lets let's
it's that's there's here's what's who's isn't aren't wasn't weren't don't doesn't didn't won't wouldn't can't cannot
couldn't shouldn't mustn't hasn't haven't hadn't i'm i've i'd i'll you're you've you'd you'll we're we've we'd we'll
they're they've they'd they'll he's she's he'd she'd it'd it'll
""".split())

_FW_TOK = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")
_SHAPEW_TOK = re.compile(r"(\w+)|(\s+)|(.)", re.S)


def shape_char(s: str) -> str:
    return "".join(("X" if ch.isupper() else "x") if ch.isalpha() else ("9" if ch.isdigit() else ch) for ch in s)


def _word_class(w: str) -> str:
    if any(c.isdigit() for c in w):
        return "9"
    if len(w) > 1 and w.isupper():
        return "A"
    return "X" if w[0].isupper() else "x"


def shape_word(s: str) -> str:
    out = []
    for m in _SHAPEW_TOK.finditer(s):
        if m.group(1) is not None:
            out.append(_word_class(m.group(1)))
        elif m.group(2) is not None:
            out.append(" ")
        else:
            out.append(m.group(3))
    return "".join(out)


def _suffix_class(w: str) -> str:
    for suf in ("ing", "tion", "sion", "ment", "ness", "ity", "ly", "ed", "es", "s", "al", "ive", "ous", "able", "er"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return "W" + suf
    return "W"


def fw_tokens(s: str, suffix: bool = False) -> str:
    out = []
    for t in _FW_TOK.findall(s):
        if t[0].isalnum() or t[0] == "_":
            low = t.lower()
            if low.replace("’", "'") in FUNCTION_WORDS:
                out.append(low)
            elif suffix:
                out.append(("C" if t[0].isupper() else "") + _suffix_class(low))
            else:
                out.append("W")
        else:
            out.append(t)
    return " ".join(out)


TRANSFORMS = {
    "raw": lambda s: s,
    "shapec": shape_char,
    "shapew": shape_word,
    "fw": lambda s: fw_tokens(s, False),
    "fwx": lambda s: fw_tokens(s, True),
}

WORD_PAT = r"(?u)\b\w+\b|[^\w\s]"
SPACE_SPLIT = r"\S+"  # fw streams are already space-joined tokens (punctuation marks are tokens)

# name -> (transform, vectorizer kwargs)
STREAMS = {
    "char": ("raw", dict(analyzer="char", ngram_range=(1, 5), lowercase=False, max_features=400_000)),
    "char16": ("raw", dict(analyzer="char", ngram_range=(1, 6), lowercase=False, max_features=600_000)),
    "char17": ("raw", dict(analyzer="char", ngram_range=(1, 7), lowercase=False, max_features=800_000)),
    "char26": ("raw", dict(analyzer="char", ngram_range=(2, 6), lowercase=False, max_features=600_000)),
    "word": ("raw", dict(analyzer="word", ngram_range=(1, 2), lowercase=True, token_pattern=WORD_PAT)),
    "word13": ("raw", dict(analyzer="word", ngram_range=(1, 3), lowercase=True, token_pattern=WORD_PAT)),
    "shapec": ("shapec", dict(analyzer="char", ngram_range=(1, 7), lowercase=False)),
    "shapew": ("shapew", dict(analyzer="char", ngram_range=(1, 8), lowercase=False)),
    "fw": ("fw", dict(analyzer="word", ngram_range=(1, 3), lowercase=False, token_pattern=SPACE_SPLIT)),
    "fw14": ("fw", dict(analyzer="word", ngram_range=(1, 4), lowercase=False, token_pattern=SPACE_SPLIT)),
    "fwx": ("fwx", dict(analyzer="word", ngram_range=(1, 3), lowercase=False, token_pattern=SPACE_SPLIT)),
}


def make_vectorizer(name: str, args) -> TfidfVectorizer:
    kw = dict(STREAMS[name][1])
    kw.update(sublinear_tf=not args.no_sublinear, min_df=args.min_df, dtype=np.float32, binary=args.binary,
              use_idf=not args.no_idf)
    if args.max_features:
        kw["max_features"] = args.max_features
    return TfidfVectorizer(**kw)


def transformed_texts(feats: str, texts) -> dict:
    need = {STREAMS[f][0] for f in feats.split("+")}
    return {t: [TRANSFORMS[t](s) for s in texts] for t in need}


# ---------------------------------------------------------------- models (all return logits for softmax)

def nb_ratios(X, y, alpha: float) -> np.ndarray:
    """(K, n_features) NB log-count ratios r_c = log(p_c/|p_c|_1) - log(q_c/|q_c|_1), p = class rows, q = the rest."""
    Y = sp.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(K, len(y)))
    Sr = np.asarray((Y @ X).todense())  # (K, F) per-class feature sums
    P = Sr + alpha
    Q = Sr.sum(0, keepdims=True) - Sr + alpha
    return (np.log(P / P.sum(1, keepdims=True)) - np.log(Q / Q.sum(1, keepdims=True))).astype(np.float32)


def fit_logits(args, A_tr, y_tr):
    """Fit the head on (A_tr, y_tr); returns f(A) -> logits (n, K)."""
    if args.model == "lr":
        m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-4).fit(A_tr, y_tr)
        return lambda A: m.predict_log_proba(A)
    if args.model == "svc":
        m = LinearSVC(C=args.C, max_iter=5000).fit(A_tr, y_tr)
        return lambda A: m.decision_function(A)
    if args.model in ("mnb", "cnb"):
        m = (MultinomialNB if args.model == "mnb" else ComplementNB)(alpha=args.alpha).fit(A_tr, y_tr)
        return lambda A: m.predict_log_proba(A)
    if args.model == "nbsvm":
        R = nb_ratios(A_tr, y_tr, args.alpha)
        heads = []
        for c in range(K):
            Xc = A_tr.multiply(R[c]).tocsr()
            yc = (y_tr == c).astype(int)
            if args.nb_base == "svc":
                h = LinearSVC(C=args.C, max_iter=5000).fit(Xc, yc)
            else:
                h = LogisticRegression(C=args.C, solver="liblinear", max_iter=2000).fit(Xc, yc)
            heads.append(h)

        def f(A):
            return np.column_stack([h.decision_function(A.multiply(R[c]).tocsr()) for c, h in enumerate(heads)])
        return f
    raise ValueError(args.model)


def needs_temperature(args) -> bool:
    return args.model != "lr" or args.temp


def run_fold(args, texts_tr: dict, y_tr, g_tr, texts_va: dict, texts_te: dict):
    """Vectorize (fit on training texts), fit head (+ inner temperature); returns (p_va, p_te, T)."""
    blocks_tr, blocks_va, blocks_te = [], [], []
    weights = [float(w) for w in args.stream_weights.split(",")] if args.stream_weights else None
    for i, f in enumerate(args.feats.split("+")):
        tname = STREAMS[f][0]
        v = make_vectorizer(f, args).fit(texts_tr[tname])
        w = weights[i] if weights else 1.0
        blocks_tr.append(v.transform(texts_tr[tname]) * w)
        blocks_va.append(v.transform(texts_va[tname]) * w)
        blocks_te.append(v.transform(texts_te[tname]) * w)
    A_tr, A_va, A_te = (sp.hstack(b).tocsr() for b in (blocks_tr, blocks_va, blocks_te))
    T = 1.0
    if needs_temperature(args):
        dec_in = np.zeros((len(y_tr), K))
        for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(A_tr, y_tr, g_tr):
            dec_in[iva] = fit_logits(args, A_tr[itr], y_tr[itr])(A_tr[iva])
        T = fit_temperature(dec_in, y_tr)
    f = fit_logits(args, A_tr, y_tr)
    return softmax(f(A_va) * T), softmax(f(A_te) * T), T


def default_tag(args) -> str:
    parts = ["ng", args.feats.replace("+", "-"), args.model]
    if args.model == "nbsvm":
        parts.append(f"{args.nb_base}a{args.alpha:g}")
    if args.model in ("mnb", "cnb"):
        parts.append(f"a{args.alpha:g}")
    else:
        parts.append(f"C{args.C:g}")
    for flag, s in ((args.binary, "bin"), (args.no_sublinear, "nosub"), (args.no_idf, "noidf"), (args.temp, "T")):
        if flag:
            parts.append(s)
    if args.min_df != 2:
        parts.append(f"md{args.min_df}")
    if args.max_features:
        parts.append(f"mf{args.max_features // 1000}k")
    if args.stream_weights:
        parts.append("w" + args.stream_weights.replace(",", "_"))
    return "_".join(parts)


def describe(args) -> str:
    head = {"lr": "LogReg" + (" + inner temperature" if args.temp else ""), "svc": "LinearSVC + inner temperature",
            "nbsvm": f"OvR NB-SVM ({args.nb_base}, alpha {args.alpha:g}) + inner temperature",
            "mnb": f"MultinomialNB alpha {args.alpha:g} + inner temperature",
            "cnb": f"ComplementNB alpha {args.alpha:g} + inner temperature"}[args.model]
    opts = [f"min_df {args.min_df}"] + (["binary tf"] if args.binary else []) + \
        (["raw tf"] if args.no_sublinear else []) + (["no idf"] if args.no_idf else []) + \
        ([f"max_features {args.max_features}"] if args.max_features else []) + \
        ([f"stream weights {args.stream_weights}"] if args.stream_weights else [])
    c = "" if args.model in ("mnb", "cnb") else f" C={args.C:g}"
    return f"per-text n-gram streams {args.feats} ({', '.join(opts)}) + {head}{c}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="char+word")
    ap.add_argument("--model", default="lr", choices=["lr", "svc", "nbsvm", "mnb", "cnb"])
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--nb-base", default="svc", choices=["svc", "lr"])
    ap.add_argument("--temp", action="store_true", help="lr: inner-split temperature on the log-probabilities")
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--no-sublinear", action="store_true")
    ap.add_argument("--no-idf", action="store_true")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=0)
    ap.add_argument("--stream-weights", default="")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--save", action="store_true", help="save as a member (outputs/preds) via run_member")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    groups = tr["prompt_id"].to_numpy()
    tag = args.tag or default_tag(args)
    T_all = transformed_texts(args.feats, tr["text"].tolist())
    T_te = transformed_texts(args.feats, te["text"].tolist())
    splits = list(iter_folds(folds))

    def take(d, idx):
        return {k: [v[i] for i in idx] for k, v in d.items()}

    res = Parallel(n_jobs=args.n_jobs)(
        delayed(run_fold)(args, take(T_all, trn), y[trn], groups[trn], take(T_all, val), T_te)
        for _, trn, val in splits)
    oof = np.zeros((len(y), K))
    test = np.zeros((len(te), K))
    for (_, trn, val), (p_va, p_te, T) in zip(splits, res):
        oof[val] = p_va
        test += p_te / len(splits)
    temps = [round(r[2], 3) for r in res]
    s = scores(y, oof, folds)
    ref, _ = load_member(REF_TAG)
    cmp = paired_bootstrap(y, ref, oof, groups)
    cmp_txt = (f"vs {REF_TAG} (0.3471 / 2.0359): diff {cmp['diff']:+.4f} [{cmp['ci_lo']:+.4f}, {cmp['ci_hi']:+.4f}]")
    print(f"[{tag}] BA {s['ba']:.4f} (folds {s['ba_fold_mean']:.4f} ± {s['ba_fold_std']:.4f}) "
          f"log loss {s['logloss']:.4f}; temps {temps}; {cmp_txt}; {time.time() - t0:.0f}s", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(SCRATCH / f"{tag}.npz", oof=oof, test=test, ba=s["ba"], logloss=s["logloss"])
    note = "; ".join(x for x in [args.note, cmp_txt, f"inner temps {temps}" if needs_temperature(args) else ""] if x)
    if args.save:
        run_member(tag, describe(args), oof, test, y, folds, t0, note=note, feats=args.feats, model=args.model,
                   C=args.C, alpha=args.alpha, binary=args.binary, min_df=args.min_df, temps=temps,
                   script="train_ngram.py")
    else:
        log_experiment(tag, describe(args), s, time.time() - t0, note=note + "; not saved as a member")


if __name__ == "__main__":
    main()
