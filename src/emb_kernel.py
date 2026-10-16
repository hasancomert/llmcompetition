"""Fast, exact linear heads on frozen-LM embeddings through per-fold Gram matrices (raw, per-text features).

Why: an L2-regularised linear model (LogReg, LinearSVC, ridge) on standardized features Z has its weight vectors in
the span of the training rows, so it can be fit on the eigen-coordinates of the training Gram matrix
K_tr = Z_tr Z_tr^T (n_tr <= 2160 columns instead of 16384) with IDENTICAL predictions: held-out rows only enter
through K_ev,tr = Z_ev Z_tr^T. Per fold, the StandardScaler is fit on the training rows only (as in train_emb.py),
every layer block of every LM gets its own Gram matrix (cached under outputs/kgram/), and any layer subset, block
weighting or concatenation of LMs is just a weighted sum of these matrices.

Held-out (and test) rows are always processed one by one: their coordinates depend on the training rows only.
Nothing here clusters, centers or matches held-out / test texts.
"""
from __future__ import annotations

import json

import numpy as np

from common import K, OUT, load_test, load_train
from train_emb import EMB

CACHE = OUT / "kgram"


def layer_info(name):
    meta = json.loads((EMB / f"meta_{name}_train.json").read_text())
    layers = meta["layers"]
    return layers, meta["dim"] // len(layers)


def raw_matrix(name, tr=None, te=None):
    """[train rows; test rows] float32 matrix of one LM (row order checked against train.csv / test.csv)."""
    tr = load_train() if tr is None else tr
    te = load_test() if te is None else te
    out = []
    for split, ids in (("train", tr["id"].tolist()), ("test", te["id"].tolist())):
        meta = json.loads((EMB / f"meta_{name}_{split}.json").read_text())
        assert meta["ids"] == ids, f"{name} {split}: rows not in csv order"
        out.append(np.load(EMB / f"emb_{name}_{split}.npy").astype(np.float32))
    return np.vstack(out)


def grams(name, folds, fold, n_test):
    """(L, N, N) float32 Gram matrices of the standardized layer blocks of `name` for one fold (N = train + test
    rows). Standardization (mean / std, std 0 -> 1, as sklearn) uses the training rows of that fold only."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_f{fold}.npy"
    if path.exists():
        G = np.load(path, mmap_mode="r")
        if G.shape[1] == len(folds) + n_test:
            return G
    X = raw_matrix(name)
    assert len(X) == len(folds) + n_test
    trn = np.flatnonzero(folds != fold)
    layers, w = layer_info(name)
    G = np.zeros((len(layers), len(X), len(X)), np.float32)
    for li in range(len(layers)):
        B = X[:, li * w:(li + 1) * w].astype(np.float64)
        mu, sd = B[trn].mean(0), B[trn].std(0)
        sd[sd == 0] = 1.0
        Z = ((B - mu) / sd).astype(np.float32)
        G[li] = Z @ Z.T
    np.save(path, G)
    return np.load(path, mmap_mode="r")


def combined_gram(blocks, folds, fold, n_test, equalize=False):
    """Weighted sum of block Grams. blocks = [(name, [layer indices] | None for all, weight)].
    equalize: weight each block by mean_width / width, so that every block contributes the same trace."""
    widths = [layer_info(n)[1] * (len(ls) if ls is not None else len(layer_info(n)[0])) for n, ls, _ in blocks]
    mean_w = float(np.mean(widths))
    Ksum = None
    for (name, ls, wt), width in zip(blocks, widths):
        G = grams(name, folds, fold, n_test)
        idx = range(G.shape[0]) if ls is None else ls
        f = wt * (mean_w / width if equalize else 1.0)
        for li in idx:
            Ksum = f * np.asarray(G[li], np.float64) if Ksum is None else Ksum + f * np.asarray(G[li], np.float64)
    return Ksum


def cosine_kernel(Kfull):
    d = np.sqrt(np.clip(np.diag(Kfull), 1e-12, None))
    return Kfull / d[:, None] / d[None, :]


def rbf_kernel(Kfull, trn, gamma_rel):
    """exp(-gamma d^2) with gamma = gamma_rel / median training squared distance."""
    dg = np.diag(Kfull)
    D2 = np.clip(dg[:, None] + dg[None, :] - 2 * Kfull, 0, None)
    sub = D2[np.ix_(trn, trn)]
    med = np.median(sub[np.triu_indices(len(trn), 1)])
    return np.exp(-gamma_rel / med * D2)


def kcoords(Kfull, trn, ev, power=1.0, drop_top=0, center=False, eps=1e-10):
    """Exact eigen-coordinates. Training rows: U * lam^(power/2); evaluation rows: K_ev,tr U lam^(power/2 - 1).
    power 1 = the plain linear model on the standardized features; power 0 = full PCA whitening (fit on the training
    rows); drop_top removes the top principal directions (fit on the training rows).
    center: double-center the Gram on the training rows first (needed for non-linear kernels, which are not
    computed from centered features)."""
    Ktr = Kfull[np.ix_(trn, trn)]
    Kev = Kfull[np.ix_(ev, trn)]
    if center:
        m_tr = Ktr.mean(0)
        Kev = Kev - Kev.mean(1, keepdims=True) - m_tr[None, :] + m_tr.mean()
        Ktr = Ktr - m_tr[None, :] - m_tr[:, None] + m_tr.mean()
    lam, U = np.linalg.eigh(Ktr)
    lam, U = lam[::-1], U[:, ::-1]
    keep = lam > eps * lam[0]
    lam, U = lam[keep], U[:, keep]
    if drop_top:
        lam, U = lam[drop_top:], U[:, drop_top:]
    Ptr = U * lam ** (power / 2)
    Pev = Kev @ (U * lam ** (power / 2 - 1))
    return Ptr, Pev, lam


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(dec, y):
    from scipy.optimize import minimize_scalar

    def nll(logt):
        p = softmax(dec * np.exp(logt))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-8, 8), method="bounded").x))


def fit_head(head, C, Ptr, ytr, Pev, groups_tr=None, sample_weight=None):
    """Probabilities for the evaluation rows. lr: multinomial LogReg; svc: LinearSVC (crammer-singer off, ovr) and
    ridge: one-vs-rest ridge on +-1 targets, both with a softmax temperature fit on an inner grouped split of the
    training rows (StratifiedGroupKFold(4) by prompt_id)."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC
    if head == "lr":
        m = LogisticRegression(C=C, max_iter=5000, tol=1e-5).fit(Ptr, ytr, sample_weight=sample_weight)
        P = np.zeros((len(Pev), K))
        P[:, m.classes_] = m.predict_proba(Pev)
        return P

    def make():
        if head == "svc":
            return LinearSVC(C=C, max_iter=20000, dual=True)
        return RidgeClassifier(alpha=1.0 / C, solver="auto")
    from sklearn.model_selection import StratifiedGroupKFold
    dec_in = np.zeros((len(ytr), K))
    for itr, iva in StratifiedGroupKFold(4, shuffle=True, random_state=7).split(Ptr, ytr, groups_tr):
        m = make().fit(Ptr[itr], ytr[itr])
        dec_in[iva] = m.decision_function(Ptr[iva])
    T = fit_temperature(dec_in, ytr)
    m = make().fit(Ptr, ytr, sample_weight=sample_weight)
    return softmax(m.decision_function(Pev) * T)
