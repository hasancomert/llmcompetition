"""Shared-prompt Gaussian group head (public idea: prvsiyan, "The Prompt Is Not the Author", Kaggle; re-implemented
here from the model description, not copied).

Model of a text's reduced feature vector u (d dims): u = m[y] + p + e, with a prompt effect p ~ N(0, Sp) shared by
every text answering the same prompt and noise e ~ N(0, Se). Centering a detected group subtracts a noisy sample mean
(for two texts the own signal is halved); this head integrates p out instead.

Fit (training rows with their true prompt ids): standardize with the training statistics; reduce to the top-d
directions of the prompt-centered rows (within-prompt PCA, through the N x N Gram); additive two-way fit of class means
and prompt effects (alternating means); Se from the residuals, Sp from the prompt effects minus their noise share,
both Ledoit-Wolf shrunk; a whitening A with A Se A' = I and A Sp A' = diag(lam).

Posterior of the labels of one group of n texts (whitened residuals r_i(c) = u_i - m_c, g = lam / (1 + n lam)):
  log p(u_1..u_n | c_1..c_n) = sum_i a_i(c_i) + sum_{i<j} B_ij(c_i, c_j) + const,
  a_i(c) = -1/2 sum_k (1 - g_k) r_ik(c)^2,   B_ij(c, c') = sum_k g_k r_ik(c) r_jk(c')
(r_k ~ N(0, I + lam_k 11'), whose inverse is I - g_k 11'). n = 1: N(m_c, I + diag lam). Labels of one group are taken
as distinct (one answer per model and prompt, as in train); n <= 3: exact marginals over distinct label tuples,
larger groups: mean-field updates of the same pairwise model with a repulsion term log(1 - q_j(c)).
Scores are divided by a temperature T before normalizing.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp


def _lw(X):
    """Ledoit-Wolf shrinkage intensity of centered rows X (sklearn, assume_centered)."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))


def _shrink(S, a):
    return (1 - a) * S + a * np.trace(S) / len(S) * np.eye(len(S))


class GaussHead:
    def __init__(self, d=256, lam_floor=1e-4):
        self.d, self.lam_floor = d, lam_floor

    def fit(self, X, y, prompts, K):
        """Memory: the (N, D) work is float32 in one buffer (standardized, then prompt-centered in place); the
        reduced (N, d) part is float64."""
        X = np.asarray(X)
        pu, pinv = np.unique(np.asarray(prompts), return_inverse=True)
        G = len(pu)
        self.mx = X.mean(0, dtype=np.float64)
        sx = X.std(0, dtype=np.float64)
        self.sx = np.where(sx > 1e-8, sx, 1.0)
        R = np.subtract(X, self.mx.astype(np.float32), dtype=np.float32)
        R /= self.sx.astype(np.float32)                             # standardized rows
        cnt = np.bincount(pinv, minlength=G).astype(float)
        members = [np.flatnonzero(pinv == i) for i in range(G)]
        Pm = np.stack([R[ix].mean(0, dtype=np.float64) for ix in members])   # prompt means (G, D)
        for i, ix in enumerate(members):                            # prompt-centered rows, in place
            R[ix] -= Pm[i].astype(np.float32)
        d = int(min(self.d, len(X) - G - 1, X.shape[1]))
        w, U = np.linalg.eigh((R @ R.T).astype(np.float64))
        o = np.argsort(w)[::-1][:d]
        V = R.T @ (U[:, o] / np.sqrt(np.clip(w[o], 1e-12, None))).astype(np.float32)   # (D, d) orthonormal
        Z = (R @ V).astype(np.float64) + (Pm @ V)[pinv]            # = standardized rows @ V
        V = V.astype(np.float64)
        del R
        mu, P = np.zeros((K, d)), np.zeros((G, d))
        for _ in range(4):                                          # additive fit: Z = mu[y] + P[prompt] + E
            Rz = Z - P[pinv]
            for c in range(K):
                mu[c] = Rz[y == c].mean(0)
            P = np.zeros((G, d))
            np.add.at(P, pinv, Z - mu[y])
            P /= cnt[:, None]
        E = Z - mu[y] - P[pinv]
        Se_raw = E.T @ E / max(len(X) - G - K + 1, 1)
        Se = _shrink(Se_raw, _lw(E))
        Pc = P - P.mean(0)
        Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / cnt) * Se_raw
        a_p = _lw(Pc)
        Sp = (1 - a_p) * Sp_raw + a_p * max(np.trace(Sp_raw), 1e-8 * np.trace(Se)) / d * np.eye(d)
        Li = np.linalg.inv(np.linalg.cholesky(Se))
        M = Li @ Sp @ Li.T
        lam, Q = np.linalg.eigh(0.5 * (M + M.T))
        A = Q.T @ Li                                                # A Se A' = I, A Sp A' = diag(lam)
        self.W = V @ A.T
        self.m = mu @ A.T
        self.lam = np.clip(lam, self.lam_floor, None)
        return self

    def transform(self, X, chunk=1024):
        """Reduced, whitened rows (float64), computed in float32 chunks of rows."""
        X = np.asarray(X)
        mx, sx, W = self.mx.astype(np.float32), self.sx.astype(np.float32), self.W.astype(np.float32)
        out = np.empty((len(X), W.shape[1]))
        for s in range(0, len(X), chunk):
            B = np.subtract(X[s:s + chunk], mx, dtype=np.float32)
            B /= sx
            out[s:s + chunk] = B @ W
        return out

    def group_logpost(self, U, T=1.0, iters=5):
        """(n, K) log posteriors of the labels of n texts that share one prompt (distinct labels)."""
        n, K = len(U), len(self.m)
        Rr = U[:, None, :] - self.m[None]                           # (n, K, d)
        if n == 1:
            L = -0.5 * (Rr ** 2 / (1 + self.lam)).sum(-1) / T
            return L - logsumexp(L, axis=1, keepdims=True)
        g = self.lam / (1 + n * self.lam)
        a = -0.5 * (Rr ** 2 * (1 - g)).sum(-1)                      # (n, K)
        Gr = Rr * np.sqrt(g)
        if n <= 3:
            Q = np.zeros([K] * n)
            for i in range(n):
                sh = [1] * n
                sh[i] = K
                Q = Q + a[i].reshape(sh)
            for i, j in itertools.combinations(range(n), 2):
                sh = [1] * n
                sh[i] = sh[j] = K
                Q = Q + (Gr[i] @ Gr[j].T).reshape(sh)
            Q = Q / T
            grids = np.meshgrid(*[np.arange(K)] * n, indexing="ij")
            same = np.zeros(Q.shape, bool)
            for i, j in itertools.combinations(range(n), 2):
                same |= grids[i] == grids[j]
            Q = np.where(same, -np.inf, Q)
            lz = logsumexp(Q)
            return np.stack([logsumexp(Q, axis=tuple(k for k in range(n) if k != i)) - lz for i in range(n)])
        B = np.einsum("ikd,jld->ijkl", Gr, Gr)                      # (n, n, K, K)
        L = a / T
        q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        for _ in range(iters):                                      # mean field + at-most-once repulsion
            inter = np.einsum("ijkl,jl->ik", B, q) - np.einsum("iikl,il->ik", B, q)
            rep = np.log(np.clip(1 - q, 1e-6, None))
            L = (a + inter) / T + rep.sum(0)[None] - rep
            q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        return L - logsumexp(L, axis=1, keepdims=True)

    def scores(self, U, groups, T=1.0):
        """(n, K) log posteriors of rows U grouped by groups (each group decoded jointly)."""
        out = np.empty((len(U), len(self.m)))
        for g in np.unique(groups):
            ix = np.flatnonzero(groups == g)
            out[ix] = self.group_logpost(U[ix], T)
        return out
