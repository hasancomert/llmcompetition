"""Two label-free post-steps of the public 0.66 notebooks (mercurydust's "V31", built on Roman Tamrazov's "Who am I?";
re-implemented here from their description and code reading, not copied):

soft_sinkhorn  damped marginal balancing: Q = Sinkhorn of P = softmax(z) towards (1 - rho) * the predicted test marginal
               + rho * the training prior; the result (1 - strength) log P + strength log Q. The hidden test has unequal
               model counts (host), so the target is not forced to be uniform. Public setting: strength 0.45, rho 0.35.
Typography     hierarchical Beta-Bernoulli likelihood of 37 typographic / formatting markers (presence per text): class
               marker rates shrunk towards the overall rate (prior strength kappa 24), markers weighted by how much their
               rates differ between classes (clipped 0.25..2), per-text log-likelihood ratio against the overall rate,
               centred over classes, scaled 1.5 / sum(weights) and clipped to +-0.30. It is added to the final scores
               with a gate: confidence (top-2 probability margin under 0.6), length (sqrt(words / 80) in 0.4..1) and
               marker coverage (sqrt((markers + 1) / 4) in 0.45..1).
"""
from __future__ import annotations

import re

import numpy as np
from scipy.special import log_softmax, softmax

MARKERS = [" — ", "—", "–", "’", "‘", "“", "”", "«", "»", '"', ";", ":", "(", ")", "[", "]", "**", "`", "…", "•",
           "\n", "\n\n", "e.g.", "i.e.", "However,", "Overall,", "In conclusion", "Therefore,", "Moreover,", "\t", " - ",
           " = ", "\n- ", "\n1.", "###", "```", "> "]


def soft_sinkhorn(z, strength, rho, prior, iters=100):
    """(n, K) scores after damped balancing of softmax(z) towards (1 - rho) * predicted marginal + rho * prior."""
    P = softmax(np.asarray(z, np.float64), axis=1)
    target = (1 - rho) * np.clip(P.mean(0), 1e-6, None) / np.clip(P.mean(0), 1e-6, None).sum() + rho * prior
    target = np.clip(target, 1e-6, None)
    target /= target.sum()
    Q = P.copy()
    for _ in range(iters):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - strength) * np.log(P + 1e-12) + strength * np.log(Q + 1e-12)


def presence(texts):
    return np.array([[float(m in t) for m in MARKERS] for t in texts], dtype=np.float64)


class Typography:
    def __init__(self, kappa=24.0, scale=1.5, clip=0.30):
        self.kappa, self.scale, self.clip = kappa, scale, clip

    def fit(self, texts, y, K):
        X = presence(texts)
        n_c = np.bincount(y, minlength=K).astype(np.float64)
        hits = np.stack([X[y == k].sum(0) for k in range(K)])
        g = (X.sum(0) + 2.0) / (len(X) + 4.0)
        rate = np.clip((hits + self.kappa * g[None, :]) / (n_c[:, None] + self.kappa), 0.01, 0.99)
        self.g = np.clip(g, 0.01, 0.99)
        self.lr, self.lnr = np.log(rate), np.log1p(-rate)
        signal = np.sqrt(np.mean((rate - self.g[None, :]) ** 2, axis=0))
        med = float(np.median(signal[signal > 0])) if np.any(signal > 0) else 1.0
        self.fw = np.clip(signal / max(med, 1e-6), 0.25, 2.0)
        return self

    def evidence(self, texts):
        """(n, K) clipped log-likelihood-ratio evidence, (n,) word counts, (n,) number of markers present."""
        X = presence(texts)
        Xw, Nw = X * self.fw[None, :], (1 - X) * self.fw[None, :]
        ev = Xw @ self.lr.T + Nw @ self.lnr.T
        ev -= (Xw @ np.log(self.g) + Nw @ np.log1p(-self.g))[:, None]
        ev -= ev.mean(1, keepdims=True)
        ev = np.clip(self.scale * ev / max(float(self.fw.sum()), 1e-6), -self.clip, self.clip)
        words = np.array([len(re.findall(r"\b\w+\b", t)) for t in texts], dtype=np.float64)
        return ev, words, X.sum(1)


def typography_apply(z, ev, words, active):
    """Gated addition of the typography evidence to final scores z (the public V31 gates)."""
    p = softmax(np.asarray(z, np.float64), axis=1)
    top2 = np.sort(p, axis=1)[:, -2:]
    gate = np.clip((0.60 - (top2[:, 1] - top2[:, 0])) / 0.60, 0.0, 1.0)
    gate *= np.clip(np.sqrt(np.maximum(words, 1.0) / 80.0), 0.40, 1.0)
    gate *= np.clip(np.sqrt((active + 1.0) / 4.0), 0.45, 1.0)
    return log_softmax(np.asarray(z, np.float64), axis=1) + gate[:, None] * ev
