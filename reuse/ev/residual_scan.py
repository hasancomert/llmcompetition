"""OOF artıklarında grup yapısı taraması (kaçırılmış bir desen var mı?); çok sınıflı, sınıf başına bire-karşı-hepsi.

S6E9'daki karşılığı (scratchpad/resid_base.py, resid_scan1-3.py; README "Yeni desen arayışı"):
- Harmanın OOF'u izotonik regresyonla kalibre edildi (p); artık r = y - p, varyans v = p(1 - p).
- Her anahtar (sütun, ikili, üçlü, ızgara, hane x hane ...) için z = (sum_g S_g^2 / V_g - G) / sqrt(2G). Burada S_g ve
  V_g, en az 20 satırlı gruplarda artık ve varyans toplamı; boş hipotezde her grup ~1 katkı verir.
  ~300 anahtarda en yüksek z 2.47 çıktı (225 testte şans düzeyi ~2.8), yani desen bulunamadı.
- gain: çapraz katlı artık hedef kodlaması (S_g / (V_g + lam)) logit'e beta ile eklenir; AUC kazancı.
  beta {0.25, 0.5, 1, 2} içinden tüm OOF'ta seçilir.

Çok sınıfa uyarlama: her sınıf için bire-karşı-hepsi artık (1[y=c] - p_c) ve varyans p_c(1 - p_c). Kazanç testi
log-olasılıklara sınıf başına artık kodlaması ekler, balanced accuracy ile ölçülür. Olasılıklar kalibre olmalı
(S6E9'da izotonik); bu modül kalibrasyon yapmaz. chance_max_z: n bağımsız N(0, 1)'in en büyüğünün ortalaması
(benzetimle). Bağımlılık: numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def balanced_accuracy(y, P) -> float:
    y = np.asarray(y)
    pred = np.asarray(P).argmax(1) if np.ndim(P) == 2 else np.asarray(P)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def group_z(y, P, key, min_size: int = 20):
    """Sınıf başına z (C,) ve en az min_size satırlı grup sayısı G."""
    y, P = np.asarray(y), np.asarray(P, dtype=float)
    k = pd.factorize(pd.Series(key), use_na_sentinel=False)[0]
    n = np.bincount(k)
    m = n >= min_size
    G = int(m.sum())
    if G < 2:
        return np.full(P.shape[1], np.nan), G
    z = np.empty(P.shape[1])
    for c in range(P.shape[1]):
        r = (y == c) - P[:, c]
        v = P[:, c] * (1 - P[:, c])
        S, V = np.bincount(k, weights=r)[m], np.bincount(k, weights=v)[m]
        z[c] = ((S ** 2 / np.maximum(V, 1e-12)).sum() - G) / np.sqrt(2 * G)
    return z, G


def scan(y, P, keys: dict, min_size: int = 20) -> list:
    """Anahtar adı -> dizi sözlüğü; dönen (z, sınıf, G, ad) listesi, en yüksek z başta."""
    res = []
    for name, key in keys.items():
        z, G = group_z(y, P, key, min_size)
        if G >= 2:
            res.extend((float(z[c]), int(c), G, name) for c in range(len(z)))
    return sorted(res, key=lambda t: -t[0])


def chance_max_z(n_tests: int, n_sim: int = 2000, seed: int = 0) -> float:
    """n_tests bağımsız standart normalin en büyüğünün ortalaması (çoklu test için şans düzeyi)."""
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal((n_sim, n_tests)).max(axis=1).mean())


def _gain_score(metric: str, y, L) -> float:
    """L: normalize edilmemiş log-skorlar. Büyük olan iyi: balanced accuracy ya da eksi log loss."""
    if metric == "balanced_accuracy":
        return balanced_accuracy(y, L)
    if metric == "logloss":
        L = L - L.max(axis=1, keepdims=True)
        logp = L - np.log(np.exp(L).sum(axis=1, keepdims=True))
        return float(logp[np.arange(len(y)), y].mean())
    raise ValueError(f"bilinmeyen metrik: {metric}")


def residual_te_gain(y, P, key, groups=None, n_splits: int = 5, seed: int = 7, lam: float = 50.0,
                     betas=(0.25, 0.5, 1.0, 2.0), metric: str = "balanced_accuracy"):
    """Çapraz katlı sınıf başına artık kodlamasını log-olasılıklara eklemenin kazancı.

    S6E9'da kazanç AUC ile (sürekli bir ölçüyle) ölçülüyordu. Balanced accuracy argmax'a dayandığı için kaba
    kalabilir; metric="logloss" sürekli bir seçenektir (kazanç = log loss düşüşü).
    Dönen: (kazanç, seçilen beta). Beta tüm OOF'ta seçilir (S6E9'daki gibi).
    """
    y, P = np.asarray(y), np.asarray(P, dtype=float)
    k = pd.factorize(pd.Series(key), use_na_sentinel=False)[0]
    K, C = k.max() + 1, P.shape[1]
    R = np.eye(C)[y] - P
    V = P * (1 - P)
    te = np.zeros_like(P)
    if groups is None:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, np.asarray(groups))
    for tr, va in splits:
        S, W = np.zeros((K, C)), np.zeros((K, C))
        np.add.at(S, k[tr], R[tr])
        np.add.at(W, k[tr], V[tr])
        te[va] = (S / (W + lam))[k[va]]
    lp = np.log(np.clip(P, 1e-12, 1.0))
    base = _gain_score(metric, y, lp)
    best = max((_gain_score(metric, y, lp + b * te), b) for b in betas)
    return best[0] - base, best[1]
