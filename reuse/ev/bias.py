"""Balanced accuracy için sınıf başına logit kaydırma (bias). YENİ: S6E9'da karşılığı yok.

S6E9'da metrik AUC'ydi (eşiksiz); eşik, sınıf ağırlığı ya da sınıf başına kaydırma araması yapılmadı. Bu modül
Name That LLM planındaki "balanced accuracy için sınıf başına logit kaydırma" fikri için yazıldı ve yalnız sahte
veriyle denendi. S6E9'dan taşınan tek ilke şu: kaydırma OOF'ta seçiliyorsa kazancı iç içe CV ile ölçülmeli,
yani seçimin görmediği katta. S6E9'da örneklem içinde bulunan kurallar görülmemiş katta −0.00037 AUC kaybettirdi.

- prior_bias: eğitim sınıf oranları π ile -log π. Balanced accuracy = sınıf başına recall ortalaması olduğundan,
  kalibre olasılıklarda bu ölçütü en büyükleyen karar argmax_c p(c|x) / π_c'dir.
- fit_class_bias: koordinat yükselişi; her sınıfın kaydırması bir ızgarada aranır, OOF balanced accuracy büyür.
- nested_bias: kaydırmayı K-1 katta seçip görülmemiş katta uygular; dürüst kazanç budur.
Bağımlılık: numpy, scikit-learn.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred).argmax(1) if np.ndim(pred) == 2 else np.asarray(pred)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def prior_bias(y_train, n_classes: int, eps: float = 1e-12) -> np.ndarray:
    """-log(eğitim sınıf oranı): kalibre log-olasılıklara eklenince karar argmax p(c|x) / π_c olur."""
    freq = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=n_classes) / len(y_train)
    b = -np.log(np.clip(freq, eps, 1.0))
    return b - b.mean()


def fit_class_bias(logp, y, grid=None, n_rounds: int = 3):
    """logp (N, C) log-olasılık; dönen (kaydırma (C,), OOF balanced accuracy).

    Eşit skorlarda sıfıra en yakın kaydırma seçilir. Kaydırmalar ortalaması sıfır olacak şekilde ötelenir;
    argmax buna duyarsızdır.
    """
    logp, y = np.asarray(logp, dtype=float), np.asarray(y)
    grid = np.linspace(-3, 3, 61) if grid is None else np.asarray(grid, dtype=float)
    C = logp.shape[1]
    b = np.zeros(C)
    best = balanced_accuracy(y, logp + b)
    for _ in range(n_rounds):
        improved = False
        for c in range(C):
            cur = b[c]
            scores = np.empty(len(grid))
            for j, g in enumerate(grid):
                b[c] = g
                scores[j] = balanced_accuracy(y, logp + b)
            top = np.flatnonzero(scores >= scores.max() - 1e-12)
            j = top[np.argmin(np.abs(grid[top]))]
            if scores[j] > best + 1e-12:
                b[c], best, improved = grid[j], scores[j], True
            else:
                b[c] = cur
        if not improved:
            break
    return b - b.mean(), best


def nested_bias(logp, y, groups=None, n_splits: int = 5, seed: int = 42, **kw) -> dict:
    """Kaydırma K-1 katta seçilip görülmemiş katta uygulanır.

    Dönen: base_score (kaydırmasız), blind_score (iç içe), gain, fold_biases, avg_bias, full_bias, full_score.
    """
    logp, y = np.asarray(logp, dtype=float), np.asarray(y)
    if groups is None:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, np.asarray(groups))
    pred = np.empty(len(y), dtype=np.int64)
    bs = []
    for tr, va in splits:
        b, _ = fit_class_bias(logp[tr], y[tr], **kw)
        bs.append(b)
        pred[va] = (logp[va] + b).argmax(1)
    full_b, full_score = fit_class_bias(logp, y, **kw)
    base = balanced_accuracy(y, logp)
    blind = balanced_accuracy(y, pred)
    return {"base_score": base, "blind_score": blind, "gain": blind - base, "fold_biases": np.array(bs),
            "avg_bias": np.mean(bs, axis=0), "full_bias": full_b, "full_score": full_score}
