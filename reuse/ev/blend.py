"""Tepe tırmanma harmanı, iç içe CV ile ağırlık doğrulama, aday ekleme testi ve LR istifleme; çok sınıflı.

S6E9'daki karşılığı:
- blend_ext.py `hill`: tekrar seçilebilir açgözlü tepe tırmanma. En iyi tek modelden başlar; her adımda metriği en
  çok artıran modeli bir kez daha ekler; kazanç 1e-7'nin altına inince durur (en çok 100 adım).
  Ağırlık = seçilme sayısı / toplam. S6E9'da tahminler önce sıraya çevriliyordu (rankdata / n), metrik AUC'ydi.
- İç içe CV: ağırlıklar OOF satırlarının 4/5'inde seçilir, kalan 1/5'te ölçülür (StratifiedKFold(5, shuffle, 42)).
  AVGW=1: son ağırlık bu 5 vektörün ortalaması (18. ve 19. gönderim).
- scratchpad/addtest.py: bir adayı (1 - w) * harman + w * aday olarak eklemenin etkisi; w ızgarası 0 .. 0.5,
  adım 0.05; w iç içe katlarda seçilir. 25 açık OOF bu yolla tek tek denendi, 3'ü katkı verdi.
- blend_ext.py STACK=1: probit(sıra) üstünde lojistik regresyon istifleme, iç içe ölçümle.

Çok sınıfa uyarlama: sıra uzayı yok. Balanced accuracy her satırda sınıflar arasında argmax alır; sütunları ayrı
ayrı sıralamak sınıfları karşılaştırılamaz yapar. Harman ya olasılık ortalaması ('prob') ya da log-olasılık
ortalaması ('logprob', geometrik ortalama) ile yapılır. Metrik balanced accuracy (varsayılan) ya da log loss
('logloss', eksi işaretle büyütülür). İç içe katlar grupla (ör. prompt_id) kurulabilir; istifleme özellikleri
modellerin log-olasılıkları. Bağımlılık: numpy, scikit-learn.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

EPS = 1e-9


def balanced_accuracy(y, P) -> float:
    """Sınıf başına recall ortalaması (y'de bulunan sınıflar üzerinden). P: (N, C) ya da (N,) etiket."""
    y = np.asarray(y)
    pred = np.asarray(P).argmax(1) if np.ndim(P) == 2 else np.asarray(P)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def log_loss(y, P) -> float:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(P)), np.asarray(y)])))


def score(metric: str, y, P) -> float:
    """Büyük olan iyi: balanced accuracy ya da eksi log loss."""
    if metric == "balanced_accuracy":
        return balanced_accuracy(y, P)
    if metric == "logloss":
        return -log_loss(y, P)
    raise ValueError(f"bilinmeyen metrik: {metric}")


def _space(P, space: str) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if space == "prob":
        return P
    if space == "logprob":
        return np.log(np.clip(P, EPS, 1.0))
    raise ValueError(f"bilinmeyen uzay: {space}")


def _finish(S, k: float, space: str) -> np.ndarray:
    """k modelin toplamı S -> olasılık matrisi."""
    Z = S / k
    if space == "prob":
        return Z
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def blend(preds, w, space: str = "prob") -> np.ndarray:
    """Ağırlıklı harman: preds (N, C) olasılık matrislerinin listesi, w ağırlıklar."""
    w = np.asarray(w, dtype=float)
    used = np.flatnonzero(w)
    S = sum(w[j] * _space(preds[j], space) for j in used)
    return _finish(S, w[used].sum(), space)


def _splits(y, groups, n_splits: int, seed: int):
    y = np.asarray(y)
    if groups is None:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, np.asarray(groups)))


def hill_climb(preds, y, idx=None, space: str = "prob", metric: str = "balanced_accuracy",
               max_steps: int = 100, tol: float = 1e-7):
    """Tekrar seçilebilir açgözlü tepe tırmanma. Dönen: (ağırlıklar (M,), harmanın idx satırlarındaki skoru)."""
    y = np.asarray(y)
    idx = np.arange(len(y)) if idx is None else np.asarray(idx)
    yy = y[idx]
    A = [_space(np.asarray(p)[idx], space) for p in preds]
    single = [score(metric, yy, _finish(a, 1.0, space)) for a in A]
    cnt = np.zeros(len(A))
    j0 = int(np.argmax(single))
    cnt[j0] = 1
    best, S = single[j0], A[j0].copy()
    for _ in range(max_steps):
        k = cnt.sum() + 1
        cand = [score(metric, yy, _finish(S + a, k, space)) for a in A]
        j = int(np.argmax(cand))
        if cand[j] <= best + tol:
            break
        best, cnt[j], S = cand[j], cnt[j] + 1, S + A[j]
    return cnt / cnt.sum(), best


def nested_hill(preds, y, groups=None, n_splits: int = 5, seed: int = 42, space: str = "prob",
                metric: str = "balanced_accuracy", **kw) -> dict:
    """İç içe CV: ağırlıklar her seferinde K-1 katta seçilir, görülmemiş katta ölçülür.

    Dönen sözlük: blind_score (iç içe skor), full_score / full_weights (tüm OOF'ta seçim), fold_weights,
    avg_weights (S6E9'daki AVGW=1) ve avg_score (ortalama ağırlıkların tüm OOF skoru), blind_pred.
    """
    y = np.asarray(y)
    blind = np.zeros((len(y), np.asarray(preds[0]).shape[1]))
    ws = []
    for tr, va in _splits(y, groups, n_splits, seed):
        w, _ = hill_climb(preds, y, idx=tr, space=space, metric=metric, **kw)
        ws.append(w)
        blind[va] = blend([np.asarray(p)[va] for p in preds], w, space)
    full_w, full_score = hill_climb(preds, y, space=space, metric=metric, **kw)
    avg_w = np.mean(ws, axis=0)
    return {"blind_score": score(metric, y, blind), "full_score": full_score, "full_weights": full_w,
            "fold_weights": np.array(ws), "avg_weights": avg_w,
            "avg_score": score(metric, y, blend(preds, avg_w, space)), "blind_pred": blind}


def add_candidate_test(base, cand, y, groups=None, n_splits: int = 5, seed: int = 42,
                       weights=np.arange(0, 0.51, 0.05), space: str = "prob",
                       metric: str = "balanced_accuracy") -> dict:
    """Harmana bir aday eklemenin etkisi: (1 - w) * base + w * cand; w tüm OOF'ta ve iç içe katlarda seçilir.

    Dönen: best_w, full_gain (tüm OOF'taki en iyi w'nin w=0'a göre kazancı), nested_gain (iç içe kazanç).
    """
    y = np.asarray(y)
    base, cand = np.asarray(base), np.asarray(cand)
    mix = lambda w, rows: blend([base[rows], cand[rows]], [1.0 - w, w], space)
    allr = np.arange(len(y))
    full = [score(metric, y, mix(w, allr)) for w in weights]
    blind = np.zeros_like(base, dtype=float)
    for tr, va in _splits(y, groups, n_splits, seed):
        wb = weights[int(np.argmax([score(metric, y[tr], mix(w, tr)) for w in weights]))]
        blind[va] = mix(wb, va)
    return {"best_w": float(weights[int(np.argmax(full))]), "full_gain": max(full) - full[0],
            "nested_gain": score(metric, y, blind) - full[0]}


def label_agreement(P1, P2) -> float:
    """İki tahmin matrisinin argmax sınıflarının aynı olduğu satır oranı.

    S6E9'da son iki gönderimden biri yedek olarak "iyi gönderimler içinde en farklı olan" diye seçildi; farklılık
    ölçüsü test tahminlerinin sıra korelasyonuydu (11 ile 19: 0.9991). Çok sınıfta sıra korelasyonunun yerine bu oran.
    """
    return float(np.mean(np.asarray(P1).argmax(1) == np.asarray(P2).argmax(1)))


def _proba_cols(model, Z, n_classes: int) -> np.ndarray:
    out = np.zeros((len(Z), n_classes))
    out[:, model.classes_] = model.predict_proba(Z)
    return out


def stack_logreg(preds_oof, y, preds_test=None, groups=None, n_splits: int = 5, seed: int = 42,
                 C: float = 1.0, max_iter: int = 2000) -> dict:
    """Modellerin log-olasılıkları üstünde çok terimli lojistik regresyon istifleme, iç içe ölçümle.

    S6E9'da STACK=1 probit(sıra) özellikleri kullanıyordu ve tepe tırmanmayla aynı iç içe skoru verdi.
    Dönen: blind_score (balanced accuracy), blind_pred, model (tüm OOF'ta eğitilmiş), test_pred (verilirse).
    """
    y = np.asarray(y)
    n_classes = np.asarray(preds_oof[0]).shape[1]
    Z = np.hstack([_space(p, "logprob") for p in preds_oof])
    blind = np.zeros((len(y), n_classes))
    for tr, va in _splits(y, groups, n_splits, seed):
        m = LogisticRegression(C=C, max_iter=max_iter).fit(Z[tr], y[tr])
        blind[va] = _proba_cols(m, Z[va], n_classes)
    full = LogisticRegression(C=C, max_iter=max_iter).fit(Z, y)
    out = {"blind_score": balanced_accuracy(y, blind), "blind_pred": blind, "model": full}
    if preds_test is not None:
        out["test_pred"] = _proba_cols(full, np.hstack([_space(p, "logprob") for p in preds_test]), n_classes)
    return out
