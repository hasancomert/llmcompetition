"""Tam veri modeli mi, kat modellerinin ortalaması mı: sahte testli benzetim; çok sınıflı.

S6E9'daki karşılığı:
- train.py hold=h: 10 katlı bölmenin h. katı etiketli sahte test olur; kalan %90 ile 9 katlı CV.
- train.py full=f: katların ortalama ağaç sayısı x f ile, kalan verinin tamamında tek model (x0.9 .. x1.3 denendi).
- scratchpad/sim_eval.py: sahte testte kat ortalaması, tam veri modeli ve sıra karışımları karşılaştırıldı.
Sonuç: XGBoost `t1`, 4 sahte test. Tam veri modeli tek başına ortalama −0.00004 AUC (üçünde −0.00005 / −0.00007,
birinde +0.00003). %25 sıra karışımı +0.00001. Bu yüzden FULLMIX kullanılmadı.

Çok sınıfa uyarlama: karışım olasılık uzayında, metrik balanced accuracy. Ağaç sayısını ölçekleme gibi modele
özgü ayarlar fit_predict içinde yapılır; bu kısım genelleştirilmedi. Bağımlılık: numpy, scikit-learn.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def balanced_accuracy(y, P) -> float:
    y = np.asarray(y)
    pred = np.asarray(P).argmax(1) if np.ndim(P) == 2 else np.asarray(P)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def _take(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    if isinstance(X, list):
        return [X[i] for i in idx]
    return X[idx]


def _splits(y, groups, n_splits: int, seed: int):
    y = np.asarray(y)
    if groups is None:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, np.asarray(groups)))


def simulate_refit(fit_predict, X, y, groups=None, holds=(0, 1, 2), n_splits: int = 10, seed: int = 42,
                   mix=(0.25, 0.5, 0.75)) -> dict:
    """fit_predict(X_tr, y_tr, X_te) -> olasılık (n_te, C).

    Her h için: bölmenin h. katı sahte test; kalan satırlarda (n_splits - 1) katlı CV modellerinin ortalaması
    ile kalan satırların tamamında eğitilmiş tek model, sahte testte karşılaştırılır; karışımlar da ölçülür.
    Dönen: rows (sahte test başına skorlar), mean_diff (kat ortalamasına göre ortalama fark).
    """
    y = np.asarray(y)
    outer = _splits(y, groups, n_splits, seed)
    rows = []
    for h in holds:
        rest, hold = outer[h]
        g_rest = None if groups is None else np.asarray(groups)[rest]
        inner = _splits(y[rest], g_rest, n_splits - 1, seed)
        X_rest, X_hold = _take(X, rest), _take(X, hold)
        p_cv = np.mean([fit_predict(_take(X_rest, tr), y[rest][tr], X_hold) for tr, _ in inner], axis=0)
        p_full = fit_predict(X_rest, y[rest], X_hold)
        row = {"hold": int(h), "cv_avg": balanced_accuracy(y[hold], p_cv),
               "full": balanced_accuracy(y[hold], p_full)}
        for w in mix:
            row[f"mix_{w}"] = balanced_accuracy(y[hold], (1 - w) * p_cv + w * p_full)
        rows.append(row)
    keys = [k for k in rows[0] if k not in ("hold", "cv_avg")]
    mean_diff = {k: float(np.mean([r[k] - r["cv_avg"] for r in rows])) for k in keys}
    return {"rows": rows, "mean_diff": mean_diff}
