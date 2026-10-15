"""Kat içinde çapraz hedef kodlama ve hiyerarşik (kaba -> ince anahtar) hedef oranı zinciri; çok sınıflı.

S6E9'daki karşılığı:
- train.py `chain_feats`: anahtar seviyeleri kabadan inceye sıralanır (gelir: ilk GPT-2 parçası -> ilk iki parça
  -> tam değer; mesafe: tam km -> tam değer). Her seviyede önsel, bir önceki seviyenin sonsalıdır:
  post = (anahtardaki alım sayısı + S * önsel) / (anahtardaki satır sayısı + S). Çıktı her seviyenin
  logit(post) değeri. Üç zincir kuruldu: S = 5 / 20 / 80.
- train.py `fold_data`: eğitim satırlarının zincir özellikleri iç 5 katla çapraz üretilir
  (StratifiedKFold(5, shuffle, random_state=seed)); doğrulama ve test satırları dış katın tüm eğitim satırlarıyla.
- Tek seviyeli hedef kodlama (te1, te3) için S6E9'da sklearn TargetEncoder(cv=5) kullanıldı. Onun iç çapraz
  katları grup bilmez; bu modülde tek seviyeli kodlama, zincirin tek seviyeli hâli olarak grup destekli
  iç katlarla yapılır.

Çok sınıfa uyarlama: önsel ve sonsal (C,) vektörleri (Dirichlet yumuşatma), çıktı log-olasılık (n, L * C).
Anahtarlar bütün satırlar (train + test) üzerinde tek dizide verilir; etiketlerden yalnız fit_idx satırları
kullanılır, test satırlarının etiketi -1 gibi herhangi bir değer olabilir.
Bağımlılık: numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def encode_keys(values) -> np.ndarray:
    """Herhangi bir anahtar dizisini (string, sayı, eksik değer) 0'dan başlayan tamsayılara çevirir."""
    codes = pd.factorize(pd.Series(values), use_na_sentinel=False)[0]
    return codes.astype(np.int64)


def chain_logp(levels, y, fit_idx, app_idx, n_classes: int, S: float = 20.0, prior=None) -> np.ndarray:
    """Hiyerarşik hedef oranı zinciri.

    levels: kabadan inceye tamsayı anahtar dizileri (her biri bütün satırlar üzerinde, encode_keys ile).
    Dönen: (len(app_idx), len(levels) * n_classes) log-olasılık; her seviyenin sonsalı bir sonrakinin önseli.
    """
    y = np.asarray(y)
    fit_idx, app_idx = np.asarray(fit_idx), np.asarray(app_idx)
    y_fit = y[fit_idx].astype(np.int64)
    if prior is None:
        prior = np.bincount(y_fit, minlength=n_classes) / len(y_fit)
    post = np.tile(np.asarray(prior, dtype=float), (len(app_idx), 1))
    out = []
    for key in levels:
        key = np.asarray(key, dtype=np.int64)
        cnt = np.zeros((int(key.max()) + 1, n_classes))
        np.add.at(cnt, (key[fit_idx], y_fit), 1.0)
        ka = key[app_idx]
        post = (cnt[ka] + S * post) / (cnt[ka].sum(axis=1, keepdims=True) + S)
        out.append(np.log(np.clip(post, 1e-12, 1.0)))
    return np.hstack(out)


def _splits(y, groups, n_splits: int, seed: int):
    y = np.asarray(y)
    if groups is None:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, np.asarray(groups)))


def cross_fit_chain(levels, y, train_idx, apply_idx, n_classes: int, S: float = 20.0, inner_splits: int = 5,
                    seed: int = 42, groups=None):
    """Dış katın eğitim satırları (train_idx) için iç katlarla çapraz zincir; apply_idx için tüm train_idx.

    groups (bütün satırlar üzerinde) verilirse iç katlar grupludur. Dönen: (eğitim özellikleri, uygulama özellikleri).
    """
    y, train_idx, apply_idx = np.asarray(y), np.asarray(train_idx), np.asarray(apply_idx)
    tr_feat = np.zeros((len(train_idx), len(levels) * n_classes))
    g = None if groups is None else np.asarray(groups)[train_idx]
    for itr, iapp in _splits(y[train_idx], g, inner_splits, seed):
        tr_feat[iapp] = chain_logp(levels, y, train_idx[itr], train_idx[iapp], n_classes, S)
    return tr_feat, chain_logp(levels, y, train_idx, apply_idx, n_classes, S)
