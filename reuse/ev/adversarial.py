"""Train ile test ayırt edilebilir mi (düşmanca doğrulama) ve id sırasında etiket yapısı var mı.

S6E9'daki karşılığı (oturum günlüğündeki tek seferlik komut; README "Yeni desen arayışı"):
- Düşmanca doğrulama: train + test ham sütunları, hedef "satır testten mi". LightGBM (300 ağaç, lr 0.1,
  63 yaprak), StratifiedKFold(3, shuffle, random_state=0) ile OOF. AUC 0.50005 çıktı, yani ayırt edilemedi.
- id sırası: ardışık satırlarda etiket korelasyonu (gecikme 1: +0.001, şans sınırı ±0.0012) ve ardışık satırlarda
  aynı kategori oranı ile rastgele beklenti (sum p^2) karşılaştırıldı. Yapı bulunamadı.
Çok sınıfa uyarlama: adversarial_auc herhangi bir sayısal özellik matrisiyle çalışır (metin için gömme ya da stil
özellikleri). order_check çok sınıflı etiketlerde ardışık aynı sınıf oranını şans düzeyiyle (sum p_c^2)
karşılaştırır. Bağımlılık: numpy, scikit-learn, lightgbm.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def adversarial_auc(F_train, F_test, n_splits: int = 3, seed: int = 0, params=None):
    """Dönen: (AUC, OOF "test olma" olasılığı). İlk len(F_train) satır train, gerisi test."""
    import lightgbm as lgb
    import pandas as pd
    A = pd.concat([pd.DataFrame(F_train), pd.DataFrame(F_test)], ignore_index=True)
    t = np.r_[np.zeros(len(F_train)), np.ones(len(F_test))]
    p = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 63, "verbose": -1} | (params or {})
    oof = np.zeros(len(t))
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(A, t):
        m = lgb.LGBMClassifier(**p).fit(A.iloc[a], t[a])
        oof[b] = m.predict_proba(A.iloc[b])[:, 1]
    return float(roc_auc_score(t, oof)), oof


def order_check(labels_in_id_order, lags=(1, 2, 10)) -> list:
    """Ardışık (id sırasındaki) satırların aynı sınıfta olma oranı ile şans düzeyi sum p_c^2.

    Dönen: gecikme başına dict (lag, observed, chance, z); z bağımsızlık varsayımıyla yaklaşık.
    """
    y = np.asarray(labels_in_id_order)
    _, counts = np.unique(y, return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())
    out = []
    for lag in lags:
        obs = float(np.mean(y[:-lag] == y[lag:]))
        se = np.sqrt(chance * (1 - chance) / (len(y) - lag))
        out.append({"lag": lag, "observed": obs, "chance": chance, "z": (obs - chance) / se})
    return out
