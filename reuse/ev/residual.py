"""Artık (residual) boosting: taban modelin logit'inden başlayan XGBoost / LightGBM / CatBoost; çok sınıflı.

S6E9'daki karşılığı:
- notebooks/glr_margin/glrm.ipynb: her dış katta doğrulama ve test satırlarının logit'i dış kat GLR'sinden gelir.
  Eğitim satırlarınınki, o katın eğitim satırlarında iç 5 katla (StratifiedKFold(5, shuffle, random_state=43))
  yeniden eğitilen GLR'den gelir. Böylece hiçbir eğitim satırının başlangıç logit'i kendi etiketini görmüş bir
  modelden gelmez. Çıktı: fit (K, N), oof (N,), test (K, M).
- train.py margin=1: XGBoost `base_margin` / `base_margin_eval_set`; LightGBM `init_score` / `eval_init_score`,
  tahmin = sigmoid(raw_score + logit); CatBoost `Pool(baseline=...)`, tahmin = sigmoid(RawFormulaVal + logit).
  Test için o katın dış GLR test logit'i. Kazanç: model başına +0.00006 ... +0.00007 AUC (10 kat).
  Erken durdurma doğrulama katının kendisiyle yapıldı; bunun OOF'a etkisi ölçülmedi.

Çok sınıfa uyarlama: logit'ler (n, C). XGBoost ve LightGBM yerel API ile çalışır, num_class açıkça verilir.
Olasılık = softmax(ham skor + logit). Bu ortamda (xgboost 3.2.0, lightgbm 4.7.0, catboost 1.2.10) denendi:
- XGBoost'ta `output_margin=True` çıktısı base_score'u da içerir. Logit'i elle eklemek bu yüzden yanlış sonuç
  verir; tahmin DMatrix(base_margin=...) ile alınmalı.
- LightGBM `raw_score` ve CatBoost `RawFormulaVal` başlangıç logit'ini içermez; softmax(ham + logit) doğru.
Erken durdurma kümesi isteğe bağlıdır; varsayılan sabit tur sayısı.
Bağımlılık: numpy, scikit-learn; xgboost, lightgbm, catboost yalnız ilgili fonksiyonda içe aktarılır.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def softmax(Z) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


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


def logreg_logits(n_classes: int, C: float = 1.0, max_iter: int = 2000):
    """Taban model örneği: çok terimli lojistik regresyon. Eğitimde olmayan sınıfın logit'i -30."""
    def base(X_fit, y_fit, X_apps):
        m = LogisticRegression(C=C, max_iter=max_iter).fit(X_fit, y_fit)
        out = []
        for Xa in X_apps:
            d = m.decision_function(Xa)
            if d.ndim == 1:  # iki sınıf: tek sütun log-odds
                d = np.c_[np.zeros(len(d)), d]
            z = np.full((len(d), n_classes), -30.0)
            z[:, m.classes_] = d
            out.append(z)
        return out
    return base


def leak_free_margins(base_logits, X, y, folds, X_test, n_classes: int, groups=None,
                      inner_splits: int = 5, inner_seed: int = 43) -> list:
    """Her dış kat için sızıntısız başlangıç logit'leri.

    base_logits(X_fit, y_fit, [X_app, ...]) -> [logit (n_app, C), ...].
    Dönen liste, kat başına dict: tr, va (dizinler); fit (len(tr), C) iç katlardan; valid (len(va), C) ve
    test (M, C) dış kat modelinden. groups verilirse iç katlar da grupludur.
    Maliyet: kat başına taban model 1 + inner_splits kez eğitilir.
    """
    y, folds = np.asarray(y), np.asarray(folds)
    out = []
    for k in np.unique(folds):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        valid, test = base_logits(_take(X, tr), y[tr], [_take(X, va), X_test])
        fit = np.full((len(tr), n_classes), np.nan)
        g_tr = None if groups is None else np.asarray(groups)[tr]
        for itr, iapp in _splits(y[tr], g_tr, inner_splits, inner_seed):
            fit[iapp] = base_logits(_take(X, tr[itr]), y[tr[itr]], [_take(X, tr[iapp])])[0]
        assert np.isfinite(fit).all(), "her eğitim satırının iç-kat logit'i olmalı"
        out.append({"fold": int(k), "tr": tr, "va": va, "fit": fit, "valid": np.asarray(valid),
                    "test": np.asarray(test)})
    return out


def fit_xgb_residual(X_tr, y_tr, m_tr, X_preds, m_preds, n_classes: int, params=None,
                     num_boost_round: int = 300, eval_set=None, early_stopping_rounds=None):
    """XGBoost (multi:softprob), başlangıç logit'i base_margin. Dönen: ([olasılık (n, C), ...], booster).

    X_preds / m_preds: tahmin edilecek matrisler ve logit'leri. eval_set=(X_es, y_es, m_es) erken durdurma için.
    """
    import xgboost as xgb
    p = {"objective": "multi:softprob", "num_class": n_classes, "eta": 0.05, "max_depth": 4,
         "eval_metric": "mlogloss", "verbosity": 0} | (params or {})
    dtr = xgb.DMatrix(X_tr, label=y_tr, base_margin=np.asarray(m_tr))
    evals = []
    if eval_set is not None:
        Xe, ye, me = eval_set
        evals = [(xgb.DMatrix(Xe, label=ye, base_margin=np.asarray(me)), "es")]
    bst = xgb.train(p, dtr, num_boost_round, evals=evals,
                    early_stopping_rounds=early_stopping_rounds if evals else None, verbose_eval=False)
    last = bst.best_iteration + 1 if (evals and early_stopping_rounds) else bst.num_boosted_rounds()
    preds = [bst.predict(xgb.DMatrix(X, base_margin=np.asarray(m)), iteration_range=(0, last))
             for X, m in zip(X_preds, m_preds)]
    return preds, bst


def fit_lgbm_residual(X_tr, y_tr, m_tr, X_preds, m_preds, n_classes: int, params=None,
                      num_boost_round: int = 300, eval_set=None, early_stopping_rounds=None):
    """LightGBM (multiclass), başlangıç logit'i init_score; olasılık = softmax(raw_score + logit)."""
    import lightgbm as lgb
    p = {"objective": "multiclass", "num_class": n_classes, "learning_rate": 0.05, "num_leaves": 15,
         "verbose": -1} | (params or {})
    dtr = lgb.Dataset(X_tr, label=y_tr, init_score=np.asarray(m_tr))
    valid_sets, callbacks = [], []
    if eval_set is not None:
        Xe, ye, me = eval_set
        valid_sets = [lgb.Dataset(Xe, label=ye, init_score=np.asarray(me), reference=dtr)]
        if early_stopping_rounds:
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    bst = lgb.train(p, dtr, num_boost_round, valid_sets=valid_sets, callbacks=callbacks)
    it = bst.best_iteration if bst.best_iteration and bst.best_iteration > 0 else None
    preds = [softmax(bst.predict(X, raw_score=True, num_iteration=it) + np.asarray(m))
             for X, m in zip(X_preds, m_preds)]
    return preds, bst


def fit_cat_residual(X_tr, y_tr, m_tr, X_preds, m_preds, n_classes: int, params=None, eval_set=None,
                     cat_features=None):
    """CatBoost (MultiClass), başlangıç logit'i Pool(baseline=...); olasılık = softmax(RawFormulaVal + logit).

    Sınıf sırası class_names=0..C-1 ile sabitlenir (etiketler 0..C-1 tamsayı olmalı). Eğitim katında bütün
    sınıflar bulunmalı: catboost 1.2.10'da eksik sınıf "Baseline data size is not equal to class count" hatası
    veriyor (denendi). XGBoost ve LightGBM sürümleri eksik sınıfla çalışıyor.
    """
    from catboost import CatBoostClassifier, Pool
    p = {"loss_function": "MultiClass", "iterations": 300, "learning_rate": 0.1, "depth": 4, "verbose": 0,
         "random_seed": 0, "class_names": list(range(n_classes))} | (params or {})
    model = CatBoostClassifier(**p)
    es = None
    if eval_set is not None:
        Xe, ye, me = eval_set
        es = Pool(Xe, ye, baseline=np.asarray(me), cat_features=cat_features)
    model.fit(Pool(X_tr, y_tr, baseline=np.asarray(m_tr), cat_features=cat_features), eval_set=es)
    preds = [softmax(model.predict(Pool(X, cat_features=cat_features), prediction_type="RawFormulaVal")
                     + np.asarray(m)) for X, m in zip(X_preds, m_preds)]
    return preds, model


def residual_cv(fit_fn, X, y, X_test, margins: list, n_classes: int, es_on_valid: bool = False, **fit_kw):
    """leak_free_margins çıktısıyla kat döngüsü. Dönen: oof (N, C), test (M, C) (kat ortalaması).

    es_on_valid=True S6E9'daki gibi erken durdurmayı doğrulama katının kendisiyle yapar (etkisi ölçülmedi).
    """
    y = np.asarray(y)
    oof = np.full((len(y), n_classes), np.nan)
    test = np.zeros((len(X_test), n_classes))
    for d in margins:
        tr, va = d["tr"], d["va"]
        kw = dict(fit_kw)
        if es_on_valid:
            kw["eval_set"] = (_take(X, va), y[va], d["valid"])
        (p_va, p_te), _ = fit_fn(_take(X, tr), y[tr], d["fit"], [_take(X, va), X_test],
                                 [d["valid"], d["test"]], n_classes, **kw)
        oof[va] = p_va
        test += p_te / len(margins)
    assert np.isfinite(oof).all()
    return oof, test
