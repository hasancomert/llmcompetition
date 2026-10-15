"""OOF ve test tahminlerini üretme, kaydetme ve yükleme; çok sınıflı (N x C olasılık matrisi).

S6E9'daki karşılığı:
- train.py: tüm modeller tek sabit bölmede (StratifiedKFold(K, shuffle=True, random_state=42)). Her kat modeli
  doğrulama katını ve testi tahmin eder; test tahmini kat modellerinin ortalamasıdır (`pred += p / FOLDS`).
  Çıktılar oof_<etiket>.npy ve pred_<etiket>.npy.
- blend.py / blend_ext.py: virgülle verilen etiketler (ör. tohumlar) önce ortalanır, tek model sayılır.
- megayak OOF kütüphanesinin bölmesinin bizimkiyle aynı olduğu çapraz tabloyla doğrulandı (same_partition).

Çok sınıfa uyarlama: diziler (N, C); sınıf sırası meta_<etiket>.json'da tutulur ve yüklerken denetlenir.
groups verilirse (ör. prompt_id) StratifiedGroupKFold kullanılır; bir grubun bütün satırları aynı katta kalır.
Bağımlılık: numpy, scikit-learn.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def balanced_accuracy(y, p) -> float:
    """Sınıf başına recall ortalaması (y'de bulunan sınıflar üzerinden). p: (N, C) olasılık ya da (N,) etiket."""
    y = np.asarray(y)
    pred = np.asarray(p).argmax(1) if np.ndim(p) == 2 else np.asarray(p)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def make_folds(y, groups=None, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Her satırın kat numarası (0 .. n_splits-1). groups verilirse aynı gruptaki satırlar aynı katta kalır."""
    y = np.asarray(y)
    folds = np.full(len(y), -1, dtype=np.int64)
    if groups is None:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    else:
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, np.asarray(groups))
    for k, (_, va) in enumerate(splits):
        folds[va] = k
    return folds


def iter_folds(folds):
    """(kat, eğitim dizinleri, doğrulama dizinleri) üçlüleri."""
    folds = np.asarray(folds)
    for k in np.unique(folds):
        yield int(k), np.flatnonzero(folds != k), np.flatnonzero(folds == k)


def same_partition(a, b) -> bool:
    """İki kat ataması aynı bölme mi (kat numaraları farklı adlandırılmış olabilir).

    S6E9'da megayak OOF kütüphanesinin `fold` sütunu bizim StratifiedKFold(10, shuffle, 42) bölmemizle bu
    yolla (çapraz tablo: her kat tek bir kata karşılık geliyor) karşılaştırıldı.
    """
    a, b = np.asarray(a).tolist(), np.asarray(b).tolist()
    if len(a) != len(b):
        return False
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


def check_proba(p, n_classes: int | None = None, atol: float = 1e-3) -> np.ndarray:
    """(N, C) olasılık matrisi denetimi: sonlu, negatif değil, satır toplamı 1."""
    p = np.asarray(p, dtype=float)
    assert p.ndim == 2, f"olasılık matrisi (N, C) olmalı, gelen şekil {p.shape}"
    if n_classes is not None:
        assert p.shape[1] == n_classes, f"{p.shape[1]} sütun var, {n_classes} bekleniyordu"
    assert np.isfinite(p).all(), "sonlu olmayan olasılık"
    assert (p >= -1e-9).all(), "negatif olasılık"
    assert np.allclose(p.sum(1), 1.0, atol=atol), "satır toplamları 1 değil"
    return p


def _take(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    if isinstance(X, list):
        return [X[i] for i in idx]
    return X[idx]


def cv_predict(fit_predict, X, y, X_test, folds, n_classes: int, verbose: bool = True):
    """Kat döngüsü. fit_predict(X_tr, y_tr, X_va, X_te) -> (p_va (n_va, C), p_te (n_te, C)).

    Dönen: oof (N, C), test (M, C) = kat modellerinin ortalaması, kat başına balanced accuracy listesi.
    """
    y = np.asarray(y)
    oof = np.full((len(y), n_classes), np.nan)
    test = np.zeros((len(X_test), n_classes))
    splits, scores = list(iter_folds(folds)), []
    for k, tr, va in splits:
        p_va, p_te = fit_predict(_take(X, tr), y[tr], _take(X, va), X_test)
        oof[va] = check_proba(p_va, n_classes)
        test += check_proba(p_te, n_classes) / len(splits)
        scores.append(balanced_accuracy(y[va], oof[va]))
        if verbose:
            print(f"  kat {k}: balanced accuracy {scores[-1]:.5f}", flush=True)
    assert np.isfinite(oof).all(), "bazı satırlar hiçbir doğrulama katında yok"
    if verbose:
        print(f"OOF balanced accuracy: {balanced_accuracy(y, oof):.5f}", flush=True)
    return oof, test, scores


def save_preds(tag: str, oof, test, classes, outdir: str = ".", **meta) -> None:
    """oof_<etiket>.npy, pred_<etiket>.npy ve sınıf sırasıyla meta_<etiket>.json yazar."""
    oof, test = np.asarray(oof), np.asarray(test)
    assert oof.shape[1] == test.shape[1] == len(classes), "sütun sayısı sınıf sayısıyla aynı olmalı"
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"oof_{tag}.npy"), oof)
    np.save(os.path.join(outdir, f"pred_{tag}.npy"), test)
    info = {"classes": [str(c) for c in classes], "n_oof": int(len(oof)), "n_test": int(len(test))} | meta
    with open(os.path.join(outdir, f"meta_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)


def load_preds(spec: str, outdir: str = ".", classes=None):
    """spec "a,b,c": etiketlerin OOF ve test tahminleri ortalanır (tohum ortalaması). Sınıf sırası denetlenir.

    Dönen: oof (N, C), test (M, C), sınıf listesi.
    """
    oofs, tests, cls = [], [], None
    for tag in spec.split(","):
        with open(os.path.join(outdir, f"meta_{tag}.json"), encoding="utf-8") as f:
            c = json.load(f)["classes"]
        cls = c if cls is None else cls
        assert c == cls, f"{tag}: sınıf sırası öncekilerden farklı"
        oofs.append(np.load(os.path.join(outdir, f"oof_{tag}.npy")))
        tests.append(np.load(os.path.join(outdir, f"pred_{tag}.npy")))
    if classes is not None:
        assert [str(x) for x in classes] == cls, "istenen sınıf sırası dosyadakiyle aynı değil"
    return np.mean(oofs, axis=0), np.mean(tests, axis=0), cls
