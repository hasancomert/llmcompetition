"""Metin için tokenizer tabanlı özellikler; S6E9'daki gpt2tok.py ve train.py 'tok' grubunun metne uyarlaması.

S6E9'daki karşılığı:
- gpt2tok.py: her gelir değeri " " + str(değer) olarak GPT-2 tokenizer'ıyla parçalandı. Tutulanlar: parça sayısı
  (ntok), ilk üç parça kimliği (t1, t2, t3), son parça (last) ve parçaların metni. Betik internetli bir Kaggle
  CPU not defterinde çalıştı; çıktı CSV, internetsiz eğitim not defterlerine girdi olarak bağlandı.
  (Mesafe için de eşleme üretildi ama hiçbir modelde kullanılmadı.)
- train.py 'tok': anahtarlar L1 = t1, L2 = t1 * 60000 + t2, LAST = ntok * 60000 + last. Her biri hedef kodlama
  anahtarı, ayrıca train + test + orijinaldeki log1p sayımı (etiketsiz) ve parça sayısı (tok_n).
  Kazanç: LightGBM 5 kat 0.94573 -> 0.94608 AUC; 10 katta model başına ~+0.0002.

Metne uyarlama:
- Tokenizer'lar ada göre sözlükle verilir; her biri metin -> kimlik listesi döndüren bir çağrılabilirdir.
- Anahtarlar string ("t1|t2"). 60000 çarpanı kimliklerin 60000'in altında olduğunu varsayar; bu GPT-2 için
  doğru, daha büyük sözlüklü tokenizer'larda anahtarlar çakışır.
- LAST burada parça sayısıyla birleştirilmez (metin uzunluğu çok değişken olduğundan anahtar tekilleşir).
- Hugging Face tokenizer'ı internetsiz ortamda yerel klasörden yüklenir (load_hf_tokenizer). Bu ortamda
  transformers kurulu olmadığı için o yol test edilmedi.
Bağımlılık: numpy, pandas (transformers yalnız load_hf_tokenizer içinde).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_hf_tokenizer(path: str, add_special_tokens: bool = False):
    """Yerel klasördeki Hugging Face tokenizer'ı -> metin -> kimlik listesi çağrılabiliri (internet gerekmez)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return lambda text: tok(text, add_special_tokens=add_special_tokens)["input_ids"]


def token_table(texts, tokenizers: dict, prefix=(1, 2), last: bool = True) -> pd.DataFrame:
    """Her tokenizer için: n_tok_<ad> (parça sayısı), key_<ad>_L<k> (ilk k parça) ve key_<ad>_LAST (son parça)."""
    cols = {}
    for name, tok in tokenizers.items():
        ids = [list(tok(t)) for t in texts]
        cols[f"n_tok_{name}"] = np.array([len(i) for i in ids], dtype=np.int64)
        for k in prefix:
            cols[f"key_{name}_L{k}"] = ["|".join(map(str, i[:k])) for i in ids]
        if last:
            cols[f"key_{name}_LAST"] = [str(i[-1]) if i else "" for i in ids]
    return pd.DataFrame(cols)


def add_key_counts(df: pd.DataFrame, key_cols=None) -> pd.DataFrame:
    """Her anahtar sütunu için log1p(anahtarın tablodaki sayısı) ekler (etiketsiz; S6E9'da train + test + orijinal)."""
    df = df.copy()
    for c in key_cols or [c for c in df.columns if c.startswith("key_")]:
        df[c + "_cnt"] = np.log1p(df[c].map(df[c].value_counts()).to_numpy(dtype=float))
    return df
