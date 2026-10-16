"""Does the external response corpus help? TF-IDF member trained on competition training folds + external texts.

External data (EXTERNAL_DATA.md): olegzholobov/name-that-llm-response-corpus, external_short_wide.parquet (40-250-word
candidates). Used ONLY for training: every validation row is a competition text of the held-out fold; external rows
never enter validation. External texts are flattened to the competition format (newlines -> spaces, markdown list /
header / emphasis / code markers removed), re-filtered to 40-250 words and de-duplicated.

Usage: python ext_tfidf_probe.py --feats char+word --C 30 --ext-weight 0.3 [--flat-only] [--tag ...]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import CLASSES, K, ROOT, encode_y, load_folds, load_test, load_train, metrics, run_member, scores
from train_tfidf import make_vec

EXT = ROOT / "data" / "external" / "ntl_corpus" / "external_short_wide.parquet"


def clean(t: str) -> str:
    t = re.sub(r"```.*?```", " ", t, flags=re.S)               # code blocks
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(?m)^\s*(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", "", t)   # headers, bullets, numbered items, quotes
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def load_external(flat_only: bool = False) -> pd.DataFrame:
    w = pd.read_parquet(EXT)
    long = w.melt(id_vars=["id", "prompt"], var_name="model", value_name="text").dropna(subset=["text"])
    long = long[long["model"].isin(CLASSES)]
    if flat_only:
        md = long["text"].str.contains(r"\n|\*\*|```|(?:^|\s)#{1,6}\s", regex=True)
        long = long[~md]
    long = long.assign(text=long["text"].map(clean))
    nw = long["text"].str.split().str.len()
    long = long[(nw >= 40) & (nw <= 250)]
    long = long.drop_duplicates(subset=["model", "text"]).drop_duplicates(subset=["text"], keep=False)
    return long.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="char+word")
    ap.add_argument("--C", type=float, default=30.0)
    ap.add_argument("--ext-weight", type=float, default=0.3, help="sample weight of an external row (competition rows 1)")
    ap.add_argument("--balance", action="store_true", help="also reweight external rows to equal total weight per class")
    ap.add_argument("--flat-only", action="store_true", help="only external texts that were already single-paragraph")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    tr, te = load_train(), load_test()
    y = encode_y(tr["model"])
    folds = load_folds(tr)
    ext = load_external(a.flat_only)
    ye = encode_y(ext["model"])
    we = np.full(len(ext), a.ext_weight)
    if a.balance:
        cnt = np.bincount(ye, minlength=K).astype(float)
        we = a.ext_weight * (len(ext) / K) / np.maximum(cnt[ye], 1.0)
    print(f"external rows {len(ext)}; per class {np.bincount(ye, minlength=K).tolist()}  ({time.time() - t0:.0f}s)",
          flush=True)
    texts, te_texts, ext_texts = tr["text"].tolist(), te["text"].tolist(), ext["text"].tolist()
    oof, test = np.zeros((len(tr), K)), np.zeros((len(te), K))
    for k in range(int(folds.max()) + 1):
        trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        fit_texts = [texts[i] for i in trn] + ext_texts
        mats = {"tr": [], "va": [], "te": []}
        for f in a.feats.split("+"):
            v = make_vec(f).fit(fit_texts)
            mats["tr"].append(v.transform(fit_texts))
            mats["va"].append(v.transform([texts[i] for i in val]))
            mats["te"].append(v.transform(te_texts))
        A = {s: sp.hstack(m).tocsr() for s, m in mats.items()}
        yy = np.concatenate([y[trn], ye])
        sw = np.concatenate([np.ones(len(trn)), we])
        m = LogisticRegression(C=a.C, max_iter=3000, tol=1e-4).fit(A["tr"], yy, sample_weight=sw)
        oof[val] = m.predict_proba(A["va"])
        test += m.predict_proba(A["te"]) / 5
        print(f"fold {k}: BA {metrics.balanced_accuracy(y[val], oof[val]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    tag = a.tag or (f"ext_tfidf_{a.feats.replace('+', '')}_lr_C{a.C:g}_w{a.ext_weight:g}"
                    f"{'_bal' if a.balance else ''}{'_flat' if a.flat_only else ''}")
    run_member(tag, f"TF-IDF {a.feats} + LogReg C={a.C:g}, trained on competition training folds + {len(ext)} EXTERNAL "
               f"texts (name-that-llm-response-corpus short view, flattened; weight {a.ext_weight:g}"
               f"{', class-balanced' if a.balance else ''}{', single-paragraph only' if a.flat_only else ''}); "
               f"validation = competition folds only", oof, test, y, folds, t0,
               note="external data experiment (EXTERNAL_DATA.md); compare with tfidf_charword_lr_C30 0.3471",
               ext_rows=len(ext), ext_weight=a.ext_weight)


if __name__ == "__main__":
    main()
