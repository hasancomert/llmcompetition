"""Larger aggregated public passage pool (Kaggle dataset dhruvtibarewal/nlm-matched-clean, file nlm_pool_v5.parquet:
119,899 passages, 46 sources) as external training rows (extdata.py --ext pool5).

The dataset (licence MIT; every row keeps the licence of its upstream dataset, column src) merges public datasets of
competitors' generations under the competition's protocol, plus ~1.2k passages its author generated through the
Kaggle Benchmarks proxy (src own_gen, w_host 2). Its builder removed exact and near duplicates and every text of the
official train set, dropped chat / trace / benchmark-style corpora, and dropped a passage when it holds a typographic
mark its label never uses in train. It contains 38,166 of the 38,642 passages of ext_pool.py's pool. Public
notebooks (mathleteutd, from dhruvtibarewal's write-up) report two-stage LoRA at 0.634 (14B) with it vs ~0.57 on the
38k pools.

Licences (kaggle datasets metadata of every source, 29 Oct; EXTERNAL_DATA.md): CC BY 4.0, CC BY-SA 4.0, CC0 or MIT,
except DROP_SOURCES (licence 'other' without terms, 'unknown', or local rows of unclear provenance), left out here.

Rules applied here (as ext_pool.py): competition format (ext_corpus.clean_text, 40-250 words), Gemini passages with
an em dash dropped, prompt blocks with fewer than `min_labels` labels dropped, texts de-duplicated (a text with two
labels is dropped). Groups 'p5:<prompt_key>' (rows without a prompt key are their own group); license class 'open'.

Usage: python ext_pool5.py profile [--min-labels 1]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from common import CLASSES, ROOT
from ext_corpus import clean_text

D = ROOT / "data" / "external" / "pools" / "dhruv"
FILE = "nlm_pool_v5.parquet"
SHA256 = "040afb3609d7f6586d57c89ddb42d9bf4b58cc26426104b57390bdf6312b4650"   # the version every local CV used
DROP_SOURCES = {"abdelghanyrageh/ntl-extra-merged": "licence 'other', no terms given",
                "illiadolenko/ntl-merged-corpus-v2": "licence 'unknown'",
                "local/kaggle_harvest": "no upstream dataset named", "local/evren_batch_pilot": "no upstream dataset named"}


def _find(root: Path) -> Path:
    p = root / FILE
    if p.exists():
        return p
    hits = sorted(root.glob(f"**/{FILE}"))
    if not hits:
        raise FileNotFoundError(f"{FILE} not found under {root}")
    return hits[0]


def load(min_words: int = 40, max_words: int = 250, min_labels: int = 1, root=None) -> pd.DataFrame:
    """Rows: ext_id, label, text, words, source, prompt_id, prompt_group_id, license_class."""
    path = _find(Path(root) if root else D)
    if root:                                          # an attached copy (notebook): say if it is another version
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[ext_pool5] {path} sha256 {sha[:12]}" + ("" if sha == SHA256 else
              f"  WARNING: differs from the version of the local CV ({SHA256[:12]})"), flush=True)
    df = pd.read_parquet(path)
    df = df[df["model"].isin(CLASSES) & df["text"].notna()].rename(columns={"model": "label", "src": "source"}).copy()
    df = df[~df["source"].str.split(":").str[0].isin(DROP_SOURCES)]
    df["text"] = df["text"].astype(str).map(clean_text)
    df["words"] = df["text"].str.split().str.len()
    df = df[(df["words"] >= min_words) & (df["words"] <= max_words)]
    df = df[~(df["label"].str.startswith("gemini") & df["text"].str.contains("—"))]   # em dash: never in train
    df = df[df.groupby("text")["label"].transform("nunique") == 1].drop_duplicates("text")
    key = df["prompt_key"].fillna("").astype(str)
    df["ext_id"] = "p5_" + (key + "\x00" + df["label"] + "\x00" + df["text"]).map(
        lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    df["prompt_id"] = key.where(key != "", df["ext_id"])
    df["prompt_group_id"] = "p5:" + df["prompt_id"]
    n_lab = df.groupby("prompt_group_id")["label"].transform("nunique")
    df = df[n_lab >= min_labels]
    df["license_class"] = "open"
    assert df["ext_id"].is_unique
    cols = ["ext_id", "label", "text", "words", "source", "prompt_id", "prompt_group_id", "license_class"]
    return df[cols].sort_values("ext_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile"])
    ap.add_argument("--min-labels", type=int, default=1)
    a = ap.parse_args()
    df = load(min_labels=a.min_labels)
    print(f"{len(df)} training rows (min_labels {a.min_labels}); {df['prompt_group_id'].nunique()} prompt groups")
    print("rows per source:", df["source"].value_counts().to_dict())
    print("labels per prompt group:", df.groupby("prompt_group_id")["label"].nunique().value_counts().sort_index().to_dict())
    print(df.groupby("label").size().reindex(CLASSES).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
