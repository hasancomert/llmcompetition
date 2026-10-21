"""Aggregated public passage pool for Name That LLM (Kaggle dataset prvsiyan/ntl-public-in-style-pool-v1, CC BY-SA 4.0)
as external training rows (extdata.py --ext pool).

The pool (pool_v1.parquet: prompt_id, model, text, source) merges 15 public datasets of competitors' generations to
prompts in the competition's style (README of the dataset; the five --ext pub sources are among them). Its builder
already keeps single-paragraph passages of 30-300 words without markdown, drops exact duplicates and passages identical
to a competition training passage, and drops a passage of model M when it holds a typographic mark (em / en dash, curly
quotes, ellipsis) that M never uses in the competition training data. No competition text or prompt is included.

Rules applied here (as ext_pub.py): competition format (ext_corpus.clean_text, 40-250 words), Gemini passages with an em
dash dropped, prompt blocks with fewer than `min_labels` labels dropped, texts de-duplicated (a text with two labels is
dropped). Groups 'pl:<prompt_id>'; every row is license class 'open' (CC BY 4.0 / CC BY-SA 4.0, attribution).

Usage: python ext_pool.py profile [--min-labels 10]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from common import CLASSES, ROOT
from ext_corpus import clean_text

D = ROOT / "data" / "external" / "pools" / "prvsiyan"
FILE = "pool_v1.parquet"
SHA256 = "4ae4f9b5ead5af3471e6f1e103e1b3add2af170d924a83d2c74ac7a8646489dc"   # the version every local CV used


def _find(root: Path) -> Path:
    """pool_v1.parquet under root (local layout) or anywhere below it (Kaggle inputs)."""
    p = root / FILE
    if p.exists():
        return p
    hits = sorted(root.glob(f"**/{FILE}"))
    if not hits:
        raise FileNotFoundError(f"{FILE} not found under {root}")
    return hits[0]


def load(min_words: int = 40, max_words: int = 250, min_labels: int = 10, root=None) -> pd.DataFrame:
    """Rows: ext_id, label, text, words, source, prompt_id, prompt_group_id, license_class."""
    path = _find(Path(root) if root else D)
    if root:                                          # an attached copy (notebook): say if it is another version
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[ext_pool] {path} sha256 {sha[:12]}" + ("" if sha == SHA256 else
              f"  WARNING: differs from the version of the local CV ({SHA256[:12]})"), flush=True)
    df = pd.read_parquet(path)
    df = df[df["model"].isin(CLASSES) & df["text"].notna()].rename(columns={"model": "label"}).copy()
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["text"] = df["text"].astype(str).map(clean_text)
    df["words"] = df["text"].str.split().str.len()
    df = df[(df["words"] >= min_words) & (df["words"] <= max_words)]
    df = df[~(df["label"].str.startswith("gemini") & df["text"].str.contains("—"))]   # em dash: never in train
    df["prompt_group_id"] = "pl:" + df["prompt_id"]
    n_lab = df.groupby("prompt_group_id")["label"].transform("nunique")
    df = df[n_lab >= min_labels]
    df = df.drop_duplicates("text")
    df = df[df.groupby("text")["label"].transform("nunique") == 1]
    df["license_class"] = "open"
    df["ext_id"] = "pl_" + (df["prompt_id"] + "\x00" + df["label"] + "\x00" + df["text"]).map(
        lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    assert df["ext_id"].is_unique
    cols = ["ext_id", "label", "text", "words", "source", "prompt_id", "prompt_group_id", "license_class"]
    return df[cols].sort_values("ext_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile"])
    ap.add_argument("--min-labels", type=int, default=10)
    a = ap.parse_args()
    df = load(min_labels=a.min_labels)
    print(f"{len(df)} training rows (min_labels {a.min_labels}); {df['prompt_group_id'].nunique()} prompt groups")
    print("rows per source:", df["source"].value_counts().to_dict())
    print("labels per prompt group:", df.groupby("prompt_group_id")["label"].nunique().value_counts().sort_index().to_dict())
    print(df.groupby("label").size().reindex(CLASSES).fillna(0).astype(int).to_string())
    print("median words:", df.groupby("label")["words"].median().round().astype(int).to_dict())


if __name__ == "__main__":
    main()
