"""Public generated passages for Name That LLM (Kaggle datasets, CC BY 4.0) as external training rows (extdata.py).

Sources (data/external/pub/<owner>_<slug>/; in a Kaggle notebook the attached inputs):
  pv  prvsiyan/prvsiyans-extended-name-that-llm-dataset: passages_clean.csv, funnel-passing complete prompt blocks
      (split new: all 18 labels; proxy_v1: the 12 Kaggle-proxy labels). Prompts written disjoint from the train topics.
  sk  seyitkaangunes/ntl-extra-api-passages: the 5 Claude labels (official API) on a 1,000-prompt bank (g0000-g0999).
  lb  lucasboesen/ntl-free-proxy-generations: the 8 GPT / Gemini labels (Kaggle proxy, reasoning low), same bank.
  dp  dophuchung/ntl-proxy-generations: GPT / Gemini on the same bank, a settings sweep.
  ab  abdullah1afify/ntl-generations: GPT / Gemini on the same bank, mostly provider defaults.
No source used competition texts or prompts (their dataset cards); only the train word-count distribution.

Rules applied here:
  - competition format: ext_corpus.clean_text (one paragraph, no markdown), 40-250 words, de-duplicated
  - bank passages: one per (prompt, label), settings preferred low > default > medium > high; reasoning 'none' dropped
    (proxy Gemini without thinking uses em dashes in 30-43% of passages, train Gemini never does: dophuchung's card)
  - Gemini passages containing an em dash are dropped (train Gemini: 0%; every source's card reports the shift)
  - prompt blocks must carry most labels: bank prompts with fewer than `min_labels` labels are dropped (a Claude-only
    prompt turns its topic into a label shortcut for an 18-way head: lucasboesen's card); they stay usable for gates
  - groups 'pv:<prompt_id>' / 'sk:<prompt_id>'; every row is license class 'open' (CC BY 4.0, attribution)

Usage: python ext_pub.py profile        (counts per source / label, block sizes)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import CLASSES, ROOT
from ext_corpus import clean_text

D = ROOT / "data" / "external" / "pub"
SRC = {"pv": "prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages_clean.csv",
       "sk": "seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv",
       "lb": "lucasboesen_ntl-free-proxy-generations/gen_passages.csv",
       "dp": "dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv",
       "ab": "abdullah1afify_ntl-generations/ntl_generations.csv"}
SETTING_RANK = {"low": 0, "default": 1, "medium": 2, "high": 3}          # 'none' and unknown settings are dropped


def _find(root: Path, rel: str) -> Path:
    """The file under root (local layout) or anywhere below it (Kaggle inputs: /kaggle/input/<slug>/...)."""
    p = root / rel
    if p.exists():
        return p
    hits = sorted(root.glob(f"**/{Path(rel).name}"))
    if not hits:
        raise FileNotFoundError(f"{rel} not found under {root}")
    return hits[0]


def _raw(root: Path) -> pd.DataFrame:
    parts = []
    pv = pd.read_csv(_find(root, SRC["pv"]))
    parts.append(pd.DataFrame({"source": "pv", "prompt_id": pv["prompt_id"].astype(str), "label": pv["model"],
                               "text_raw": pv["text"], "setting": "default", "split": pv["split"]
                               if "split" in pv.columns else "", "orig_id": pv["id"].astype(str)}))
    sk = pd.read_csv(_find(root, SRC["sk"]))
    parts.append(pd.DataFrame({"source": "sk", "prompt_id": sk["prompt_id"].astype(str), "label": sk["model"],
                               "text_raw": sk["text"], "setting": "default", "split": "bank",
                               "orig_id": sk["id"].astype(str)}))
    for s, lab in (("lb", "model"), ("dp", "label"), ("ab", "label")):
        d = pd.read_csv(_find(root, SRC[s]))
        parts.append(pd.DataFrame({"source": s, "prompt_id": d["prompt_id"].astype(str), "label": d[lab],
                                   "text_raw": d["text"], "setting": d["setting"].astype(str), "split": "bank",
                                   "orig_id": [f"{s}{i}" for i in range(len(d))]}))
    return pd.concat(parts, ignore_index=True)


def load(min_words: int = 40, max_words: int = 250, min_labels: int = 10, raw: bool = False,
         root=None) -> pd.DataFrame:
    """Rows: ext_id, label, text, words, source, prompt_id, prompt_group_id, license_class, setting, split."""
    df = _raw(Path(root) if root else D)
    df = df[df["label"].isin(CLASSES) & df["text_raw"].notna()].copy()
    if raw:
        return df.reset_index(drop=True)
    df["text"] = df["text_raw"].astype(str).map(clean_text)
    df["words"] = df["text"].str.split().str.len()
    df = df[(df["words"] >= min_words) & (df["words"] <= max_words)]
    df = df[~(df["label"].str.startswith("gemini") & df["text"].str.contains("—"))]   # em dash: never in train
    bank = df["split"] == "bank"
    df["rank"] = df["setting"].map(SETTING_RANK)
    df = df[~bank | df["rank"].notna()]
    df = pd.concat([df[df["split"] != "bank"],
                    df[df["split"] == "bank"].sort_values(["rank", "source"]).drop_duplicates(["prompt_id", "label"])])
    df["prompt_group_id"] = (df["source"].where(df["source"] == "pv", "sk") + ":" + df["prompt_id"])
    n_lab = df.groupby("prompt_group_id")["label"].transform("nunique")
    df = df[n_lab >= min_labels]
    df = df.drop_duplicates("text")
    df = df[df.groupby("text")["label"].transform("nunique") == 1]
    df["license_class"] = "open"
    df["ext_id"] = "pub_" + df["source"] + "_" + df["orig_id"]
    cols = ["ext_id", "label", "text", "words", "source", "prompt_id", "prompt_group_id", "license_class", "setting",
            "split"]
    return df[cols].sort_values("ext_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile"])
    ap.add_argument("--min-labels", type=int, default=10)
    a = ap.parse_args()
    raw = load(raw=True)
    df = load(min_labels=a.min_labels)
    print(f"raw {len(raw)} rows -> {len(df)} training rows (min_labels {a.min_labels}); "
          f"{df['prompt_group_id'].nunique()} prompt groups")
    print("rows per source:", df["source"].value_counts().to_dict())
    print("labels per prompt group:", df.groupby("prompt_group_id")["label"].nunique().value_counts().sort_index().to_dict())
    print(df.groupby("label").size().reindex(CLASSES).fillna(0).astype(int).to_string())
    print("median words:", df.groupby("label")["words"].median().round().astype(int).to_dict())


if __name__ == "__main__":
    main()
