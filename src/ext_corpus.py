"""External response corpus: Kaggle dataset olegzholobov/name-that-llm-response-corpus (snapshot 15 Oct 2026).

Only the short view is used (external_short_wide.parquet: 40-250-word answer candidates). The all view overlaps it and is
never concatenated with it; the quarantine_luna_* files (author-asserted gpt-5.6-luna labels) are excluded by default.
Every short-view cell is joined to its cell metadata (response_id, prompt_group_id), canonical response metadata
(source, domain, unit, finish reason, evidence level) and to all of its provenance occurrences (all sources).

Texts are brought to the competition format: the competition texts are single paragraphs without newlines or markdown,
so code blocks, markdown headers / bullets / emphasis are removed and newlines become spaces (`flat` records whether the
raw answer was already a single plain paragraph). After cleaning, rows outside 40-250 whitespace words are dropped,
exact duplicate (label, text) pairs are kept once and texts that occur under two labels are dropped.

License classes (per source, from source_catalog.csv; the corpus does not relicense anything):
  open          the source declares an open license for the released rows (Apache-2.0 / MIT / CC-BY-4.0)
  open_terms    open repository / prompt license, but the model outputs are declared to stay under the providers' terms
  unknown       no license, no output license or an unspecified upstream license

Usage: python ext_corpus.py profile        (counts, lengths, sources, licenses -> data/external/ntl_corpus/profile_short.json)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, ROOT

D = ROOT / "data" / "external" / "ntl_corpus"
KAGGLE_DATASET = "olegzholobov/name-that-llm-response-corpus"

LICENSE_CLASS = {
    "Met4physics/claude-opus-4-8-xhigh-reasoning-8.7k": "open",          # apache-2.0
    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k": "open",           # apache-2.0
    "Zek-Takai/glm53-flash-harvest": "open",                             # mit
    "evaleval/EEE_datastore": "open",                                    # mit
    "greghavens/kimi-k3-coding-and-debugging-traces": "open",            # cc-by-4.0
    "artindnr/khayyam-challenge": "open",                                # cc-by-4.0
    "zlab-princeton/SWEeper-Bench-traces": "open",                       # cc-by-4.0 (quoted code keeps its licenses)
    "pamela-zenodo-21278557": "open",                                    # CC-BY-4.0
    "Tongyi-MAI/MobileWorld": "open",                                    # Apache-2.0
    "arcprize_arc_agi_v2_public_eval": "open",                           # MIT dataset declaration
    "speechmap": "open_terms",                                           # Apache-2.0 repo; output terms may apply
    "petergpt/bullshit-benchmark": "open_terms",                         # MIT repo; provider output terms may apply
    "woog/arena-prose-100-49-models": "open_terms",                      # CC-BY-4.0 prompts; provider output terms
    "novigens/token-efficiency-benchmark": "open_terms",                 # MIT repo; provider output terms
    "periospot_dental_benchmark": "open_terms",                          # MIT repo; output rights not stated
    "endless_exam": "open_terms",                                        # MIT code; provider terms for responses
    "josefchen/flavourbench_complete_core": "open_terms",                # CC BY 4.0 prompts; provider output terms
    "Eventual-Inc/VLADBench-reeval": "open_terms",                       # CC-BY-4.0 answers; questions unlicensed
    "Postman APIFlow-Bench raw trial transcripts": "open_terms",         # Apache-2.0 compilation; output reuse conditional
    "rise_unibas_humanities_data_benchmark": "open_terms",               # GPL-3.0 software only; outputs need review
    "eqbench4": "unknown",
    "eqbench_creative_writing_longform": "unknown",
    "eqbench_creative_writing_v3": "unknown",
    "glm_flash_analysis": "unknown",
    "artificialanalysis_microevals": "unknown",                          # no output-specific reuse license verified
    "copilot_model_value_benchmark": "unknown",
    "ModelLineup": "unknown",
    "namethatllm_public_game": "unknown",
    "anthropic_agentic_misalignment_summer_2026": "unknown",
    "PawanKrd/claude-fable-5-code": "unknown",                           # unspecified output license
    "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces": "unknown",               # unspecified upstream; mirror declares cc-by-4.0
}

_MD = re.compile(r"\n|\*\*|__|```|`|(?m:^\s*(?:#{1,6}\s|[-*+•]\s|\d+[.)]\s|>\s))")


def clean_text(t: str) -> str:
    """Competition format: one paragraph, no markdown (code blocks dropped, list / header / quote markers removed)."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(?m)^\s*(#{1,6}\s+|[-*+•]\s+|\d+[.)]\s+|>\s*)", "", t)
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t, flags=re.S)
    t = re.sub(r"\s*\n+\s*", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def load(min_words: int = 40, max_words: int = 250, raw: bool = False, root=None) -> pd.DataFrame:
    """Short-view cells with metadata; cleaned, length-filtered and de-duplicated unless raw=True. `root`: the dataset
    directory (default data/external/ntl_corpus; in a Kaggle notebook the attached input)."""
    D_ = Path(root) if root else D
    w = pd.read_parquet(D_ / "external_short_wide.parquet")
    long = w.melt(id_vars=["id", "prompt"], var_name="model", value_name="text_raw").dropna(subset=["text_raw"])
    long = long[long["model"].isin(CLASSES)]
    cm = pd.read_parquet(D_ / "external_cell_metadata.parquet")
    cm = cm[cm["view"] == "short"][["id", "model", "cell_id", "response_id", "prompt_group_id", "prompt_exact",
                                    "license_unknown", "output_terms_review", "truncation_metadata_missing"]]
    df = long.merge(cm, on=["id", "model"], how="left", validate="one_to_one")
    rm = pd.read_parquet(D_ / "external_response_metadata.parquet", columns=[
        "response_id", "source", "domain", "unit", "response_kind", "finish_reason", "is_dialogue_turn",
        "has_code_fence", "evidence_level", "occurrences"])
    df = df.merge(rm, on="response_id", how="left", validate="many_to_one")
    pv = pd.read_parquet(D_ / "external_cell_provenance.parquet", columns=["cell_id_short", "source"])
    allsrc = pv.dropna(subset=["cell_id_short"]).groupby("cell_id_short")["source"].agg(lambda s: ";".join(sorted(set(s))))
    df["all_sources"] = df["cell_id"].map(allsrc)
    df["license_class"] = df["source"].map(LICENSE_CLASS).fillna("unknown")
    df["flat"] = ~df["text_raw"].str.contains(_MD)
    df["text"] = df["text_raw"] if raw else df["text_raw"].map(clean_text)
    df["words"] = df["text"].str.split().str.len()
    if not raw:
        df = df[(df["words"] >= min_words) & (df["words"] <= max_words)]
        df = df.drop_duplicates(subset=["model", "text"])
        df = df[~df.duplicated(subset=["text"], keep=False)]
    df = df.rename(columns={"model": "label"}).reset_index(drop=True)
    df["ext_id"] = "ext_" + df["cell_id"].astype(str)
    return df


def qtab(words: pd.Series) -> dict:
    q = words.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(words)), "q05": int(q[0.05]), "q25": int(q[0.25]), "median": int(q[0.5]),
            "q75": int(q[0.75]), "q95": int(q[0.95]), "mean": round(float(words.mean()), 1)}


def profile():
    raw = load(raw=True)
    ext = load()
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    tr["words"] = tr["text"].str.split().str.len()
    rows = []
    for c in CLASSES:
        r, e = raw[raw.label == c], ext[ext.label == c]
        src = e["source"].value_counts()
        lic = e["license_class"].value_counts(normalize=True)
        rows.append({"label": c, "cells_short_view": len(r), "after_cleaning": len(e),
                     "prompt_groups": int(e["prompt_group_id"].nunique()),
                     "flat_raw_share": round(float(e["flat"].mean()), 3) if len(e) else np.nan,
                     "dialogue_turn_share": round(float(e["is_dialogue_turn"].mean()), 3) if len(e) else np.nan,
                     "open": round(float(lic.get("open", 0)), 3), "open_terms": round(float(lic.get("open_terms", 0)), 3),
                     "unknown": round(float(lic.get("unknown", 0)), 3),
                     "median_words": int(e["words"].median()) if len(e) else np.nan,
                     "official_median_words": int(tr.loc[tr.model == c, "words"].median()),
                     "top_sources": "; ".join(f"{s} {n}" for s, n in src.head(3).items())})
    per_class = pd.DataFrame(rows)
    srcs = []
    for s, g in ext.groupby("source"):
        srcs.append({"source": s, "license_class": LICENSE_CLASS.get(s, "unknown"), "rows": len(g),
                     "labels": int(g["label"].nunique()), "top_domain": g["domain"].value_counts().index[0],
                     "flat_raw_share": round(float(g["flat"].mean()), 3),
                     "median_words": int(g["words"].median())})
    per_source = pd.DataFrame(srcs).sort_values("rows", ascending=False)
    lengths = {"external_clean": qtab(ext["words"]), "external_raw_short_view": qtab(raw["words"]),
               "official_train": qtab(tr["words"])}
    out = {"per_class": per_class.to_dict("records"), "per_source": per_source.to_dict("records"),
           "lengths": lengths, "license_class_rows": ext["license_class"].value_counts().to_dict(),
           "rows_raw": len(raw), "rows_clean": len(ext),
           "dropped": {"length_after_cleaning": int(((raw["text_raw"].map(clean_text).str.split().str.len() < 40) |
                                                     (raw["text_raw"].map(clean_text).str.split().str.len() > 250)).sum())},
           "domains_top": ext["domain"].value_counts().head(12).to_dict()}
    (D / "profile_short.json").write_text(json.dumps(out, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(per_class.to_string(index=False))
    print(per_source.to_string(index=False))
    print(json.dumps({k: out[k] for k in ("lengths", "license_class_rows", "rows_raw", "rows_clean", "dropped",
                                          "domains_top")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile"])
    a = ap.parse_args()
    if a.cmd == "profile":
        profile()


if __name__ == "__main__":
    sys.exit(main())
