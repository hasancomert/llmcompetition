"""Fit test v2: prompt rows for generation (Kaggle Benchmarks task for 12 labels, Featherless for 3).

Design (user-approved, 16 Oct):
- 30 reconstructed train prompts (data/external/reconstructed_prompts.csv), 6 per fold, chosen among the 38 prompts
  whose step-3 Featherless T0 outputs are usable for all three Featherless labels (so T0 is reused there).
- 10 of them (2 per fold) = DEV: the style template is chosen there; 20 (4 per fold) = EVAL: the per-label threshold
  is tested there only, with the chosen template (choosing and judging on the same prompts would be optimistic).
- Templates (none prescribes punctuation or wording habits, so the models' own style stays visible):
    T0  prompt + "Answer in a single paragraph of about N words."             (= fit test v1)
    T1  prompt + a literary-register brief (vivid prose, concrete imagery, varied rhythm, plain text)
    T2  essayist system prompt + prompt + "Keep it to one paragraph of about N words."
- N = word count of the official train text of the same (prompt_id, label) (paired comparison, as in v1).
Rows: row_id, split, template, prompt_id, fold, label, provider, model, system, user_message, target_words.
Outputs (gitignored; derived from competition data: never public):
  data/external/ft2/rows_<split>.csv and data/external/ft2/kaggle_ds/ (private Kaggle dataset: Kaggle labels only).

Usage: python ft2_rows.py dev                 (all three templates on the 10 dev prompts)
       python ft2_rows.py eval --template T1   (the chosen template on the 20 eval prompts)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT

EXT = ROOT / "data" / "external"
OUT = EXT / "ft2"
SEED = 20261017

KAGGLE = {"gpt-5.5": "gpt-5.5-2026-04-23", "gpt-5.6-luna": "gpt-5.6-luna", "gpt-5.6-sol": "gpt-5.6-sol",
          "gpt-5.6-terra": "gpt-5.6-terra", "gpt-6-astra": "gpt-6-astra",
          "gemini-3.1-pro-preview": "gemini-3.1-pro-preview", "gemini-3.6-flash": "gemini-3.6-flash",
          "gemini-3.8-flash": "gemini-3.8-flash", "claude-opus-4-6": "claude-opus-4-6-default",
          "claude-opus-4-8": "claude-opus-4-8-default", "claude-opus-5": "claude-opus-5-default",
          "claude-sonnet-5": "claude-sonnet-5-default"}
FEATHERLESS = {"glm-5.3": "zai-org/GLM-5.3", "glm-5.3-flash": "zai-org/GLM-5.3-Flash", "kimi-k3": "moonshotai/Kimi-K3"}
T0_RUNS = {"glm-5.3": "glm53_low", "glm-5.3-flash": "flash_low", "kimi-k3": "kimik3_low"}   # step-3 outputs (T0)

TEMPLATES = {
    "T0": ("", "{p}\n\nAnswer in a single paragraph of about {n} words."),
    "T1": ("", "{p}\n\nWrite a single paragraph of about {n} words in vivid, literary prose, with concrete imagery and "
               "varied sentence rhythm. Plain text only: no title, no lists, no markdown."),
    "T2": ("You are a skilled essayist who writes short, polished prose pieces for a general audience.",
           "{p}\n\nKeep it to one paragraph of about {n} words."),
}


def t0_usable(label: str) -> set:
    rows = [json.loads(l) for l in open(EXT / "featherless" / f"gen_{T0_RUNS[label]}.jsonl") if l.strip()]
    return {r["prompt_id"] for r in rows if len(r["text"].split()) >= 5}


def choose_prompts():
    """30 prompts, 6 per fold, 2 dev + 4 eval per fold (deterministic)."""
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    folds = pd.read_csv(ROOT / "outputs" / "folds.csv")
    pf = tr[["id", "prompt_id"]].merge(folds[["id", "fold"]], on="id").drop_duplicates("prompt_id").set_index(
        "prompt_id")["fold"]
    cand = sorted(set.intersection(*(t0_usable(l) for l in FEATHERLESS)))
    rng = np.random.default_rng(SEED)
    dev, ev = [], []
    for k in sorted(pf.unique()):
        ids = [p for p in cand if pf[p] == k]
        pick = list(rng.choice(ids, size=6, replace=False))
        dev += pick[:2]
        ev += pick[2:]
    return sorted(dev), sorted(ev), pf


def build(split: str, templates: list):
    dev, ev, pf = choose_prompts()
    pids = dev if split == "dev" else ev
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    nw = tr.assign(n=tr["text"].str.split().str.len()).set_index(["prompt_id", "model"])["n"]
    P = pd.read_csv(EXT / "reconstructed_prompts.csv").set_index("prompt_id")["reconstructed_prompt"]
    rows = []
    for t in templates:
        system, user = TEMPLATES[t]
        for pid in pids:
            for label, model in list(KAGGLE.items()) + list(FEATHERLESS.items()):
                n = int(nw.get((pid, label), 110))
                rows.append({"row_id": f"{split}_{t}_{pid}_{label}", "split": split, "template": t, "prompt_id": pid,
                             "fold": int(pf[pid]), "label": label,
                             "provider": "kaggle" if label in KAGGLE else "featherless", "model": model,
                             "system": system, "user_message": user.format(p=P[pid], n=n), "target_words": n})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"rows_{split}.csv", index=False)
    kd = OUT / "kaggle_ds"
    kd.mkdir(parents=True, exist_ok=True)
    df[df.provider == "kaggle"].to_csv(kd / f"rows_{split}.csv", index=False)
    (kd / "dataset-metadata.json").write_text(json.dumps(
        {"title": "nllm-ft2-prompts", "id": "hasancmert/nllm-ft2-prompts", "licenses": [{"name": "other"}],
         "isPrivate": True}, indent=1))
    print(f"{split}: {len(pids)} prompts {pids}; {len(df)} rows ({(df.provider == 'kaggle').sum()} kaggle, "
          f"{(df.provider == 'featherless').sum()} featherless); templates {templates}")
    print("rows per fold:", df.drop_duplicates("prompt_id")["fold"].value_counts().sort_index().to_dict())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "eval"])
    ap.add_argument("--template", default="", help="eval: the template chosen on dev")
    a = ap.parse_args()
    if a.split == "dev":
        build("dev", list(TEMPLATES))
    else:
        if a.template not in TEMPLATES:
            ap.error("eval needs --template T0|T1|T2 (chosen on dev)")
        build("eval", [a.template])


if __name__ == "__main__":
    main()
