"""Reconstruct the (unknown) user prompt of every train prompt_id from 4 of its answers (Featherless, cheap model).

The 4 answers are drawn (seeded per prompt) from models that will NOT be generated later (glm-5.3, glm-5.3-flash,
kimi-k3 are excluded), so the reconstructed prompt does not carry the target models' own wording. The model is told
that every answer had its own word limit, so the reconstruction leaves out length and format requirements (they are
added per generation). Output: data/external/reconstructed_prompts.csv (derived from competition data: never public,
never in git). Resumable: prompt_ids already in the file are skipped.

Usage: python fx_reconstruct.py [--model Qwen/Qwen3-32B] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import featherless as F

OUT = F.ROOT / "data" / "external" / "reconstructed_prompts.csv"
TARGETS = {"glm-5.3", "glm-5.3-flash", "kimi-k3"}
SYSTEM = "You reverse-engineer the instructions that were given to AI writing assistants."
TEMPLATE = """Below are {k} independent responses, written by different AI models, to the SAME user prompt.
Infer the most likely original user prompt.

Rules:
- Each response was given its own word limit, so the responses differ in length. Do NOT include any length, word-count or paragraph requirement.
- Do NOT include formatting instructions.
- Write the prompt the way the user most likely wrote it: usually one or two sentences (a question, a task or a creative brief), keeping any specific angle, constraint or style that all responses share.
- Output ONLY the prompt text. No quotes, no preamble, no explanation.

{responses}"""


def degenerate(t: str) -> bool:
    """Repetition glitches: a character repeated 4+ times or too much punctuation."""
    return bool(re.search(r"([^\w\s])\1{3,}", t)) or sum(not c.isalnum() and not c.isspace() for c in t) > 0.12 * len(t)


def build_messages(texts):
    blocks = "\n\n".join(f"Response {i + 1} ({len(t.split())} words):\n{t}" for i, t in enumerate(texts))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(k=len(texts), responses=blocks)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--n-answers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step-cap", type=float, default=0.15, help="USD cap for this step")
    ap.add_argument("--redo", default="", help="comma-separated prompt_ids to reconstruct again")
    a = ap.parse_args()
    tr = pd.read_csv(F.ROOT / "data" / "train.csv")
    redo = set(a.redo.split(",")) if a.redo else set()
    if redo and OUT.exists():
        prev = pd.read_csv(OUT)
        prev[~prev.prompt_id.isin(redo)].to_csv(OUT, index=False)
    done = set(pd.read_csv(OUT)["prompt_id"]) if OUT.exists() else set()
    pids = [p for p in sorted(tr["prompt_id"].unique()) if p not in done]
    if a.limit:
        pids = pids[:a.limit]
    settings = {"temperature": 0.2, "top_p": 0.8, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False}}

    def one(pid):
        sub = tr[(tr.prompt_id == pid) & (~tr.model.isin(TARGETS))].sort_values("model")
        rng = np.random.default_rng(int(pid[1:]) + 12345)
        pick = sub.iloc[rng.choice(len(sub), size=min(a.n_answers, len(sub)), replace=False)]
        for attempt, temp in enumerate((0.2, 0.6, 0.9)):        # degenerate outputs (char runs) are retried
            r = F.chat(a.model, build_messages(pick["text"].tolist()), purpose="step2:reconstruct", max_tokens=160,
                       temperature=temp, top_p=0.8, extra={"chat_template_kwargs": {"enable_thinking": False}},
                       step_cap=a.step_cap, step_prefix="step2:")
            text = r["content"].strip().strip('"').strip()
            if text and not degenerate(text):
                break
        settings_used = dict(settings, temperature=temp, attempts=attempt + 1)
        return {"prompt_id": pid, "reconstructed_prompt": text, "source_models": ";".join(pick["model"]),
                "source_ids": ";".join(pick["id"]), "model_id": a.model, "settings": json.dumps(settings_used),
                "date": datetime.now(timezone.utc).date().isoformat(), "finish_reason": r["finish_reason"],
                "cost_usd": r["cost_usd"]}

    rows = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, row in enumerate(ex.map(one, pids)):
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(pids)} done, step spend {F.spent('step2:'):.4f} USD", flush=True)
    df = pd.DataFrame(rows)
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
    df.sort_values("prompt_id").to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} prompts); step-2 spend {F.spent('step2:'):.4f} USD, total {F.spent():.4f} USD")


if __name__ == "__main__":
    main()
