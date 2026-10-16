"""Generate labelled extra texts with the exact model behind a competition label (Featherless), from the
reconstructed train prompts (data/external/reconstructed_prompts.csv).

One generation per (prompt_id, label): the target length is the word count of that label's train text for the same
prompt (paired style comparison; the train lengths look like per-sample word targets, texts are complete single
paragraphs). Sampling: the model's recommended defaults (generation_config.json / model card: temperature 1.0,
top_p 0.95); thinking: `reasoning_effort` as given (GLM-5.3 / GLM-5.3-Flash: "low" gives no reasoning; Kimi-K3
always reasons). Reasoning is never part of the text (returned separately as reasoning_content). Post-processing,
recorded per row: newlines -> spaces, markdown markers removed.

Rows (data/external/featherless/gen_<run>.jsonl) carry full provenance: model_id, label, settings, date, prompt
source, target words, finish reason, tokens, cost. Resumable (prompt_ids already in the file are skipped).

Usage: python fx_generate.py --model-id zai-org/GLM-5.3 --label glm-5.3 --effort low --run glm53_low [--n 150]
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

PROMPTS = F.ROOT / "data" / "external" / "reconstructed_prompts.csv"
OUTDIR = F.EXT
INSTRUCTION = "\n\nAnswer in a single paragraph of about {n} words."


def clean(t: str) -> tuple[str, dict]:
    info = {"newlines_removed": t.count("\n"), "markdown_removed": 0}
    md = re.findall(r"\*\*|__|^#{1,6}\s|^\s*[-*•]\s|^\s*\d+[.)]\s", t, flags=re.M)
    info["markdown_removed"] = len(md)
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?m)^\s*(#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", "", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip(), info


def pick_prompts(n: int, seed: int = 0) -> list:
    """n prompt_ids, stratified over the shared folds (equal share per fold), deterministic."""
    tr = pd.read_csv(F.ROOT / "data" / "train.csv")
    folds = pd.read_csv(F.ROOT / "outputs" / "folds.csv")
    pf = tr[["id", "prompt_id"]].merge(folds[["id", "fold"]], on="id").drop_duplicates("prompt_id").set_index("prompt_id")["fold"]
    if n >= len(pf):
        return sorted(pf.index)
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(pf.unique()):
        ids = sorted(pf[pf == k].index)
        out += list(rng.choice(ids, size=n // pf.nunique(), replace=False))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--label", required=True, help="competition label (exactly this model)")
    ap.add_argument("--effort", default="low", help="reasoning_effort: low | high | max | default (not sent)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--prompts", default="", help="comma-separated prompt_ids (overrides --n)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reasoning-allowance", type=int, default=0, help="extra max_tokens for reasoning")
    ap.add_argument("--step-cap", type=float, default=1.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retry-empty", type=int, default=1, help="retries when the content comes back empty")
    ap.add_argument("--step", default="step3", help="cost-log purpose prefix (step3 = fit test, step4 = scaling)")
    a = ap.parse_args()
    tr = pd.read_csv(F.ROOT / "data" / "train.csv")
    P = pd.read_csv(PROMPTS).set_index("prompt_id")
    out = OUTDIR / f"gen_{a.run}.jsonl"
    done = {json.loads(l)["prompt_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    pids = a.prompts.split(",") if a.prompts else pick_prompts(a.n)
    pids = [p for p in pids if p not in done]
    tgt = tr[tr.model == a.label].set_index("prompt_id")["text"].str.split().str.len()
    extra = {} if a.effort == "default" else {"reasoning_effort": a.effort}

    def one(pid):
        n = int(tgt.get(pid, 110))
        msgs = [{"role": "user", "content": P.loc[pid, "reconstructed_prompt"] + INSTRUCTION.format(n=n)}]
        max_tokens = int(n * 2.2) + 200 + a.reasoning_allowance
        for attempt in range(1 + a.retry_empty):           # empty content (reasoning hit the cap / backend glitch)
            r = F.chat(a.model_id, msgs, purpose=f"{a.step}:gen_{a.run}", max_tokens=max_tokens,
                       temperature=a.temperature, top_p=a.top_p, extra=extra, step_cap=a.step_cap,
                       step_prefix=f"{a.step}:")
            text, info = clean(r["content"])
            if len(text.split()) >= 5:
                break
            max_tokens = int(max_tokens * 1.5) + 1000
        info["attempts"] = attempt + 1
        return {"id": f"fx_{a.run}_{pid}", "prompt_id": pid, "label": a.label, "model_id": a.model_id,
                "settings": {"temperature": a.temperature, "top_p": a.top_p, "max_tokens": max_tokens, **extra},
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prompt_source": f"reconstructed_prompts.csv ({P.loc[pid, 'model_id']}) + length instruction",
                "user_message": msgs[0]["content"], "target_words": n, "words": len(text.split()), "text": text,
                "raw_text": r["content"], "reasoning_chars": len(r["reasoning"]), "finish_reason": r["finish_reason"],
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                "cost_usd": r["cost_usd"], **info}

    n_ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        try:
            for row in ex.map(one, pids):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"{a.run}: {n_ok}/{len(pids)}  {a.step} spend {F.spent(a.step + ':'):.4f} USD", flush=True)
        except F.BudgetStop as e:
            print(f"BUDGET STOP: {e}", flush=True)
    print(f"{a.run}: wrote {n_ok} rows to {out}; {a.step} spend {F.spent(a.step + ':'):.4f} USD, total {F.spent():.4f} USD")


if __name__ == "__main__":
    main()
