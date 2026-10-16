"""Fit test v2, Featherless side: glm-5.3 (zai-org/GLM-5.3), glm-5.3-flash (zai-org/GLM-5.3-Flash), kimi-k3
(moonshotai/Kimi-K3) on the rows of data/external/ft2/rows_<split>.csv (ft2_rows.py).

Template T0 rows are not generated again: the step-3 outputs (gen_<run>.jsonl, same instruction, same target words,
same settings) are reused when usable. Settings as in step 3: temperature 1.0, top_p 0.95, reasoning_effort low;
max_tokens = 2.2 x target words + 200 + 1500 reasoning allowance (step 3: GLM-5.3-Flash ran out at low effort,
Kimi-K3 always reasons); an empty answer is retried once with more room. Every call is logged by featherless.chat
(cost log, budget guard); this step is capped at --cap USD (purpose prefix 'ft2:').
Output rows (data/external/ft2/featherless_<split>.jsonl) carry full provenance.

Usage: python ft2_featherless.py dev [--cap 1.0]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

import featherless as F
from ft2_rows import OUT, T0_RUNS
from fx_generate import clean

SETTINGS = {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "low"}


def reusable_t0() -> dict:
    out = {}
    for label, run in T0_RUNS.items():
        for line in open(F.EXT / f"gen_{run}.jsonl"):
            if line.strip():
                r = json.loads(line)
                if len(r["text"].split()) >= 5:
                    out[(r["prompt_id"], label)] = (run, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "eval"])
    ap.add_argument("--cap", type=float, default=1.0, help="USD cap for the fit test v2 Featherless calls")
    ap.add_argument("--allowance", type=int, default=1500)
    ap.add_argument("--labels", default="", help="';'-separated labels to generate (default: all three)")
    a = ap.parse_args()
    rows = pd.read_csv(OUT / f"rows_{a.split}.csv", keep_default_na=False)
    rows = rows[rows.provider == "featherless"]
    if a.labels:
        rows = rows[rows.label.isin(a.labels.split(";"))]
    out_path = OUT / f"featherless_{a.split}.jsonl"
    done = {json.loads(l)["row_id"] for l in out_path.read_text().splitlines() if l.strip()} if out_path.exists() else set()
    t0 = reusable_t0()
    todo, reused = [], []
    for r in rows.itertuples():
        if r.row_id in done:
            continue
        if r.template == "T0" and (r.prompt_id, r.label) in t0:
            run, g = t0[(r.prompt_id, r.label)]
            assert g["target_words"] == r.target_words and g["user_message"] == r.user_message, r.row_id
            reused.append({**g, "row_id": r.row_id, "split": r.split, "template": "T0", "system": "",
                           "reused_from": f"featherless/gen_{run}.jsonl"})
        else:
            todo.append(r)

    def one(r):
        msgs = ([{"role": "system", "content": r.system}] if r.system else []) + \
               [{"role": "user", "content": r.user_message}]
        max_tokens = int(r.target_words * 2.2) + 200 + a.allowance
        for attempt in range(2):
            res = F.chat(r.model, msgs, purpose=f"ft2:{a.split}_{r.template}_{r.label}", max_tokens=max_tokens,
                         temperature=SETTINGS["temperature"], top_p=SETTINGS["top_p"],
                         extra={"reasoning_effort": SETTINGS["reasoning_effort"]}, step_cap=a.cap, step_prefix="ft2:")
            text, info = clean(res["content"])
            if len(text.split()) >= 5:
                break
            max_tokens = int(max_tokens * 1.5) + 1000
        info["attempts"] = attempt + 1
        return {"row_id": r.row_id, "split": r.split, "template": r.template, "prompt_id": r.prompt_id,
                "label": r.label, "model_id": r.model, "settings": {**SETTINGS, "max_tokens": max_tokens},
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prompt_source": "reconstructed_prompts.csv (Qwen/Qwen3-32B) + template " + r.template,
                "system": r.system, "user_message": r.user_message, "target_words": int(r.target_words),
                "words": len(text.split()), "text": text, "raw_text": res["content"],
                "reasoning_chars": len(res["reasoning"]), "finish_reason": res["finish_reason"],
                "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                "cost_usd": res["cost_usd"], **info}

    with open(out_path, "a", encoding="utf-8") as f:
        for g in reused:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
        n = 0
        with ThreadPoolExecutor(max_workers=2) as ex:
            try:
                for row in ex.map(one, todo):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
            except F.BudgetStop as e:
                print(f"BUDGET STOP: {e}", flush=True)
    print(f"{a.split}: reused {len(reused)} T0 rows, generated {n}/{len(todo)}; ft2 spend {F.spent('ft2:'):.4f} USD, "
          f"workstream total {F.spent():.4f} USD")


if __name__ == "__main__":
    main()
