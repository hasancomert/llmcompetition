# %% [markdown]
# Private research task (Name That LLM, fit test v2). For the model under test it writes one paragraph per prompt row
# of the attached private dataset hasancmert/nllm-ft2-prompts; nothing is scored. The rows come from competition
# training data, so this task and its outputs stay private (never published).
#
# Push / run (from the repo root; generation runs on Kaggle, not with a local proxy token):
#   kaggle benchmarks tasks push nllm-gen-ft2 -f kaggle/bench_gen_task.py -d hasancmert/nllm-ft2-prompts --wait
#   kaggle benchmarks tasks run nllm-gen-ft2 -m gpt-5.6-luna --wait
#   kaggle benchmarks tasks download nllm-gen-ft2 -m gpt-5.6-luna -o data/external/ft2/kaggle_out

# %%
import glob
import json
import os
import re
import time

import pandas as pd

import kaggle_benchmarks as kbench

SPLIT = "dev"          # rows_<SPLIT>.csv of the attached dataset (dev: 3 templates x 10 prompts; eval: 1 x 20)
REASONING = "low"      # same as the Featherless runs (reasoning_effort=low); falls back to the model default
TEMPERATURE = 1.0      # provider default for these models; ignored where the model does not take a temperature
MAX_OUT = 8000         # output-token cap: the proxy reserves the worst-case cost (model maximum output) per call and
                       # refuses calls (403) when the reservations exceed the quota; answers use ~200-2000 tokens
CAP_KEYS = ("max_completion_tokens", "max_output_tokens", "max_tokens")   # tried in order (client paths differ)
SKIP = {"dev": {"gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview", "gpt-5.6-luna", "gpt-5.6-terra",
                "claude-sonnet-5", "claude-opus-4-8",          # dev already generated (version 1 of this task)
                "gpt-5.5", "gpt-6-astra"}}                     # dev skipped for cost (5 USD cap): eval only

LABEL_OF = {           # Kaggle Benchmarks model (proxy slug and catalog slug) -> competition label
    "openai/gpt-5.5-2026-04-23": "gpt-5.5", "gpt-5.5-2026-04-23": "gpt-5.5",
    "openai/gpt-5.6-luna": "gpt-5.6-luna", "gpt-5.6-luna": "gpt-5.6-luna",
    "openai/gpt-5.6-sol": "gpt-5.6-sol", "gpt-5.6-sol": "gpt-5.6-sol",
    "openai/gpt-5.6-terra": "gpt-5.6-terra", "gpt-5.6-terra": "gpt-5.6-terra",
    "openai/gpt-6-astra": "gpt-6-astra", "gpt-6-astra": "gpt-6-astra",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-preview", "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "google/gemini-3.6-flash": "gemini-3.6-flash", "gemini-3.6-flash": "gemini-3.6-flash",
    "google/gemini-3.8-flash": "gemini-3.8-flash", "gemini-3.8-flash": "gemini-3.8-flash",
    "anthropic/claude-opus-4-6@default": "claude-opus-4-6", "claude-opus-4-6-default": "claude-opus-4-6",
    "anthropic/claude-opus-4-8@default": "claude-opus-4-8", "claude-opus-4-8-default": "claude-opus-4-8",
    "anthropic/claude-opus-5@default": "claude-opus-5", "claude-opus-5-default": "claude-opus-5",
    "anthropic/claude-sonnet-5@default": "claude-sonnet-5", "claude-sonnet-5-default": "claude-sonnet-5",
}

MODEL = os.environ.get("LLM_DEFAULT", "")
LABEL = LABEL_OF.get(MODEL)
paths = sorted(glob.glob(f"/kaggle/input/**/rows_{SPLIT}.csv", recursive=True))
assert paths, f"rows_{SPLIT}.csv not found: attach hasancmert/nllm-ft2-prompts"
rows = pd.read_csv(paths[0], keep_default_na=False)
# an unknown model, or a label already done / skipped for this split (e.g. the run made when the task is created),
# writes a single row only: no cost leaks
todo = rows[rows["label"] == LABEL] if LABEL and LABEL not in SKIP.get(SPLIT, set()) else rows.head(1)
print(f"model {MODEL!r} -> label {LABEL!r}; {len(todo)} rows of {paths[0]}", flush=True)


def _ask(llm, row_id, system, user_message, reasoning):
    """One fresh chat per attempt; the output cap is passed under the first parameter name the client accepts.
    A quota refusal (403) is not retried with another name."""
    last = None
    for key in CAP_KEYS:
        kw = {"temperature": TEMPERATURE, "extra_api_params": {key: MAX_OUT}}
        if reasoning:
            kw["reasoning"] = reasoning
        try:
            with kbench.chats.new(f"{row_id}-{reasoning or 'default'}-{key}", system_instructions=system or None) as chat:
                text = llm.prompt(user_message, **kw)
                return text, chat.usage, key
        except Exception as e:
            if "403" in str(e) or "PermissionDenied" in type(e).__name__:
                raise
            last = e
    raise last


# %%
@kbench.task(name="nllm-gen-ft2", description="Private research task: one paragraph per prompt row, no scoring.")
def nllm_gen_ft2(llm, row_id: str, system: str, user_message: str) -> dict:
    t0 = time.time()
    try:
        text, usage, cap_key = _ask(llm, row_id, system, user_message, REASONING)
        mode = REASONING
    except Exception as e:  # model without a reasoning setting: its default, in a fresh chat
        if "403" in str(e) or "PermissionDenied" in type(e).__name__:
            raise
        text, usage, cap_key = _ask(llm, row_id, system, user_message, None)
        mode = f"default ({type(e).__name__})"
    text = re.sub(r"<think>.*?</think>\s*", "", str(text), flags=re.S).strip()
    return {"row_id": row_id, "text": text, "reasoning": mode, "max_out": MAX_OUT, "cap_key": cap_key,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "cost_nanodollars": usage.total_cost_nanodollars,
            "seconds": round(time.time() - t0, 1)}


# %%
runs = nllm_gen_ft2.evaluate(llm=[kbench.llm], evaluation_data=todo[["row_id", "system", "user_message"]],
                             n_jobs=2, timeout=900, max_attempts=4, retry_delay=30, on_failure="continue")
out = []
for r in runs.runs:
    rec = {"row_id": r.params.get("row_id")}
    if isinstance(r.result, dict):
        rec.update(r.result)
    rec.update({"model_env": MODEL, "label": LABEL, "split": SPLIT, "temperature": TEMPERATURE,
                "error": r.error_message, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out.append(rec)
name = f"/kaggle/working/ft2_{SPLIT}_{LABEL or 'unknown'}.jsonl"
with open(name, "w", encoding="utf-8") as f:
    for rec in out:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
ok = sum(bool(r.get("text")) for r in out)
cost = sum((r.get("cost_nanodollars") or 0) for r in out) / 1e9
print(f"wrote {name}: {ok}/{len(out)} rows with text, cost {cost:.4f} USD", flush=True)
