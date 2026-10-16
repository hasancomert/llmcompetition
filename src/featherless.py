"""Featherless (OpenAI-compatible) client for the extra-data workstream (branch featherless-extra-data).

- The API key is read from FEATHERLESS_API_KEY and is never printed, logged or written anywhere.
- At most 2 requests in flight (threading.Semaphore); 429 / 5xx / network errors are retried with backoff
  (Retry-After honoured).
- Every request is appended to data/external/featherless/costs.jsonl: time, model, purpose, input/output tokens,
  cost (token prices from GET /v1/models, cached in models.json), sampling parameters, status, finish reason.
- Budget guard: the workstream budget is 8 USD. Before a request, the spent total + this request's worst case
  (estimated prompt tokens + max_tokens at the model's prices) must stay below ASK_AT (7 USD); otherwise BudgetStop
  is raised so that the user can be asked. An optional per-step cap (step_cap, USD, summed over `purpose` prefix)
  is enforced the same way. Requests in flight count with their worst case until their cost is logged (without this,
  two concurrent requests could both pass the check and overshoot a cap: ft2 eval, 1.0105 USD for a 1 USD cap).
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "featherless"
COSTS = EXT / "costs.jsonl"
MODELS_JSON = EXT / "models.json"
BASE = "https://api.featherless.ai/v1"
BUDGET_TOTAL = 8.0
ASK_AT = 7.0
_SEM = threading.Semaphore(2)
_LOCK = threading.Lock()
_PRICES: dict | None = None
_HELD: list = []      # [purpose, worst-case USD] of requests in flight


class BudgetStop(RuntimeError):
    """Raised instead of sending a request that could take the spend past the stop threshold."""


def _headers():
    return {"Authorization": "Bearer " + os.environ["FEATHERLESS_API_KEY"], "Content-Type": "application/json"}


def refresh_models() -> list:
    """GET /v1/models (no tokens billed) -> cached json with pricing."""
    EXT.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{BASE}/models", headers=_headers(), timeout=120)
    r.raise_for_status()
    MODELS_JSON.write_text(r.text)
    global _PRICES
    _PRICES = None
    return r.json()["data"]


def prices(model: str) -> tuple[float, float]:
    """(USD per prompt token, USD per completion token) from the cached model list."""
    global _PRICES
    if _PRICES is None:
        if not MODELS_JSON.exists():
            refresh_models()
        data = json.loads(MODELS_JSON.read_text())["data"]
        _PRICES = {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}
    if model not in _PRICES:
        raise KeyError(f"no price for {model}")
    return _PRICES[model]


def spent(purpose_prefix: str = "") -> float:
    if not COSTS.exists():
        return 0.0
    tot = 0.0
    for line in COSTS.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("purpose", "").startswith(purpose_prefix):
                tot += float(e.get("cost_usd", 0.0))
    return tot


def _log(entry: dict):
    EXT.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(COSTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def chat(model: str, messages: list, purpose: str, max_tokens: int, temperature: float | None = None,
         top_p: float | None = None, extra: dict | None = None, step_cap: float | None = None,
         step_prefix: str | None = None, timeout: int = 900, max_attempts: int = 8) -> dict:
    """One chat completion. Returns {content, reasoning, finish_reason, prompt_tokens, completion_tokens, cost_usd}.
    `extra` is merged into the request body (e.g. chat_template_kwargs). Raises BudgetStop before sending when the
    worst-case cost would cross ASK_AT (or step_cap for spends whose purpose starts with step_prefix)."""
    pin, pout = prices(model)
    est_prompt = sum(len(str(m.get("content", ""))) for m in messages) / 3.0 + 50
    worst = est_prompt * pin + max_tokens * pout
    with _LOCK:
        total = spent() + sum(w for _, w in _HELD)
        if total + worst > ASK_AT:
            raise BudgetStop(f"spent/held {total:.4f} USD + worst case {worst:.4f} would pass {ASK_AT} USD (budget {BUDGET_TOTAL})")
        if step_cap is not None:
            pre = step_prefix or purpose
            step = spent(pre) + sum(w for q, w in _HELD if q.startswith(pre))
            if step + worst > step_cap:
                raise BudgetStop(f"step '{pre}' cap {step_cap} USD reached ({step:.4f} spent or held)")
        ticket = [purpose, worst]
        _HELD.append(ticket)
    try:
        return _send(model, messages, purpose, max_tokens, temperature, top_p, extra, timeout, max_attempts, pin, pout)
    finally:
        with _LOCK:
            _HELD.remove(ticket)


def _send(model, messages, purpose, max_tokens, temperature, top_p, extra, timeout, max_attempts, pin, pout) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if extra:
        body.update(extra)
    params = {k: v for k, v in body.items() if k not in ("messages", "model")}
    t0 = time.time()
    status, j, err = None, None, None
    with _SEM:
        for attempt in range(max_attempts):
            try:
                r = requests.post(f"{BASE}/chat/completions", headers=_headers(), json=body, timeout=timeout)
                status = r.status_code
                if status == 429 or status >= 500:
                    ra = r.headers.get("Retry-After")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(120, 5 * 2 ** attempt)
                    time.sleep(wait + random.random())
                    continue
                if status != 200:
                    err = r.text[:500]
                    break
                j = r.json()
                break
            except requests.RequestException as e:          # network hiccup: retry
                err = type(e).__name__
                time.sleep(min(120, 5 * 2 ** attempt))
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model, "purpose": purpose,
             "params": params, "status": status, "seconds": round(time.time() - t0, 1)}
    if j is None:
        entry.update(prompt_tokens=0, completion_tokens=0, cost_usd=0.0, error=err)
        _log(entry)
        raise RuntimeError(f"featherless request failed (status {status}): {err}")
    u = j.get("usage") or {}
    pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    cost = pt * pin + ct * pout
    msg = j["choices"][0]["message"]
    fin = j["choices"][0].get("finish_reason")
    entry.update(prompt_tokens=pt, completion_tokens=ct, cost_usd=round(cost, 8), finish_reason=fin)
    _log(entry)
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish_reason": fin, "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost}
