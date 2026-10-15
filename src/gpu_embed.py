"""GPU feature extraction with frozen causal LMs (PLAN.md steps A and I, cheap version).

For every text and every model: mean-pooled hidden states of a few layers (embedding member, step A) and
statistics of the model's own token log-probabilities of the text (log-likelihood "fingerprint", step I):
mean / std / quantiles of log p(token), mean log rank of the true token, mean entropy, share of tokens that are
the argmax. Runs offline on attached Kaggle Models; nothing here uses labels, so features of all texts can be
computed once and the classifiers are fitted per fold elsewhere (src/train_emb.py).

Usage: python gpu_embed.py --model-dir DIR --name qwen3_0.6b --texts train.csv [--texts test.csv] --out emb/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    n_gpu = torch.cuda.device_count()
    kw = dict(dtype=dtype, local_files_only=True)
    if n_gpu > 1:
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    if n_gpu <= 1:
        model = model.to("cuda" if n_gpu else "cpu")
    model.eval()
    return tok, model, dtype


@torch.no_grad()
def featurize(texts, tok, model, layer_fracs=(0.25, 0.5, 0.75, 1.0), batch=8, max_len=512):
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else \
        model.config.get_text_config().num_hidden_layers
    layers = sorted({max(1, int(round(f * n_layers))) for f in layer_fracs})
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: less padding
    emb, lp = [None] * len(texts), [None] * len(texts)
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = [(out.hidden_states[l].to(dev).float() * mask).sum(1) / mask.sum(1) for l in layers]  # multi-GPU safe
        E = torch.cat(pooled, dim=1).cpu().numpy().astype(np.float16)
        lens = enc["attention_mask"].sum(1).tolist()
        for j, i in enumerate(idx):
            n = int(lens[j])
            lg = out.logits[j, :n - 1].to(dev).float()             # per text: avoids (batch, len, vocab) float32
            tgt = enc["input_ids"][j, 1:n]
            logp_all = torch.log_softmax(lg, dim=-1)
            v_t = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            r_t = (lg > lg.gather(-1, tgt.unsqueeze(-1))).sum(-1).float()
            e_t = -(logp_all.exp() * logp_all).sum(-1)
            v, r, e = v_t.cpu().numpy(), r_t.cpu().numpy(), e_t.cpu().numpy()
            q = np.quantile(v, [0.05, 0.1, 0.25, 0.5, 0.75]) if len(v) else np.zeros(5)
            lp[i] = np.array([v.mean(), v.std(), *q, np.log1p(r).mean(), (r == 0).mean(), e.mean(), e.std(),
                              (v + e).mean(), len(v)], dtype=np.float32)
            emb[i] = E[j]
            del lg, logp_all
        del out
    names = ["lp_mean", "lp_std", "lp_q05", "lp_q10", "lp_q25", "lp_q50", "lp_q75", "logrank_mean", "top1_rate",
             "ent_mean", "ent_std", "lp_minus_negent", "n_tok"]
    return np.stack(emb), np.stack(lp), names, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--texts", action="append", required=True, help="csv files with id,text (train first)")
    ap.add_argument("--out", default="emb")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, dtype = load(a.model_dir)
    print(f"[{a.name}] loaded {a.model_dir} dtype {dtype} gpus {torch.cuda.device_count()} in {time.time() - t0:.0f}s",
          flush=True)
    for path in a.texts:
        df = pd.read_csv(path)
        split = Path(path).stem
        t1 = time.time()
        E, L, names, layers = featurize(df["text"].tolist(), tok, model, batch=a.batch, max_len=a.max_len)
        np.save(out / f"emb_{a.name}_{split}.npy", E)
        np.save(out / f"lp_{a.name}_{split}.npy", L)
        (out / f"meta_{a.name}_{split}.json").write_text(json.dumps(
            {"ids": df["id"].tolist(), "layers": layers, "lp_names": names, "dim": int(E.shape[1]),
             "model_dir": a.model_dir, "seconds": time.time() - t1}))
        print(f"[{a.name}] {split}: {len(df)} texts -> emb {E.shape}, lp {L.shape} in {time.time() - t1:.0f}s; "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f} GiB",
              flush=True)


if __name__ == "__main__":
    main()
