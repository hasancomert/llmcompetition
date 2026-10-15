"""LoRA fine-tuning of a causal LM as an 18-class sequence classifier on the shared folds (PLAN.md step C).

Per fold k: train on folds != k for a FIXED number of epochs (no early stopping on the validation fold, U4),
predict fold k (OOF) and the attached test file, save the LoRA adapter + score head for the submission notebook.
Several folds can run in parallel, one per GPU (--parallel). Outputs in lora/: oof_f<k>.npy (rows of fold k in
train order), test_f<k>.npy, adapter_f<k>/, log_f<k>.json.

Usage: python gpu_lora.py --model-dir DIR --folds-csv folds.csv --train train.csv --test test.csv --folds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
])


def batches(texts, tok, max_len, bs, order):
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        yield idx, enc


def run_fold(a, k):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(a.seed + k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(a.train)
    te = pd.read_csv(a.test)
    fdf = pd.read_csv(a.folds_csv)
    assert (fdf["id"].values == tr["id"].values).all(), "folds.csv not in train order"
    folds = fdf["fold"].to_numpy()
    y = np.array([CLASSES.index(m) for m in tr["model"]])
    trn, val = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    texts, te_texts = tr["text"].tolist(), te["text"].tolist()

    dev = "cuda"
    cap = torch.cuda.get_device_capability(0)
    amp_dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # frozen base in half precision (fp32 4B = 15 GiB = a whole T4); trainable LoRA + score head cast to fp32 below
    base_dtype = amp_dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_dir, num_labels=len(CLASSES), dtype=base_dtype, local_files_only=True)
    model.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="SEQ_CLS",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for prm in model.parameters():          # fp32 master weights for everything trained (GradScaler needs fp32 grads)
        if prm.requires_grad:
            prm.data = prm.data.float()
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    n_batches = math.ceil(len(trn) / a.bs)
    steps = a.epochs * math.ceil(n_batches / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    yt = torch.tensor(y)
    t0 = time.time()
    log = {"fold": k, "model_dir": a.model_dir, "amp": str(amp_dtype), "base_dtype": str(base_dtype), "losses": []}
    rng = np.random.default_rng(a.seed + k)
    step = 0
    model.train()
    for ep in range(a.epochs):
        order = trn[rng.permutation(len(trn))]
        run_loss, n_b = 0.0, 0
        for bi, (idx, enc) in enumerate(batches(texts, tok, a.max_len, a.bs, order)):
            enc = {kk: v.to(dev) for kk, v in enc.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), yt[idx].to(dev), label_smoothing=a.smooth)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep} batch {bi} (amp {amp_dtype})")
            group = a.accum if bi < (n_batches // a.accum) * a.accum else n_batches - (n_batches // a.accum) * a.accum
            scaler.scale(loss / group).backward()
            run_loss, n_b = run_loss + loss.item(), n_b + 1
            if (bi + 1) % a.accum == 0 or bi + 1 == n_batches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
            if (bi + 1) % 50 == 0:
                print(f"fold {k} ep {ep} batch {bi + 1}: loss {run_loss / n_b:.4f} ({time.time() - t0:.0f}s)", flush=True)
        log["losses"].append(run_loss / max(n_b, 1))
    model.eval()

    def predict(tx):
        P = np.zeros((len(tx), len(CLASSES)), dtype=np.float32)
        order = np.argsort([len(t) for t in tx])
        with torch.no_grad():
            for idx, enc in batches(tx, tok, a.max_len, a.bs * 2, order):
                enc = {kk: v.to(dev) for kk, v in enc.items()}
                with torch.autocast("cuda", dtype=amp_dtype):
                    lg = model(**enc).logits.float()
                P[idx] = torch.softmax(lg, -1).cpu().numpy()
        return P
    p_val = predict([texts[i] for i in val])
    p_te = predict(te_texts)
    np.save(out / f"oof_f{k}.npy", p_val)
    np.save(out / f"val_idx_f{k}.npy", val)
    np.save(out / f"test_f{k}.npy", p_te)
    model.save_pretrained(out / f"adapter_f{k}")
    (out / f"adapter_f{k}" / "nllm_meta.json").write_text(json.dumps(
        {"classes": CLASSES, "num_labels": len(CLASSES), "pad_token_id": tok.pad_token_id, "max_len": a.max_len,
         "amp_dtype": str(amp_dtype), "base_dtype": str(base_dtype), "padding_side": "right",
         "model_dir": a.model_dir, "fold": k}, indent=1))
    acc = float((p_val.argmax(1) == y[val]).mean())
    rec = [float((p_val.argmax(1)[y[val] == c] == c).mean()) for c in range(len(CLASSES))]
    log.update(seconds=time.time() - t0, val_acc=acc, val_ba=float(np.mean(rec)), n_train=len(trn), n_val=len(val),
               peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30, args=vars(a))
    (out / f"log_f{k}.json").write_text(json.dumps(log, indent=1))
    print(f"fold {k}: val BA {log['val_ba']:.4f} acc {acc:.4f}, {log['seconds']:.0f}s, "
          f"peak {log['peak_mem_gib']:.1f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--folds-csv", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--out", default="lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--parallel", action="store_true", help="spread folds over the visible GPUs (subprocesses)")
    a = ap.parse_args()
    folds = [int(f) for f in a.folds.split(",")]
    if a.parallel and os.environ.get("NLLM_LORA_CHILD") != "1":
        import torch
        n_gpu = max(1, torch.cuda.device_count())
        queues = [folds[i::n_gpu] for i in range(n_gpu)]
        procs = []
        for g, q in enumerate(queues):
            if not q:
                continue
            argv = [sys.executable, __file__]
            for key, val in vars(a).items():           # rebuilt from the parsed namespace, not sys.argv
                if key == "parallel":
                    continue
                flag = "--" + key.replace("_", "-")
                if isinstance(val, bool):
                    argv += [flag] if val else []
                else:
                    argv += [flag, ",".join(map(str, q)) if key == "folds" else str(val)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), NLLM_LORA_CHILD="1")
            procs.append((q, subprocess.Popen(argv, env=env)))
        codes = [(q, p.wait()) for q, p in procs]
        print("fold workers exit codes", codes, flush=True)
        sys.exit(0 if all(c == 0 for _, c in codes) else 1)
    for k in folds:
        run_fold(a, k)


if __name__ == "__main__":
    main()
