"""LoRA24 on the shared folds: a re-implementation (written anew, not copied) of the public two-head 24-layer recipe
(lightningv08, "NTL LoRA24 Fivefold Training", Kaggle notebook, Apache 2.0).

Model: a Qwen3 base LM truncated to its first 24 decoder layers (fp16 on one GPU per job, gradient checkpointing), LoRA
r16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down. Hidden states of layers 2, 4, 6, 8, 12, 16, 20, 24 are
mean-pooled (attention mask, first token excluded), each through its own LayerNorm (fp32), and concatenated
(8 x hidden). Two linear heads on the pooled vector z (dropout 0.1): plain(z) and centered(z - mean of z over the
batch's same-prompt triple); loss = CE(plain) + CE(centered). The centered head is the LoRA's own prompt-centered twin.

Training: every epoch covers every fitting answer once in same-prompt triples (an incomplete last triple is filled
with other answers of that prompt), 6 triples per update (effective batch 18), 3 epochs, AdamW (adapter 1e-4,
heads 1e-3, weight decay 0.01), 10% linear warmup then cosine decay, fp16 autocast with a grad scaler (start 256, no
growth; an update whose gradients overflow is redone with half the scale, at most 3 times), gradient clipping 1.0.
Fixed epochs: held-out labels never select anything.

Two-stage variant (public "Two-stage LoRA on the public pool", Nishant Dahal; re-implemented): job 'pre' trains the
same network on the public passage pool only (ext_pool.load; every prompt block topic-matched to an official prompt
is dropped (ext_leak), so the stage-1 model never saw a validated prompt and is shared by all folds), class-weighted
cross-entropy (the pool is unbalanced), --pre-epochs (1) at --pre-lr / --pre-head-lr; fold jobs then start from its
adapter and heads (--init-pre) and train on the official rows (public recipe: 2 epochs at lr 5e-5 / 5e-4).
--ddp (torchrun, one process per GPU; gloo on CPU): every job runs on all processes, each process takes its share of
an update's triples and the gradients are summed across processes before the step (same update as one process).
--micro m: m triples per forward pass (centering stays within each triple). --max-seconds: training stops at that
wall-clock time of the job; outputs are still written (log 'truncated': true).

Jobs: f0..f4 (fit = rows of the other folds, held out = fold k) and 'final' (all rows). Outputs in <out>/<job>/:
adapter/ (peft), heads.pt (norms + plain + centered state dicts), readouts.npz (head weights as numpy, pooled layers),
valid_features.npy (float16, held-out rows) + valid_idx.npy (their train row numbers), anchor_features.npy (train rows
0..2, for the inference parity check), log.json. --worker w --workers n runs jobs[w::n] (one process per GPU).

Usage: python gpu_lora24.py --model-dir D --train train.csv --folds-csv folds.csv --jobs f0,f1,f2,f3,f4,final
       [--worker 0 --workers 2] [--out lora24]       smoke: --device cpu --limit 72 --epochs 1 --n-layers 12
       two-stage: torchrun --nproc_per_node 2 gpu_lora24.py ... --ddp --jobs pre,f0,f1,f2,f3,f4 --init-pre
       --epochs 2 --lr 5e-5 --head-lr 5e-4 [--pool-min-labels 10]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_lora import CLASSES, disable_torchao_check  # noqa: E402

POOL_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize(tok, texts, max_len):
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    return [x + [tok.eos_token_id] for x in ids]


def epoch_triples(fit, groups, rng):
    """One epoch: every fitting row once, in same-prompt triples (a short last triple is filled with other rows of
    the prompt), triples shuffled, then padded with repeats to a multiple of 6 (one update = 6 triples)."""
    out = []
    for g in np.unique(groups[fit]):
        members = rng.permutation(fit[groups[fit] == g])
        assert len(members) >= 3, f"prompt {g} has fewer than 3 fitting rows"
        for s in range(0, len(members), 3):
            t = members[s:s + 3]
            if len(t) < 3:
                t = np.concatenate([t, rng.choice(np.setdiff1d(members, t), 3 - len(t), replace=False)])
            out.append(t)
    out = [out[i] for i in rng.permutation(len(out))]
    n = len(out)
    while len(out) % 6:
        out.append(out[int(rng.integers(n))].copy())
    return out


class Lora24:
    """Truncated trunk + LoRA + per-layer LayerNorms + plain / centered heads."""

    def __init__(self, model_dir, n_layers, pool_layers, device, dtype, r=16, alpha=32, train=True, init=None):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        disable_torchao_check()
        self.torch, self.device, self.layers = torch, device, list(pool_layers)
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        cfg.num_hidden_layers, cfg.use_cache = n_layers, False
        if getattr(cfg, "layer_types", None) is not None:
            cfg.layer_types = cfg.layer_types[:n_layers]
        kw = dict(config=cfg, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
        if device.type == "cuda":
            kw.update(device_map={"": device.index or 0}, attn_implementation="sdpa")
        base = AutoModel.from_pretrained(model_dir, **kw)
        if device.type != "cuda":                       # cuda: already placed by device_map
            base = base.to(device)
        base.requires_grad_(False)
        H = cfg.hidden_size
        self.hidden = H
        if train and init is not None:                  # stage 2: continue the stage-1 adapter
            self.model = PeftModel.from_pretrained(base, Path(init) / "adapter", is_trainable=True)
        elif train:
            self.model = get_peft_model(base, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                                                         target_modules=TARGETS))
        if train:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        else:
            self.model = base
        nn = torch.nn
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in self.layers]).to(device).float()
        self.plain = nn.Linear(H * len(self.layers), len(CLASSES)).to(device).float()
        self.centered = nn.Linear(H * len(self.layers), len(CLASSES)).to(device).float()
        self.dropout = nn.Dropout(0.1)
        if init is not None:
            h = torch.load(Path(init) / "heads.pt", map_location=device)
            self.norms.load_state_dict(h["norms"])
            self.plain.load_state_dict(h["plain"])
            self.centered.load_state_dict(h["centered"])

    def pooled(self, ids_batch):
        """(n, 8 * H) fp32 pooled vectors of a list of token id lists."""
        torch = self.torch
        b = self.tok.pad({"input_ids": ids_batch}, return_tensors="pt").to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            hs = self.model(**b, output_hidden_states=True, use_cache=False).hidden_states
        m = b["attention_mask"].unsqueeze(-1).float()
        m[:, 0] = 0
        with torch.autocast(self.device.type, enabled=False):
            z = torch.cat([norm((hs[li].float() * m).sum(1) / m.sum(1).clamp(min=1))
                           for li, norm in zip(self.layers, self.norms)], -1)
        assert bool(torch.isfinite(z).all()), "non-finite pooled features"
        return z, int(b["attention_mask"].sum())

    def loss(self, ids_batch, y, weight=None):
        """CE(plain) + CE(centered) of consecutive same-prompt triples (len(ids_batch) = 3 x number of triples);
        weight: optional per-class weights of the cross-entropy."""
        torch = self.torch
        z, n_tok = self.pooled(ids_batch)
        yt = torch.as_tensor(y, device=self.device, dtype=torch.long)
        with torch.autocast(self.device.type, enabled=False):
            zt = z.view(-1, 3, z.shape[1])
            zc = (zt - zt.mean(1, keepdim=True)).view_as(z)
            lp = torch.nn.functional.cross_entropy(self.plain(self.dropout(z)), yt, weight=weight)
            lc = torch.nn.functional.cross_entropy(self.centered(self.dropout(zc)), yt, weight=weight)
        return lp + lc, float(lp.detach()), float(lc.detach()), n_tok

    def features(self, ids, bs=3):
        """Eval-mode pooled vectors (float32 numpy) of every token id list, length-sorted batches."""
        torch = self.torch
        out = np.zeros((len(ids), self.hidden * len(self.layers)), np.float32)
        order = np.argsort([len(x) for x in ids], kind="stable")
        with torch.inference_mode():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                out[idx] = self.pooled([ids[i] for i in idx])[0].cpu().numpy()
        return out


def pool_rows(train_csv, min_labels, limit=0, sha=""):
    """Stage-1 rows: the public passage pool (ext_pool.load) minus every prompt block topic-matched to an official
    prompt (ext_leak.compute, as extdata): stage 1 is shared by all folds, so it must not see any validated prompt.
    Blocks need >= 3 rows (same-prompt triples). limit (smoke tests): keep the first N blocks. sha: expected sha256
    of the pool file (the version the local CV used)."""
    import hashlib
    import ext_leak
    import ext_pool
    if sha:
        got = hashlib.sha256(ext_pool._find(ext_pool.D).read_bytes()).hexdigest()
        assert got == sha, f"pool file differs from the local copy (sha256 {got})"
    tr = pd.read_csv(train_csv)
    df = ext_pool.load(min_labels=min_labels)
    lm, _ = ext_leak.compute(tr, ext_pool.load(min_labels=1))
    flagged = set(lm.loc[lm["flagged"], "ext_id"])
    n0 = len(df)
    df = df[~df["ext_id"].isin(flagged)]
    df = df[df.groupby("prompt_group_id")["ext_id"].transform("size") >= 3]
    if limit:
        df = df[df["prompt_group_id"].isin(df["prompt_group_id"].drop_duplicates().iloc[:limit])]
    df = df.reset_index(drop=True)
    print(f"stage-1 pool: {n0} rows (min_labels {min_labels}) -> {len(df)} rows in {df['prompt_group_id'].nunique()} "
          f"blocks after dropping topic-matched blocks and blocks under 3 rows", flush=True)
    return df


def class_weights(y):
    """Per-class cross-entropy weights n / (K n_c): every class carries the same total weight."""
    cnt = np.bincount(y, minlength=len(CLASSES)).astype(float)
    return np.where(cnt > 0, len(y) / (len(CLASSES) * np.maximum(cnt, 1)), 0.0)


def run_job(a, name, d, rank=0, world=1):
    """Train one job. d: ids / y / groups / folds of the official rows, and pool_ids / pool_y / pool_groups for
    the stage-1 job 'pre'. Under --ddp every process runs this with its rank; process 0 writes the outputs."""
    import torch
    dist = None
    if world > 1:
        import torch.distributed as dist
    if name == "pre":
        ids, y, groups = d["pool_ids"], d["pool_y"], d["pool_groups"]
        fit, valid = np.arange(len(y)), np.zeros(0, int)
        epochs, lr, head_lr, init = a.pre_epochs, a.pre_lr, a.pre_head_lr, None
        weight = class_weights(y)
        max_seconds = a.pre_max_seconds
    else:
        ids, y, groups, folds = d["ids"], d["y"], d["groups"], d["folds"]
        fit = np.flatnonzero(folds != int(name[1:])) if name != "final" else np.arange(len(y))
        valid = np.flatnonzero(folds == int(name[1:])) if name != "final" else np.zeros(0, int)
        epochs, lr, head_lr, weight = a.epochs, a.lr, a.head_lr, None
        max_seconds = a.max_seconds
        init = Path(a.out) / "pre" if a.init_pre else None
        if init is not None:
            assert (init / "COMPLETE.json").exists(), f"{init}: stage-1 job not complete"
    assert not set(groups[fit]) & set(groups[valid]), "prompt overlap between fit and held-out rows"
    out = Path(a.out) / name
    assert not (out / "COMPLETE.json").exists(), f"{out} already complete (no silent re-use)"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    plan = [epoch_triples(fit, groups, rng) for _ in range(epochs)]
    total = sum(len(p) // 6 for p in plan)
    if a.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if world > 1 else "cuda:0")
    t0 = time.time()
    net = Lora24(a.model_dir, a.n_layers, [l for l in POOL_LAYERS if l <= a.n_layers], device,
                 torch.float16 if device.type == "cuda" else torch.float32, init=init)
    adapter = [p for p in net.model.parameters() if p.requires_grad]
    heads = list(net.norms.parameters()) + list(net.plain.parameters()) + list(net.centered.parameters())
    params = adapter + heads
    if world > 1:                                    # identical start on every process
        for p in params:
            dist.broadcast(p.data, 0)
    wt = torch.as_tensor(weight, dtype=torch.float32, device=device) if weight is not None else None
    print(f"[{name}] rank {rank}/{world}: loaded in {time.time() - t0:.0f}s; fit {len(fit)} rows, held out "
          f"{len(valid)}, {total} updates, {epochs} epochs, lr {lr:g} / {head_lr:g}"
          f"{', init ' + str(init) if init else ''}{', class-weighted' if wt is not None else ''}", flush=True)
    opt = torch.optim.AdamW([{"params": adapter, "lr": lr}, {"params": heads, "lr": head_lr}], weight_decay=0.01)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, growth_interval=1_000_000, enabled=use_amp)
    log = {"job": name, "fit_rows": int(len(fit)), "valid_rows": int(len(valid)), "updates": total, "epochs": epochs,
           "n_layers": a.n_layers, "pool_layers": net.layers, "max_len": a.max_len, "lr": lr, "head_lr": head_lr,
           "seed": a.seed, "init": str(init) if init else None, "class_weighted": wt is not None, "world": world,
           "micro": a.micro, "history": [], "overflow_retries": 0, "truncated": False}
    net.model.train()
    net.norms.train()
    net.plain.train()
    net.centered.train()
    step, t1 = 0, time.time()
    stop = False
    for ep, triples in enumerate(plan):
        for s in range(0, len(triples), 6):
            group = triples[s:s + 6]
            mine = group[rank::world]                # this process's share of the update's triples
            for attempt in range(4):                 # redo an overflowing update with half the scale (<= 3 times)
                opt.zero_grad(set_to_none=True)
                stats = []
                for m in range(0, len(mine), a.micro):
                    chunk = mine[m:m + a.micro]
                    rows = np.concatenate(chunk)
                    loss, lp, lc, n_tok = net.loss([ids[i] for i in rows], y[rows], wt)
                    scaler.scale(loss * len(chunk) / len(group)).backward()
                    stats.append((lp, lc, n_tok))
                if world > 1:                        # sum of the processes' gradients = one-process gradient
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                    flat = torch.cat([p.grad.reshape(-1) for p in params])
                    dist.all_reduce(flat)
                    o = 0
                    for p in params:
                        p.grad.copy_(flat[o:o + p.numel()].view_as(p))
                        o += p.numel()
                if use_amp:
                    scaler.unscale_(opt)
                    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                    if not finite:
                        log["overflow_retries"] += 1
                        assert attempt < 3, f"update {step + 1}: overflow after 3 retries"
                        scaler.update(new_scale=scaler.get_scale() * 0.5)
                        continue
                gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                scaler.step(opt)
                scaler.update()
                break
            sched.step()
            step += 1
            if step % 10 == 0 or step == total:
                h = {"step": step, "epoch": ep + 1, "plain": float(np.mean([x[0] for x in stats])),
                     "center": float(np.mean([x[1] for x in stats])), "grad_norm": gn,
                     "seconds": round(time.time() - t1, 1)}
                log["history"].append(h)
                if rank == 0:
                    print(f"[{name}] {json.dumps(h)}", flush=True)
            if max_seconds:                          # wall-clock cap: process 0 decides for everyone
                flag = torch.tensor([float(time.time() - t0 > max_seconds)], device=device)
                if world > 1:
                    dist.broadcast(flag, 0)
                stop = bool(flag.item())
            if stop:
                break
        if stop:
            log["truncated"] = True
            if rank == 0:
                print(f"[{name}] TIME CAP {max_seconds}s reached after {step}/{total} updates", flush=True)
            break
    log["updates_done"] = step
    log["train_seconds"] = round(time.time() - t1, 1)
    net.model.eval()
    net.norms.eval()
    net.plain.eval()
    net.centered.eval()
    if hasattr(net.model, "gradient_checkpointing_disable"):
        net.model.gradient_checkpointing_disable()
    if rank == 0:
        if len(valid):
            Z = net.features([ids[i] for i in valid])
            np.save(out / "valid_features.npy", Z.astype(np.float16))
            np.save(out / "valid_idx.npy", valid)
        np.save(out / "anchor_features.npy", net.features([d["ids"][i] for i in range(3)]))   # official rows 0..2
        np.savez(out / "readouts.npz", classes=np.array(CLASSES), pool_layers=np.array(net.layers),
                 plain_coef=net.plain.weight.detach().cpu().numpy(),
                 plain_intercept=net.plain.bias.detach().cpu().numpy(),
                 center_coef=net.centered.weight.detach().cpu().numpy(),
                 center_intercept=net.centered.bias.detach().cpu().numpy())
        net.model.save_pretrained(out / "adapter")
        torch.save({"norms": net.norms.state_dict(), "plain": net.plain.state_dict(),
                    "centered": net.centered.state_dict()}, out / "heads.pt")
        if device.type == "cuda":
            log["peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
        log["total_seconds"] = round(time.time() - t0, 1)
        (out / "log.json").write_text(json.dumps(log, indent=1))
        (out / "COMPLETE.json").write_text(json.dumps({"job": name, "updates": step, "truncated": log["truncated"]}))
        print(f"[{name}] COMPLETE in {log['total_seconds']}s", flush=True)
    if world > 1:
        dist.barrier()                               # the next job may read this job's outputs
    del net, opt, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds-csv", required=True)
    ap.add_argument("--jobs", default="f0,f1,f2,f3,f4,final")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="lora24")
    ap.add_argument("--n-layers", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=447)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="SMOKE TEST ONLY: keep the first N prompts' rows")
    ap.add_argument("--micro", type=int, default=1, help="triples per forward pass")
    ap.add_argument("--max-seconds", type=float, default=0, help="wall-clock cap of each fold job (0: none)")
    ap.add_argument("--pre-max-seconds", type=float, default=0, help="wall-clock cap of the job 'pre' (0: none)")
    ap.add_argument("--ddp", action="store_true", help="torchrun: every job on all processes, gradients summed")
    ap.add_argument("--init-pre", action="store_true", help="fold jobs start from <out>/pre (two-stage)")
    ap.add_argument("--pool-min-labels", type=int, default=10, help="job 'pre': minimum labels per pool block")
    ap.add_argument("--pool-limit", type=int, default=0, help="SMOKE TEST ONLY: keep the first N pool blocks")
    ap.add_argument("--pool-sha", default="", help="expected sha256 of the pool file")
    ap.add_argument("--pre-epochs", type=int, default=1)
    ap.add_argument("--pre-lr", type=float, default=1e-4)
    ap.add_argument("--pre-head-lr", type=float, default=1e-3)
    a = ap.parse_args()
    rank, world = 0, 1
    if a.ddp:
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        dist.init_process_group("nccl" if a.device == "cuda" else "gloo", timeout=timedelta(hours=1))
        rank, world = dist.get_rank(), dist.get_world_size()
        if a.device == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    tr = pd.read_csv(a.train)
    fd = pd.read_csv(a.folds_csv)
    assert (fd["id"].values == tr["id"].values).all(), "folds.csv does not match train.csv row order"
    if a.limit:
        keep = tr["prompt_id"].isin(tr["prompt_id"].drop_duplicates().iloc[:a.limit])
        tr, fd = tr[keep].reset_index(drop=True), fd[keep].reset_index(drop=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    d = {"ids": tokenize(tok, tr["text"].tolist(), a.max_len),
         "y": tr["model"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(),
         "groups": tr["prompt_id"].astype(str).to_numpy(), "folds": fd["fold"].to_numpy()}
    jobs = a.jobs.split(",") if a.ddp else a.jobs.split(",")[a.worker::a.workers]
    if "pre" in jobs:
        pool = pool_rows(a.train, a.pool_min_labels, a.pool_limit, a.pool_sha)
        d.update(pool_ids=tokenize(tok, pool["text"].tolist(), a.max_len),
                 pool_y=pool["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(),
                 pool_groups=pool["prompt_group_id"].astype(str).to_numpy())
    print(f"rank {rank}/{world}, worker {a.worker}/{a.workers}: jobs {jobs}; {len(tr)} official rows", flush=True)
    for name in jobs:
        run_job(a, name, d, rank, world)
    if a.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
