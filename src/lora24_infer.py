"""Test-time features of trained LoRA24 jobs (gpu_lora24.py) for the submission replay (GPU).

The truncated trunk is loaded once; every job's adapter is attached in turn (peft), its LayerNorms loaded from
heads.pt, and the pooled vectors of the test texts are written to <out>/test_features_<job>.npy (float16). Before
featurizing, the train rows 0..2 are re-encoded and compared with the job's saved anchor_features.npy (mean relative
difference must stay under 2 %: same tokenizer, truncation, pooling and LayerNorms as in training).
Probabilities, OOF and twin evidence are computed on CPU by lora24_member.py.

Usage: python lora24_infer.py --model-dir D --jobs-dir lora24 --jobs f0,f1,f2,f3,f4 --texts test.csv
       --train train.csv --out outputs/lora24 [--n-layers 24 --max-len 447]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_lora24 import POOL_LAYERS, Lora24, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--jobs", default="f0,f1,f2,f3,f4")
    ap.add_argument("--texts", required=True, help="csv with id,text (the test file)")
    ap.add_argument("--train", required=True, help="train.csv (anchor parity check on rows 0..2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-layers", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=447)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if a.device == "cuda" else "cpu")
    net = Lora24(a.model_dir, a.n_layers, [l for l in POOL_LAYERS if l <= a.n_layers], device,
                 torch.float16 if a.device == "cuda" else torch.float32, train=False)
    net.model.eval()
    trunk = net.model
    te = pd.read_csv(a.texts, dtype={"id": str})
    ids = tokenize(net.tok, te["text"].fillna("").astype(str).tolist(), a.max_len)
    anchors = tokenize(net.tok, pd.read_csv(a.train)["text"].iloc[:3].tolist(), a.max_len)
    print(f"trunk loaded in {time.time() - t0:.0f}s; {len(ids)} texts", flush=True)
    for job in a.jobs.split(","):
        d = Path(a.jobs_dir) / job
        assert (d / "COMPLETE.json").exists(), f"{d}: job not complete"
        net.model = PeftModel.from_pretrained(trunk, d / "adapter").eval()
        heads = torch.load(d / "heads.pt", map_location="cpu")
        net.norms.load_state_dict(heads["norms"])
        net.norms.eval()
        ref = np.load(d / "anchor_features.npy")
        got = net.features(anchors, bs=3)
        rel = float(np.abs(got - ref).mean() / (np.abs(ref).mean() + 1e-9))
        print(f"[{job}] anchor parity: mean rel diff {rel:.5f}", flush=True)
        assert rel < 0.02, f"{job}: features do not match the training run (rel {rel:.4f})"
        Z = net.features(ids, bs=a.bs)
        np.save(out / f"test_features_{job}.npy", Z.astype(np.float16))
        (out / f"test_meta_{job}.json").write_text(json.dumps({"ids": te["id"].tolist(), "job": job,
                                                               "anchor_rel_diff": rel}))
        trunk = net.model.unload()                      # drop this adapter, keep the trunk
        print(f"[{job}] test features {Z.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
