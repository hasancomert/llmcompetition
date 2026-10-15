"""Kaggle GPU driver for gpu_embed.py: finds the attached models and competition csvs under /kaggle/input and
runs one subprocess per model (frees GPU memory between models). Failures are logged, the next model still runs."""
import glob
import os
import subprocess
import sys
import time

MODELS = [(m.split(":") + ["auto"])[:3] for m in os.environ.get("EMB_MODELS", "qwen3_0.6b_base:0.6b-base").split(",")]


def find_csv(name):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if "name-that-llm" in p]
    assert hits, name
    return hits[0]


def find_model(hint):
    hits = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                  if f"/{hint}/" in p + "/" or p.split("/")[-3] == hint or hint in p.split("/"))
    return hits[0] if hits else None


if __name__ == "__main__":
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True,
                         text=True).stdout, flush=True)
    train, test = find_csv("train.csv"), find_csv("test.csv")
    for name, hint, dtype in MODELS:
        d = find_model(hint)
        if d is None:
            print(f"### {name}: no model dir matching {hint}; config.json files:",
                  glob.glob("/kaggle/input/**/config.json", recursive=True), flush=True)
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, "src/gpu_embed.py", "--model-dir", d, "--name", name,
                            "--texts", train, "--texts", test, "--out", "emb", "--dtype", dtype])
        print(f"### {name}: exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
