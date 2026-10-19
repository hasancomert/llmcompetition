"""Kaggle GPU driver for gpu_lora24.py: finds the attached base model, the competition train.csv and folds.csv, then
runs one worker process per visible GPU (CUDA_VISIBLE_DEVICES=w; worker w takes jobs[w::n]) and streams their logs.
Extra arguments are forwarded to every worker (e.g. --jobs f0,f1,f2,f3,f4,final --epochs 3).
Smoke test on a CPU kernel: LORA24_DEVICE=cpu (one worker, e.g. --limit 6 --epochs 1 --n-layers 12 --jobs f0).
Outputs go straight to /kaggle/working/lora24 (LORA24_OUT): the generic runner only keeps top-level files of its
output dirs, and each job writes a directory (adapter, heads, features)."""
import glob
import os
import subprocess
import sys
import time


def find(name, hint):
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p)
    assert hits, (name, hint)
    return hits[0]


if __name__ == "__main__":
    hint = os.environ.get("LORA24_MODEL_HINT", "8b-base")
    cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if f"/{hint}/" in p]
    assert cfg, f"no model dir for {hint}"
    device = os.environ.get("LORA24_DEVICE", "cuda")
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True,
                         text=True).stdout, flush=True)
    n = int(os.environ.get("LORA24_WORKERS", "2" if device == "cuda" else "1"))
    base = [sys.executable, "src/gpu_lora24.py", "--model-dir", os.path.dirname(cfg[0]),
            "--train", find("train.csv", "name-that-llm"), "--folds-csv", find("folds.csv", "nllm-folds"),
            "--out", os.environ.get("LORA24_OUT", "/kaggle/working/lora24"), "--workers", str(n),
            "--device", device] + sys.argv[1:]
    procs = []
    for w in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w)) if device == "cuda" else dict(os.environ)
        log = open(f"lora24_worker{w}.log", "w")
        procs.append((subprocess.Popen(base + ["--worker", str(w)], env=env, stdout=log, stderr=subprocess.STDOUT), log))
    seen = [0] * n
    while True:
        for w in range(n):
            lines = open(f"lora24_worker{w}.log", errors="replace").read().splitlines()
            for line in lines[seen[w]:]:
                print(f"w{w} {line}", flush=True)
            seen[w] = len(lines)
        codes = [p.poll() for p, _ in procs]
        if any(c not in (None, 0) for c in codes):
            for p, _ in procs:
                if p.poll() is None:
                    p.terminate()
            sys.exit(f"worker failed: exit codes {codes}")
        if all(c == 0 for c in codes):
            for w in range(n):                       # last lines written after the final read
                for line in open(f"lora24_worker{w}.log", errors="replace").read().splitlines()[seen[w]:]:
                    print(f"w{w} {line}", flush=True)
            break
        time.sleep(30)
    print("all workers done", flush=True)
