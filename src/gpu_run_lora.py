"""Kaggle GPU driver for gpu_lora.py: finds the attached model, competition csvs and folds.csv under /kaggle/input
and forwards every other argument (e.g. --folds 0,1 --parallel --epochs 2)."""
import glob
import os
import subprocess
import sys


def find(name, hint):
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p)
    assert hits, (name, hint)
    return hits[0]


if __name__ == "__main__":
    hint = os.environ["LORA_MODEL_HINT"]                 # e.g. "1.7b-base"
    cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if f"/{hint}/" in p]
    assert cfg, f"no model dir for {hint}"
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True,
                         text=True).stdout, flush=True)
    argv = [sys.executable, "src/gpu_lora.py", "--model-dir", os.path.dirname(cfg[0]),
            "--train", find("train.csv", "name-that-llm"), "--test", find("test.csv", "name-that-llm"),
            "--folds-csv", find("folds.csv", "nllm-folds"), "--out", "lora"] + sys.argv[1:]
    sys.exit(subprocess.run(argv).returncode)
