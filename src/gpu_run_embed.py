"""Kaggle GPU driver for gpu_embed.py: finds the attached models and competition csvs under /kaggle/input and
runs one subprocess per model (frees GPU memory between models). Failures are logged, the next model still runs.

EMB_MODELS: comma-separated name:hint[:dtype[:fracs]] (fracs '/'-separated, e.g. 0.25/0.5; default EMB_LAYER_FRACS or
gpu_embed's 4 layers). EMB_TEXTS: comma-separated splits, train / test (competition csvs) and ext_pub (the public
generated passages used as external training rows: ext_pub.load on the attached datasets, ids = ext_id).
"""
import glob
import os
import subprocess
import sys
import time

def model_spec(m):
    """name:hint[:dtype[:fracs]] -> (name, hint, dtype or 'auto', fracs or '')."""
    p = (m.split(":") + ["", "", "", ""])[:4]
    return p[0], p[1], p[2] or "auto", p[3]


MODELS = [model_spec(m) for m in os.environ.get("EMB_MODELS", "qwen3_0.6b_base:0.6b-base").split(",")]
TEXTS = os.environ.get("EMB_TEXTS", "train,test").split(",")


def find_csv(name):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if "name-that-llm" in p]
    assert hits, name
    return hits[0]


def find_model(hint):
    hits = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                  if f"/{hint}/" in p + "/" or p.split("/")[-3] == hint or hint in p.split("/"))
    return hits[0] if hits else None


def ext_pub_csv(path="/tmp/ext_pub.csv"):
    """The external passages exactly as extdata.py --ext pub trains on them (ext_pub.load defaults), id = ext_id."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ext_pub
    df = ext_pub.load(root=os.environ.get("EXT_PUB_ROOT", "/kaggle/input"))
    df[["ext_id", "text"]].rename(columns={"ext_id": "id"}).to_csv(path, index=False)
    print(f"ext_pub: {len(df)} passages -> {path}", flush=True)
    return path


if __name__ == "__main__":
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True,
                         text=True).stdout, flush=True)
    paths = []
    for t in TEXTS:
        paths += ["--texts", ext_pub_csv() if t == "ext_pub" else find_csv(f"{t}.csv")]
    for name, hint, dtype, fr in MODELS:
        d = find_model(hint)
        if d is None:
            print(f"### {name}: no model dir matching {hint}; config.json files:",
                  glob.glob("/kaggle/input/**/config.json", recursive=True), flush=True)
            continue
        t0 = time.time()
        fr = fr.replace("/", ",") or os.environ.get("EMB_LAYER_FRACS", "")
        fracs = ["--layer-fracs", fr] if fr else []
        r = subprocess.run([sys.executable, "src/gpu_embed.py", "--model-dir", d, "--name", name,
                            *paths, "--out", "emb", "--dtype", dtype, *fracs])
        print(f"### {name}: exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
