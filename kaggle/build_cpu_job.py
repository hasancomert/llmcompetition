"""Run one src/ member script on a private Kaggle CPU kernel (30 GB RAM; the local machine has 15 GB, too little for
the n-gram SVC with the 120k pool) and fetch the saved member (outputs/preds) and the EXPERIMENTS.md row it logged.

The kernel gets the code (src/*.py, reuse/{ev,airline}/*.py as base64), the competition files, the folds
(hasancmert/nllm-folds) and the public pools the command needs (--pools), linked at their local paths. No GPU, no
internet. Members are saved in the kernel exactly as locally (same code, same folds); --fetch copies them into
outputs/preds without overwriting an existing tag and prints the logged row (append it to EXPERIMENTS.md by hand).

Usage (repo root):
  python kaggle/build_cpu_job.py --slug nllm-cpu-ng4pool5 --pools pool5 \
      --cmd "python train_ngram.py --feats char+word13+shapec+fw ... --save" [--push]
  python kaggle/build_cpu_job.py --slug nllm-cpu-ng4pool5 --fetch
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reuse" / "airline"))
import kaggle_runner  # noqa: E402

POOLS = {"pool": ("prvsiyan/ntl-public-in-style-pool-v1", "pool_v1.parquet", "ntl-public-in-style-pool",
                  "data/external/pools/prvsiyan/pool_v1.parquet"),
         "pool5": ("dhruvtibarewal/nlm-matched-clean", "nlm_pool_v5.parquet", "nlm-matched-clean",
                   "data/external/pools/dhruv/nlm_pool_v5.parquet")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--cmd", default="", help="command run in src/ of the kernel's project copy")
    ap.add_argument("--pools", default="", help="comma-separated: " + ", ".join(POOLS))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    build = ROOT / "outputs" / "kaggle_build"
    if a.fetch:
        d = build / a.slug / "output"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        subprocess.run(["kaggle", "kernels", "output", f"hasancmert/{a.slug}", "-p", str(d)], check=True)
        preds = ROOT / "outputs" / "preds"
        for p in sorted(d.glob("**/outputs/preds/*")):
            if p.name == "experiments_rows.md":
                print(p.read_text(encoding="utf-8"))
            elif (preds / p.name).exists():
                print("exists, not copied:", p.name)
            else:
                shutil.copy(p, preds / p.name)
                print("fetched", p.name)
        sys.exit(0)
    if not a.cmd:
        ap.error("--cmd is required to build")
    pools = [POOLS[p] for p in a.pools.split(",") if p]
    code = {f"src/{p.name}": p for p in sorted((ROOT / "src").glob("*.py"))}
    for pkg in ("ev", "airline"):
        code.update({f"reuse/{pkg}/{p.name}": p for p in sorted((ROOT / "reuse" / pkg).glob("*.py"))})
    links = [("train.csv", "name-that-llm", "data/train.csv"), ("test.csv", "name-that-llm", "data/test.csv"),
             ("folds.csv", "nllm-folds", "outputs/folds.csv")] + [(f, hint, dest) for _, f, hint, dest in pools]
    d = kaggle_runner.build(
        build, a.slug, "hasancmert", code_files=code,
        commands=[f"cd src && {a.cmd}", "cp EXPERIMENTS.md outputs/preds/experiments_rows.md || true"],
        competition="name-that-llm", datasets=["hasancmert/nllm-folds"] + [ds for ds, _, _, _ in pools],
        links=links, out_dirs=("outputs/preds",), gpu=False, internet=False)
    print("built", d)
    if a.push:
        kaggle_runner.push(d)
