"""Build the offline Kaggle submission notebook (private, internet off, CPU).

The notebook embeds src/*.py and reuse/{ev,airline}/*.py as base64 (code only: competition data and folds are
never embedded, they come from the attached competition and the private dataset hasancmert/nllm-folds), finds its
inputs under /kaggle/input, and runs src/submission_pipeline.py, which writes /kaggle/working/submission.parquet.

Usage (from the repo root):  python kaggle/build_submission.py [--push]
Output: outputs/kaggle_build/nllm-submit/{submission.ipynb, kernel-metadata.json}
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "nllm-submit"
USER = "hasancmert"

CELL = r'''
import base64, glob, os, runpy, sys
FILES = __FILES__
print("code bundle", __CODE_SHA__, "git", __GIT__)
proj = "/tmp/proj"
for rel, b64 in FILES.items():
    os.makedirs(os.path.dirname(f"{proj}/{rel}"), exist_ok=True)
    open(f"{proj}/{rel}", "wb").write(base64.b64decode(b64))
def find(name, hint):
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p)
    assert hits, f"{name} ({hint}) not found under /kaggle/input"
    return hits[0]
comp_dir = os.path.dirname(find("test.csv", "name-that-llm"))
tok_dir = os.path.dirname(find("gemma__tokenizer.json", "nllm-tokenizers"))
folds_csv = find("folds.csv", "nllm-folds")
print("inputs:", comp_dir, tok_dir, folds_csv)
sys.argv = [__PIPELINE__, "--data-dir", comp_dir, "--folds", folds_csv, "--tokenizers-flat", tok_dir,
            "--out", "/kaggle/working/submission.parquet"] + __EXTRA_ARGS__
runpy.run_path(f"{proj}/src/{__PIPELINE__}", run_name="__main__")
'''


def code_files() -> dict[str, Path]:
    files = {f"src/{p.name}": p for p in sorted((ROOT / "src").glob("*.py"))}
    for pkg in ("ev", "airline"):
        files.update({f"reuse/{pkg}/{p.name}": p for p in sorted((ROOT / "reuse" / pkg).glob("*.py"))})
    return files


GPU_MODELS = [f"qwen-lm/qwen-3/transformers/{h}/1" for h in ("0.6b-base", "1.7b-base", "4b-base", "8b-base")]


EXT_DATASET = "olegzholobov/name-that-llm-response-corpus"   # attached as an input when external rows are used


def build(gpu: bool = False, slug: str = "", pipeline: str = "", extra_args: list | None = None,
          ext_dataset: bool = False) -> Path:
    files = code_files()
    payload = {rel: base64.b64encode(p.read_bytes()).decode() for rel, p in files.items()}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or "unknown"
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src", "reuse"], capture_output=True,
                           text=True).stdout.strip()
    git += "+dirty" if dirty else ""
    pipeline = pipeline or ("submission_pipeline_gpu.py" if gpu else "submission_pipeline.py")
    # GPU: train LM features come from the private dataset nllm-lmfeats when it matches (saves ~25 GPU-min per run)
    extra = (["--work", "/tmp/nllm_work", "--train-lm", "auto"] if gpu else []) + list(extra_args or [])
    src = (CELL.replace("__FILES__", repr(payload)).replace("__CODE_SHA__", repr(sha)).replace("__GIT__", repr(git))
           .replace("__PIPELINE__", repr(pipeline)).replace("__EXTRA_ARGS__", repr(extra)))
    nb = {"cells": [{"cell_type": "code", "id": "nllm-submit-main", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": src.strip().splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    slug = slug or SLUG + ("-gpu" if gpu else "")
    d = ROOT / "outputs" / "kaggle_build" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "submission.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "submission.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": False,
            "competition_sources": ["name-that-llm"],
            "dataset_sources": [f"{USER}/nllm-tokenizers", f"{USER}/nllm-folds"] + ([f"{USER}/nllm-lmfeats"] if gpu else [])
                               + ([EXT_DATASET] if ext_dataset else []),
            "model_sources": GPU_MODELS if gpu else [], "kernel_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"built {d} with {len(files)} code files, bundle {sha}, git {git}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="GPU notebook (submission_pipeline_gpu.py, Qwen3 models attached)")
    ap.add_argument("--slug", default="", help="kernel slug (default nllm-submit / nllm-submit-gpu)")
    ap.add_argument("--pipeline", default="", help="src/ script to run (default by --gpu)")
    ap.add_argument("--extra", default="", help="extra pipeline arguments, space-separated (ablations)")
    ap.add_argument("--ext-dataset", action="store_true",
                    help=f"attach {EXT_DATASET} as an input (needed by replay.py --ext; never copied or re-uploaded)")
    a = ap.parse_args()
    if "--ext" in a.extra.split() and not a.ext_dataset:
        ap.error("--extra uses --ext: add --ext-dataset so the corpus is attached as a notebook input")
    d = build(a.gpu, a.slug, a.pipeline, a.extra.split(), a.ext_dataset)
    if a.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)
