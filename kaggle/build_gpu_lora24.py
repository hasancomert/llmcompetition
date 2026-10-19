"""Build (and optionally push / fetch) the private kernel that trains LoRA24 on the shared folds (src/gpu_lora24.py):
5 fold jobs + 'final' over two T4 workers (one job per GPU at a time), internet off, Qwen3 base model mounted from
Kaggle Models, folds from the private dataset nllm-folds. Outputs: /kaggle/working/lora24/<job>/.

GPU budget: ~3 h per job (public recipe's measurement), 3 jobs per worker -> ~9-10 GPU hours for one run.
User decision (19 Oct): no GPU before the 24 Oct reset; GPU runs need the user's go-ahead.

Usage (repo root):
  python kaggle/build_gpu_lora24.py --slug nllm-lora24 [--push]                       GPU run (8B, all 6 jobs)
  python kaggle/build_gpu_lora24.py --slug nllm-lora24-smoke --smoke [--push]         CPU smoke (0.6B, tiny)
  python kaggle/build_gpu_lora24.py --slug nllm-lora24 --fetch
  python kaggle/build_gpu_lora24.py --slug nllm-lora24-infer-smoke --infer-from nllm-lora24-smoke [--push]
      CPU smoke of lora24_infer.py on the smoke run's job f0 (anchor parity, test features)
"""
import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reuse" / "airline"))
import kaggle_runner  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--hint", default="8b-base")
    ap.add_argument("--args", default="--jobs f0,f1,f2,f3,f4,final --epochs 3")
    ap.add_argument("--smoke", action="store_true", help="CPU kernel, 0.6B, 6 prompts, 1 epoch, 12 layers, job f0")
    ap.add_argument("--infer-from", default="", help="CPU inference smoke on the output of this (smoke) kernel")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    build = ROOT / "outputs" / "kaggle_build"
    if a.fetch:
        d = build / a.slug / "output"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        subprocess.run(["kaggle", "kernels", "output", f"hasancmert/{a.slug}", "-p", str(d)], check=True)
        print("fetched into", d)
        sys.exit(0)
    hint, args, env = a.hint, a.args, ""
    if a.infer_from:
        d = kaggle_runner.build(
            build, a.slug, "hasancmert",
            code_files={f"src/{f}": ROOT / "src" / f for f in ("gpu_lora.py", "gpu_lora24.py", "lora24_infer.py")},
            commands=["find /kaggle/input -maxdepth 4 | head -40; "
                      "python src/lora24_infer.py --model-dir $(dirname $(find /kaggle/input -path '*0.6b-base*' "
                      "-name config.json | head -1)) --jobs-dir $(dirname $(find /kaggle/input -type d -path '*lora24/f0' "
                      "| head -1)) --jobs f0 --texts $(find /kaggle/input -name test.csv | grep name-that-llm | head -1) "
                      "--train $(find /kaggle/input -name train.csv | grep name-that-llm | head -1) "
                      "--out /kaggle/working/lora24_feats --n-layers 12 --max-len 128 --device cpu"],
            competition="name-that-llm", datasets=[], models=["qwen-lm/qwen-3/transformers/0.6b-base/1"],
            out_dirs=(), gpu=False, internet=False, machine_shape=None)
        meta = json.loads((d / "kernel-metadata.json").read_text())
        meta["kernel_sources"] = [f"hasancmert/{a.infer_from}"]
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print("built", d)
        if a.push:
            kaggle_runner.push(d)
        sys.exit(0)
    if a.smoke:
        hint, args = "0.6b-base", "--jobs f0 --limit 6 --epochs 1 --n-layers 12 --max-len 128"
        env = "LORA24_DEVICE=cpu "
    d = kaggle_runner.build(
        build, a.slug, "hasancmert",
        code_files={f"src/{f}": ROOT / "src" / f for f in ("gpu_lora.py", "gpu_lora24.py", "gpu_run_lora24.py")},
        commands=[f"{env}LORA24_MODEL_HINT={hint} python src/gpu_run_lora24.py {args}"],
        competition="name-that-llm", datasets=["hasancmert/nllm-folds"],
        models=[f"qwen-lm/qwen-3/transformers/{hint}/1"], out_dirs=(),
        gpu=not a.smoke, internet=False, machine_shape=None if a.smoke else "NvidiaTeslaT4")
    print("built", d)
    if a.push:
        kaggle_runner.push(d)
