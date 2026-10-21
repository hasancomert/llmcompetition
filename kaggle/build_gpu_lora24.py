"""Build (and optionally push / fetch) the private kernel that trains LoRA24 on the shared folds (src/gpu_lora24.py):
5 fold jobs + 'final' over two T4 workers (one job per GPU at a time), internet off, Qwen3 base model mounted from
Kaggle Models, folds from the private dataset nllm-folds. Outputs: /kaggle/working/lora24/<job>/.

GPU budget: the public recipe caps its run at 10 h (3 jobs per worker); our probe2 (4B, 27 layers, 4 epochs, ~1 h per
fold on a T4) scales to ~1.4 h per 8B-24 job, i.e. ~4.5 h for 3 jobs per worker. --budget-h (default 5) stops the
training at a hard wall-clock cap; complete jobs are kept. User decisions: no GPU before the 24 Oct reset; the week of
24 Oct gives NLLM ~7 GPU hours (S6E10 ~23 h).

Two-stage (--two-stage; public "Two-stage LoRA on the public pool", re-implemented in gpu_lora24.py): job 'pre' on
the public passage pool (Kaggle prvsiyan/ntl-public-in-style-pool-v1, CC BY-SA 4.0, attached; topic-matched blocks
dropped inside the kernel), then f0..f4 from its adapter (2 epochs, lr 5e-5 / 5e-4); torchrun over both T4s
(LORA24_DDP=1). The pool file is checked against the sha256 of the local copy.
--fetch downloads only features, readouts and logs (never adapter or head weights: user rule).

Usage (repo root):
  python kaggle/build_gpu_lora24.py --slug nllm-lora24 [--push]                       GPU run (8B, all 6 jobs)
  python kaggle/build_gpu_lora24.py --slug nllm-lora2s --two-stage --hint 4b-base [--push]   two-stage GPU run
  python kaggle/build_gpu_lora24.py --slug nllm-lora2s-smoke --two-stage --smoke [--push]    CPU smoke (DDP, gloo)
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
    ap.add_argument("--budget-h", type=float, default=5.0, help="wall-clock cap of the training (LORA24_BUDGET_H)")
    ap.add_argument("--infer-from", default="", help="CPU inference smoke on the output of this (smoke) kernel")
    ap.add_argument("--two-stage", action="store_true", help="stage 1 on the public pool, then f0..f4 (DDP)")
    ap.add_argument("--pool-min-labels", type=int, default=10)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    build = ROOT / "outputs" / "kaggle_build"
    if a.fetch:
        d = build / a.slug / "output"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        subprocess.run(["kaggle", "kernels", "output", f"hasancmert/{a.slug}", "-p", str(d), "--file-pattern",
                        r"(log\.json|COMPLETE\.json|readouts\.npz|valid_features\.npy|valid_idx\.npy|"
                        r"anchor_features\.npy|\.log|\.txt)$"], check=True)
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
    files = ("gpu_lora.py", "gpu_lora24.py", "gpu_run_lora24.py")
    code_files = {f"src/{f}": ROOT / "src" / f for f in files}
    datasets, links = ["hasancmert/nllm-folds"], []
    if a.two_stage:
        import hashlib
        pool = ROOT / "data" / "external" / "pools" / "prvsiyan" / "pool_v1.parquet"
        sha = hashlib.sha256(pool.read_bytes()).hexdigest()
        code_files.update({f"src/{f}": ROOT / "src" / f for f in ("common.py", "ext_corpus.py", "ext_pool.py",
                                                                   "ext_leak.py")})
        code_files.update({f"reuse/{f}": ROOT / "reuse" / f for f in ("airline/metrics.py", "ev/__init__.py",
                                                                       "ev/cv_preds.py")})
        datasets.append("prvsiyan/ntl-public-in-style-pool-v1")
        links = [("pool_v1.parquet", "ntl-public-in-style-pool", "data/external/pools/prvsiyan/pool_v1.parquet")]
        args = (f"--jobs pre,f0,f1,f2,f3,f4 --init-pre --epochs 2 --lr 5e-5 --head-lr 5e-4 "
                f"--pool-min-labels {a.pool_min_labels} --pool-sha {sha}")
        env = "LORA24_DDP=1 "
        if a.smoke:
            args += " --jobs pre,f0 --limit 6 --pool-limit 12 --n-layers 12 --max-len 128 --micro 2"
    if a.smoke:
        hint = "0.6b-base"
        if not a.two_stage:
            args = "--jobs f0 --limit 6 --epochs 1 --n-layers 12 --max-len 128"
        env += "LORA24_DEVICE=cpu "
    d = kaggle_runner.build(
        build, a.slug, "hasancmert", code_files=code_files,
        commands=[f"{env}LORA24_MODEL_HINT={hint} LORA24_BUDGET_H={0 if a.smoke else a.budget_h:g} "
                  f"python src/gpu_run_lora24.py {args}"],
        competition="name-that-llm", datasets=datasets, links=links,
        models=[f"qwen-lm/qwen-3/transformers/{hint}/1"], out_dirs=(),
        gpu=not a.smoke, internet=False, machine_shape=None if a.smoke else "NvidiaTeslaT4")
    print("built", d)
    if a.push:
        kaggle_runner.push(d)
