"""Build (and optionally push / fetch) the private GPU kernel that LoRA-fine-tunes a Qwen3 base model per fold
(src/gpu_lora.py). Internet off, model mounted from Kaggle Models, folds from the private dataset nllm-folds.
Outputs: /kaggle/working/lora/{oof_f<k>.npy, val_idx_f<k>.npy, test_f<k>.npy, log_f<k>.json, adapter_f<k>/}.

Usage (repo root):
  python kaggle/build_gpu_lora.py --slug nllm-lora-probe --hint 1.7b-base --args "--folds 0,1 --parallel" --push
  python kaggle/build_gpu_lora.py --slug nllm-lora-probe --fetch
"""
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
    ap.add_argument("--hint", default="1.7b-base")
    ap.add_argument("--args", default="--folds 0,1 --parallel")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    build = ROOT / "outputs" / "kaggle_build"
    if a.fetch:
        d = build / a.slug / "output"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        subprocess.run(["kaggle", "kernels", "output", f"hasancmert/{a.slug}", "-p", str(d)], check=True)
        dst = ROOT / "outputs" / "lora" / a.slug
        dst.mkdir(parents=True, exist_ok=True)
        for p in list(d.glob("**/lora/*.npy")) + list(d.glob("**/lora/*.json")):
            shutil.copy(p, dst / p.name)
        print("fetched into", dst, sorted(x.name for x in dst.iterdir()))
        sys.exit(0)
    d = kaggle_runner.build(
        build, a.slug, "hasancmert",
        code_files={"src/gpu_lora.py": ROOT / "src" / "gpu_lora.py",
                    "src/gpu_run_lora.py": ROOT / "src" / "gpu_run_lora.py"},
        commands=[f"LORA_MODEL_HINT={a.hint} python src/gpu_run_lora.py {a.args}"],
        competition="name-that-llm", datasets=["hasancmert/nllm-folds"],
        models=[f"qwen-lm/qwen-3/transformers/{a.hint}/1"], out_dirs=("lora",),
        gpu=True, internet=False, machine_shape="NvidiaTeslaT4")
    print("built", d)
    if a.push:
        kaggle_runner.push(d)
