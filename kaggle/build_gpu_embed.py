"""Build (and optionally push) the private GPU kernel that extracts frozen-LM features (src/gpu_embed.py).

Uses reuse/airline/kaggle_runner.build: code is embedded as base64, the attached Qwen3 base models are mounted
(not downloaded), internet is off (the same models must later load offline in the submission notebook).
Outputs land in /kaggle/working/emb/ and are fetched with kaggle_runner.fetch(..., subdirs=("emb",)).

Usage (repo root): python kaggle/build_gpu_embed.py [--push] [--models qwen3_0.6b_base:0.6b-base,...]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reuse" / "airline"))
import kaggle_runner  # noqa: E402

DEFAULT = "qwen3_0.6b_base:0.6b-base,qwen3_1.7b_base:1.7b-base,qwen3_4b_base:4b-base,qwen3_8b_base:8b-base"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="nllm-gpu-embed")
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    sources = [f"qwen-lm/qwen-3/transformers/{m.split(':')[1]}/1" for m in a.models.split(",")]
    d = kaggle_runner.build(
        ROOT / "outputs" / "kaggle_build", a.slug, "hasancmert",
        code_files={"src/gpu_embed.py": ROOT / "src" / "gpu_embed.py",
                    "src/gpu_run_embed.py": ROOT / "src" / "gpu_run_embed.py"},
        commands=[f"EMB_MODELS={a.models} python src/gpu_run_embed.py"],
        competition="name-that-llm", models=sources, out_dirs=("emb",),
        gpu=True, internet=False, machine_shape="NvidiaTeslaT4")
    print("built", d, "models", sources)
    if a.push:
        kaggle_runner.push(d)
