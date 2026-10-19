"""Build (and optionally push) the private GPU kernel that extracts frozen-LM features (src/gpu_embed.py).

Uses reuse/airline/kaggle_runner.build: code is embedded as base64, the attached Qwen3 base models are mounted
(not downloaded), internet is off (the same models must later load offline in the submission notebook).
Outputs land in /kaggle/working/emb/. Fetch them with --fetch: kaggle_runner.fetch copies only *.npy, so the
meta_*.json files (row ids, layers) that src/train_emb.py checks are copied here as well.

Usage (repo root): python kaggle/build_gpu_embed.py [--push] [--fetch] [--models qwen3_0.6b_base:0.6b-base,...]
  model spec name:hint[:dtype[:kaggle source[:fracs]]], fracs '/'-separated (per-model layers, else --layer-fracs)
  --texts train,test (default) | ext_pub (with --ext-pub: the public passages of extdata.py --ext pub; the 5 CC BY 4.0
  datasets are attached as inputs, the texts are rebuilt in the kernel by ext_pub.load, nothing is uploaded)
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
    ap.add_argument("--layer-fracs", default="", help="comma-separated layer fractions (default: gpu_embed's 4)")
    ap.add_argument("--texts", default="train,test", help="splits to featurize: train, test, ext_pub")
    ap.add_argument("--ext-pub", action="store_true", help="attach the public passage datasets (needed by ext_pub)")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download the kernel output into outputs/emb")
    a = ap.parse_args()
    if a.fetch:
        import shutil
        build = ROOT / "outputs" / "kaggle_build"
        d = kaggle_runner.fetch(build, "hasancmert", a.slug, ROOT / "outputs", subdirs=("emb",))
        for js in d.glob("**/emb/*.json"):
            shutil.copy(js, ROOT / "outputs" / "emb" / js.name)
            print("fetched emb", js.name)
        sys.exit(0)
    # model spec: name:hint[:dtype[:kaggle source]]; the source defaults to the official Qwen3 instance of <hint>
    specs = [m.split(":") for m in a.models.split(",")]
    sources = [sp[3] if len(sp) > 3 else f"qwen-lm/qwen-3/transformers/{sp[1]}/1" for sp in specs]
    run_models = ",".join(":".join([sp[0], sp[1], sp[2] if len(sp) > 2 and sp[2] else "auto"] +
                                   ([sp[4]] if len(sp) > 4 else [])) for sp in specs)
    if "ext_pub" in a.texts and not a.ext_pub:
        ap.error("--texts ext_pub needs --ext-pub (the passage datasets must be attached)")
    code = {"src/gpu_embed.py": ROOT / "src" / "gpu_embed.py", "src/gpu_run_embed.py": ROOT / "src" / "gpu_run_embed.py"}
    datasets = []
    if a.ext_pub:                                   # ext_pub.load needs common.py (and its reuse/ imports), ext_corpus
        from build_submission import PUB_DATASETS
        datasets = list(PUB_DATASETS)
        for f in ("common.py", "ext_pub.py", "ext_corpus.py"):
            code[f"src/{f}"] = ROOT / "src" / f
        for pkg in ("ev", "airline"):
            code.update({f"reuse/{pkg}/{q.name}": q for q in sorted((ROOT / "reuse" / pkg).glob("*.py"))})
    d = kaggle_runner.build(
        ROOT / "outputs" / "kaggle_build", a.slug, "hasancmert",
        code_files=code,
        commands=[f"EMB_MODELS={run_models} EMB_TEXTS={a.texts}" + (f" EMB_LAYER_FRACS={a.layer_fracs}" if a.layer_fracs else "")
                  + " python src/gpu_run_embed.py"],
        competition="name-that-llm", datasets=datasets, models=sources, out_dirs=("emb",),
        gpu=True, internet=False, machine_shape="NvidiaTeslaT4")
    print("built", d, "models", sources)
    if a.push:
        kaggle_runner.push(d)
