"""Run the self-test of every reuse module (random synthetic data only, no competition files).
Usage: python reuse/run_selftests.py"""
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
MODULES = ["metrics", "oof_io", "hill_climb", "stacker", "residual", "adversarial", "submission", "kaggle_runner"]

failed = []
for name in MODULES:
    t = time.time()
    try:
        importlib.import_module(name)._selftest()
        print(f"  ({time.time() - t:.1f}s)")
    except Exception:
        traceback.print_exc()
        failed.append(name)
print("FAILED: " + ", ".join(failed) if failed else f"all {len(MODULES)} self-tests passed")
sys.exit(1 if failed else 0)
