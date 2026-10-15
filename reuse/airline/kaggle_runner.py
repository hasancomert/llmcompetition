"""Run local code in a private Kaggle kernel (GPU or CPU), split long jobs into units, collect the units.

Adapted from S6E10:
  - kaggle/gpu_kernel.py          bundle src/*.py as base64 into one run.py, symlink the competition files found
                                  under /kaggle/input, run shell commands from the project root, copy oof/ and
                                  preds/ *.npy and the experiment log to /kaggle/working; push / status / fetch
  - kaggle/tabpfn/build_kernel.py (seed, fold) units, one per GPU, with a deadline: the next unit starts only if
                                  now + 1.05 x (median duration of the finished units, first guess given) fits
                                  before the hard limit; the worker itself gets the limit minus 10 minutes
  - kaggle/tabpfn/aggregate.py    assemble units f<fold>_s<seed>_{val,test}.npy from several kernels, check that
                                  every fold has a validation unit and every unit a test file, average seeds
Changes: competition slug, datasets, user, file links and output folders are parameters; predictions are
(n, K) matrices; paths inside the kernel can be overridden by environment variables so that a generated run.py
can be executed locally (the self-test does that). Pushing needs the kaggle CLI and credentials in the
environment; this module never reads, prints or writes the credentials.

Not carried over: in S6E10 every kernel had internet on (pip install, Hugging Face download). The new task's
submission notebook runs without internet; that setup (wheels and weights as attached inputs) was never done in
these projects.
"""
import base64
import collections
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

RUNNER = r'''
import base64, glob, os, shutil, subprocess, sys, time
FILES = __FILES__
COMMANDS = __COMMANDS__
LINKS = __LINKS__
OUT_DIRS = __OUT_DIRS__
inp = os.environ.get("RUNNER_INPUT", "/kaggle/input")
out = os.environ.get("RUNNER_OUT", "/kaggle/working")
proj = os.environ.get("RUNNER_PROJ", "/tmp/proj")
for rel, b64 in FILES.items():
    os.makedirs(os.path.dirname(f"{proj}/{rel}"), exist_ok=True)
    open(f"{proj}/{rel}", "wb").write(base64.b64decode(b64))
def find(name, hint):
    hits = sorted(p for p in glob.glob(f"{inp}/**/{name}", recursive=True) if hint in p)
    assert hits, (name, hint)
    return hits[0]
for name, hint, dest in LINKS:
    os.makedirs(os.path.dirname(f"{proj}/{dest}"), exist_ok=True)
    if not os.path.exists(f"{proj}/{dest}"):
        os.symlink(find(name, hint), f"{proj}/{dest}")
for c in COMMANDS:
    t0 = time.time()
    print(f"### {c}", flush=True)
    r = subprocess.run(c, shell=True, cwd=proj)
    print(f"### exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
    for d in OUT_DIRS:
        os.makedirs(f"{out}/{d}", exist_ok=True)
        for p in glob.glob(f"{proj}/{d}/*"):
            if os.path.isfile(p):
                shutil.copy(p, f"{out}/{d}/")
    if os.path.exists(f"{proj}/experiments.md"):
        shutil.copy(f"{proj}/experiments.md", f"{out}/experiments_rows.md")
'''


def build(build_dir, slug, user, code_files, commands, competition=None, datasets=(), models=(), links=(),
          out_dirs=("oof", "preds"), gpu=True, internet=True, machine_shape=None):
    """Write <build_dir>/<slug>/run.py and kernel-metadata.json (nothing is pushed).
    code_files: {path inside the project: local path}; links: [(file name, path hint, destination)], e.g.
    ("train.parquet", "name-that-llm", "data/train.parquet"); models: Kaggle model sources ("owner/model/...")."""
    d = Path(build_dir) / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    files = {rel: base64.b64encode(Path(p).read_bytes()).decode() for rel, p in code_files.items()}
    code = (RUNNER.replace("__FILES__", repr(files)).replace("__COMMANDS__", repr(list(commands)))
            .replace("__LINKS__", repr([tuple(x) for x in links])).replace("__OUT_DIRS__", repr(list(out_dirs))))
    (d / "run.py").write_text(code)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": bool(gpu), "enable_internet": bool(internet),
            "competition_sources": [competition] if competition else [], "dataset_sources": list(datasets),
            "model_sources": list(models), "kernel_sources": []}
    if machine_shape:
        meta["machine_shape"] = machine_shape          # S6E10 used "NvidiaTeslaT4" for the 2 x T4 TabPFN kernels
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return d


def bundle(root, pattern="*.py", prefix="src"):
    """{prefix/<name>: path} for the files matching pattern in root (the S6E10 runner bundled src/*.py)."""
    return {f"{prefix}/{p.name}": p for p in sorted(Path(root).glob(pattern))}


def push(kernel_dir):
    subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)


def status(user, slug):
    subprocess.run(["kaggle", "kernels", "status", f"{user}/{slug}"])


def fetch(build_dir, user, slug, root, subdirs=("oof", "preds")):
    """Download the kernel output to <build_dir>/<slug>/output and copy its oof/ preds/ *.npy into root."""
    d = Path(build_dir) / slug / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(["kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(d)], check=True)
    for sub in subdirs:
        (Path(root) / sub).mkdir(parents=True, exist_ok=True)
        for p in set(d.glob(f"{sub}/*.npy")) | set(d.glob(f"**/{sub}/*.npy")):
            shutil.copy(p, Path(root) / sub / p.name)
            print("fetched", sub, p.name)
    return d


def schedule(units, run_unit, slots, limit_s, first_estimate_s, t0=None, margin=1.05, log=print):
    """Run units on parallel slots (e.g. GPUs) until the queue is empty or the next unit would not fit.
    run_unit(slot, unit, deadline) -> True on success. Returns (finished, not_started) unit lists."""
    t0 = time.time() if t0 is None else t0
    hard = t0 + limit_s
    queue = collections.deque(units)
    lock = threading.Lock()
    durations, done = [], []

    def loop(slot):
        while True:
            with lock:
                if not queue:
                    return
                est = float(np.median(durations)) if durations else first_estimate_s
                if time.time() + margin * est > hard:
                    log(f"slot {slot}: not enough time for another unit (estimate {est:.1f}s) -> stop")
                    return
                u = queue.popleft()
            t = time.time()
            ok = run_unit(slot, u, hard)
            with lock:
                if ok:
                    durations.append(time.time() - t)
                    done.append(u)
            log(f"slot {slot}: unit {u} {'ok' if ok else 'FAILED'} ({time.time() - t:.1f}s)")
    th = [threading.Thread(target=loop, args=(s,)) for s in slots]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return done, list(queue)


def save_unit(out_dir, fold, seed, val_pred, test_pred, **meta):
    """f<fold>_s<seed>_val.npy / _test.npy / _meta.json, the S6E10 unit layout."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"f{fold}_s{seed}_val.npy", np.asarray(val_pred, dtype=np.float32))
    np.save(out_dir / f"f{fold}_s{seed}_test.npy", np.asarray(test_pred, dtype=np.float32))
    (out_dir / f"f{fold}_s{seed}_meta.json").write_text(json.dumps(meta))


def aggregate_units(unit_dirs, folds, n_train, n_test):
    """Assemble OOF (seeds averaged per fold) and test (all units averaged) from unit files spread over
    several directories (one per kernel). Fails if a fold has no validation unit or a unit has no test file."""
    vals, tests = collections.defaultdict(dict), {}
    for d in unit_dirs:
        for p in sorted(Path(d).glob("f*_s*_val.npy")):
            unit = p.name[:-len("_val.npy")]
            k = int(unit.split("_")[0][1:])
            vals[k][unit] = np.load(p)
            t = Path(d) / f"{unit}_test.npy"
            if t.exists():
                tests[unit] = np.load(t)
    missing = [k for k in range(len(folds)) if k not in vals]
    no_test = sorted({u for k in vals for u in vals[k]} - set(tests))
    assert not missing, f"folds without a validation unit: {missing}"
    assert not no_test, f"units without a test prediction: {no_test}"
    first = next(iter(tests.values()))
    oof = np.zeros((n_train,) + first.shape[1:])
    for k, (_, va) in enumerate(folds):
        v = np.mean(list(vals[k].values()), axis=0)
        assert v.shape[0] == len(va), (k, v.shape, len(va))
        oof[va] = v
    pred = np.mean(list(tests.values()), axis=0)
    assert pred.shape[0] == n_test
    return oof, pred, {k: len(vals[k]) for k in sorted(vals)}


def _selftest():
    import os
    import sys
    import tempfile
    from oof_io import make_folds
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1) build a kernel and run the generated run.py locally against a fake /kaggle/input
        (tmp / "code").mkdir()
        (tmp / "code" / "job.py").write_text(
            "import numpy as np, os, sys\n"
            "rows = open('data/train.csv').read().splitlines()[1:]\n"
            "os.makedirs('oof', exist_ok=True); os.makedirs('preds', exist_ok=True)\n"
            "np.save('oof/m.npy', np.full((len(rows), 3), 1 / 3)); np.save('preds/m.npy', np.zeros((2, 3)))\n")
        (tmp / "input" / "fake-comp").mkdir(parents=True)
        (tmp / "input" / "fake-comp" / "train.csv").write_text("id,text\n1,a\n2,b\n3,c\n")
        kd = build(tmp / "build", "unit-test", "someone", bundle(tmp / "code"), ["python src/job.py"],
                   competition="fake-comp", links=[("train.csv", "fake-comp", "data/train.csv")], gpu=False,
                   internet=False)
        meta = json.loads((kd / "kernel-metadata.json").read_text())
        assert meta["enable_internet"] is False and meta["competition_sources"] == ["fake-comp"]
        env = dict(os.environ, RUNNER_INPUT=str(tmp / "input"), RUNNER_OUT=str(tmp / "working"),
                   RUNNER_PROJ=str(tmp / "proj"))
        r = subprocess.run([sys.executable, str(kd / "run.py")], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and "### exit 0" in r.stdout, r.stdout + r.stderr
        assert np.load(tmp / "working" / "oof" / "m.npy").shape == (3, 3)
        # 2) deadline-aware scheduler: 2 slots, 0.05 s units, 0.2 s budget -> about 3 units per slot, rest left
        log = []
        done, left = schedule(list(range(12)), lambda s, u, dl: time.sleep(0.05) or True, slots=[0, 1],
                              limit_s=0.2, first_estimate_s=0.05, log=log.append)
        assert 2 <= len(done) <= 8 and len(done) + len(left) == 12, (done, left, log)
        assert any("not enough time" in x for x in log)
        # 3) units from two "kernels" -> OOF / test
        n, m, k = 180, 7, 18
        y = rng.integers(0, k, n)
        folds = make_folds(y, k=5, seed=42)
        for i, (_, va) in enumerate(folds):
            for s in (0, 1) if i < 3 else (0,):
                save_unit(tmp / ("kA" if i < 3 else "kB"), i, s, rng.dirichlet(np.ones(k), len(va)),
                          rng.dirichlet(np.ones(k), m), total=1.0)
        oof, pred, per_fold = aggregate_units([tmp / "kA", tmp / "kB"], folds, n, m)
        assert oof.shape == (n, k) and pred.shape == (m, k) and per_fold == {0: 2, 1: 2, 2: 2, 3: 1, 4: 1}
        try:
            aggregate_units([tmp / "kA"], folds, n, m)
            raise RuntimeError("missing folds were not detected")
        except AssertionError:
            pass
    print(f"kaggle_runner selftest OK: generated run.py ran locally, scheduler finished {len(done)} / left "
          f"{len(left)} units at the deadline, units from 2 kernels aggregated (units per fold {per_fold})")


if __name__ == "__main__":
    _selftest()
