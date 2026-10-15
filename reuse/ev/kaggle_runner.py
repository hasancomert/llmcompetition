"""Kaggle not defterinde betik çalıştırma: kernel-metadata.json kurma, gönderme, bekleme, günlük ve çıktı indirme.

S6E9'daki karşılığı: kaggle_run.py. train.py işlerini özel bir not defterinde T4 GPU ya da CPU (PVEP_CPU=1) ile
çalıştırır. "Maximum batch GPU" hatasında 60 sn bekleyip yeniden dener (hesapta aynı anda en çok 2 GPU oturumu).
Bitince stdout/stderr günlüğünü basar ve .npy çıktılarını indirir; çıktı www.kaggleusercontent.com'dan iner, o
adres kapalıysa indirme başarısız olur. Önceden hesaplanan dosyalar (GPT-2 eşlemesi, GLR logit'leri) başka not
defterlerinin çıktısı olarak `kernel_sources` ile bağlandı.

Bu sürüm yarışma, veri seti ve not defteri kaynaklarını parametre alır. Test: yalnız build_kernel (dosya üretimi)
sahte değerlerle denendi. push / wait / logs / fetch Kaggle hesabı gerektirdiği için test edilmedi.
Bağımlılık: kaggle (yalnız push / wait / logs / fetch içinde).
"""
from __future__ import annotations

import json
import os
import shutil
import time

# Not defterinin başına eklenecek yardımcı: Kaggle girdi klasörlerinde dosya bulma (kaggle_run.py'deki find).
FIND_INPUT = '''import glob
def find_input(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]
'''


def build_kernel(build_dir: str, code: str, slug: str, username: str, competition: str | None = None,
                 datasets=(), kernel_sources=(), gpu: bool = True, internet: bool = False, private: bool = True,
                 machine_shape: str = "NvidiaTeslaT4", add_find_input: bool = True) -> dict:
    """build_dir'e run.py ve kernel-metadata.json yazar (klasör önce silinir). Dönen: metadata sözlüğü."""
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)
    with open(os.path.join(build_dir, "run.py"), "w", encoding="utf-8") as f:
        f.write((FIND_INPUT + "\n" if add_find_input else "") + code)
    meta = dict(id=f"{username}/{slug}", title=slug, code_file="run.py", language="python", kernel_type="script",
                is_private=private, enable_gpu=gpu, enable_tpu=False, enable_internet=internet,
                **({"machine_shape": machine_shape} if gpu else {}),
                competition_sources=[competition] if competition else [], dataset_sources=list(datasets),
                kernel_sources=list(kernel_sources), model_sources=[])
    with open(os.path.join(build_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def push(build_dir: str, retry_wait: int = 60):
    """Not defterinin yeni sürümünü başlatır; GPU yuvaları doluysa bekleyip yeniden dener."""
    from kaggle import api
    r = api.kernels_push(build_dir)
    while r is not None and r.error and "Maximum batch GPU" in r.error:
        time.sleep(retry_wait)
        r = api.kernels_push(build_dir)
    if r is None or r.error:
        raise RuntimeError(f"push hatası: {r and r.error}")
    return r


def wait(ref: str, poll: int = 30) -> str:
    """Bitene kadar bekler; son durumu döndürür (COMPLETE / ERROR / CANCEL_ACKNOWLEDGED)."""
    from kaggle import api
    while True:
        status = api.kernels_status(ref).status.name
        if status in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            return status
        time.sleep(poll)


def logs(ref: str):
    """(stdout, stderr) metinleri."""
    from kaggle import api
    log = api.kernels_logs(ref)
    try:
        entries = json.loads(log)
        out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
        err = "".join(e["data"] for e in entries if e.get("stream_name") == "stderr")
    except (ValueError, TypeError, KeyError):
        out, err = log, ""
    return out, err


def fetch(ref: str, outdir: str, file_pattern: str = r"\.npy$") -> list:
    """Çıktı dosyalarını outdir'e indirir (www.kaggleusercontent.com erişimi gerekir)."""
    from kaggle import api
    files, _ = api.kernels_output(ref, outdir, file_pattern=file_pattern, force=True)
    return files
