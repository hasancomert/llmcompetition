"""Kaggle CLI wrapper for benchmark tasks: kagglesdk 0.1.37 (the newest on PyPI, 16 Oct 2026) does not know the
server's newer enum values (e.g. BENCHMARK_TASK_VERSION_CREATION_STATE_VALIDATING) and crashes while reading a task's
status. Unknown values are mapped to a known neighbour (VALIDATING -> RUNNING, otherwise UNSPECIFIED) and reported.

Usage: python kaggle/kcli.py benchmarks tasks status nllm-gen-ft2      (same arguments as the kaggle command)
"""
import sys

import kagglesdk.kaggle_object as ko

_orig = ko.EnumSerializer._from_str


def _from_str(cls, v):
    try:
        return _orig(cls, v)
    except KeyError:
        for alt in (v.replace("VALIDATING", "RUNNING"), v.rsplit("_", 1)[0] + "_UNSPECIFIED"):
            if alt in cls.__members__:
                print(f"[kcli] unknown {cls.__name__} value {v} -> {alt}", file=sys.stderr)
                return cls[alt]
        return list(cls)[0]


ko.EnumSerializer._from_str = staticmethod(_from_str)

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
