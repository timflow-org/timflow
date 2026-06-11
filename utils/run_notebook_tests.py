"""Run pytest with --nbval-lax and -n auto for notebook directories tested in CI.

Runs notebooks in parallel on all available CPU cores.

Mirrors the directories used in ci-steady.yml and ci-transient.yml.
Directories like 01concepts/, 04pumpingtests/, and _projects/ are intentionally
excluded, matching CI behaviour.

Slow notebooks (benchmarking/timing notebooks) are skipped to keep the local
run fast. These are the same notebooks excluded from output-clearing in
clear_notebooks.py.
"""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent.parent

# Mirrors the directories tested in ci-steady.yml and ci-transient.yml.
NOTEBOOK_DIRS = [
    "docs/steady/00userguide",
    "docs/steady/02examples",
    "docs/steady/03xsections",
    "docs/steady/04benchmarks",
    "docs/transient/00userguide/howtos",
    "docs/transient/00userguide/tutorials",
    "docs/transient/02examples",
    "docs/transient/03xsections",
    "docs/transient/05benchmarks",
]

# Slow notebooks kept out of CI notebook runs (same list as clear_notebooks.py).
SLOW_NOTEBOOKS = [
    "docs/steady/02examples/vertical_anisotropy.ipynb",
    "docs/steady/04benchmarks/besselnumba_timing.ipynb",
    "docs/steady/04benchmarks/benchmarking_besselaes.ipynb",
]

if __name__ == "__main__":
    dirs = [d for d in NOTEBOOK_DIRS if (root / d).exists()]
    ignores = [
        arg for nb in SLOW_NOTEBOOKS if (root / nb).exists() for arg in ("--ignore", nb)
    ]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--nbval-lax",
        "--dist",
        "loadscope",
        "-n",
        "auto",
        *dirs,
        *ignores,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=root)
    sys.exit(result.returncode)
