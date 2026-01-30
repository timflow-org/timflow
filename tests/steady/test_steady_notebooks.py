import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

nbdirs = [
    os.path.join("docs/steady/00userguide/tutorials"),
    os.path.join("docs/steady/00userguide/howtos"),
    os.path.join("docs/steady/02examples"),
    os.path.join("docs/steady/03xsections"),
    os.path.join("docs/steady/04benchmarks"),
]


def get_notebooks():
    skip = ["benchmarking_besselaes.ipynb", "vertical_anisotropy.ipynb"]
    nblist = []
    for nbdir in nbdirs:
        nblist += [nb for nb in nbdir.glob("*.ipynb") if nb.name not in skip]
    return nblist


# @pytest.mark.notebooks
@pytest.mark.skip(reason="Use pytest --nbval on notebooks directly for coverage.")
@pytest.mark.parametrize("pth", get_notebooks())
def test_notebook_py(pth):
    pth = Path(pth)
    with open(pth, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert ep.preprocess(nb, {"metadata": {"path": pth.parent}}) is not None, (
                f"Got empty notebook for {os.path.basename(pth)}"
            )
        except Exception as e:
            pytest.fail(reason=f"Failed executing {os.path.basename(pth)}: {e}")
