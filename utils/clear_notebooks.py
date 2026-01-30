from pathlib import Path

import nbformat
from nbconvert.preprocessors import (
    ClearMetadataPreprocessor,
    ClearOutputPreprocessor,
)

nbdirs = [
    # steady
    Path("docs/steady/00userguide/tutorials"),
    Path("docs/steady/00userguide/howtos"),
    Path("docs/steady/02examples"),
    Path("docs/steady/03xsections"),
    Path("docs/steady/04benchmarks"),
    # transient
    Path("docs/transient/00userguide/tutorials"),
    Path("docs/transient/00userguide/howtos"),
    Path("docs/transient/02examples"),
    Path("docs/transient/03xsections"),
    Path("docs/transient/05benchmarks"),
]

skip = [
    "benchmarking_besselaes.ipynb",
    "vertical_anisotropy.ipynb",
    "besselnumbanew_timing.ipynb",
]


def get_notebooks():
    nblist = []
    for nbdir in nbdirs:
        nblist += [nb for nb in nbdir.glob("*.ipynb") if nb.name not in skip]
    return nblist


clear_output = ClearOutputPreprocessor()
clear_metadata = ClearMetadataPreprocessor()

for notebook in get_notebooks():
    print("Clearing notebook:", notebook)
    nb = nbformat.read(notebook, as_version=4)

    # run nbconvert preprocessors to clear outputs and metadata
    clear_output.preprocess(nb, {})
    clear_metadata.preprocess(nb, {})

    nbformat.write(nb, notebook)
