from pathlib import Path

import nbformat
from nbconvert.preprocessors import (
    ClearMetadataPreprocessor,
    ClearOutputPreprocessor,
)

nbdirs = [
    Path("docs/steady/00userguide/tutorials"),
    Path("docs/steady/00userguide/howtos"),
    Path("docs/steady/02examples"),
    Path("docs/steady/03xsections"),
    Path("docs/steady/04benchmarks"),
    Path("docs/transient/00userguide/tutorials"),
    Path("docs/transient/00userguide/howtos"),
    Path("docs/transient/02examples"),
    Path("docs/transient/03xsections"),
    Path("docs/transient/05benchmarks"),
]

skip = {
    "benchmarking_besselaes.ipynb",
    "vertical_anisotropy.ipynb",
}


def get_notebooks():
    nblist = []
    for nbdir in nbdirs:
        nblist += [
            nb_file for nb_file in nbdir.glob("*.ipynb") if nb_file.name not in skip
        ]
    return nblist


clear_output = ClearOutputPreprocessor()
clear_metadata = ClearMetadataPreprocessor()

for notebook in get_notebooks():
    print("Clearing notebook:", notebook)
    with open(notebook, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # run nbconvert preprocessors to clear outputs and metadata
    clear_output.preprocess(nb, {})
    clear_metadata.preprocess(nb, {})

    with open(notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
