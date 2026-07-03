from pathlib import Path

import nbformat
from nbconvert.preprocessors import (
    ClearMetadataPreprocessor,
    ClearOutputPreprocessor,
)

nbroots = [Path("docs/transient/04pumpingtests")]


def get_notebooks():
    nblist = []
    for root in nbroots:
        for nb in root.glob("*.ipynb"):
            if "_build" in nb.parts:
                continue
            nblist.append(nb)
    return sorted(nblist)


clear_output = ClearOutputPreprocessor()

# By default nbconvert keeps `metadata.language_info.name`. Use an empty preserve
# mask so kernel/language metadata is stripped from notebook files.
clear_metadata = ClearMetadataPreprocessor(preserve_nb_metadata_mask=set())

for notebook in get_notebooks():
    print("Clearing notebook:", notebook)
    nb = nbformat.read(notebook, as_version=4)

    # run nbconvert preprocessors to clear outputs and metadata
    clear_output.preprocess(nb, {})
    clear_metadata.preprocess(nb, {})

    # Ensure a portable kernel is defined for CI/ReadTheDocs notebook execution.
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }

    nbformat.write(nb, notebook)
