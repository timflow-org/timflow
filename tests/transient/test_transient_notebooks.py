from pathlib import Path

import papermill as pm
import pytest

NB_DIR = Path.cwd().parent.parent / "docs/transient"


def get_notebooks() -> list[Path]:
    nb_subdirs = [
        NB_DIR / "00userguide",
        NB_DIR / "02examples",
        NB_DIR / "03xsections",
        NB_DIR / "05benchmarks",
    ]

    return [nb for nb_dir in nb_subdirs for nb in nb_dir.glob("*.ipynb")]


PARAMETERS = {
    "meandering_river.ipynb": {"NGR": 2},
    "horizontal_well.ipynb": {"N": 2},
    "line_sink_well_sol.ipynb": {"NGR": 2},
    "river_in_cross_section.ipynb": {"SOLVE_KWARGS": {"max_nfev": 1}},
    "well_near_leaky_river.ipynb": {"N": 2},
}


# @pytest.mark.notebooks
@pytest.mark.skip(reason="Use pytest --nbval on notebooks directly for coverage.")
@pytest.mark.parametrize("pth", get_notebooks())
def test_notebook(pth):
    input_path = pth
    output_path = pth.with_suffix(".out.ipynb")
    pm.execute_notebook(
        input_path,
        str(output_path),
        timeout=600,
        cwd=pth.parent,
        parameters=PARAMETERS.get(pth.name),
    )
    output_path.unlink()  # Remove the output notebook after execution


# local run
if __name__ == "__main__":
    from time import time

    times = {}
    for file in get_notebooks():
        start = time()
        test_notebook(file)
        end = time()
        times[file] = end - start
        print(f"Execution time for {file}: {times[file]:.2f} seconds")

    # Print summary
    print("\nSummary:")
    for file, duration in times.items():
        print(f"{file}: {duration:.2f} seconds")
