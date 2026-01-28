import os

import nbformat
import pytest
from nbconvert.preprocessors import (
    ClearMetadataPreprocessor,
    ClearOutputPreprocessor,
    ExecutePreprocessor,
)

# import subprocess

# %%
nbdirs = [
    # os.path.join("docs/00tutorials"),
    os.path.join("docs/02examples"),
    # os.path.join("notebooks"),
]


# testdir = tempfile.mkdtemp()


def get_notebooks():
    nblist = []
    for nbdir in nbdirs:
        nblist += [
            os.path.join(nbdir, f) for f in os.listdir(nbdir) if f.endswith(".ipynb")
        ]
    return nblist


# def get_jupyter_kernel():
#     try:
#         jklcmd = ("jupyter", "kernelspec", "list")
#         b = subprocess.Popen(
#             jklcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
#         ).communicate()[0]
#         if isinstance(b, bytes):
#             b = b.decode("utf-8")
#         print(b)
#         for line in b.splitlines():
#             if "python" in line:
#                 kernel = line.split()[0]
#     except:  # noqa
#         kernel = None

#     return kernel


# @pytest.mark.notebooks
# @pytest.mark.parametrize("pth", get_notebooks())
# def test_notebook(pth):
#     kernel = get_jupyter_kernel()
#     print("available jupyter kernel {}".format(kernel))

#     fn = os.path.basename(pth)

#     cmd = (
#         "jupyter "
#         + "nbconvert "
#         + "--ExecutePreprocessor.timeout=600 "
#         + "--to "
#         + "notebook "
#         + "--execute "
#         + "{} ".format(pth)
#         + "--output-dir "
#         + "{} ".format(testdir)
#         + "--output "
#         + "{}".format(fn)
#     )
#     ival = os.system(cmd)
#     assert ival == 0, "could not run {}".format(fn)


# @pytest.mark.notebooks
# @pytest.mark.parametrize("pth", get_notebooks())
# def test_notebook_cmd(pth):
#     cmd = (
#         "jupyter "
#         + "nbconvert "
#         + "--ExecutePreprocessor.timeout=600 "
#         + "--ExecutePreprocessor.kernel_name=python3 "
#         + "--to "
#         + "notebook "
#         + "--execute "
#         + "{} ".format(pth)
#     )
#     ival = os.system(cmd)
#     assert ival == 0, f"could not run {os.path.basename(pth)}"


@pytest.mark.notebooks
@pytest.mark.parametrize("pth", get_notebooks())
def test_notebook_py(pth):
    with open(pth, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert (
                ep.preprocess(nb, {"metadata": {"path": "docs/02examples"}}) is not None
            ), f"Got empty notebook for {os.path.basename(pth)}"
        except Exception as e:
            pytest.fail(reason=f"Failed executing {os.path.basename(pth)}: {e}")


# %% clear output and metadata of all notebooks
if __name__ == "__main__":
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
