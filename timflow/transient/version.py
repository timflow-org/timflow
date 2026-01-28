from importlib import import_module, metadata
from platform import python_version

__version__ = "0.8.0"


def show_versions(optional=True) -> None:
    """Print the version of dependencies.

    Parameters
    ----------
    optional : bool, optional
        Print the version of optional dependencies, by default False
    """
    msg = (
        f"TTim version       : {__version__}\n\n"
        f"Python version     : {python_version()}\n"
        f"Numpy version      : {metadata.version('numpy')}\n"
        f"Numba version      : {metadata.version('numba')}\n"
        f"Scipy version      : {metadata.version('scipy')}\n"
        f"Pandas version     : {metadata.version('pandas')}\n"
        f"Matplotlib version : {metadata.version('matplotlib')}"
    )
    if optional:
        msg += "\nLmFit version      : "
        try:
            import_module("lmfit")
            msg += f"{metadata.version('lmfit')}"
        except ImportError:
            msg += "Not Installed"

    print(msg)
