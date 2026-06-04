import warnings
from importlib import import_module, metadata
from platform import python_version

__version__ = "0.3.0.dev0"


def show_versions(optional=True) -> None:
    """Print the version of dependencies.

    Parameters
    ----------
    optional : bool, optional
        Print the version of optional dependencies, by default False
    """
    msg = (
        f"timflow version    : {__version__}\n\n"
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


def check_tqdm_parallel(parallel):
    """Check if tqdm is installed when parallel processing is requested.

    Parameters
    ----------
    parallel : bool
        Whether parallel processing is requested.

    Returns
    -------
    parallel : bool
        Whether parallel processing was requested and can be used.
    thread_map : function or None
        The thread_map function from tqdm if parallel processing is available, else None.
    tqdm : class or None
        The tqdm class from tqdm if parallel processing is available, else None.
    """
    if not parallel:  # short circuit when no parallel requested
        return parallel, None, None
    try:
        from tqdm import tqdm
        from tqdm.contrib.concurrent import thread_map
    except ImportError:
        warnings.warn(
            "Parallel requires 'tqdm'. Install 'timflow[parallel]' or 'tqdm' to"
            " enable parallel execution. Falling back to serial execution.",
            category=ImportWarning,
            stacklevel=2,
        )
        parallel = False
        thread_map = None
        tqdm = None
    return parallel, thread_map, tqdm
