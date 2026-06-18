# %%
from typing import NamedTuple

import numpy as np

# %%
line_element_dtype = np.dtype(
    [
        ("etype", np.int64),
        ("bctype", np.int8),
        ("aq_id", np.int64),
        ("nparam", np.int64),
        ("z1", np.complex128),
        ("z2", np.complex128),
        ("L", np.float64),
        ("order", np.int64),
        ("rzero", np.float64),
        ("p0", np.int64),
        ("p1", np.int64),
    ]
)

well_element_dtype = np.dtype(
    [
        ("etype", np.int64),
        ("bctype", np.int8),
        ("aq_id", np.int64),
        ("nparam", np.int64),
        ("xw", np.float64),
        ("yw", np.float64),
        ("rw", np.float64),
        ("rzero", np.float64),
        ("p0", np.int64),
        ("p1", np.int64),
    ]
)


class ModelTuple(NamedTuple):
    """Data structure for timflow Model parameters and arrays."""

    nint: np.int64
    npint: np.int64
    npval: np.int64
    ngvbc: np.int64
    M: np.int64
    tstart: np.float64
    p: np.ndarray[np.complex128]
    tintervals: np.ndarray[np.float64]
    enumber: np.ndarray[np.int64]
    etstart: np.ndarray[np.float64]
    ebc: np.ndarray[np.float64]


class AquiferTuple(NamedTuple):
    """Data structure for Aquifer parameters and arrays."""

    naq: np.int64
    lab2: np.ndarray[np.complex128]
    lababs: np.ndarray[np.float64]
    eigvec: np.ndarray[np.complex128]
    Tcol: np.ndarray[np.float64]


class ElementTuple(NamedTuple):
    """Unified element tuple for all element types (lines, wells, etc.)."""

    meta: np.ndarray
    layers: np.ndarray[np.int32]
    layer_ptr: np.ndarray[np.int32]
    term2: np.ndarray[np.complex128]
    parameters: np.ndarray[np.complex128]
