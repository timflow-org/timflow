# %%
import numba as nb
import numpy as np

from timflow.bessel.besselnumba import besselk0, bessellsv2
from timflow.transient.element import BCType, ElementType
from timflow.transient.invlapnumba import invlapcomp
from timflow.transient.parallel.dtypes import (
    AquiferTuple,
    ModelTuple,
    line_element_dtype,
    well_element_dtype,
)

# %%


@nb.njit(nogil=True, cache=True, fastmath=True)
def potinf_linesink(
    x: float,
    y: float,
    nint: int,
    naq: int,
    lab2: np.ndarray,
    meta: np.ndarray,
    term2: np.ndarray,
    out4d: np.ndarray,
) -> None:
    """Evaluates analytic line element contributions straight into thread memory."""
    nls = meta.shape[0]
    for i in range(nls):
        m = meta[i]
        p0, p1 = m["p0"], m["p1"]
        z1, z2, L, order, rzero = m["z1"], m["z2"], m["L"], m["order"], m["rzero"]

        for a in range(naq):
            for j in range(nint):
                out4d[p0:p1, a, j, :] = (
                    term2[p0:p1, a, j, :]
                    * bessellsv2(x, y, z1, z2, lab2[a, j, :], order, rzero)
                    / L
                )


@nb.njit(nogil=True, cache=True, fastmath=True)
def potinf_well(
    x: float,
    y: float,
    nint: int,
    npint: int,
    naq: int,
    lab2: np.ndarray,
    meta: np.ndarray,
    term2: np.ndarray,
    out4d: np.ndarray,
) -> None:
    """Evaluates analytic well element contributions straight into thread memory."""
    nwells = meta.shape[0]
    for i in range(nwells):
        m = meta[i]
        p0, p1 = m["p0"], m["p1"]
        xw, yw, rw, rzero = m["xw"], m["yw"], m["rw"], m["rzero"]

        dx = x - xw
        dy = y - yw
        r = np.sqrt(dx**2 + dy**2)
        if r < rw:
            dx = rw
            dy = 0.0
            r = rw

        for a in range(naq):
            for j in range(nint):
                if r / abs(lab2[a, j, 0]) < rzero:
                    for k in range(lab2.shape[2]):
                        out4d[p0:p1, a, j, k] = term2[p0:p1, a, j, k] * besselk0(
                            dx, dy, lab2[a, j, k]
                        )
                else:
                    # Explicitly zero out
                    out4d[p0:p1, a, j, :] = 0.0 + 0.0j


@nb.njit(nogil=True, cache=True, fastmath=True)
def pot_linesink(
    x: float,
    y: float,
    nint: int,
    npint: int,
    naq: int,
    ngvbc: int,
    lab2: np.ndarray,
    meta: np.ndarray,
    term2: np.ndarray,
    params: np.ndarray,
    pot_view: np.ndarray,
) -> None:
    """Evaluates analytic line elements and accumulates directly into thread memory."""
    nls = meta.shape[0]
    for i in range(nls):
        m = meta[i]
        p0, p1 = m["p0"], m["p1"]
        z1, z2, L, order, rzero = m["z1"], m["z2"], m["L"], m["order"], m["rzero"]

        for a in range(naq):
            for j in range(nint):
                bessel_vals = bessellsv2(x, y, z1, z2, lab2[a, j, :], order, rzero) / L
                for g in range(ngvbc):
                    for ip in range(p0, p1):
                        order_idx = ip - p0
                        params_slice = params[g, ip, :]
                        for k in range(npint):
                            v = j * npint + k
                            t_val = term2[ip, a, j, k] * bessel_vals[order_idx, k]
                            pot_view[g, a, v] += params_slice[v] * t_val


@nb.njit(nogil=True, cache=True, fastmath=True)
def pot_well(
    x: float,
    y: float,
    nint: int,
    npint: int,
    naq: int,
    ngvbc: int,
    lab2: np.ndarray,
    meta: np.ndarray,
    term2: np.ndarray,
    params: np.ndarray,
    pot_view: np.ndarray,
) -> None:
    """Evaluates analytic well elements and accumulates directly into thread memory."""
    nwells = meta.shape[0]
    for i in range(nwells):
        m = meta[i]
        p0, p1 = m["p0"], m["p1"]
        xw, yw, rw, rzero = m["xw"], m["yw"], m["rw"], m["rzero"]

        dx = x - xw
        dy = y - yw
        r = np.sqrt(dx**2 + dy**2)
        if r < rw:
            dx = rw
            dy = 0.0
            r = rw

        for a in range(naq):
            for j in range(nint):
                if r / abs(lab2[a, j, 0]) < rzero:
                    for k in range(npint):
                        # The fix for the Issue A broadcast bug is included here
                        b_val = besselk0(dx, dy, lab2[a, j, k])
                        v = j * npint + k
                        for ip in range(p0, p1):
                            t_val = term2[ip, a, j, k] * b_val
                            for g in range(ngvbc):
                                pot_view[g, a, v] += params[g, ip, v] * t_val


def get_element_data(ml):
    """Prepare element data for numba processing.

    Parameters
    ----------
    ml : Model3D or ModelMaq
        The model object containing the elements.

    Returns
    -------
    dict
        A dictionary with ElementType keys and lists of element data tuples as values.
    """
    edict = {i: [] for i in ElementType}
    for e in ml.elementlist:
        if not hasattr(e, "to_numba_tuple"):
            raise NotImplementedError(
                f"Element of type {type(e)} does not have a to_numba_tuple() method."
            )
        edata = e.to_numba_tuple()
        edict[edata.meta["etype"][0]].append(edata)

    return edict


@nb.njit(nogil=True, cache=True)
def elements_to_numba_arrays(etuples, element_dtype, mtuple, aqtuple):
    """Convert a list of element data tuples to a structured numpy array.

    Parameters
    ----------
    etuples : list
        List of element data tuples.
    element_dtype : np.dtype
        The numpy dtype schema for the structured array.

    Returns
    -------
    meta : np.rec.array
        Structured numpy array containing all element data and indexers for
        contiguous stacked arrays.
    term2 : np.ndarray
        4D array of term2 values for each element.
    params : np.ndarray
        3D array of parameters for each element.
    """
    # dimensions
    nint, npint, npval, ngvbc = mtuple.nint, mtuple.npint, mtuple.npval, mtuple.ngvbc
    naq = aqtuple.naq

    # no. of elements and parameters
    n_params = 0
    n_elements = 0
    for e in etuples:
        n_elements += e.meta.shape[0]
        n_params += np.sum(e.meta["nparam"])

    # build metadata, term2 and param arrays
    meta = np.empty(n_elements, dtype=element_dtype)
    term2 = np.empty((n_params, naq, nint, npint), dtype=np.complex128)
    params = np.empty((ngvbc, n_params, npval), dtype=np.complex128)

    ie = 0
    ip = 0
    for e in etuples:
        for i in range(e.meta.shape[0]):
            imeta = e.meta[i]
            meta[ie] = imeta
            p0, p1 = int(imeta["p0"]), int(imeta["p1"])
            n = p1 - p0
            meta[ie]["p0"] = ip
            meta[ie]["p1"] = ip + n
            term2[ip : ip + n] = e.term2[p0:p1]
            params[:, ip : ip + n, :] = e.parameters[:, p0:p1, :]
            ip += n
            ie += 1

    return meta, term2, params


def prepare_element_data(ml, mtuple, aqtuple):

    edict = get_element_data(ml)

    # 1. Gather and build line arrays
    ltuples = tuple(edict[ElementType.LINESINK] + edict[ElementType.LINESINKSTRING])
    line_meta, line_term2, line_params = elements_to_numba_arrays(
        ltuples, line_element_dtype, mtuple, aqtuple
    )

    # 2. Gather and build well arrays
    wtuples = tuple(edict[ElementType.WELL] + edict[ElementType.WELLSTRING])
    well_meta, well_term2, well_params = elements_to_numba_arrays(
        wtuples, well_element_dtype, mtuple, aqtuple
    )

    return line_meta, line_term2, line_params, well_meta, well_term2, well_params


@nb.njit(nogil=True, cache=True, parallel=True)
def _headgrid_numba(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    mtuple: ModelTuple,
    aqtuple: AquiferTuple,
    line_meta: np.ndarray,
    line_term2: np.ndarray,
    line_params: np.ndarray,
    well_meta: np.ndarray,
    well_term2: np.ndarray,
    well_params: np.ndarray,
):
    # get model and aquifer data
    nint, npint, npval, ngvbc = (mtuple.nint, mtuple.npint, mtuple.npval, mtuple.ngvbc)
    naq = aqtuple.naq
    lab2 = aqtuple.lab2
    npts = len(x)

    # create output array and compute delta time
    out = np.empty((naq, len(t), npts), dtype=np.float64)
    time = np.atleast_1d(t) - mtuple.tstart


    # allocate computation arrays (with dimension per thread)
    num_threads = nb.config.NUMBA_NUM_THREADS
    pot_scratch = np.empty((num_threads, ngvbc, naq, npval), dtype=np.complex128)
    pot_tx_scratch = np.empty((num_threads, ngvbc, naq, npval), dtype=np.complex128)

    # ensure parameter and eigvec arrays are contiguous
    eigvec_contiguous = np.ascontiguousarray(aqtuple.eigvec)

    # parallel loop
    for i in nb.prange(npts):
        xi, yi = x[i], y[i]
        tid = nb.get_thread_id()

        # Zero out the reused array views for this iteration
        pot_view = pot_scratch[tid]
        pot_tx_view = pot_tx_scratch[tid]
        pot_view[:, :, :] = 0.0 + 0.0j
        pot_tx_view[:, :, :] = 0.0 + 0.0j

        # Compute potential line sinks
        if line_meta.shape[0] > 0:
            pot_linesink(
                xi,
                yi,
                nint,
                npint,
                naq,
                ngvbc,
                lab2,
                line_meta,
                line_term2,
                line_params,
                pot_view,
            )

        # Compute potential wells
        if well_meta.shape[0] > 0:
            pot_well(
                xi,
                yi,
                nint,
                npint,
                naq,
                ngvbc,
                lab2,
                well_meta,
                well_term2,
                well_params,
                pot_view,
            )

        # compute potentials
        for g in range(ngvbc):
            for a_out in range(naq):
                for a_in in range(naq):
                    for v in range(npval):
                        pot_tx_view[g, a_out, v] += (
                            pot_view[g, a_in, v] * eigvec_contiguous[a_out, a_in, v]
                        )

        # inverse laplace transform
        rv_inv = invlapcomp(
            time,
            pot_tx_view,
            npint,
            mtuple.M,
            mtuple.tintervals,
            mtuple.enumber,
            mtuple.etstart,
            mtuple.ebc,
            naq,
        )
        out[:, :, i] = rv_inv / aqtuple.Tcol

    return out


def headgrid(
    ml,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
):
    """ """
    mtuple = ml.to_numba_tuple()
    aqtuple = ml.aq.to_numba_tuple()

    lmeta, lterm2, lparams, wmeta, wterm2, wparams = prepare_element_data(
        ml, mtuple, aqtuple
    )

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    return _headgrid_numba(
        x,
        y,
        t,
        mtuple,
        aqtuple,
        lmeta,
        lterm2,
        lparams,
        wmeta,
        wterm2,
        wparams,
    )
