# Copyright 2019 Kristopher L. Kuhlman <klkuhlm _at_ sandia _dot_ gov>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def invlap(t, tmax, fp, M, alpha=1e-10, tol=1e-9):
    """Inverse Laplace tansform with algorithm of De Hoog, Knight and Stokes.

    Parameters
    ----------
    t : array
        times for which inverse is computed
    tmax : float
        maximum time
    fp : complex array
        Laplace transformed solution
    M : integer
        number of terms (number of values of fp)
    alpha:  is the real part of the rightmost pole or singularity, which
            is chosen based on the desired accuracy (assuming the rightmost
            singularity is 0), and tol=10α is the desired tolerance

    Returns
    -------
    result : array
        time domain solution for specified times

    Reference
    ---------

    de Hoog, F., J. Knight, A. Stokes (1982). An improved method for
    numerical inversion of Laplace transforms. SIAM Journal of Scientific
    and Statistical Computing 3:357-366, http://dx.doi.org/10.1137/0903022

    https://bitbucket.org/klkuhlm/invlap/src/default/invlap.py
    """
    # return zeros if all fp equal to zero
    if np.all(np.abs(fp) == 0):
        return np.zeros(len(t))

    NP = 2 * M + 1
    scale = 2.0
    T = scale * tmax
    Nt = len(t)
    #
    # tol = alpha * 10.0
    gamma = alpha - np.log(tol) / (scale * T)

    # would it be useful to try re-using
    # space between e&q and A&B?
    # Kris programmed this as np.complex64
    e = np.empty((2 * M + 1, M + 1), dtype=np.complex128)
    q = np.empty((2 * M, M), dtype=np.complex128)
    d = np.empty(2 * M + 1, dtype=np.complex128)
    A = np.empty((2 * M + 2, Nt), dtype=np.complex128)
    B = np.empty((2 * M + 2, Nt), dtype=np.complex128)

    # initialize Q-D table
    e[:, 0] = 0.0
    q[0, 0] = fp[1] / (fp[0] / 2.0)
    for i in range(1, 2 * M):
        q[i, 0] = fp[i + 1] / fp[i]

    # rhombus rule for filling triangular Q-D table (e & q)
    for r in range(1, M + 1):
        # start with e, column 1, 0:2*M-2
        mr = 2 * (M - r) + 1
        e[0:mr, r] = q[1 : mr + 1, r - 1] - q[0:mr, r - 1] + e[1 : mr + 1, r - 1]
        if not r == M:
            rq = r + 1
            mr = 2 * (M - rq) + 2
            for i in range(mr):
                q[i, rq - 1] = q[i + 1, rq - 2] * e[i + 1, rq - 1] / e[i, rq - 1]

    # build up continued fraction coefficients (d)
    d[0] = fp[0] / 2.0
    for r in range(1, M + 1):
        d[2 * r - 1] = -q[0, r - 1]  # even terms
        d[2 * r] = -e[0, r]  # odd terms

    # seed A and B for recurrence
    A[0] = 0.0
    A[1] = d[0]
    B[0:2] = 1.0 + 0j

    # base of the power series
    z = np.exp(1j * np.pi * t / T)

    # coefficients of Pade approximation (A & B)
    # using recurrence for all but last term
    for i in range(1, 2 * M):
        A[i + 1] = A[i] + d[i] * A[i - 1] * z
        B[i + 1] = B[i] + d[i] * B[i - 1] * z

    # "improved remainder" to continued fraction
    brem = (1.0 + (d[2 * M - 1] - d[2 * M]) * z) / 2.0
    rem = -brem * (1.0 - np.sqrt(1.0 + d[2 * M] * z / brem**2))

    # last term of recurrence using new remainder
    A[NP] = A[2 * M] + rem * A[2 * M - 1]
    B[NP] = B[2 * M] + rem * B[2 * M - 1]

    # diagonal Pade approximation
    # F=A/B represents accelerated trapezoid rule
    result = np.exp(gamma * t) / T * (A[NP] / B[NP]).real
    # print('results:', result)
    # print('A', A)
    # print('B', B)

    return result


@numba.njit(nogil=True, cache=True)
def compute_laplace_parameters_numba(tmax, M=20, alpha=1e-10, tol=1e-9):
    # 2*M+1 terms in approximation
    # desired tolerance (here simply related to alpha)
    # tol = alpha * 10.0
    nump = 2 * M + 1  # number of terms in approximation
    # scaling factor (likely tune-able, but 2 is typical)
    scale = 2.0
    T = scale * tmax
    gamma = alpha - np.log(tol) / (scale * T)
    p = gamma + 1j * np.pi * np.arange(nump) / T
    return p


def invlaptest():
    p = compute_laplace_parameters_numba(tmax=10, alpha=1e-10)
    fp = 1 / (p + 1) ** 2
    t = np.arange(1.0, 10)
    ft = invlap(t, 10, fp, 20, alpha=1e-10)
    print("approximate from invlap:", ft)
    print("exact:", t * np.exp(-t))


@numba.njit(nogil=True, cache=True)
def invlapcomp(time, pot, npint, M, tintervals, enumber, etstart, ebc, nlayers):
    """Compute time domain solution for given laplace domain solution.

    Parameters
    ----------
    time : array, must be ordered
        times for which time domain solution is computed, must start at 0
    pot : array
        laplace domain solution conform the timflow.transient shape
    npint : int
        number of p values per interval (=2M + 1)
    M : int
        order of the approximation
    tintervals :
        time intervals
    enumber : array
        number of element
    etstart : array
        starting time of bc in element
    ebc : array
        boundary condition value of element
    nlayers : integer, optional
        number of layers, default is None

    Notes
    -----
    enumber, etstart, and ebc are used because numba cannot deal with a list
    of arrays of different lengths (makes sense, actually)

    Returns
    -------
    pot[naq, ntimes] if layers=None,
    otherwise pot[len(layers) ,ntimes]
    t must be ordered
    """
    print_tmin_warning = True  # set to False if warning is printed once
    print_tmax_warning = True
    nelements, naq, npval = pot.shape
    nint = len(tintervals) - 1
    rv = np.zeros((nlayers, len(time)))
    #
    # assuming that first time of all bc's is 0
    for j in range(len(enumber)):
        t = time - etstart[j]
        it = 0
        if t[0] <= 0:  # there are times before start of bc
            if t[-1] <= 0:  # all times before start of bc, also for len(t)=1
                continue
            else:
                # no effect for any t <= 0
                it = np.argmax(t > 0)  # find_first
        if t[it] < tintervals[0]:  # there are times before first interval
            if print_tmin_warning:
                print("Warning, some of the times are smaller than tmin after")
                print("a change in boundary condition. nans are substituted")
                print_tmin_warning = False
            if t[-1] < tintervals[0]:  # all times before first interval
                itnew = len(t)
            else:
                itnew = np.argmax(t >= tintervals[0])  # find_first
            rv[:, it:itnew] = np.nan
            it = itnew
        for n in range(nint):
            if n == 0:
                tp = t[(t >= tintervals[n]) & (t <= tintervals[n + 1])]
            else:
                tp = t[(t > tintervals[n]) & (t <= tintervals[n + 1])]
            nt = len(tp)
            # if nt > 0:  # if all zero, don't do the inv transform
            if nt == 0:
                continue
            for i in range(nlayers):
                # I used to check the first value only, but got to check if
                # none of the values are zero
                if not np.any(pot[enumber[j], i, n * npint : (n + 1) * npint] == 0):
                    rv[i, it : it + nt] += ebc[j] * invlap(
                        tp,
                        tintervals[n + 1],
                        pot[enumber[j], i, n * npint : (n + 1) * npint],
                        M,
                    )
            it = it + nt
        if it < len(t):  # there are times above tintervals[-1]
            if print_tmax_warning:
                print("Warning, some of the times are larger than tmax after")
                print("a change in boundary condition. nans are substituted")
                print_tmax_warning = False
            rv[:, it:] = np.nan
    return rv


# The following general function is currently not used but very useful
# to do numerical Laplace transformation
@numba.njit(nogil=True, cache=True)
def invlapgen(time, pot, M, tintervals, tstart, ebc):
    """Compute time domain solution for given Laplace domain solution.

    Parameters
    ----------
    time : array, must be ordered
           times for which time domain solution is computed, must start at 0
    pot : 1D array of laplace domain solution of length nint * (2M + 1)
    M : int order of the approximation (i.e., (2M + 1) p values per interval)
    tintervals : 1D array of time intervals, length nint + 1
    tstart : 1D array with starting times of bc in element, length nstart
    ebc : 1D array with change in boundary condition value, length nstart

    Notes
    -----
    ebc is the difference with the previous value

    Returns
    -------
    rv[ntimes]
    """
    print_tmin_warning = True  # set to False if warning is printed once
    print_tmax_warning = True
    npint = 2 * M + 1
    nint = len(tintervals) - 1
    rv = np.zeros(len(time))
    #
    # assuming that first time of all bc's is 0
    for j in range(len(tstart)):
        t = time - tstart[j]
        it = 0
        if t[0] <= 0:  # there are times before start of bc
            if t[-1] <= 0:  # all times before start of bc, also for len(t)=1
                continue
            else:
                # no effect for any t <= 0
                it = np.argmax(t > 0)  # find_first
        if t[it] < tintervals[0]:  # there are times before first interval
            if print_tmin_warning:
                print("Warning, some of the times are smaller than tmin after")
                print("a change in boundary condition. nans are substituted")
                print_tmin_warning = False
            if t[-1] < tintervals[0]:  # all times before first interval
                itnew = len(t)
            else:
                itnew = np.argmax(t >= tintervals[0])  # find_first
            rv[it:itnew] = np.nan
            it = itnew
        for n in range(nint):
            if n == 0:
                tp = t[(t >= tintervals[n]) & (t <= tintervals[n + 1])]
            else:
                tp = t[(t > tintervals[n]) & (t <= tintervals[n + 1])]
            nt = len(tp)
            # if nt > 0:  # if all zero, don't do the inv transform
            if nt == 0:
                continue
            # I used to check the first value only, but got to check if
            # none of the values are zero
            if not np.any(pot[n * npint : (n + 1) * npint] == 0):
                rv[it : it + nt] += ebc[j] * invlap(
                    tp, tintervals[n + 1], pot[n * npint : (n + 1) * npint], M
                )
            it = it + nt
        if it < len(t):  # there are times above tintervals[-1]
            if print_tmax_warning:
                print("Warning, some of the times are larger than tmax after")
                print("a change in boundary condition. nans are substituted")
                print_tmax_warning = False
            rv[it:] = np.nan
    return rv
