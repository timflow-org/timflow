import numba
import numpy as np

# defenition of parameters

tiny = 1e-10
c = np.log(0.5) + 0.577215664901532860

fac = 1.0

nrange = np.arange(21, dtype=np.float64)

a = np.zeros(21, dtype=np.float64)
a[0] = 1.0
b = np.zeros(21, dtype=np.float64)

for n in range(1, 21):
    fac = n * fac
    a[n] = 1.0 / (4.0 ** nrange[n] * fac**2)
    b[n] = b[n - 1] + 1 / nrange[n]

b = (b - c) * a
a = -a / 2.0

gam = np.zeros((21, 21), dtype=np.float64)
for n in range(21):
    for m in range(n + 1):
        gam[n, m] = np.prod(nrange[m + 1 : n + 1]) / np.prod(nrange[1 : n - m + 1])

# gotta predefine these i.o. gam which is used in the old code
binom = np.zeros((21, 21), dtype=np.float64)
for n in range(21):
    for m in range(n + 1):
        binom[n, m] = np.prod(nrange[m + 1 : n + 1]) / np.prod(nrange[1 : n - m + 1])

# coefficients K1
fac = 1.0
bot = np.zeros(21, dtype=np.float64)
bot[0] = 4.0
for n in range(1, 21):
    fac = n * fac
    bot[n] = fac * (n + 1) * fac * 4.0 ** (n + 1)

psi = np.zeros(21, dtype=np.float64)
for n in range(2, 22):
    psi[n - 1] = psi[n - 2] + 1 / (n - 1)
psi = psi - 0.577215664901532860

a1 = np.empty(21, dtype=np.float64)
b1 = np.empty(21, dtype=np.float64)
twologhalf = 2 * np.log(0.5)
for n in range(21):
    a1[n] = 1 / bot[n]
    b1[n] = (twologhalf - (2.0 * psi[n] + 1 / (n + 1))) / bot[n]

# Laplace line doublet elements


@numba.njit(nogil=True, cache=True)
def lapld_int_ho(x, y, z1, z2, order):
    """lapld_int_ho.

    ! Near field only
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: omega, qm
    integer :: m, n
    real(kind=8) :: L
    complex(kind=8) :: z, zplus1, zmin1
    """
    omega = np.zeros(order + 1, dtype=np.complex128)
    qm = np.zeros(order + 1, dtype=np.complex128)

    # L = np.abs(z2 - z1)
    z = (2.0 * complex(x, y) - (z1 + z2)) / (z2 - z1)
    zplus1 = z + 1.0
    zmin1 = z - 1.0
    # Not sure if this gives correct answer at corner point (z also appears in qm);
    # should really be caught in code that calls this function
    if np.abs(zplus1) < tiny:
        zplus1 = tiny
    if np.abs(zmin1) < tiny:
        zmin1 = tiny

    omega[0] = np.log(zmin1 / zplus1)
    for n in range(1, order + 1):
        omega[n] = z * omega[n - 1]

    if order > 0:
        qm[1] = 2.0
    for m in range(3, order + 1, 2):
        qm[m] = qm[m - 2] * z * z + 2.0 / m

    for m in range(2, order + 1, 2):
        qm[m] = qm[m - 1] * z

    omega = 1.0 / (complex(0.0, 2.0) * np.pi) * (omega + qm)
    return omega


@numba.njit(nogil=True, cache=True)
def lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2):
    """lapld_int_ho_d1d2.

    Near field only
    Returns integral from d1 to d2 along real axis while strength is still
    Delta^order from -1 to +1
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: omega, omegac
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """
    omega = np.zeros(order + 1, dtype=np.complex128)

    bigz1 = complex(d1, 0.0)
    bigz2 = complex(d2, 0.0)
    z1p = 0.5 * (z2 - z1) * bigz1 + 0.5 * (z1 + z2)
    z2p = 0.5 * (z2 - z1) * bigz2 + 0.5 * (z1 + z2)
    omegac = lapld_int_ho(x, y, z1p, z2p, order)
    dc = (d1 + d2) / (d2 - d1)
    for n in range(order + 1):
        for m in range(n + 1):
            omega[n] = omega[n] + gam[n, m] * dc ** (n - m) * omegac[m]
        omega[n] = (0.5 * (d2 - d1)) ** n * omega[n]

    return omega


@numba.njit(nogil=True, cache=True)
def lapld_int_ho_wdis(x, y, z1, z2, order):
    """lapld_int_ho_wdis.

    # Near field only
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: wdis
    complex(kind=8), dimension(0:10) :: qm  # Max order is 10
    integer :: m, n
    complex(kind=8) :: z, zplus1, zmin1, term1, term2, zterm
    """
    qm = np.zeros(11, dtype=np.complex128)
    wdis = np.zeros(order + 1, dtype=np.complex128)

    z = (2.0 * complex(x, y) - (z1 + z2)) / (z2 - z1)
    zplus1 = z + 1.0
    zmin1 = z - 1.0
    # Not sure if this gives correct answer at corner point (z also appears in qm);
    # should really be caught in code that calls this function
    if np.abs(zplus1) < tiny:
        zplus1 = tiny
    if np.abs(zmin1) < tiny:
        zmin1 = tiny

    qm[0:1] = 0.0
    for m in range(2, order + 1):
        qm[m] = 0.0
        for n in range(1, m // 2 + 1):
            qm[m] = qm[m] + (m - 2 * n + 1) * z ** (m - 2 * n) / (2 * n - 1)

    term1 = 1.0 / zmin1 - 1.0 / zplus1
    term2 = np.log(zmin1 / zplus1)
    wdis[0] = term1
    zterm = complex(1.0, 0.0)
    for m in range(1, order + 1):
        wdis[m] = m * zterm * term2 + z * zterm * term1 + 2.0 * qm[m]
        zterm = zterm * z

    wdis = -wdis / (np.pi * complex(0.0, 1.0) * (z2 - z1))
    return wdis


@numba.njit(nogil=True, cache=True)
def lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2):
    """lapld_int_ho_wdis_d1d2.

    # Near field only
    # Returns integral from d1 to d2 along real axis while strength is still
    # Delta^order from -1 to +1
    implicit none
    integer, intent(in) :: order
    real(kind=8), intent(in) :: x,y,d1,d2
    complex(kind=8), intent(in) :: z1,z2
    complex(kind=8), dimension(0:order) :: wdis, wdisc
    integer :: n, m
    real(kind=8) :: xp, yp, dc, fac
    complex(kind=8) :: z1p,z2p,bigz1,bigz2
    """
    wdis = np.zeros(order + 1, dtype=np.complex128)

    bigz1 = complex(d1, 0.0)
    bigz2 = complex(d2, 0.0)
    z1p = 0.5 * (z2 - z1) * bigz1 + 0.5 * (z1 + z2)
    z2p = 0.5 * (z2 - z1) * bigz2 + 0.5 * (z1 + z2)
    wdisc = lapld_int_ho_wdis(x, y, z1p, z2p, order)
    dc = (d1 + d2) / (d2 - d1)
    wdis[0 : order + 1] = 0.0
    for n in range(order + 1):
        for m in range(n + 1):
            wdis[n] = wdis[n] + gam[n, m] * dc ** (n - m) * wdisc[m]
        wdis[n] = (0.5 * (d2 - d1)) ** n * wdis[n]
    return wdis


# Fp function


@numba.njit(nogil=True, cache=True)
def Fp(x, y, z1, z2, biga, order, d1, d2, a, b, nt):
    tol = 1e-12
    zeta = (2 * complex(x, y) - (z1 + z2)) / (z2 - z1) / biga
    zetabar = np.conj(zeta)
    zminzbar = np.zeros(nt + 1, dtype=np.complex128)
    zminzbar[0] = 1
    for n in range(1, nt + 1):
        zminzbar[n] = zminzbar[n - 1] * (zeta - zetabar)

    eta = np.zeros((nt + 1, nt + 1), dtype=np.complex128)  # lower triangular
    etabar = np.zeros((nt + 1, nt + 1), dtype=np.complex128)
    for n in range(nt + 1):
        for m in range(0, n + 1):
            eta[n, m] = binom[n, m] * zminzbar[n - m]
            etabar[n, m] = np.conj(eta[n, m])

    atil = np.zeros(2 * nt + 1, dtype=np.complex128)
    btil = np.zeros(2 * nt + 1, dtype=np.complex128)
    ctil = np.zeros(2 * nt + 1, dtype=np.complex128)
    for n in range(2 * nt + 1):
        for m in range(max(0, n - nt), int(n / 2) + 1):
            atil[n] = atil[n] + a[n - m] * eta[n - m, m]
            btil[n] = btil[n] + b[n - m] * eta[n - m, m]
            ctil[n] = ctil[n] + a[n - m] * etabar[n - m, m]

    d1minzeta = d1 / biga - zeta
    d2minzeta = d2 / biga - zeta
    if np.abs(d1minzeta) < tol:
        d1minzeta = d1minzeta + complex(tol, 0)
    if np.abs(d2minzeta) < tol:
        d2minzeta = d2minzeta + complex(tol, 0)
    log1 = np.log(d1minzeta)
    log2 = np.log(d2minzeta)

    alpha = np.zeros(2 * nt + order + 1, dtype=np.complex128)
    beta = np.zeros(2 * nt + order + 1, dtype=np.complex128)
    gamma = np.zeros(2 * nt + order + 1, dtype=np.complex128)

    omega = np.zeros(order + 1, dtype=np.complex128)

    for p in range(order + 1):
        alpha[0 : 2 * nt + p + 1] = 0
        beta[0 : 2 * nt + p + 1] = 0
        gamma[0 : 2 * nt + p + 1] = 0

        d = np.zeros(p + 1, dtype=np.complex128)
        dbar = np.zeros(p + 1, dtype=np.complex128)
        for m in range(p + 1):
            d[m] = biga**p * binom[p, m] * zeta ** (p - m)
            dbar[m] = np.conj(d[m])
        for n in range(2 * nt + p + 1):
            for m in range(max(0, n - 2 * nt), min(p, n) + 1):
                alpha[n] = alpha[n] + d[m] * atil[n - m]
                beta[n] = beta[n] + d[m] * btil[n - m]
                gamma[n] = gamma[n] + dbar[m] * ctil[n - m]

        term1 = 1
        term2 = 1
        for n in range(2 * nt + p + 1):
            term1 = term1 * d1minzeta
            term2 = term2 * d2minzeta
            omega[p] = omega[p] + (
                alpha[n] * log2 - alpha[n] / (n + 1) + beta[n]
            ) * term2 / (n + 1)
            omega[p] = omega[p] - (
                alpha[n] * log1 - alpha[n] / (n + 1) + beta[n]
            ) * term1 / (n + 1)
            omega[p] = omega[p] + (
                gamma[n] * np.conj(log2) - gamma[n] / (n + 1)
            ) * np.conj(term2) / (n + 1)
            omega[p] = omega[p] - (
                gamma[n] * np.conj(log1) - gamma[n] / (n + 1)
            ) * np.conj(term1) / (n + 1)

    return biga * omega


# Bessel line elements


@numba.njit(nogil=True, cache=True)
def bessells_int_ho_new(x, y, z1, z2, lab, order, d1, d2, nt=20):
    """
    Docs.

    To come here
    """
    L = np.abs(z2 - z1)
    ang = np.arctan2(lab.imag, lab.real)
    biga = 2 * np.abs(lab) / L

    exprange = np.exp(-complex(0, 2) * ang * nrange)
    ahat = a * exprange
    bhat = (b - a * complex(0, 2) * ang) * exprange

    omega = Fp(x, y, z1, z2, biga, order, d1, d2, ahat, bhat, nt)
    return -L / (4 * np.pi) * omega


@numba.njit(nogil=True, cache=True)
def bessells_int_ho_qxqy_new(x, y, z1, z2, lab, order, d1, d2):
    """
    Docs.

    To come here
    """
    nt = 20  # number of terms in series is nt + 1
    bigz = (2 * complex(x, y) - (z1 + z2)) / (z2 - z1)
    bigx = bigz.real
    bigy = bigz.imag
    L = np.abs(z2 - z1)
    ang = np.arctan2(lab.imag, lab.real)
    angz = np.arctan2((z2 - z1).imag, (z2 - z1).real)
    biglab = 2 * lab / L
    biga = np.abs(biglab)

    exprange = np.exp(-complex(0, 2) * ang * nrange)
    ahat = a * exprange
    bhat = (b - a * complex(0, 2) * ang) * exprange

    atil = 2 * nrange[1:] * ahat[1:]
    btil = 2 * nrange[1:] * bhat[1:] + 2 * ahat[1:]

    omega = Fp(x, y, z1, z2, biga, order + 1, d1, d2, atil, btil, nt - 1)
    omegalap = lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    term1 = 1 / (2 * np.pi * biga**2) * bigx * omega[:-1]
    term2 = -1 / (2 * np.pi * biga**2) * omega[1:]
    term3 = 2 * ahat[0] * omegalap.imag
    qx = term1 + term2 + term3
    term1 = 1 / (2 * np.pi * biga**2) * bigy * omega[:-1]
    term3 = 2 * ahat[0] * omegalap.real
    qy = term1 + term3

    qxqy = np.zeros(2 * order + 2, dtype=np.complex128)
    qxqy[: order + 1] = qx * np.cos(angz) - qy * np.sin(angz)
    qxqy[order + 1 :] = qx * np.sin(angz) + qy * np.cos(angz)
    return qxqy


@numba.njit(nogil=True, cache=True)
def besselld_int_ho_new(x, y, z1, z2, lab, order, d1, d2):
    """
    Docs.

    To come here
    """
    nt = 20  # number of terms in series is nt + 1
    bigz = (2 * complex(x, y) - (z1 + z2)) / (z2 - z1)
    bigy = bigz.imag
    L = np.abs(z2 - z1)
    ang = np.arctan2(lab.imag, lab.real)
    biglab = 2 * lab / L
    biga = np.abs(biglab)

    exprange = np.exp(-complex(0, 2) * ang * nrange)
    ahat = a1 * exprange
    bhat = (b1 - a1 * complex(0, 2) * ang) * exprange

    omega = Fp(x, y, z1, z2, biga, order, d1, d2, ahat, bhat, nt)

    rv = (
        bigy / (2.0 * np.pi * biglab**2) * omega
        + lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2).real
    )

    return rv


@numba.njit(nogil=True, cache=True)
def besselld_int_ho_qxqy_new(x, y, z1, z2, lab, order, d1, d2):
    """
    Docs.

    To come here
    """
    nt = 20  # number of terms in series is nt + 1
    bigz = (2 * complex(x, y) - (z1 + z2)) / (z2 - z1)
    bigx = bigz.real
    bigy = bigz.imag
    L = np.abs(z2 - z1)
    ang = np.arctan2(lab.imag, lab.real)
    angz = np.arctan2((z2 - z1).imag, (z2 - z1).real)
    biglab = 2 * lab / L
    biga = np.abs(biglab)

    exprange = np.exp(-complex(0, 2) * ang * nrange)
    ahat = a1 * exprange
    bhat = (b1 - a1 * complex(0, 2) * ang) * exprange

    atil = 2 * nrange[1:] * ahat[1:]
    btil = 2 * nrange[1:] * bhat[1:] + 2 * ahat[1:]

    omega_pot = Fp(x, y, z1, z2, biga, order, d1, d2, ahat, bhat, nt)
    omega = Fp(x, y, z1, z2, biga, order + 1, d1, d2, atil, btil, nt - 1)
    omegalap = lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    wlap = lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2)

    term1 = bigx / (2 * np.pi * biga**2) * omega[:-1]
    term2 = -1 / (2 * np.pi * biga**2) * omega[1:]
    term3 = 2 * ahat[0] * omegalap.imag
    qx = -2 * bigy / (L * biglab**2) * (term1 + term2 + term3)  # + wlap.real

    term1 = 1 / (2.0 * np.pi * biglab**2) * 2 / L * omega_pot
    term2 = bigy / (2 * np.pi * biga**2) * omega[:-1]
    term3 = 2 * ahat[0] * omegalap.real
    qy = -term1 - 2 * bigy / (L * biglab**2) * (term2 + term3)  # - wlap.imag

    qxqy = np.zeros(2 * order + 2, dtype=np.complex128)
    qxqy[: order + 1] = qx * np.cos(angz) - qy * np.sin(angz) + wlap.real
    qxqy[order + 1 :] = qx * np.sin(angz) + qy * np.cos(angz) - wlap.imag
    return qxqy
