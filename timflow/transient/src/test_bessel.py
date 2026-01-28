import numpy as np
from ttim.bessel import bessel  # fortran compiled funcs

from ttim.besselnumba import besselnumba  # numba funcs

bessel.initialize()


def test_variables():
    list_of_vars = ["a", "a1", "afar", "b", "b1", "gam", "nrange", "tiny", "wg", "xg"]

    result = []
    for var in list_of_vars:
        numba_var = getattr(besselnumba, var)
        frtrn_var = getattr(bessel, var)
        # print(var, np.allclose(numba_var, frtrn_var))
        result.append(np.allclose(numba_var, frtrn_var))
    assert np.all(result), "Not all variables equal!"
    return result


def test_besselk0far():
    z = 1 + 1j
    Nt = 11
    a = bessel.besselk0far(z, Nt)
    b = besselnumba.besselk0far(z, Nt)
    assert a == b, "not equal"
    return a, b


def test_besselk0near():
    z = 1 + 1j
    Nt = 17
    a = bessel.besselk0near(z, Nt)
    b = besselnumba.besselk0near(z, Nt)
    assert a == b, "not equal"
    return a, b


def test_besselk1near():
    z = 1 + 1j
    Nt = 20
    a = bessel.besselk1near(z, Nt)
    b = besselnumba.besselk1near(z, Nt)
    assert a == b, "not equal"
    return a, b


def test_besselk0cheb():
    z = 1 + 1j
    Nt = 6
    a = bessel.besselk0cheb(z, Nt)
    b = besselnumba.besselk0cheb(z, Nt)
    assert a == b, "not equal"
    return a, b


def test_besselk1cheb():
    z = 1 + 1j
    Nt = 6
    a = bessel.besselk1cheb(z, Nt)
    b = besselnumba.besselk1cheb(z, Nt)
    assert a == b, "not equal"
    return a, b


def test_besselk0():
    x = 10.0
    y = 10.0
    lab = 100.0
    a = bessel.besselk0(x, y, lab)
    b = besselnumba.besselk0(x, y, lab)
    assert a == b, "not equal"
    return a, b


def test_besselk1():
    x = 10.0
    y = 10.0
    lab = 100.0
    a = bessel.besselk1(x, y, lab)
    b = besselnumba.besselk1(x, y, lab)
    assert a == b, "not equal"
    return a, b


def test_k0bessel():
    z = 1.0 + 1.0j
    a = bessel.k0bessel(z)
    b = besselnumba.k0bessel(z)
    assert a == b, "not equal"
    return a, b


def test_besselk0v():
    x = 10.0
    y = 10.0
    lab = np.array([100.0])
    nlab = 1
    omega = np.zeros(1, dtype=np.complex128)
    bessel.besselk0v(x, y, lab, omega=omega)
    b = besselnumba.besselk0v(x, y, lab, nlab, np.zeros(1, dtype=np.complex128))
    assert omega == b, "not equal"
    return omega, b


def test_k0besselv():
    z = np.array([1 + 1j])
    nlab = 1
    omega = np.zeros(1, dtype=np.complex128)
    bessel.k0besselv(z, omega=omega)
    b = besselnumba.k0besselv(z, nlab, np.zeros(1, dtype=np.complex128))
    assert omega == b, "not equal"
    return omega, b


# def test_besselcheb():
#    z = 1. + 1.j
#    Nt = 6
#    a = bessel.besselcheb(z, Nt)
#    b = besselnumba.besselcheb(z, Nt)
#    assert a == b, "not equal"
#    return a, b


# def test_ucheb():
#    a = 1.
#    c = 1.
#    z = 1.+1.j
#    n0 = 1.
#    a = bessel.ucheb(a, c, z, n0)
#    b = besselnumba.ucheb(a, c, z, n0)
#    assert np.allclose(a, b), "not equal"
#    return a, b


# def test_besselk0complex():
#    x = 10.
#    y = 10.
#    a = bessel.besselk0complex(x, y)
#    b = besselnumba.besselk0complex(x, y)
#    assert np.allclose(a, b), "not equal"
#    return a, b


def test_lapls_int_ho():
    x = 10.0
    y = 10.0
    z1 = 1.0 + 1.0j
    z2 = 2.0 + 2.0j
    order = 1
    a = bessel.lapls_int_ho(x, y, z1, z2, order)
    b = besselnumba.lapls_int_ho(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsreal():
    x = 5.0
    y = 5.0
    x1 = 0.0
    y1 = 0.0
    x2 = 10.0
    y2 = 10.0
    lab = 100.0
    a = bessel.bessellsreal(x, y, x1, y1, x2, y2, lab)
    b = besselnumba.bessellsreal(x, y, x1, y1, x2, y2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsrealho():
    x = 5.0
    y = 5.0
    x1 = 0.0
    y1 = 0.0
    x2 = 10.0
    y2 = 10.0
    lab = 100.0
    order = 1
    a = bessel.bessellsrealho(x, y, x1, y1, x2, y2, lab, order)
    b = besselnumba.bessellsrealho(x, y, x1, y1, x2, y2, lab, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_int():
    x = 5.0
    y = 5.0
    lab = 100.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    a = bessel.bessells_int(x, y, z1, z2, lab)
    b = besselnumba.bessells_int(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_int_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_int_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_gauss():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    a = bessel.bessells_gauss(x, y, z1, z2, lab)
    b = besselnumba.bessells_gauss(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_gauss_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    a = bessel.bessells_gauss(x, y, z1, z2, lab)
    b = besselnumba.bessells_gauss(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_gauss_ho_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_gauss_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    a = bessel.bessells_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    b = besselnumba.bessells_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_gauss_ho_qxqy_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    # assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    a = bessel.bessells(x, y, z1, z2, lab, order, d1in, d2in)
    b = besselnumba.bessells(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsv():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    nlab = 1
    a = bessel.bessellsv(x, y, z1, z2, lab, order, R, nlab)
    b = besselnumba.bessellsv(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    nlab = 1
    a = bessel.bessellsv2(x, y, z1, z2, lab, order, R, nlab)
    b = besselnumba.bessellsv2(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsqxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    a = bessel.bessellsqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    b = besselnumba.bessellsqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsqxqyv():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    nlab = 1
    a = bessel.bessellsqxqyv(x, y, z1, z2, lab, order, R, nlab)
    b = besselnumba.bessellsqxqyv(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsqxqyv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    nlab = 1
    a = bessel.bessellsqxqyv2(x, y, z1, z2, lab, order, R, nlab)
    b = besselnumba.bessellsqxqyv2(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsuni():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    a = bessel.bessellsuni(x, y, z1, z2, lab)
    b = besselnumba.bessellsuni(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessellsuniv():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    nlab = 1
    a = bessel.bessellsuniv(x, y, z1, z2, lab)
    b = besselnumba.bessellsuniv(x, y, z1, z2, lab, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapld_int_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    a = bessel.lapld_int_ho(x, y, z1, z2, order)
    b = besselnumba.lapld_int_ho(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapld_int_ho_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    b = besselnumba.lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapld_int_ho_wdis():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    a = bessel.lapld_int_ho_wdis(x, y, z1, z2, order)
    b = besselnumba.lapld_int_ho_wdis(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_lapld_int_ho_wdis_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2)
    b = besselnumba.lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld_int_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld_gauss_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    a = bessel.besselld_gauss_ho(x, y, z1, z2, lab, order)
    b = besselnumba.besselld_gauss_ho(x, y, z1, z2, lab, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld_gauss_ho_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    a = bessel.besselld(x, y, z1, z2, lab, order, d1in, d2in)
    b = besselnumba.besselld(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselldv():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    nlab = 1
    R = 1.0
    a = bessel.besselldv(x, y, z1, z2, lab, order, R)
    b = besselnumba.besselldv(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselldv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    nlab = 1
    R = 1.0
    a = bessel.besselldv2(x, y, z1, z2, lab, order, R)
    b = besselnumba.besselldv2(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselldpart():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.besselldpart(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.besselldpart(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld_int_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld_gauss_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    a = bessel.besselld_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    b = besselnumba.besselld_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselld_gauss_ho_qxqy_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    a = bessel.besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    b = besselnumba.besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselldqxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    a = bessel.besselldqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    b = besselnumba.besselldqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselldqxqyv():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    nlab = 1
    R = 1.0
    a = bessel.besselldqxqyv(x, y, z1, z2, lab, order, R)
    b = besselnumba.besselldqxqyv(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_besselldqxqyv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    nlab = 1
    a = bessel.besselldqxqyv2(x, y, z1, z2, lab, order, R)
    b = besselnumba.besselldqxqyv2(x, y, z1, z2, lab, order, R, nlab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_bessells_circcheck():
    x = 5.0
    y = 5.0
    z1in = 1.0 + 1.0j
    z2in = 5.0 + 5.0j
    lab = 100.0
    a = bessel.bessells_circcheck(x, y, z1in, z2in, lab)
    b = besselnumba.bessells_circcheck(x, y, z1in, z2in, lab)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_circle_line_intersection():
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    zc = 2.0 + 2.0j
    R = 10.0
    # xouta = 0.0
    # youta = 0.0
    # xoutb = 1.0
    # youtb = 1.0
    # N = 0
    xyn = bessel.circle_line_intersection_func(z1, z2, zc, R)
    a = (xyn[0], xyn[1], xyn[2], xyn[3], int(xyn[4]))
    b = besselnumba.circle_line_intersection(z1, z2, zc, R)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_find_d1d2():
    z1 = -1 - 2j
    z2 = 2 + 1j
    zc = 2 + 0.5j
    R = 2.0
    # d1 = 0.0
    # d2 = 0.0
    a = bessel.find_d1d2_func(z1, z2, zc, R)
    b = besselnumba.find_d1d2(z1, z2, zc, R)
    assert np.allclose(a, b), "not equal"
    return a, b


def test_isinside():
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    zc = 2.0 + 2.0j
    R = 10.0
    a = bessel.isinside(z1, z2, zc, R)
    b = besselnumba.isinside(z1, z2, zc, R)
    assert np.allclose(a, b), "not equal"
    return a, b


if __name__ == "__main__":
    t0 = test_variables()
    t1 = test_besselk0far()
    t2 = test_besselk0near()
    t3 = test_besselk1near()
    t4 = test_besselk0cheb()
    t5 = test_besselk1cheb()
    t6 = test_besselk0()
    t7 = test_besselk1()
    t8 = test_k0bessel()
    t9 = test_besselk0v()
    t10 = test_k0besselv()
    # t11 = test_besselcheb()
    # t12 = test_ucheb()  # fails
    # t13 = test_besselk0complex()
    t14 = test_lapls_int_ho()
    t15 = test_bessellsreal()
    t16 = test_bessellsrealho()
    t17 = test_bessells_int()
    t18 = test_bessells_int_ho()
    t19 = test_bessells_int_ho_qxqy()
    t20 = test_bessells_gauss()
    t21 = test_bessells_gauss_ho()
    t22 = test_bessells_gauss_ho_d1d2()
    t23 = test_bessells_gauss_ho_qxqy()
    t24 = test_bessells_gauss_ho_qxqy_d1d2()
    t25 = test_bessells()
    t26 = test_bessellsv()
    t27 = test_bessellsv2()
    t28 = test_bessellsqxqy()
    t29 = test_bessellsqxqyv()
    t30 = test_bessellsqxqyv2()
    t31 = test_bessellsuni()
    t32 = test_bessellsuniv()
    t33 = test_lapld_int_ho()
    t34 = test_lapld_int_ho_d1d2()
    t35 = test_lapld_int_ho_wdis()
    t36 = test_lapld_int_ho_wdis_d1d2()
    t37 = test_besselld_int_ho()
    t38 = test_besselld_gauss_ho()
    t39 = test_besselld_gauss_ho_d1d2()
    t40 = test_besselld()
    t41 = test_besselldv()
    t42 = test_besselldv2()
    t43 = test_besselldpart()  # fails with numba
    t44 = test_besselld_int_ho_qxqy()  # fails with numba
    t45 = test_besselld_gauss_ho_qxqy()  # fails with numba
    t46 = test_besselld_gauss_ho_qxqy_d1d2()  # fails with numba
    t47 = test_besselldqxqy()  # fails with numba
    t48 = test_besselldqxqyv()  # fails with numba
    t49 = test_besselldqxqyv2()  # fails with numba
    t50 = test_bessells_circcheck()
    t51 = test_circle_line_intersection()  # fails
    t52 = test_find_d1d2()  # fails
    t53 = test_isinside()
