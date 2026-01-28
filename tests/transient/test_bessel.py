import numpy as np

from ttim import besselnumba
from ttim.besselnumba import (
    besselld_gauss_ho_d1d2,
    besselld_gauss_ho_qxqy_d1d2,
    besselld_int_ho,
    besselld_int_ho_qxqy,
    bessells_gauss_ho_d1d2,
    bessells_gauss_ho_qxqy_d1d2,
    bessells_int_ho,
    bessells_int_ho_qxqy,
)


def test_variables():
    values = {
        "a": np.array(
            [
                -5.0000000000000000e-01,
                -1.2500000000000000e-01,
                -7.8125000000000000e-03,
                -2.1701388888888888e-04,
                -3.3908420138888887e-06,
                -3.3908420138888889e-08,
                -2.3547513985339507e-10,
                -1.2014037747622198e-12,
                -4.6929834951649210e-15,
                -1.4484516960385559e-17,
                -3.6211292400963895e-20,
                -7.4816719836702267e-23,
                -1.2989013860538587e-25,
                -1.9214517545175424e-28,
                -2.4508313195376818e-31,
                -2.7231459105974243e-34,
                -2.6593221783177971e-37,
                -2.3004517113475752e-40,
                -1.7750399007311537e-43,
                -1.2292520088165885e-46,
                -7.6828250551036773e-50,
            ],
            dtype=float,
        ),
        "a1": np.array(
            [
                2.5000000000000000e-01,
                3.1250000000000000e-02,
                1.3020833333333333e-03,
                2.7126736111111110e-05,
                3.3908420138888890e-07,
                2.8257016782407406e-09,
                1.6819652846671074e-11,
                7.5087735922638736e-14,
                2.6072130528694005e-16,
                7.2422584801927787e-19,
                1.6459678364074498e-21,
                3.1173633265292609e-24,
                4.9957745617456107e-27,
                6.8623276947055087e-30,
                8.1694377317922725e-33,
                8.5098309706169508e-36,
                7.8215358185817554e-39,
                6.3901436426321534e-42,
                4.6711576335030367e-45,
                3.0731300220414711e-48,
                1.8292440607389709e-51,
            ],
            dtype=float,
        ),
        "afar": np.array(
            [
                1.2533141373155001e00,
                -1.5666426716443752e-01,
                8.8123650279996107e-02,
                -9.1795469041662622e-02,
                1.4056181197004589e-01,
                -2.8463766923934292e-01,
                7.1752412454084369e-01,
                -2.1653853044179034e00,
                7.6126827108441919e00,
                -3.0556462547694050e01,
                1.3788603724646941e02,
                -6.9099707301923877e02,
                3.8076817877830972e03,
                -2.2882703051581113e04,
                1.4894187968395205e05,
                -1.0438343401183640e06,
                7.8369125066699041e06,
                -6.2752924410025924e07,
                5.3383564168251222e08,
                -4.8080328517326269e09,
                4.5706362296783279e10,
            ],
            dtype=float,
        ),
        "b": np.array(
            [
                1.1593151565841242e-01,
                2.7898287891460311e-01,
                2.5248929932162694e-02,
                8.4603509070822292e-04,
                1.4914719299260427e-05,
                1.6271056104815986e-07,
                1.2084261650077971e-09,
                6.5086978387473547e-12,
                2.6597846806398086e-14,
                8.5310901319585936e-17,
                2.2051951177915761e-19,
                4.6922186596030460e-22,
                8.3626965150420416e-25,
                1.2666460795135899e-27,
                1.6506318753729742e-30,
                1.8703440292223820e-33,
                1.8597493682664549e-36,
                1.6358438681148632e-39,
                1.2819478724424317e-42,
                9.0071502976208028e-46,
                5.7062971865640375e-49,
            ],
            dtype=float,
        ),
        "b1": np.array(
            [
                -3.0796575782920621e-01,
                -8.5370719728650776e-02,
                -4.6421827664715606e-03,
                -1.1253607036630565e-04,
                -1.5592887702038207e-06,
                -1.4030163700386776e-08,
                -8.8718962192938521e-11,
                -4.1617958191203950e-13,
                -1.5066271898317757e-15,
                -4.3379676507812249e-18,
                -1.0173247611453297e-20,
                -1.9810691358890129e-23,
                -3.2548507716449826e-26,
                -4.5727526246535748e-29,
                -5.5565691694551969e-32,
                -5.8980115348862992e-35,
                -5.5158601173635832e-38,
                -4.5795115427781316e-41,
                -3.3981320729195735e-44,
                -2.2671532245154079e-47,
                -1.3673528732806709e-50,
            ],
            dtype=float,
        ),
        "wg": np.array(
            [
                0.101228536290378,
                0.22238103445338,
                0.31370664587789,
                0.36268378337836,
                0.3626837833836,
                0.313706636428833,
                0.22238103445338,
                0.10122853629038,
            ],
            dtype=float,
        ),
        "xg": np.array(
            [
                -0.960289856497536,
                -0.796666477413626,
                -0.525532409916329,
                -0.18343464249565,
                0.18343464249565,
                0.525532409916329,
                0.796666477413626,
                0.960289856497536,
            ],
            dtype=float,
        ),
    }
    for var, vals in values.items():
        numba_var = getattr(besselnumba, var)
        # print(var, np.allclose(numba_var, frtrn_var))
        assert np.allclose(numba_var, vals), f"Variable {var} not equal"


def test_besselk0near():
    z = 1 + 1j
    Nt = 17
    # a = bessel.besselk0near(z, Nt)
    a = 0.08019772694651779 - 0.35727745928533017j
    b = besselnumba.besselk0near(z, Nt)
    assert a == b, "not equal"


def test_besselk1near():
    z = 1 + 1j
    Nt = 20
    # a = bessel.besselk1near(z, Nt)
    a = 0.02456830552374039 - 0.4597194738011894j
    b = besselnumba.besselk1near(z, Nt)
    assert a == b, "not equal"


def test_besselk0cheb():
    z = 1 + 1j
    Nt = 6
    # a = bessel.besselk0cheb(z, Nt)
    a = 0.08019774438980808 - 0.3572774444355785j
    b = besselnumba.besselk0cheb(z, Nt)
    assert a == b, "not equal"


def test_besselk1cheb():
    z = 1 + 1j
    Nt = 6
    # a = bessel.besselk1cheb(z, Nt)
    a = 0.02456827435427396 - 0.4597194926092556j
    b = besselnumba.besselk1cheb(z, Nt)
    assert np.allclose(a, b), "not equal"


def test_besselk0():
    x = 10.0
    y = 10.0
    lab = 100.0
    # a = bessel.besselk0(x, y, lab)
    a = 2.0873250716727094 + 0j
    b = besselnumba.besselk0(x, y, lab)
    assert a == b, "not equal"


def test_besselk1():
    x = 10.0
    y = 10.0
    lab = 100.0
    # a = bessel.besselk1(x, y, lab)
    a = 6.888616183848397 + 0j
    b = besselnumba.besselk1(x, y, lab)
    assert a == b, "not equal"


def test_bessells_int():
    x = 5.0
    y = 5.0
    lab = 100.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    # a = bessel.bessells_int(x, y, z1, z2, lab)
    a = -3.5917095941591426 - 0j
    b = besselnumba.bessells_int(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"


def test_bessells_int_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    a = np.array([3.5917095941591426 - 0.0j, 0.449669310313802 - 0.0j], dtype=complex)
    b = besselnumba.bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_bessells_int_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    a = np.array(
        [
            3.02251936780004 + 0.0j,
            2.79789916586647 + 0.0j,
            2.3154125866134923 + 0.0j,
            2.090792384679922 + 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_bessells_gauss():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    # a = bessel.bessells_gauss(x, y, z1, z2, lab)
    a = -3.583821946423638 - 0j
    b = besselnumba.bessells_gauss(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"


def test_bessells_gauss_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    # a = bessel.bessells_gauss(x, y, z1, z2, lab)
    a = -3.583821946423638 - 0j
    b = besselnumba.bessells_gauss(x, y, z1, z2, lab)
    assert np.allclose(a, b), "not equal"


def test_bessells_gauss_ho_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    a = np.array([-3.5838219513840075 + 0.0j, -0.4416936603433489 + 0.0j], dtype=complex)
    b = besselnumba.bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_bessells_gauss_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    # a = bessel.bessells_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    a = np.array(
        [
            -0.6113736236154267 + 0.0j,
            -0.3867534224134755 + 0.0j,
            -0.6113736236154266 - 0.0j,
            -0.3867534224134755 - 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.bessells_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    assert np.allclose(a, b), "not equal"


def test_bessells_gauss_ho_qxqy_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    a = np.array(
        [
            -0.6113736251610367 + 0.0j,
            -0.3867534239558352 + 0.0j,
            -0.6113736251610368 + 0.0j,
            -0.3867534239558352 + 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_bessells():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    # a = bessel.bessells(x, y, z1, z2, lab, order, d1in, d2in)
    a = np.array([3.5917095941591426 - 0.0j, 0.449669310313802 - 0.0j], dtype=complex)
    b = besselnumba.bessells(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"


def test_bessellsqxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    # a = bessel.bessellsqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    a = np.array(
        [
            3.02251936780004 + 0.0j,
            2.79789916586647 + 0.0j,
            2.3154125866134923 + 0.0j,
            2.090792384679922 + 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.bessellsqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"


def test_bessellsqxqyv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    # nlab = 1
    # a = bessel.bessellsqxqyv2(x, y, z1, z2, lab, order, R, nlab)
    a = np.array(
        [
            [-2.6689659772067666 + 0.0j],
            [-2.444345775273196 + 0.0j],
            [-2.668965977206766 - 0.0j],
            [-2.444345775273196 - 0.0j],
        ],
        dtype=complex,
    )
    b = besselnumba.bessellsqxqyv2(x, y, z1, z2, lab, order, R)
    assert np.allclose(a, b), "not equal"


# def test_bessellsuni():
#     x = 5.0
#     y = 5.0
#     z1 = 1.0 + 1.0j
#     z2 = 5.0 + 5.0j
#     lab = 100.0
#     # a = bessel.bessellsuni(x, y, z1, z2, lab)
#     a = -3.5917095941591426 - 0j
#     b = besselnumba.bessellsuni(x, y, z1, z2, lab)
#     assert np.allclose(a, b), "not equal"


# def test_bessellsuniv():
#     x = 5.0
#     y = 5.0
#     z1 = 1.0 + 1.0j
#     z2 = 5.0 + 5.0j
#     lab = np.array([100.0])
#     nlab = 1
#     # a = bessel.bessellsuniv(x, y, z1, z2, lab)
#     a = -3.5917095941591426 - 0j
#     b = besselnumba.bessellsuniv(x, y, z1, z2, lab, nlab)
#     assert np.allclose(a, b), "not equal"


def test_lapld_int_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    # a = bessel.lapld_int_ho(x, y, z1, z2, order)
    a = np.array([0.0 + 3.7749957944734644j, 0.0 + 3.456685908289674j], dtype=complex)
    b = besselnumba.lapld_int_ho(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"


def test_lapld_int_ho_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    a = np.array([0.5 - 3.7749957944734644j, 0.5 - 3.456685908289674j], dtype=complex)
    b = besselnumba.lapld_int_ho_d1d2(x, y, z1, z2, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_lapld_int_ho_wdis():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    # a = bessel.lapld_int_ho_wdis(x, y, z1, z2, order)
    a = np.array(
        [
            3.9788735770984399e08 + 3.9788735770984399e08j,
            3.9788735676609504e08 + 3.9788735676609504e08j,
        ],
        dtype=complex,
    )
    b = besselnumba.lapld_int_ho_wdis(x, y, z1, z2, order)
    assert np.allclose(a, b), "not equal"


def test_lapld_int_ho_wdis_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2)
    a = np.array(
        [
            3.9788735774963272e08 + 3.9788735774963272e08j,
            3.9788735856838167e08 + 3.9788735881838167e08j,
        ],
        dtype=complex,
    )
    b = besselnumba.lapld_int_ho_wdis_d1d2(x, y, z1, z2, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_besselld_int_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
    a = np.array([0.5 + 0.0j, 0.5 + 0.0j], dtype=complex)
    b = besselnumba.besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_besselld_gauss_ho():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    # a = bessel.besselld_gauss_ho(x, y, z1, z2, lab, order)
    a = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    b = besselnumba.besselld_gauss_ho(x, y, z1, z2, lab, order)
    assert np.allclose(a, b), "not equal"


def test_besselld_gauss_ho_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    a = np.array([0.0 + 0.0j, -0.0 + 0.0j], dtype=complex)
    b = besselnumba.besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_besselld():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    # a = bessel.besselld(x, y, z1, z2, lab, order, d1in, d2in)
    a = np.array([0.5 + 0.0j, 0.5 + 0.0j])
    b = besselnumba.besselld(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"


def test_besselldv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    # nlab = 1
    R = 1.0
    # a = bessel.besselldv2(x, y, z1, z2, lab, order, R)
    a = np.array([[0.0 + 0.0j], [0.0 + 0.0j]], dtype=complex)
    b = besselnumba.besselldv2(x, y, z1, z2, lab, order, R)
    assert np.allclose(a, b), "not equal"


def test_besselldpart():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.besselldpart(x, y, z1, z2, lab, order, d1, d2)
    a = np.array(
        [5.715372856758380e-04 + 0.0j, 6.362529038138779e-05 + 0.0j], dtype=complex
    )
    b = besselnumba.besselldpart(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_besselld_int_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    a = np.array(
        [
            3.978873577497756e08 + 0.0j,
            3.978873585683976e08 + 0.0j,
            -3.978873577497756e08 + 0.0j,
            -3.978873588183976e08 + 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_besselld_gauss_ho_qxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    # a = bessel.besselld_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    a = np.array(
        [
            2.86464636853321 + 0.0j,
            2.64849315098468 + 0.0j,
            -2.8646463685332106 - 0.0j,
            -2.6484931509846805 - 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.besselld_gauss_ho_qxqy(x, y, z1, z2, lab, order)
    assert np.allclose(a, b), "not equal"


def test_besselld_gauss_ho_qxqy_d1d2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1 = 1.0
    d2 = -1.0
    # a = bessel.besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    a = np.array(
        [
            -2.864646370041508 + 0.0j,
            -2.6484931519465227 + 0.0j,
            2.8646463700415077 + 0.0j,
            2.6484931519465222 - 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(a, b), "not equal"


def test_besselldqxqy():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = 100.0
    order = 1
    d1in = 1.0
    d2in = -1.0
    # a = bessel.besselldqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    a = np.array(
        [
            3.978873577497756e08 + 0.0j,
            3.978873585683976e08 + 0.0j,
            -3.978873577497756e08 + 0.0j,
            -3.978873588183976e08 + 0.0j,
        ],
        dtype=complex,
    )
    b = besselnumba.besselldqxqy(x, y, z1, z2, lab, order, d1in, d2in)
    assert np.allclose(a, b), "not equal"


def test_besselldqxqyv2():
    x = 5.0
    y = 5.0
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    lab = np.array([100.0])
    order = 1
    R = 1.0
    # nlab = 1
    # a = bessel.besselldqxqyv2(x, y, z1, z2, lab, order, R)
    a = np.array(
        [
            [3.978873577097011e08 + 0.0j],
            [3.978873567660791e08 + 0.0j],
            [-3.978873577097011e08 + 0.0j],
            [-3.978873567660791e08 + 0.0j],
        ],
        dtype=complex,
    )
    b = besselnumba.besselldqxqyv2(x, y, z1, z2, lab, order, R)
    assert np.allclose(a, b), "not equal"


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
    # xyn = bessel.circle_line_intersection_func(z1, z2, zc, R)
    # a = (xyn[0], xyn[1], xyn[2], xyn[3], int(xyn[4]))
    a = ((1.0 + 1.0j), (5.0 + 5.0j), 2)
    b = besselnumba.circle_line_intersection(z1, z2, zc, R)
    assert np.allclose(a, b), "not equal"


def test_find_d1d2():
    z1 = -1 - 2j
    z2 = 2 + 1j
    zc = 2 + 0.5j
    R = 2.0
    # d1 = 0.0
    # d2 = 0.0
    # a = bessel.find_d1d2_func(z1, z2, zc, R)
    a = (-0.0946273938050036, 1.0)
    b = besselnumba.find_d1d2(z1, z2, zc, R)
    assert np.allclose(a, b), "not equal"


def test_isinside():
    z1 = 1.0 + 1.0j
    z2 = 5.0 + 5.0j
    zc = 2.0 + 2.0j
    R = 10.0
    # a = bessel.isinside(z1, z2, zc, R)
    a = 1
    b = besselnumba.isinside(z1, z2, zc, R)
    assert np.allclose(a, b), "not equal"


def test_bessellspotnew():
    x = 2
    y = 3
    z1 = -1 - 2j
    z2 = 2 + 1j
    d1 = -0.5
    d2 = 0.2
    lab = 8.0
    order = 7
    pot = bessells_int_ho(x, y, z1, z2, lab, order, d1, d2).real
    potgauss = bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2).real
    assert np.allclose(pot, potgauss), "not equal"
    lab = 8 + 3j
    order = 3
    pot = bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    potgauss = bessells_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(pot, potgauss), "not equal"


def test_bessellsqxqynew():
    x = 2
    y = 3
    d = 1e-4
    z1 = -1 - 2j
    z2 = 2 + 1j
    d1 = -0.5
    d2 = 0.2
    lab = 8.0
    order = 7
    qxqy = bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2).real
    qxqygauss = bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2).real
    assert np.allclose(qxqy, qxqygauss), "not equal"
    qxqynum = np.zeros(2 * (order + 1))
    pot1 = bessells_int_ho(x + d, y, z1, z2, lab, order, d1, d2).real
    pot2 = bessells_int_ho(x - d, y, z1, z2, lab, order, d1, d2).real
    qxqynum[: order + 1] = (pot2 - pot1) / (2 * d)
    pot1 = bessells_int_ho(x, y + d, z1, z2, lab, order, d1, d2).real
    pot2 = bessells_int_ho(x, y - d, z1, z2, lab, order, d1, d2).real
    qxqynum[order + 1 :] = (pot2 - pot1) / (2 * d)
    assert np.allclose(qxqy, qxqynum), "not equal"
    lab = 8 + 3j
    order = 3
    qxqy = bessells_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    qxqygauss = bessells_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(qxqy, qxqygauss), "not equal"
    qxqynum = np.zeros(2 * (order + 1), dtype="complex")
    pot1 = bessells_int_ho(x + d, y, z1, z2, lab, order, d1, d2)
    pot2 = bessells_int_ho(x - d, y, z1, z2, lab, order, d1, d2)
    qxqynum[: order + 1] = (pot2 - pot1) / (2 * d)
    pot1 = bessells_int_ho(x, y + d, z1, z2, lab, order, d1, d2)
    pot2 = bessells_int_ho(x, y - d, z1, z2, lab, order, d1, d2)
    qxqynum[order + 1 :] = (pot2 - pot1) / (2 * d)
    assert np.allclose(qxqy, qxqynum), "not equal"


def test_besselldpotnew():
    x = 2
    y = 3
    z1 = -1 - 2j
    z2 = 2 + 1j
    d1 = -0.5
    d2 = 0.2
    lab = 8.0
    order = 7
    pot = besselld_int_ho(x, y, z1, z2, lab, order, d1, d2).real
    potgauss = besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2).real
    assert np.allclose(pot, potgauss), "not equal"
    lab = 8 + 3j
    order = 3
    pot = besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
    potgauss = besselld_gauss_ho_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(pot, potgauss), "not equal"


def test_besselldqxqynew():
    x = 2
    y = 3
    d = 1e-4
    z1 = -1 - 2j
    z2 = 2 + 1j
    d1 = -0.5
    d2 = 0.2
    lab = 8.0
    order = 7
    qxqy = besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2).real
    qxqygauss = besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2).real
    assert np.allclose(qxqy, qxqygauss), "not equal"
    qxqynum = np.zeros(2 * (order + 1))
    pot1 = besselld_int_ho(x + d, y, z1, z2, lab, order, d1, d2).real
    pot2 = besselld_int_ho(x - d, y, z1, z2, lab, order, d1, d2).real
    qxqynum[: order + 1] = (pot2 - pot1) / (2 * d)
    pot1 = besselld_int_ho(x, y + d, z1, z2, lab, order, d1, d2).real
    pot2 = besselld_int_ho(x, y - d, z1, z2, lab, order, d1, d2).real
    qxqynum[order + 1 :] = (pot2 - pot1) / (2 * d)
    assert np.allclose(qxqy, qxqynum), "not equal"
    lab = 8 + 3j
    order = 3
    qxqy = besselld_int_ho_qxqy(x, y, z1, z2, lab, order, d1, d2)
    qxqygauss = besselld_gauss_ho_qxqy_d1d2(x, y, z1, z2, lab, order, d1, d2)
    assert np.allclose(qxqy, qxqygauss), "not equal"
    qxqynum = np.zeros(2 * (order + 1), dtype="complex")
    pot1 = besselld_int_ho(x + d, y, z1, z2, lab, order, d1, d2)
    pot2 = besselld_int_ho(x - d, y, z1, z2, lab, order, d1, d2)
    qxqynum[: order + 1] = (pot2 - pot1) / (2 * d)
    pot1 = besselld_int_ho(x, y + d, z1, z2, lab, order, d1, d2)
    pot2 = besselld_int_ho(x, y - d, z1, z2, lab, order, d1, d2)
    qxqynum[order + 1 :] = (pot2 - pot1) / (2 * d)
    assert np.allclose(qxqy, qxqynum), "not equal"


def test_bessellslap():
    x = 2
    y = 3
    d = 1e-3
    z1 = -1 - 2j
    z2 = 2 + 1j
    d1 = -0.5
    d2 = 0.2
    lab = 8.0
    order = 5
    pot0 = bessells_int_ho(x, y, z1, z2, lab, order, d1, d2).real
    pot1 = bessells_int_ho(x + d, y, z1, z2, lab, order, d1, d2).real
    pot2 = bessells_int_ho(x, y + d, z1, z2, lab, order, d1, d2).real
    pot3 = bessells_int_ho(x - d, y, z1, z2, lab, order, d1, d2).real
    pot4 = bessells_int_ho(x, y - d, z1, z2, lab, order, d1, d2).real
    lapnum = (pot1 + pot2 + pot3 + pot4 - 4 * pot0) / (d**2)
    lap = pot0 / (lab**2)
    assert np.allclose(lapnum, lap, atol=1e-6), "not equal"
    lab = 8 + 3j
    order = 3
    pot0 = bessells_int_ho(x, y, z1, z2, lab, order, d1, d2)
    pot1 = bessells_int_ho(x + d, y, z1, z2, lab, order, d1, d2)
    pot2 = bessells_int_ho(x, y + d, z1, z2, lab, order, d1, d2)
    pot3 = bessells_int_ho(x - d, y, z1, z2, lab, order, d1, d2)
    pot4 = bessells_int_ho(x, y - d, z1, z2, lab, order, d1, d2)
    lapnum = (pot1 + pot2 + pot3 + pot4 - 4 * pot0) / (d**2)
    lap = pot0 / (lab**2)
    assert np.allclose(lapnum, lap, atol=1e-6), "not equal"


def test_besselldlap():
    x = 2
    y = 3
    d = 1e-3
    z1 = -1 - 2j
    z2 = 2 + 1j
    d1 = -0.5
    d2 = 0.2
    lab = 8.0
    order = 5
    pot0 = besselld_int_ho(x, y, z1, z2, lab, order, d1, d2).real
    pot1 = besselld_int_ho(x + d, y, z1, z2, lab, order, d1, d2).real
    pot2 = besselld_int_ho(x, y + d, z1, z2, lab, order, d1, d2).real
    pot3 = besselld_int_ho(x - d, y, z1, z2, lab, order, d1, d2).real
    pot4 = besselld_int_ho(x, y - d, z1, z2, lab, order, d1, d2).real
    lapnum = (pot1 + pot2 + pot3 + pot4 - 4 * pot0) / (d**2)
    lap = pot0 / (lab**2)
    assert np.allclose(lapnum, lap, atol=1e-6), "not equal"
    lab = 8 + 3j
    order = 3
    pot0 = besselld_int_ho(x, y, z1, z2, lab, order, d1, d2)
    pot1 = besselld_int_ho(x + d, y, z1, z2, lab, order, d1, d2)
    pot2 = besselld_int_ho(x, y + d, z1, z2, lab, order, d1, d2)
    pot3 = besselld_int_ho(x - d, y, z1, z2, lab, order, d1, d2)
    pot4 = besselld_int_ho(x, y - d, z1, z2, lab, order, d1, d2)
    lapnum = (pot1 + pot2 + pot3 + pot4 - 4 * pot0) / (d**2)
    lap = pot0 / (lab**2)
    assert np.allclose(lapnum, lap, atol=1e-8), "not equal"
