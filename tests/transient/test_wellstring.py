import numpy as np

from timflow import transient as ttim


def test_wellstring_multilayer_common_head_and_total_discharge():
    ml = ttim.ModelMaq(
        kaq=[10, 20],
        z=[20, 10, 8, 0],
        c=[200],
        Saq=1e-4,
        tmin=0.01,
        tmax=100,
    )
    ws = ttim.WellString(
        ml,
        xy=[(0, -10), (0, 10)],
        tsandQ=[(0, 1000)],
        rw=0.3,
        layers=[0, 1],
    )
    ml.solve(silent=True)

    h0 = ws.wlist[0].headinside(7.0)
    h1 = ws.wlist[1].headinside(7.0)

    # All inside heads for all screens in both wells should be equal.
    hall = np.array([h0[0, 0], h0[1, 0], h1[0, 0], h1[1, 0]])
    assert np.allclose(hall, hall[0], rtol=1e-8, atol=1e-10)

    # Total discharge should equal specified Q.
    qsum = ws.discharge(7.0).sum()
    assert np.allclose(qsum, 1000.0, rtol=1e-6, atol=1e-4)


def test_wellstring_multilayer_with_resistance():
    ml = ttim.ModelMaq(
        kaq=[10, 20],
        z=[20, 10, 8, 0],
        c=[200],
        Saq=1e-4,
        tmin=0.01,
        tmax=100,
    )
    ws = ttim.WellString(
        ml,
        xy=[(0, -10), (0, 10)],
        tsandQ=[(0, 1000)],
        rw=0.3,
        res=0.1,
        layers=[0, 1],
    )
    ml.solve(silent=True)

    h0 = ws.wlist[0].headinside(7.0)
    h1 = ws.wlist[1].headinside(7.0)

    # All inside heads for all screens in both wells should be equal.
    hall = np.array([h0[0, 0], h0[1, 0], h1[0, 0], h1[1, 0]])
    assert np.allclose(hall, hall[0], rtol=1e-8, atol=1e-10)

    # Total discharge should equal specified Q.
    qsum = ws.discharge(7.0).sum()
    assert np.allclose(qsum, 1000.0, rtol=1e-6, atol=1e-4)
