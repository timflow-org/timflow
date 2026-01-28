import numpy as np

import ttim

ml = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=1e-3,
    tmax=1e3,
    M=20,
)
w1 = ttim.HeadWell(ml, xw=0, yw=0, rw=0.3, tsandh=[(0, 1)], layers=0)
ml.solve()


def gausslag1(tau, p0, p1, f):
    rv = np.cos(-p1 / p0 * tau) * f(tau / p0)
    return rv


def gausslag2(tau, p0, p1, f):
    rv = np.sin(-p1 / p0 * tau) * f(tau / p0)
    return rv


def glag(p0, p1, func):
    x, w = np.polynomial.laguerre.laggauss(50)
    g1 = 0.0
    g2 = 0.0
    for i in range(len(x)):
        g1 += w[i] * gausslag1(x[i], p0, p1, func)
        g2 += w[i] * gausslag2(x[i], p0, p1, func)
    return (g1 + 1j * g2) / p0


# t = np.linspace(100, 1e5, 1000)
# Q = w1.strength(t)
# func = interp1d(t - 100, Q[0], 'cubic')


def func(t):
    return w1.strength(t + 100)[0, 0]


ml2 = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=0.1,
    tmax=100,
    M=20,
)
ml2.initialize()
fp = np.zeros(len(ml2.p), "D")
p = ml2.p
for i in range(len(p)):
    p0 = p[i].real
    p1 = p[i].imag
    fp[i] = glag(p0, p1, func)
# ml2 = ttim.ModelMaq(kaq=[1, 5], z=[3, 2, 1, 0], c=[10], Saq=[0.3, 0.01], Sll=[0.001],
# tmin=10, tmax=100, M=20)
w2 = ttim.WellTest(ml2, xw=0, yw=0, rw=0.3, tsandQ=[(0, 1)], layers=0, fp=fp)
ml2.solve()
