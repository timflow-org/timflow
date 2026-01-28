import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

import ttim

ml = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=1e-4,
    tmax=1e5,
    M=20,
)
w1 = ttim.HeadWell(ml, xw=0, yw=0, rw=0.3, tsandh=[(0, 1)], layers=0)
ml.solve()


def func1(tau, p0, p1, f):
    rv = np.exp(-tau) * np.cos(-p1 / p0 * tau) * f(tau / p0)
    return rv


def func2(tau, p0, p1, f):
    rv = np.exp(-tau) * np.sin(-p1 / p0 * tau) * f(tau / p0)
    return rv


t = np.linspace(100, 1e5, 1000)
Q = w1.strength(t)
func = interp1d(t - 100, Q[0], "cubic")


def funcnew(t):
    print("time:", t)
    if t > 5000:
        print("t too large:", t)
    return func(t)


def quadfunc(p0, p1, func):
    f1 = quad(func1, 0, np.inf, args=(p0, p1, func))[0]
    f2 = quad(func2, 0, np.inf, args=(p0, p1, func))[0]
    return (f1 + 1j * f2) / p0


ml2 = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=10,
    tmax=100,
    M=20,
)
ml2.initialize()
fp = np.zeros(41, "D")
p = ml2.p
for i in range(41):
    p0 = p[i].real
    p1 = p[i].imag
    fp[i] = quadfunc(p0, p1, funcnew)

ml2 = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=10,
    tmax=100,
    M=20,
)
w2 = ttim.WellTest(ml2, xw=0, yw=0, rw=0.3, tsandQ=[(0, 1)], layers=0, fp=fp)
ml2.solve()
