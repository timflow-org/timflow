import numpy as np
from scipy.integrate import quad


def func1(tau, p0, p1, f):
    rv = np.exp(-tau) * np.cos(-p1 / p0 * tau) * f(tau / p0)
    return rv


def func2(tau, p0, p1, f):
    rv = np.exp(-tau) * np.sin(-p1 / p0 * tau) * f(tau / p0)
    return rv


def func(t):
    return np.exp(-t)


p0 = 0.2
p1 = 0.4


def quadfunc():
    f1 = quad(func1, 0, np.inf, args=(p0, p1, func))[0]
    f2 = quad(func2, 0, np.inf, args=(p0, p1, func))[0]
    return (f1 + 1j * f2) / p0


f = quadfunc()


def gausslag1(tau, p0, p1, f):
    rv = np.cos(-p1 / p0 * tau) * f(tau / p0)
    return rv


def gausslag2(tau, p0, p1, f):
    rv = np.sin(-p1 / p0 * tau) * f(tau / p0)
    return rv


def lag(func):
    x, w = np.polynomial.laguerre.laggauss(50)
    g1 = 0.0
    g2 = 0.0
    for i in range(len(x)):
        g1 += w[i] * gausslag1(x[i], p0, p1, func)
        g2 += w[i] * gausslag2(x[i], p0, p1, func)
    return (g1 + 1j * g2) / p0


g = lag(func)
