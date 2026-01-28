import numpy as np
from scipy.special import exp1

import ttim


def theis(r, t, T, S, Q):
    u = r**2 * S / (4 * T * t)
    h = -Q / (4 * np.pi * T) * exp1(u)
    return h


def theisQr(r, t, T, S, Q):
    u = r**2 * S / (4 * T * t)
    Qr = -Q / (2 * np.pi) * np.exp(-u) / r
    return Qr


T = 500
S = 1e-3
t = np.logspace(-4, 1, 10)
r = 30
Q = 788

h1 = theis(r, t, T, S, Q)

ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=1e-4, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q)])
ml.solve()
h2 = ml.head(r, 0, t)[0]

assert np.allclose(h1, h2, atol=1e-4), "h1 and h2 not all close Theis well 1"

# turn Theis off
t = np.logspace(-1, 1, 10)

h1 = theis(r, t, T, S, Q)
h1[t > 5] -= theis(r, t[t > 5] - 5, T, S, Q)

ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=1e-4, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q), (5, 0)])
ml.solve()
h2 = ml.head(r, 0, t)[0]

assert np.allclose(h1, h2, atol=1e-4), "h1 and h2 not all close Theis well 2"

# test nan values for Theis well 1
t = np.array([0.08, 0.09, 0.1, 1, 5, 9])
h1 = theis(r, t, T, S, Q)

tmin = 0.1
ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=tmin, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q)])
ml.solve()
h2 = ml.head(r, 0, t)[0]

a = np.isnan(h2)
b = t < tmin

assert np.all(a == b), "nans not in the right spot for tmin"
assert np.allclose(h1[~b], h2[~b], atol=1e-4)

# test nan values for Theis well 2

t = np.array([0.08, 0.09, 0.1, 1, 5, 5.03, 9])
h1 = theis(r, t, T, S, Q)
h1[t > 5] -= theis(r, t[t > 5] - 5, T, S, Q)

tmin = 0.1
ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=tmin, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q), (5, 0)])
ml.solve()
h2 = ml.head(r, 0, t)[0]

a = np.isnan(h2)
b = (t < tmin) | ((t > 5) & (t < 5.1))

assert np.all(a == b), "nans not in the right spot for tmin"
assert np.allclose(h1[~b], h2[~b], atol=1e-4)

### test for Qr ########################

Qr1 = theisQr(r, t, T, S, Q)

ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=1e-4, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q)])
ml.solve()
Qr2 = ml.disvec(r, 0, t)[0][0]

assert np.allclose(Qr1, Qr2, atol=1e-4), "Qr1 and Qr2 not all close Theis well 1"

# turn Theis off
t = np.logspace(-1, 1, 10)

Qr1 = theisQr(r, t, T, S, Q)
Qr1[t > 5] -= theisQr(r, t[t > 5] - 5, T, S, Q)

ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=1e-4, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q), (5, 0)])
ml.solve()
Qr2 = ml.disvec(r, 0, t)[0][0]

assert np.allclose(Qr1, Qr2, atol=1e-4), "Qr1 and Qr2 not all close Theis well 2"

# test nan values for Theis well 1
t = np.array([0.08, 0.09, 0.1, 1, 5, 9])
Qr1 = theisQr(r, t, T, S, Q)

tmin = 0.1
ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=tmin, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q)])
ml.solve()
Qr2 = ml.disvec(r, 0, t)[0][0]

a = np.isnan(Qr2)
b = t < tmin

assert np.all(a == b), "nans not in the right spot for tmin"
assert np.allclose(Qr1[~b], Qr2[~b], atol=1e-4)

# test nan values for Theis well 2

t = np.array([0.08, 0.09, 0.1, 1, 5, 5.03, 9])
Qr1 = theisQr(r, t, T, S, Q)
Qr1[t > 5] -= theisQr(r, t[t > 5] - 5, T, S, Q)

tmin = 0.1
ml = ttim.ModelMaq(kaq=50, z=[10, 0], Saq=S / 10, tmin=tmin, tmax=10)
w = ttim.Well(ml, 0, 0, rw=0.3, tsandQ=[(0, Q), (5, 0)])
ml.solve()
Qr2 = ml.disvec(r, 0, t)[0][0]

a = np.isnan(h2)
b = (t < tmin) | ((t > 5) & (t < 5.1))

assert np.all(a == b), "nans not in the right spot for tmin"
assert np.allclose(Qr1[~b], Qr2[~b], atol=1e-4)
