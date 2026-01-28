import numpy as np

import ttim

ml = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=10,
    tmax=1000,
    M=20,
)
w1 = ttim.HeadWell(ml, xw=0, yw=0, rw=0.3, tsandh=[(0, 1)], layers=0)
ml.solve()

ml2 = ttim.ModelMaq(
    kaq=[1, 5],
    z=[3, 2, 1, 0],
    c=[10],
    Saq=[0.3, 0.01],
    Sll=[0.001],
    tmin=10,
    tmax=1000,
    M=20,
)
w2 = ttim.DischargeWell(ml2, xw=0, yw=0, rw=0.3, tsandQ=[(0, 0), (100, 2.15)], layers=0)
ml2.solve()

x = np.linspace(-10, 10, 101)
h1 = ml.headalongline(x, np.zeros_like(x), 110, [0, 1])
h2 = ml2.headalongline(x, np.zeros_like(x), 110, [0, 1])
