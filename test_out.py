import matplotlib.pyplot as plt
import numpy as np

import timflow.steady as tfs

ml = tfs.ModelXsection(naq=2)
tfs.XsectionMaq(
    ml,
    x1=-np.inf,
    x2=-50,
    kaq=[1, 2],
    z=[4, 3, 2, 1, 0],
    c=[1000, 1000],
    npor=0.3,
    topboundary="semi",
    hstar=5,
)
tfs.XsectionMaq(
    ml,
    x1=-50,
    x2=50,
    kaq=[1, 2],
    z=[4, 3, 2, 1, 0],
    c=[1000, 1000],
    npor=0.3,
    topboundary="semi",
    hstar=4.5,
)
tfs.XsectionMaq(
    ml,
    x1=50,
    x2=np.inf,
    kaq=[1, 2],
    z=[4, 3, 2, 1, 0],
    c=[1000, 1000],
    npor=0.3,
    topboundary="semi",
    hstar=4,
)
ml.solve()

x = np.linspace(-200, 200, 101)
h = ml.headalongline(x, np.zeros(101))
plt.plot(x, h[0], label="layer 0")
plt.plot(x, h[1], label="layer 1")
plt.xlabel("x (m)")
plt.ylabel("head (m)")
plt.legend(loc="best")
plt.grid()

ml.to_json("./test.json")
