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


# ml = tfs.ModelMaq(kaq=[1, 2, 4], z=[5, 4, 3, 2, 1, 0], c=[5000, 1000])
# uf = tfs.Uflow(ml, 0.002, 0)
# rf = tfs.Constant(ml, 100, 0, 20)
# ld1 = tfs.ImpermeableWall1D(ml, xld=0, layers=[0, 1])

# ml.solve()
# x = np.linspace(-100, 100, 101)
# h = ml.headalongline(x, np.zeros_like(x))
# Qx, _ = ml.disvecalongline(x, np.zeros_like(x))

# plt.figure(figsize=(10, 3))
# plt.subplot(121)
# plt.title("head")
# plt.plot(x, h[0], label="layer 0")
# plt.plot(x, h[1], label="layer 1")
# plt.plot(x, h[2], label="layer 2")
# plt.xlabel("x (m)")
# plt.ylabel("head (m)")
# plt.legend(loc="best")
# plt.grid()
# plt.subplot(122)
# plt.title("Qx")
# plt.plot(x, Qx[0], label="layer 0")
# plt.plot(x, Qx[1], label="layer 1")
# plt.plot(x, Qx[2], label="layer 2")
# plt.xlabel("x (m)")
# plt.ylabel("$Q_x$ (m$^2$/d)")
# plt.legend(loc="best")
# plt.grid()

ml.to_json("./test.json")
