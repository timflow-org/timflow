import matplotlib.pyplot as plt
import numpy as np

import timflow.steady as tfs

ml = tfs.ModelXsection.from_json("./test.json")
ml.solve()

x = np.linspace(-200, 200, 101)
h = ml.headalongline(x, np.zeros(101))
plt.plot(x, h[0], label="layer 0")
plt.plot(x, h[1], label="layer 1")
plt.xlabel("x (m)")
plt.ylabel("head (m)")
plt.legend(loc="best")
plt.grid()

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