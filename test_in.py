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

