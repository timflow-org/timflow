import numpy as np

import ttim

xls = 100 * np.cos(np.linspace(np.pi, 0, 7))
yls = 50 * np.ones(len(xls))

ml1 = ttim.ModelMaq(
    kaq=[1, 20, 2],
    z=[25, 20, 18, 10, 8, 0],
    c=[100, 200],
    Saq=[0.1, 1e-4, 1e-4],
    Sll=[0, 0],
    phreatictop=True,
    tmin=0.1,
    tmax=10,
    M=20,
    f2py=False,
)
ls1 = ttim.HeadLineSinkString(
    ml1, list(zip(xls, yls, strict=False)), tsandh=[(0, 2)], layers=0, label="river"
)
ml1.solve()

ml2 = ttim.ModelMaq(
    kaq=[1, 20, 2],
    z=[25, 20, 18, 10, 8, 0],
    c=[100, 200],
    Saq=[0.1, 1e-4, 1e-4],
    Sll=[0, 0],
    phreatictop=True,
    tmin=0.1,
    tmax=10,
    M=20,
    f2py=True,
)
ls2 = ttim.HeadLineSinkString(
    ml2, list(zip(xls, yls, strict=False)), tsandh=[(0, 2)], layers=0, label="river"
)
ml2.solve()

print("ml1:", ml1.disvec(40, 20, 10))
print("ml2:", ml2.disvec(40, 20, 10))

ml3 = ttim.ModelMaq(
    kaq=[1, 20, 2],
    z=[25, 20, 18, 10, 8, 0],
    c=[100, 200],
    Saq=[0.1, 1e-4, 1e-4],
    Sll=[0, 0],
    phreatictop=True,
    tmin=0.1,
    tmax=10,
    M=20,
    f2py=False,
)
ls3 = ttim.HeadLineSinkHo(
    ml3, -100, 0, 100, 0, tsandh=[(0, 2)], layers=0, order=5, label="river"
)
ml3.solve()

ml4 = ttim.ModelMaq(
    kaq=[1, 20, 2],
    z=[25, 20, 18, 10, 8, 0],
    c=[100, 200],
    Saq=[0.1, 1e-4, 1e-4],
    Sll=[0, 0],
    phreatictop=True,
    tmin=0.1,
    tmax=10,
    M=20,
    f2py=True,
)
ls4 = ttim.HeadLineSinkHo(
    ml4, -100, 0, 100, 0, tsandh=[(0, 2)], layers=0, order=5, label="river"
)
ml4.solve()

print("ml3:", ml3.disvec(40, 20, 10))
print("ml4:", ml4.disvec(40, 20, 10))

# x = np.linspace(-200, 200, 101)
# h1 = ml1.headalongline(x, 50, t=100)

# ml2 = ttim.ModelMaq(kaq=[1, 20, 2], z=[25, 20, 18, 10, 8, 0], c=[100, 200],
#               Saq=[0.1, 1e-4, 1e-4], Sll=[0, 0], phreatictop=True,
#               tmin=0.1, tmax=1000, M=20, f2py=False)
# ls2 = ttim.HeadLineSinkHo(ml2, x1=-100, y1=50, x2=100, y2=50, tsandh=[(0.0,2.0)],\
#                  order=5, layers=0)
# ml2.solve()

# print('ml1:', ml1.disvec(40, 20, 10))
# print('ml2:', ml2.disvec(40, 20, 10))

# # x = np.linspace(-200, 200, 101)
# # h2 = ml2.headalongline(x, 50, t=100)

# # plt.figure(figsize=(12, 4))
# # plt.subplot(121)
# # for i in range(3):
# #     plt.plot(x, h1[i, 0])
# # plt.grid()
# # plt.subplot(122)
# # for i in range(3):
# #     plt.plot(x, h2[i, 0])
# # plt.grid()
# # plt.show()
