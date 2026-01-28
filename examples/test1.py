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
w1 = ttim.DischargeWell(ml, xw=0, yw=0, rw=1e-5, tsandQ=[(0, 1)], layers=0)
ml.solve()
