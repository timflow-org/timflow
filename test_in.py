import timflow.steady as tfs

ml = tfs.ModelXsection.from_json("./test.json")
print(ml.elementlist)
ml.initialize()
# ml.solve()