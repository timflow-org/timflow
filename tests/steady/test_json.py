from pathlib import Path

import numpy as np

import timflow.steady as tfs


def test_river1D(tmp_path: Path) -> None:
    file_path: Path = tmp_path / "river.json"
    model_in = tfs.ModelMaq(
        kaq=[1, 2],
        z=np.array([4, 3, 2, 1, 0]),
        c=np.array([1000, 1000]),
        topboundary="semi",
        hstar=5,
    )
    tfs.River1D(model_in, xls=0, hls=2, layers=0)
    model_in.to_json(file_path)
    
    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)
    
    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()