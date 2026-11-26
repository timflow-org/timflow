from pathlib import Path

import numpy as np
import pytest

import timflow.steady as tfs


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("PolygonInhomMaq", {}),
        ("BuildingPitMaq", {}),
        ("LeakyBuildingPitMaq", {}),
    ],
)
def test_model_maq_inhoms(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "maq.json"

    model_in = tfs.ModelMaq()
    xy = [[-10, -10], [10, -10], [10, 10], [-10, 10]]

    getattr(tfs, component)(model_in, xy, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("Well", {}),
        ("HeadWell", {}),
        ("TargetHeadWell", {}),
        ("WellString", {"xy": [[-1, 0], [0, 0], [1, 0]]}),
        ("HeadWellString", {"xy": [[-1, 0], [0, 0], [1, 0]]}),
        ("TargetHeadWellString", {"xy": [[-1, 0], [0, 0], [1, 0]]}),
    ],
)
def test_model_maq_wells(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "well.json"

    model_in = tfs.ModelMaq()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("River", {}),
        ("Ditch", {}),
        ("RiverString", {}),
        ("DitchString", {}),
        ("CollectorWell", {"xy":[[1, 0, 10, 0], [0, 1, 0, 10]]}),
        ("RadialCollectorWell", {}),
    ],
)
def test_model_maq_linesink(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linesink.json"

    model_in = tfs.ModelMaq()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("ImpermeableWall", {}),
        ("ImpermeableWallString", {}),
        ("LeakyWall", {}),
        ("LeakyWallString", {}),
    ],
)
def test_model_maq_linedoublet(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linedoublet.json"

    model_in = tfs.ModelMaq()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("CircAreaSink", {}),
        ("Constant", {}),
        ("Uflow", {"slope":1e-4, "angle":0.0}),
    ],
)
def test_model_maq_other(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linesink.json"

    model_in = tfs.ModelMaq()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()
    
@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("ImpermeableWall1D", {}),
        ("LeakyWall1D", {}),
        ("LineSink1D", {}),
        ("River1D", {}),
    ],
)
def test_model_maq_1D(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linesink.json"

    model_in = tfs.ModelMaq()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelMaq = tfs.ModelMaq.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("PolygonInhom3D", {}),
        ("BuildingPit3D", {}),
        ("LeakyBuildingPit3D", {}),
    ],
)
def test_model_3D_inhoms(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "maq.json"

    model_in = tfs.Model3D()
    xy = [[-10, -10], [10, -10], [10, 10], [-10, 10]]

    getattr(tfs, component)(model_in, xy, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.Model3D = tfs.Model3D.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("Well", {"xw": 0, "yw": 0}),
        ("HeadWell", {"xw": 0, "yw": 0}),
        ("TargetHeadWell", {"xw": 0, "yw": 0}),
        ("WellString", {"xy": [[-1, 0], [0, 0], [1, 0]]}),
        ("HeadWellString", {"xy": [[-1, 0], [0, 0], [1, 0]]}),
        ("TargetHeadWellString", {"xy": [[-1, 0], [0, 0], [1, 0]]}),
    ],
)
def test_model_3D_wells(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "well.json"

    model_in = tfs.Model3D()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.Model3D = tfs.Model3D.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("River", {}),
        ("Ditch", {}),
        ("RiverString", {}),
        ("DitchString", {}),
        ("CollectorWell", {"xy":[[1, 0, 10, 0], [0, 1, 0, 10]]}),
        ("RadialCollectorWell", {}),
    ],
)
def test_model_3D_linesink(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linesink.json"

    model_in = tfs.Model3D()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.Model3D = tfs.Model3D.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("ImpermeableWall", {}),
        ("ImpermeableWallString", {}),
        ("LeakyWall", {}),
        ("LeakyWallString", {}),
    ],
)
def test_model_3D_linedoublet(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linedoublet.json"

    model_in = tfs.Model3D()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.Model3D = tfs.Model3D.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("CircAreaSink", {}),
        ("Constant", {}),
        ("Uflow", {"slope":1e-4, "angle":0.0}),
    ],
)
def test_model_3D_other(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linesink.json"

    model_in = tfs.Model3D()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.Model3D = tfs.Model3D.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()

@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("ImpermeableWall1D", {}),
        ("LeakyWall1D", {}),
        ("LineSink1D", {}),
        ("River1D", {}),
    ],
)
def test_model_3D_1D(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "linesink.json"

    model_in = tfs.Model3D()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.Model3D = tfs.Model3D.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("XsectionMaq", {"x1":-np.inf, "x2":np.inf}),
        ("Xsection3D", {"x1":-np.inf, "x2":np.inf, "kaq":1}),
    ],
)
def test_model_xsection(component: str, kwargs: dict, tmp_path) -> None:
    file_path: Path = tmp_path / "xsection.json"

    model_in = tfs.ModelXsection()

    getattr(tfs, component)(model_in, **kwargs)

    model_in.to_json(file_path)

    model_out: tfs.ModelXsection = tfs.ModelXsection.from_json(file_path)

    # Assert that model_out is not a reference to model_in
    assert model_out is not model_in
    # Assert all kwargs are the same
    assert model_out.to_dict() == model_in.to_dict()
