"""Copyright (C), 2015, Mark Bakker.

Mark Bakker, Delft University of Technology
mark dot bakker at tudelft dot nl

TimML is a computer program for the simulation of steady-state multi-aquifer flow with
analytic elements and consists of a library of Python scripts and FORTRAN extensions.
"""

# ruff: noqa: F401
from timflow.steady import bessel
from timflow.steady.circareasink import CircAreaSink
from timflow.steady.constant import Constant, ConstantStar
from timflow.steady.inhomogeneity import (
    BuildingPit3D,
    BuildingPitMaq,
    LeakyBuildingPit3D,
    LeakyBuildingPitMaq,
    PolygonInhom3D,
    PolygonInhomMaq,
)
from timflow.steady.inhomogeneity1d import (
    StripInhom3D,
    StripInhomMaq,
    Xsection3D,
    XsectionMaq,
)
from timflow.steady.linedoublet import (
    ImpLineDoublet,
    ImpLineDoubletString,
    LeakyLineDoublet,
    LeakyLineDoubletString,
)
from timflow.steady.linedoublet1d import ImpLineDoublet1D, LeakyLineDoublet1D
from timflow.steady.linesink import (
    CollectorWell,
    Ditch,
    DitchString,
    HeadLineSink,  # Deprecated alias for River
    # HeadLineSinkContainer,
    HeadLineSinkString,  # Deprecated alias for RiverString
    # HeadLineSinkZero,
    LineSinkBase,
    LineSinkDitch,  # Deprecated alias for Ditch
    LineSinkDitchString,  # Deprecated alias for DitchString
    RadialCollectorWell,
    River,
    RiverString,
)
from timflow.steady.linesink1d import HeadLineSink1D, LineSink1D
from timflow.steady.model import Model, Model3D, ModelMaq, ModelXsection
from timflow.steady.stripareasink import XsectionAreaSink
from timflow.steady.trace import timtraceline, timtracelines
from timflow.steady.uflow import Uflow
from timflow.steady.well import (
    HeadWell,
    HeadWellString,
    LargeDiameterWell,
    TargetHeadWell,
    TargetHeadWellString,
    Well,
    WellBase,
    WellString,
    WellStringBase,
)

# default bessel module is numba
bessel.set_bessel_method(method="numba")
