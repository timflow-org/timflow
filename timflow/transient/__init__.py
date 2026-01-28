"""Copyright (C), 2017, Mark Bakker.

TTim is a computer program for the simulation of transient multi-layer flow with
analytic elements and consists of a library of Python scripts and FORTRAN extensions.

Mark Bakker, Delft University of Technology mark dot bakker at tudelft dot nl.
"""

# ruff : noqa: F401
__name__ = "ttim"
__author__ = "Mark Bakker"
from ttim.circareasink import CircAreaSink
from ttim.fit import Calibrate
from ttim.inhom1d import Xsection3D, XsectionMaq
from ttim.linedoublet import LeakyLineDoublet, LeakyLineDoubletString
from ttim.linedoublet1d import LeakyLineDoublet1D, LineDoublet1DBase
from ttim.linesink import (
    HeadLineSink,
    HeadLineSinkHo,
    HeadLineSinkString,
    LineSink,
    LineSinkDitchString,
)
from ttim.linesink1d import (
    DischargeLineSink1D,
    FluxDiffLineSink1D,
    HeadDiffLineSink1D,
    HeadLineSink1D,
    LineSink1D,
    LineSink1DBase,
)

# Import all classes and functions
from ttim.model import Model3D, ModelMaq, ModelXsection
from ttim.trace import timtrace, timtraceline
from ttim.version import __version__, show_versions
from ttim.well import DischargeWell, HeadWell, Well, WellTest
