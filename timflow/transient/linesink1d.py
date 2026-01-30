"""1D line-sink elements for transient flow.

Implements simplified line-sinks for transient cross-section (1D) models.

Example::

    HeadLineSink1D(ml, xls=0, tsandbc=[(0, 1)], layers=0)
"""

import matplotlib.pyplot as plt
import numpy as np

from timflow.transient.element import Element
from timflow.transient.equation import (
    FluxDiffEquation,
    HeadDiffEquation,
    HeadEquation,
    MscreenEquation,
)


class LineSink1DBase(Element):
    """LineSink1D Base Class.

    All LineSink1D elements are derived from this class

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xls : float
        x-coordinate of the line sink
    tsandbc : list of tuples
        list of tuples of the form (time, bc) for boundary conditions
    res : float
        resistance of the line sink
    wh : string or float
        wetted perimeter of the linesink, "H" for aquifer height,
        "2H" for 2x aquifer height (two-sided flow) or specify any float value
    layers : int, array or list
        layer (int) or layers (list or array) in which line sink is located
    type : string
        type of element, "g" for given, "v" for variable and  "z" for zero.
    name : string
        name of the element
    label : string, optional
        label of the element
    aq : Aquifer object
        aquifer in which the element is located
    inhomelement : boolean
        set to True if element is part of an inhomogeneity
    """

    tiny = 1e-8

    def __init__(
        self,
        model,
        xls=0,
        tsandbc=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        type="",
        name="LineSink1DBase",
        label=None,
        aq=None,
        inhomelement=False,
    ):
        super().__init__(
            model,
            nparam=1,
            nunknowns=0,
            layers=layers,
            tsandbc=tsandbc,
            type=type,
            name=name,
            label=label,
            inhomelement=inhomelement,
        )
        # Defined here and not in Element as other elements can have multiple
        # parameters per layers
        self.nparam = len(self.layers)
        self.xls = float(xls)
        self.res = np.atleast_1d(res).astype(np.float64)
        self.wh = wh
        self.aq = aq
        self.model.addelement(self)

    def __repr__(self):
        return self.name + " at " + str(self.xls)

    def initialize(self):
        # Control point to make sure the point is always the same for
        # all elements
        self.xc = np.array([self.xls])
        self.yc = np.zeros(1)
        self.ncp = 1
        if self.aq is None:
            self.aq = self.model.aq.find_aquifer_data(self.xc[0], self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.layers, :]
        self.setflowcoef()
        # term is shape (self.nparam,self.aq.naq,self.model.npval)
        self.term = self.flowcoef * coef
        self.term2 = self.term.reshape(
            self.nparam, self.aq.naq, self.model.nint, self.model.npint
        )
        self.dischargeinf = self.flowcoef * coef
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )
        if isinstance(self.wh, str):
            if self.wh == "H":
                self.wh = self.aq.Haq[self.layers]
            elif self.wh == "2H":
                self.wh = 2.0 * self.aq.Haq[self.layers]
        else:
            self.wh = np.atleast_1d(self.wh) * np.ones(self.nlayers)
        # Q = (h - hls) / resfach
        self.resfach = self.res / (self.wh)
        # Q = (Phi - Phils) / resfacp
        self.resfacp = self.resfach * self.aq.T[self.layers]

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, y=0, aq=None):
        """Can be called with only one x value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            if (x - self.xls) < 0.0:
                pot = -0.5 * aq.lab * np.exp((x - self.xls) / aq.lab)
            elif (x - self.xls) >= 0.0:
                pot = -0.5 * aq.lab * np.exp(-(x - self.xls) / aq.lab)
            else:
                raise ValueError("Something wrong with passed x value.")
            rv[:] = self.term * pot
        return rv

    def disvecinf(self, x, y=0, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        rvy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            if (x - self.xls) < 0.0:
                qx = 0.5 * np.exp((x - self.xls) / aq.lab)
            elif (x - self.xls) >= 0.0:
                qx = -0.5 * np.exp(-(x - self.xls) / aq.lab)
            else:
                raise ValueError("Something wrong with passed x value.")
            rvx[:] = self.term * qx
        return rvx, rvy

    def changetrace(
        self, xyzt1, xyzt2, aq, layer, ltype, modellayer, direction, hstepmax
    ):
        raise NotImplementedError("changetrace not implemented for this element")

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        aq = self.model.aq.find_aquifer_data(self.xls, 0.0)
        for ilay in self.layers:
            ax.plot(
                [self.xls, self.xls],
                [aq.zaqtop[ilay], aq.zaqbot[ilay]],
                "k-",
            )


class DischargeLineSink1D(LineSink1DBase):
    r"""Linesink1D with a specified discharge for each layer the linesink is in.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    x : float
        x-coordinate of the linesink
    tsandq : list of tuples
        tuples of starting time and specific discharge after starting time
    res : float
        resistance of the linesink
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string or None (default: None)
        label of the linesink

    Examples
    --------
    Example of an infinitely long linesink that pumps with a specific discharge of
    100 between times 10 and 50, with a specific discharge of 20 between
    times 50 and 200, and zero speficic discharge after time 200.

    >>> DischargeLineSink1D(ml, tsandq=[(10, 100), (50, 20), (200, 0)])
    """

    def __init__(
        self, model, xls=0, tsandq=[(0, 1)], res=0, wh="H", layers=0, label=None
    ):
        super().__init__(
            model,
            xls,
            tsandbc=tsandq,
            res=res,
            wh=wh,
            layers=layers,
            type="g",
            name="DischargeLineSink1D",
            label=label,
        )


class LineSink1D(LineSink1DBase, MscreenEquation):
    r"""Linesink1D with a specified discharge.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    x : float
        x-coordinate of the linesink
    tsandq : list of tuples
        tuples of starting time and specific discharge after starting time
    res : float
        resistance of the linesink
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string or None (default: None)
        label of the linesink

    Examples
    --------
    Example of an infinitely long linesink that pumps with a specific discharge of
    100 between times 10 and 50, with a specific discharge of 20 between
    times 50 and 200, and zero specific discharge after time 200.

    >>> LineSink1D(ml, tsandq=[(10, 100), (50, 20), (200, 0)])
    """

    def __init__(
        self,
        model,
        xls=0,
        tsandq=[(0, 1)],
        res=0,
        wh="H",
        vres=0.0,
        wv=1.0,
        layers=0,
        label=None,
    ):
        super().__init__(
            model,
            xls,
            tsandbc=tsandq,
            res=res,
            wh=wh,
            layers=layers,
            type="v",
            name="LineSink1D",
            label=label,
        )
        self.nunknowns = self.nparam
        # Vertical resistance inside line-sink
        self.vres = np.atleast_1d(vres)
        self.wv = wv
        if len(self.vres) == 1:
            self.vres = self.vres[0] * np.ones(self.nlayers - 1)

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        # qv = (hn - hn-1) / vresfac[n - 1]
        self.vresfac = self.vres / self.wv


class HeadLineSink1D(LineSink1DBase, HeadEquation):
    """1D head-specified linesink element.

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xls : float
        x-coordinate of the linesink
    tsandh : list of tuples
        list of tuples of the form (time, head) for head conditions
    res : float
        resistance of the linesink
    wh : string or float
        wetted perimeter of the linesink, "H" for aquifer height,
        "2H" for 2x aquifer height (two-sided flow) or specify any float value
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string, optional
        label of the element
    """

    def __init__(
        self,
        model,
        xls=0,
        tsandh=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        label=None,
    ):
        super().__init__(
            model,
            xls,
            tsandbc=tsandh,
            res=res,
            wh=wh,
            layers=layers,
            type="v",
            name="HeadLineSink1D",
            label=label,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        # Needed in solving, solve for a unit head
        self.pc = self.aq.T[self.layers]


class HeadDiffLineSink1D(LineSink1DBase, HeadDiffEquation):
    """1D head-difference linesink element.

    Used to ensure continuity of head in a cross-section model, e.g. at the boundary
    of an inhomogeneity.

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xls : float
        x-coordinate of the linesink
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string, optional
        label of the element
    aq : Aquifer object
        aquifer in which the element is located
    """

    def __init__(
        self,
        model,
        xls=0,
        layers=0,
        label=None,
        aq=None,
    ):
        super().__init__(
            model,
            xls,
            tsandbc=[(0, 0)],
            res=0.0,
            wh="H",
            layers=layers,
            type="z",
            name="HeadDiffLineSink1D",
            label=label,
            aq=aq,
            inhomelement=True,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.xcout = self.xc + self.tiny
        self.xcin = self.xc - self.tiny
        self.ycout = np.zeros(1)
        self.ycin = np.zeros(1)
        self.cosout = -np.ones(1)
        self.sinout = np.zeros(1)
        self.aqout = self.model.aq.find_aquifer_data(self.xcout[0], 0)
        self.aqin = self.model.aq.find_aquifer_data(self.xcin[0], 0)

        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        aqout = self.model.aq.find_aquifer_data(self.xcout, 0.0)
        aqin = self.model.aq.find_aquifer_data(self.xcin, 0.0)
        ztop_out = aqout.z[0]
        ztop_in = aqin.z[0]

        ax.plot(
            [self.xls, self.xls],
            [np.max([ztop_in, ztop_out]), aqout.z[-1]],
            "k--",
            lw=1.0,
        )
        return ax


class FluxDiffLineSink1D(LineSink1DBase, FluxDiffEquation):
    """1D flux-difference linesink element.

    Used to ensure continuity of flux in a cross-section model, e.g. at the boundary
    of an inhomogeneity.

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xls : float
        x-coordinate of the linesink
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string, optional
        label of the element
    """

    def __init__(self, model, xls=0, layers=0, label=None, aq=None):
        super().__init__(
            model,
            xls,
            tsandbc=[(0, 0)],
            res=0.0,
            wh="H",
            layers=layers,
            type="z",
            name="FluxDiffLineSink1D",
            label=label,
            aq=aq,
            inhomelement=True,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.xcout = self.xc - self.tiny
        self.xcin = self.xc + self.tiny
        self.ycout = np.zeros(1)
        self.ycin = np.zeros(1)
        self.cosout = -np.ones(1)
        self.sinout = np.zeros(1)
        self.aqout = self.model.aq.find_aquifer_data(self.xcout[0], 0)
        self.aqin = self.model.aq.find_aquifer_data(self.xcin[0], 0)
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        aqout = self.model.aq.find_aquifer_data(self.xcout, 0.0)
        aqin = self.model.aq.find_aquifer_data(self.xcin, 0.0)
        ztop_out = aqout.z[0]
        ztop_in = aqin.z[0]

        ax.plot(
            [self.xls, self.xls],
            [np.max([ztop_in, ztop_out]), aqout.z[-1]],
            "k--",
            lw=1.0,
        )
        return ax
