import inspect  # Used for storing the input

import matplotlib.pyplot as plt
import numpy as np

from timflow import besselnumba
from timflow.transient.element import Element
from timflow.transient.equation import (
    HeadEquation,
    HeadEquationNores,
    MscreenDitchEquation,
    MscreenEquation,
)


class LineSinkBase(Element):
    """LineSink Base Class.

    All LineSink elements are derived from this class
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandbc=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        type="",
        name="LineSinkBase",
        label=None,
        addtomodel=True,
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
        )
        self.nparam = len(self.layers)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.res = np.atleast_1d(res).astype(float)
        self.wh = wh
        if addtomodel:
            self.model.addelement(self)

    def __repr__(self):
        return (
            self.name
            + " from "
            + str((self.x1, self.y1))
            + " to "
            + str((self.x2, self.y2))
        )

    def initialize(self):
        self.xc = np.array([0.5 * (self.x1 + self.x2)])
        self.yc = np.array([0.5 * (self.y1 + self.y2)])
        self.ncp = 1
        self.z1 = self.x1 + 1j * self.y1
        self.z2 = self.x2 + 1j * self.y2
        self.L = np.abs(self.z1 - self.z2)
        self.order = 0  # This is for univform discharge only
        self.aq = self.model.aq.find_aquifer_data(self.xc, self.yc)
        self.setbc()
        coef = self.aq.coef[self.layers, :]
        self.setflowcoef()
        # self.term shape (self.nparam,self.aq.naq,self.model.npval)
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
        self.resfach = self.res / (self.wh * self.L)
        # Q = (Phi - Phils) / resfacp
        self.resfacp = self.resfach * self.aq.T[self.layers]

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros(
            (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
        )
        if aq == self.aq:
            pot = np.zeros(self.model.npint, dtype=complex)
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    # pot[:] = besselnumba.bessellsuniv(
                    #     x, y, self.z1, self.z2, self.aq.lab2[i, j, :], self.rzero
                    # )
                    # note that self.order=0
                    pot[:] = besselnumba.bessellsv2(
                        x,
                        y,
                        self.z1,
                        self.z2,
                        self.aq.lab2[i, j, :],
                        self.order,
                        self.rzero,
                    )
                    # Divide by L as the parameter is total discharge
                    rv[:, i, j, :] = self.term2[:, i, j, :] * pot / self.L
        rv.shape = (self.nparam, aq.naq, self.model.npval)
        return rv

    def disvecinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros(
            (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
        )
        rvy = np.zeros(
            (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
        )
        if aq == self.aq:
            qxqy = np.zeros((2, self.model.npint), dtype=complex)
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    if besselnumba.isinside(
                        self.z1, self.z2, x + y * 1j, self.rzero * self.aq.lababs[i, j]
                    ):
                        qxqy[:, :] = (
                            besselnumba.bessellsqxqyv2(
                                x,
                                y,
                                self.z1,
                                self.z2,
                                self.aq.lab2[i, j, :],
                                self.order,
                                self.rzero,
                            )
                            / self.L
                        )
                    rvx[:, i, j, :] = self.term2[:, i, j, :] * qxqy[0]
                    rvy[:, i, j, :] = self.term2[:, i, j, :] * qxqy[1]
        rvx.shape = (self.nparam, aq.naq, self.model.npval)
        rvy.shape = (self.nparam, aq.naq, self.model.npval)
        return rvx, rvy

    def headinside(self, t):
        """The head inside the line-sink.

        Parameters
        ----------
        t : array or float
            time(s) for whih head is computed

        Returns
        -------
        array (length number of layers)
            Head inside the line-sink for each layer that
            the line-sink is screened in
        """
        return self.model.head(self.xc[0], self.yc[0], t)[self.layers] - self.resfach[
            :, np.newaxis
        ] * self.discharge(t)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot([self.x1, self.x2], [self.y1, self.y2], "k")


class LineSink(LineSinkBase):
    """LineSink with non-zero and potentially variable discharge through time.

    Really only used for testing.
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandQ=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        label=None,
        addtomodel=True,
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            tsandbc=tsandQ,
            res=res,
            wh=wh,
            layers=layers,
            type="g",
            name="LineSink",
            label=label,
            addtomodel=addtomodel,
        )


# class ZeroHeadLineSink(LineSinkBase,HeadEquation):
#'''HeadLineSink that remains zero and constant through time'''
# def __init__(self, model, x1=-1, y1=0, x2=1, y2=0, res=0.0, wh='H', \
#             layers=0, label=None, addtomodel=True):
#    self.storeinput(inspect.currentframe())
#    LineSinkBase.__init__(self, model, x1=x1, y1=y1, x2=x2, y2=y2, \
#                          tsandbc=[(0, 0)], res=res, wh=wh, layers=layers,\
#                          type='z', name='ZeroHeadLineSink', label=label, \
#                          addtomodel=addtomodel)
#    self.nunknowns = self.nparam
#
# def initialize(self):
#    LineSinkBase.initialize(self)
#    self.parameters = np.zeros((self.model.ngvbc, self.nparam,
#                                self.model.npval), 'D')


class HeadLineSink(LineSinkBase, HeadEquation):
    r"""Create a head-specified line-sink with optional width and resistance.

    Inflow per unit length of line-sink is computed as:

    .. math::
        \sigma = w(h_{aq} - h_{ls})/c

    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    x1 : scalar
        x-coordinate of fist point of line-sink
    y1 : scalar
        y-coordinate of fist point of line-sink
    x2 : scalar
        x-coordinate of second point of line-sink
    y2 : scalar
        y-coordinate of second point of line-sink
    tsandh : list or 2D array of (time, head) values or string
        if list or 2D array: pairs of time and head after that time
        if 'fixed': head is fixed (no change in head) during entire simulation
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer
        layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when
        flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the
        aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element

    See Also
    --------
    :class:`.HeadLineSinkString`
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandh=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        label=None,
        addtomodel=True,
    ):
        self.storeinput(inspect.currentframe())
        if tsandh == "fixed":
            tsandh = [(0, 0)]
            etype = "z"
        else:
            etype = "v"
        super().__init__(
            model,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            tsandbc=tsandh,
            res=res,
            wh=wh,
            layers=layers,
            type=etype,
            name="HeadLineSink",
            label=label,
            addtomodel=addtomodel,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        # Needed in solving, solve for a unit head
        self.pc = self.aq.T[self.layers]


class LineSinkStringBase(Element):
    def __init__(
        self,
        model,
        tsandbc=[(0, 1)],
        layers=0,
        type="",
        name="LineSinkStringBase",
        label=None,
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
        )
        self.lslist = []

    def __repr__(self):
        return self.name + " with nodes " + str(zip(self.x, self.y, strict=False))

    def initialize(self):
        self.ncp = self.nls
        self.nparam = self.nlayers * self.nls
        self.nunknowns = self.nparam
        self.xls, self.yls = np.empty((self.nls, 2)), np.empty((self.nls, 2))
        for i, ls in enumerate(self.lslist):
            ls.initialize()
            self.xls[i, :] = [ls.x1, ls.x2]
            self.yls[i, :] = [ls.y1, ls.y2]
        # Only used for layout when it is a continuous string
        self.xlslayout = np.hstack((self.xls[:, 0], self.xls[-1, 1]))
        self.ylslayout = np.hstack((self.yls[:, 0], self.yls[-1, 1]))
        self.aq = self.model.aq.find_aquifer_data(self.lslist[0].xc, self.lslist[0].yc)
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        self.setbc()
        # As parameters are only stored for the element not the list
        # we need to combine the following
        self.resfach = []
        self.resfacp = []
        for ls in self.lslist:
            ls.initialize()
            self.resfach.extend(ls.resfach.tolist())  # Needed in solving
            self.resfacp.extend(ls.resfacp.tolist())  # Needed in solving
        self.resfach = np.array(self.resfach)
        self.resfacp = np.array(self.resfacp)
        self.dischargeinf = np.zeros(
            (self.nparam, self.aq.naq, self.model.npval), dtype=complex
        )
        self.dischargeinflayers = np.zeros((self.nparam, self.model.npval), dtype=complex)
        self.xc, self.yc = np.zeros(self.nls), np.zeros(self.nls)
        for i in range(self.nls):
            self.dischargeinf[i * self.nlayers : (i + 1) * self.nlayers, :] = self.lslist[
                i
            ].dischargeinf[:]
            self.dischargeinflayers[i * self.nlayers : (i + 1) * self.nlayers, :] = (
                self.lslist[i].dischargeinflayers
            )
            self.xc[i] = self.lslist[i].xc
            self.yc[i] = self.lslist[i].yc

    def potinf(self, x, y, aq=None):
        """Returns array (nunknowns, Nperiods)."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        for i in range(self.nls):
            rv[i * self.nlayers : (i + 1) * self.nlayers, :] = self.lslist[i].potinf(
                x, y, aq
            )
        return rv

    def disvecinf(self, x, y, aq=None):
        """Returns array (nunknowns,Nperiods)."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        rvy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        for i in range(self.nls):
            qx, qy = self.lslist[i].disvecinf(x, y, aq)
            rvx[i * self.nlayers : (i + 1) * self.nlayers, :] = qx
            rvy[i * self.nlayers : (i + 1) * self.nlayers, :] = qy
        return rvx, rvy

    def headinside(self, t, derivative=0):
        """The head inside the line-sink string.

        Parameters
        ----------
        t : array or float
            time(s) for whih head is computed

        Returns
        -------
        array size nline-sinks, nlayers, ntimes
            Head inside the line-sink for each line-sink, each layer that
            the line-sink is screened in, and each time
        """
        rv = np.zeros((self.nls, self.nlayers, np.size(t)))
        Q = self.discharge_list(t, derivative=derivative)
        for i in range(self.nls):
            rv[i, :, :] = (
                self.model.head(self.xc[i], self.yc[i], t, derivative=derivative)[
                    self.layers
                ]
                - self.resfach[i * self.nlayers : (i + 1) * self.nlayers, np.newaxis]
                * Q[i]
            )
        return rv

    def plot(self, ax):
        ax.plot(self.xlslayout, self.ylslayout, "k")

    def run_after_solve(self):
        for i in range(self.nls):
            self.lslist[i].parameters[:] = self.parameters[
                :, i * self.nlayers : (i + 1) * self.nlayers, :
            ]

    def discharge_list(self, t, derivative=0):
        """The discharge of each line-sink in the string.

        Parameters
        ----------
        t : array or float
            time(s) for whih discharge is computed

        Returns
        -------
        array size nline-sinks, nlayers, ntimes
            Discharge for each line-sink, each layer that
            the line-sink is screened in, and each time
        """
        rv = np.zeros((self.nls, self.nlayers, np.size(t)))
        for i in range(self.nls):
            rv[i, :, :] = self.lslist[i].discharge(t, derivative=derivative)
        return rv


class HeadLineSinkString(LineSinkStringBase, HeadEquation):
    r"""String of head-specified line-sinks with optional width and resistance.

    Inflow per unit length of line-sink is computed as:

    .. math::
        \sigma = w(h_{aq} - h_{ls})/c

    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        line-sinks in string
    tsandh : list or 2D array of (time, head) values or string
        if list or 2D array: pairs of time and head after that time
        if 'fixed': head is fixed (no change in head) during entire simulation
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer
        layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when
        flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the
        aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element

    See Also
    --------
    :class:`.HeadLineSink`
    """

    def __init__(
        self,
        model,
        xy=[(-1, 0), (1, 0)],
        tsandh=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        label=None,
    ):
        if tsandh == "fixed":
            tsandh = [(0, 0)]
            etype = "z"
        else:
            etype = "v"
        super().__init__(
            model,
            tsandbc=tsandh,
            layers=layers,
            type=etype,
            name="HeadLineSinkString",
            label=label,
        )
        xy = np.atleast_2d(xy).astype(float)
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        self.nls = len(self.x) - 1
        self.tsandh = tsandh
        self.res = np.atleast_1d(float(res))
        self.wh = wh
        self.model.addelement(self)

    def initialize(self):
        self.lslist = []
        for i in range(self.nls):
            self.lslist.append(
                HeadLineSink(
                    self.model,
                    x1=self.x[i],
                    y1=self.y[i],
                    x2=self.x[i + 1],
                    y2=self.y[i + 1],
                    tsandh=self.tsandh,
                    res=self.res,
                    wh=self.wh,
                    layers=self.layers,
                    label=None,
                    addtomodel=False,
                )
            )
        super().initialize()
        self.pc = np.zeros(self.nls * self.nlayers)
        for i in range(self.nls):
            self.pc[i * self.nlayers : (i + 1) * self.nlayers] = self.lslist[i].pc


class MscreenLineSink(LineSinkBase, MscreenEquation):
    """MscreenLineSink that varies through time.

    Must be screened in multiple layers but heads are same in all screened layers
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandQ=[(0.0, 1.0)],
        res=0.0,
        wh="H",
        layers=[0, 1],
        vres=0.0,
        wv=1.0,
        label=None,
        addtomodel=True,
    ):
        # assert len(layers) > 1, "number of layers must be at least 2"
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            tsandbc=tsandQ,
            res=res,
            wh=wh,
            layers=layers,
            type="v",
            name="MscreenLineSink",
            label=label,
            addtomodel=addtomodel,
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
        # Qv = (hn - hn-1) / vresfac[n - 1]
        self.vresfac = self.vres / (self.wv * self.L)


class LineSinkDitchString(LineSinkStringBase, MscreenDitchEquation):
    r"""Create ditch consisting of a string of line-sink.

    The total discharge for the string is specified and divided over the line-sinks such
    that the head at the center inside each line-sink is equal. A width and resistance
    may optionally be specified.

    Inflow per unit length of line-sink is computed as:

    .. math::
        \sigma = w(h_{aq} - h_{ls})/c

    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        line-sinks in string
    tsandQ : list or 2D array of (time, discharge) values
        if list or 2D array: pairs of time and discharge after that time
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer
        layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when
        flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the
        aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element
    """

    def __init__(
        self,
        model,
        xy=[(-1, 0), (1, 0)],
        tsandQ=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        Astorage=None,
        label=None,
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            tsandbc=tsandQ,
            layers=layers,
            type="v",
            name="LineSinkDitchString",
            label=label,
        )
        xy = np.atleast_2d(xy).astype(float)
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.nls = len(self.x) - 1
        for i in range(self.nls):
            self.lslist.append(
                MscreenLineSink(
                    model,
                    x1=self.x[i],
                    y1=self.y[i],
                    x2=self.x[i + 1],
                    y2=self.y[i + 1],
                    tsandQ=tsandQ,
                    res=res,
                    wh=wh,
                    layers=layers,
                    label=None,
                    addtomodel=False,
                )
            )
        self.Astorage = Astorage
        self.model.addelement(self)

    def initialize(self):
        super().initialize()
        # set vresfac to zero, as I don't quite know what it would mean if
        # it is not zero
        self.vresfac = np.zeros_like(self.resfach)


class LineSinkDitchString2(LineSinkStringBase, MscreenDitchEquation):
    r"""Create ditch consisting of a string of line-sink.

    The total discharge for the string is specified and divided over the line-sinks such
    that the head at the center inside each line-sink is equal. A width and resistance
    may optionally be specified.

    Inflow per unit length of line-sink is computed as:

    .. math::
        \sigma = w(h_{aq} - h_{ls})/c

    where :math:`c` is the resistance of the bottom of the line-sink,
    :math:`w` is the width over which water enters the line-sink,
    :math:`h_{aq}` is the head in the aquifer at the center of the line-sink,
    :math:`h_{ls}` is the specified head inside the line-sink
    Note that all that matters is the conductance term :math:`w/c` but
    both are specified separately

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        line-sinks in string
    tsandQ : list or 2D array of (time, discharge) values
        if list or 2D array: pairs of time and discharge after that time
    res : scalar (default is 0)
        resistance of line-sink
    wh : scalar or str
        distance over which water enters line-sink
        if 'H': (default) distance is equal to the thickness of the aquifer
        layer (when flow comes mainly from one side)
        if '2H': distance is twice the thickness of the aquifer layer (when
        flow comes from both sides)
        if scalar: the width of the stream that partially penetrates the
        aquifer layer
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element
    """

    def __init__(
        self,
        model,
        xy=[(-1, 0), (1, 0)],
        tsandQ=[(0, 1)],
        res=0,
        wh="H",
        layers=0,
        Astorage=None,
        label=None,
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            tsandbc=tsandQ,
            layers=layers,
            type="v",
            name="LineSinkDitchString",
            label=label,
        )
        xy = np.atleast_2d(xy).astype(float)
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.nls = len(self.x) - 1
        for i in range(self.nls):
            self.lslist.append(
                MscreenLineSink(
                    model,
                    x1=self.x[i],
                    y1=self.y[i],
                    x2=self.x[i + 1],
                    y2=self.y[i + 1],
                    tsandQ=tsandQ,
                    res=res,
                    wh=wh,
                    layers=layers,
                    label=None,
                    addtomodel=False,
                )
            )
        self.Astorage = Astorage
        self.model.addelement(self)

    def initialize(self):
        super().initialize()
        # set vresfac to zero, as I don't quite know what it would mean if
        # it is not zero
        self.vresfac = np.zeros_like(self.resfach)


class LineSinkHoBase(Element):
    """Higher Order LineSink Base Class.

    All Higher Order Line Sink elements are derived from this class
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandbc=[(0.0, 1.0)],
        res=0.0,
        wh="H",
        order=0,
        layers=0,
        type="",
        name="LineSinkBase",
        label=None,
        addtomodel=True,
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
        )
        self.order = order
        self.nparam = (self.order + 1) * len(self.layers)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.res = res
        self.wh = wh
        if addtomodel:
            self.model.addelement(self)

    def __repr__(self):
        return (
            self.name
            + " from "
            + str((self.x1, self.y1))
            + " to "
            + str((self.x2, self.y2))
        )

    def initialize(self):
        self.ncp = self.order + 1
        self.z1 = self.x1 + 1j * self.y1
        self.z2 = self.x2 + 1j * self.y2
        self.L = np.abs(self.z1 - self.z2)
        #
        thetacp = np.arange(np.pi, 0, -np.pi / self.ncp) - 0.5 * np.pi / self.ncp
        Zcp = np.zeros(self.ncp, dtype=complex)
        Zcp.real = np.cos(thetacp)
        # control point just on positive site (this is handy later on)
        Zcp.imag = 1e-6
        zcp = Zcp * (self.z2 - self.z1) / 2 + 0.5 * (self.z1 + self.z2)
        self.xc = zcp.real
        self.yc = zcp.imag
        #
        self.aq = self.model.aq.find_aquifer_data(self.xc[0], self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.layers, :]
        self.setflowcoef()
        # shape of term (self.nlayers, self.aq.naq, self.model.npval)
        self.term = self.flowcoef * coef
        self.term2 = self.term.reshape(
            self.nlayers, self.aq.naq, self.model.nint, self.model.npint
        )
        self.dischargeinf = self.flowcoef * coef
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )
        if self.wh == "H":
            self.wh = self.aq.Haq[self.layers]
        elif self.wh == "2H":
            self.wh = 2.0 * self.aq.Haq[self.layers]
        else:
            self.wh = np.atleast_1d(self.wh) * np.ones(self.nlayers)
        # Q = (h - hls) / resfach
        self.resfach = self.res / (self.wh * self.L)
        # Q = (Phi - Phils) / resfacp
        self.resfacp = self.resfach * self.aq.T[self.layers]

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = 1 / self.model.p  # Step function

    def potinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros(
            (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
        )
        if aq == self.aq:
            pot = np.zeros((self.order + 1, self.model.npint), dtype=complex)
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    if besselnumba.isinside(
                        self.z1, self.z2, x + y * 1j, self.rzero * self.aq.lababs[i, j]
                    ):
                        pot[:, :] = (
                            besselnumba.bessellsv2(
                                x,
                                y,
                                self.z1,
                                self.z2,
                                self.aq.lab2[i, j, :],
                                self.order,
                                self.rzero,
                            )
                            / self.L
                        )
                    for k in range(self.nlayers):
                        rv[k :: self.nlayers, i, j, :] = self.term2[k, i, j, :] * pot

        rv.shape = (self.nparam, aq.naq, self.model.npval)
        return rv

    def disvecinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros(
            (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
        )
        rvy = np.zeros(
            (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
        )
        if aq == self.aq:
            qxqy = np.zeros((2 * (self.order + 1), self.model.npint), dtype=complex)
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    if besselnumba.isinside(
                        self.z1, self.z2, x + y * 1j, self.rzero * self.aq.lababs[i, j]
                    ):
                        qxqy[:, :] = (
                            besselnumba.bessellsqxqyv2(
                                x,
                                y,
                                self.z1,
                                self.z2,
                                self.aq.lab2[i, j, :],
                                self.order,
                                self.rzero,
                            )
                            / self.L
                        )
                    for k in range(self.nlayers):
                        rvx[k :: self.nlayers, i, j, :] = (
                            self.term2[k, i, j, :] * qxqy[: self.order + 1, :]
                        )
                        rvy[k :: self.nlayers, i, j, :] = (
                            self.term2[k, i, j, :] * qxqy[self.order + 1 :, :]
                        )
        rvx.shape = (self.nparam, aq.naq, self.model.npval)
        rvy.shape = (self.nparam, aq.naq, self.model.npval)
        return rvx, rvy

    def headinside(self, t):
        """The head inside the line-sink.

        Returns
        -------
        array (length number of screens)
            Head inside the well for each screen
        """
        return self.model.head(self.xc, self.yc, t)[self.layers] - self.resfach[
            :, np.newaxis
        ] * self.discharge(t)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot([self.x1, self.x2], [self.y1, self.y2], "k")


class HeadLineSinkHo(LineSinkHoBase, HeadEquationNores):
    """HeadLineSink of which the head varies through time.

    May be screened in multiple layers but all with the same head
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandh=[(0.0, 1.0)],
        order=0,
        layers=0,
        label=None,
        addtomodel=True,
    ):
        self.storeinput(inspect.currentframe())
        if tsandh == "fixed":
            tsandh = [(0, 0)]
            etype = "z"
        else:
            etype = "v"
        super().__init__(
            model,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            tsandbc=tsandh,
            res=0.0,
            wh="H",
            order=order,
            layers=layers,
            type=etype,
            name="HeadLineSinkHo",
            label=label,
            addtomodel=addtomodel,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        self.pc = np.empty(self.nparam)
        for i, T in enumerate(self.aq.T[self.layers]):
            # Needed in solving; solve for a unit head
            self.pc[i :: self.nlayers] = T
