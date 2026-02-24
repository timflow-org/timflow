"""Line-doublet elements for transient flow.

Implements line doublets used to model barriers in transient simulations.

Example::

    LeakyLineDoublet(ml, x1=-10, y1=0, x2=10, y2=0, layers=0)
"""

import inspect  # Used for storing the input
import warnings

import matplotlib.pyplot as plt
import numpy as np

from timflow.bessel import besselnumba
from timflow.transient.element import Element
from timflow.transient.equation import LeakyWallEquation


class LineDoubletHoBase(Element):
    """Higher Order LineDoublet Base Class.

    All Higher Order Line Doublet elements are derived from this class
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        tsandbc=[(0.0, 0.0)],
        res="imp",
        order=0,
        layers=0,
        type="",
        name="LineDoubletHoBase",
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
        if res == "imp":
            self.res = np.inf
        else:
            self.res = float(res)
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
        self.thetanormOut = np.arctan2(self.y2 - self.y1, self.x2 - self.x1) - np.pi / 2
        self.cosout = np.cos(self.thetanormOut) * np.ones(self.ncp)
        self.sinout = np.sin(self.thetanormOut) * np.ones(self.ncp)
        #
        thetacp = np.arange(np.pi, 0, -np.pi / self.ncp) - 0.5 * np.pi / self.ncp
        Zcp = np.zeros(self.ncp, dtype=complex)
        Zcp.real = np.cos(thetacp)
        # control point just on positive site (this is handy later on)
        Zcp.imag = 1e-6
        zcp = Zcp * (self.z2 - self.z1) / 2 + 0.5 * (self.z1 + self.z2)
        self.xc = zcp.real
        self.yc = zcp.imag
        # control point just on negative side
        # (this is needed for building the system of equations)
        Zcp.imag = -1e-6
        zcp = Zcp * (self.z2 - self.z1) / 2 + 0.5 * (self.z1 + self.z2)
        self.xcneg = zcp.real
        self.ycneg = zcp.imag  # control points just on negative side
        #
        self.aq = self.model.aq.find_aquifer_data(self.xc[0], self.yc[0])
        self.setbc()
        coef = self.aq.coef[self.layers, :]
        self.setflowcoef()
        # shape (self.nlayers,self.aq.naq,self.model.npvalval)
        self.term = self.flowcoef * coef
        self.term2 = self.term.reshape(
            self.nlayers, self.aq.naq, self.model.nint, self.model.npint
        )
        self.resfac = self.aq.Haq[self.layers] / self.res
        self.dischargeinf = self.flowcoef * coef
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )

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
            pot = np.zeros((self.order + 1, self.model.npint), dtype=complex)
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    if besselnumba.isinside(
                        self.z1, self.z2, x + y * 1j, self.rzero * self.aq.lababs[i, j]
                    ):
                        pot[:, :] = (
                            besselnumba.besselldv2(
                                x,
                                y,
                                self.z1,
                                self.z2,
                                self.aq.lab2[i, j, :],
                                self.order,
                                self.rzero * self.aq.lababs[i, j],
                            )
                            / self.L
                        )
                        for k in range(self.nlayers):
                            rv[k :: self.nlayers, i, j, :] = self.term2[k, i, j, :] * pot
        return rv.reshape((self.nparam, aq.naq, self.model.npval))

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
                            besselnumba.besselldqxqyv2(
                                x,
                                y,
                                self.z1,
                                self.z2,
                                self.aq.lab2[i, j, :],
                                self.order,
                                self.rzero * self.aq.lababs[i, j],
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
        return (
            rvx.reshape((self.nparam, aq.naq, self.model.npval)),
            rvy.reshape((self.nparam, aq.naq, self.model.npval)),
        )

    def plot(self, ax=None, layer=None):
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="datalim")
        if layer is None or np.isin(self.layers, layer).any():
            ax.plot([self.x1, self.x2], [self.y1, self.y2], "k")


class LeakyWall(LineDoubletHoBase, LeakyWallEquation):
    """Create a segment of a leaky wall, which is simulated with a line-doublet.

    The specific discharge through the wall is equal to the head difference
    across the wall divided by the resistance of the wall.

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    x1 : scalar
        x-coordinate of fist point of line-doublet
    y1 : scalar
        y-coordinate of fist point of line-doublet
    x2 : scalar
        x-coordinate of second point of line-doublet
    y2 : scalar
        y-coordinate of second point of line-doublet
    res : scalar or string
        if string: 'imp' for an impermeable wall (same as res = np.inf)
        if scalar: resistance of leaky wall
    order : int (default is 0)
        polynomial order of potential jump along line-doublet
        (head jump if transmissivity is equal on each side of wall)
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element

    See Also
    --------
    :class:`.LeakyLineDoubletString`
    """

    def __init__(
        self,
        model,
        x1=-1,
        y1=0,
        x2=1,
        y2=0,
        res="imp",
        order=0,
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
            tsandbc=[(0, 0)],
            res=res,
            order=order,
            layers=layers,
            type="z",
            name="LeakyWall",
            label=label,
            addtomodel=addtomodel,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )


class LeakyWallString(Element, LeakyWallEquation):
    """Create a string of leaky wall segements consisting of line-doublets.

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xy : array or list
        list or array of (x,y) pairs of coordinates of end-points of
        the segements in the string
    res : scalar or string
        if string: 'imp' for an impermeable wall (same as res = np.inf)
        if scalar: resistance of leaky wall
    order : int (default is 0)
        polynomial order of potential jump along line-doublet
        (head jump if transmissivity is equal on each side of wall)
    layers : scalar, list or array
        layer(s) in which element is placed
        if scalar: element is placed in this layer
        if list or array: element is placed in all these layers
    label: str or None
        label of element

    See Also
    --------
    :class:`.LeakyLineDoublet`
    """

    def __init__(
        self, model, xy=[(-1, 0), (1, 0)], res="imp", order=0, layers=0, label=None
    ):
        self.storeinput(inspect.currentframe())
        Element.__init__(
            self,
            model,
            nparam=1,
            nunknowns=0,
            layers=layers,
            tsandbc=[(0, 0)],
            type="z",
            name="LeakyWallString",
            label=label,
        )
        self.res = res
        self.order = order
        self.ldlist = []
        xy = np.atleast_2d(xy).astype(float)
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.nld = len(self.x) - 1
        for i in range(self.nld):
            self.ldlist.append(
                LeakyWall(
                    model,
                    x1=self.x[i],
                    y1=self.y[i],
                    x2=self.x[i + 1],
                    y2=self.y[i + 1],
                    res=self.res,
                    order=self.order,
                    layers=layers,
                    label=label,
                    addtomodel=False,
                )
            )
        self.model.addelement(self)

    def __repr__(self):
        return self.name + " with nodes " + str(zip(self.x, self.y, strict=False))

    def initialize(self):
        for ld in self.ldlist:
            ld.initialize()
        # Same order for all elements in string
        self.ncp = self.nld * self.ldlist[0].ncp
        self.nparam = self.nld * self.ldlist[0].nparam
        self.nunknowns = self.nparam
        self.xld, self.yld = np.empty((self.nld, 2)), np.empty((self.nld, 2))
        for i, ld in enumerate(self.ldlist):
            self.xld[i, :] = [ld.x1, ld.x2]
            self.yld[i, :] = [ld.y1, ld.y2]
        # Only used for layout when it is a continuous string
        self.xldlayout = np.hstack((self.xld[:, 0], self.xld[-1, 1]))
        self.yldlayout = np.hstack((self.yld[:, 0], self.yld[-1, 1]))
        self.aq = self.model.aq.find_aquifer_data(self.ldlist[0].xc, self.ldlist[0].yc)
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        self.setbc()
        # As parameters are only stored for the element not the list,
        # we need to combine the following
        self.resfac = self.ldlist[0].resfac  # same for all elements in the list
        self.xc, self.yc = np.zeros(self.ncp), np.zeros(self.ncp)
        self.xcneg, self.ycneg = np.zeros(self.ncp), np.zeros(self.ncp)
        self.cosout, self.sinout = np.zeros(self.ncp), np.zeros(self.ncp)
        for i, ld in enumerate(self.ldlist):
            self.xc[i * ld.ncp : (i + 1) * ld.ncp] = ld.xc
            self.yc[i * ld.ncp : (i + 1) * ld.ncp] = ld.yc
            self.xcneg[i * ld.ncp : (i + 1) * ld.ncp] = ld.xcneg
            self.ycneg[i * ld.ncp : (i + 1) * ld.ncp] = ld.ycneg
            self.cosout[i * ld.ncp : (i + 1) * ld.ncp] = ld.cosout
            self.sinout[i * ld.ncp : (i + 1) * ld.ncp] = ld.sinout

    def potinf(self, x, y, aq=None):
        """Returns array (nunknowns,nperiods)."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        for i, ld in enumerate(self.ldlist):
            rv[i * ld.nparam : (i + 1) * ld.nparam, :] = ld.potinf(x, y, aq)
        return rv

    def disvecinf(self, x, y, aq=None):
        """Returns array (nunknowns,nperiods)."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        rvy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        for i, ld in enumerate(self.ldlist):
            qx, qy = ld.disvecinf(x, y, aq)
            rvx[i * ld.nparam : (i + 1) * ld.nparam, :] = qx
            rvy[i * ld.nparam : (i + 1) * ld.nparam, :] = qy
        return rvx, rvy

    def plot(self, ax=None, layer=None):
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="datalim")
        for ld in self.ldlist:
            ld.plot(ax=ax, layer=layer)


class LeakyLineDoublet(LeakyWall):
    """Deprecated alias for :class:`.LeakyWall`.

    .. deprecated::
        Use :class:`.LeakyWall` instead. This alias will be removed in a
        future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LeakyLineDoublet is deprecated. Use LeakyWall instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class LeakyLineDoubletString(LeakyWallString):
    """Deprecated alias for :class:`.LeakyWallString`.

    .. deprecated::
        Use :class:`.LeakyWallString` instead. This alias will be removed in a
        future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LeakyLineDoubletString is deprecated. Use LeakyWallString instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
