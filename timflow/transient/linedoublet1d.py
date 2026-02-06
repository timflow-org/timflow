"""1D line-doublet elements for transient flow.

Doublet features for modeling barriers in a transient cross-section.

Example::

    LeakyLineDoublet1D(ml, xld=0, layers=0)

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from timflow.transient.element import Element
from timflow.transient.equation import LeakyWallEquation


class LineDoublet1DBase(Element):
    """LineDoublet1D Base Class.

    All LineDoublet1D elements are derived from this class

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xld : float
        x-coordinate of the line doublet
    tsandbc : list of tuples
        list of tuples of the form (time, bc) for boundary conditions
    res : float
        resistance of the line doublet
    layers : int, array or list
        layer (int) or layers (list or array) in which line doublet is located
    type : string
        type of element, "g" for given, "v" for variable and  "z" for zero.
    name : string
        name of the element
    label : string, optional
        label of the element
    """

    tiny = 1e-8

    def __init__(
        self,
        model,
        xld=0,
        tsandbc=[(0, 0)],
        res="imp",
        layers=0,
        type="",
        name="LineDoublet1DBase",
        label=None,
    ):
        Element.__init__(
            self,
            model,
            nparam=1,
            nunknowns=0,
            layers=layers,
            tsandbc=tsandbc,
            type=type,
            name=name,
            label=label,
        )
        # Defined here and not in Element as other elements can have multiple
        # parameters per layers
        self.nparam = len(self.layers)
        self.xld = float(xld)
        if res == "imp":
            self.res = np.inf
        else:
            self.res = float(res)
        self.model.addelement(self)

    def __repr__(self):
        return self.name + " at " + str(self.xld)

    def initialize(self):
        # control point just on the positive side
        self.xc = np.array([self.xld + self.tiny])
        self.yc = np.zeros(1)
        # control point on the negative side
        self.xcneg = np.array([self.xld - self.tiny])
        self.ycneg = np.zeros(1)
        self.cosout = -np.ones(1)
        self.sinout = np.zeros(1)

        self.ncp = 1
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
        self.resfac = self.aq.Haq[self.layers] / self.res

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, y=0, aq=None):
        """Can be called with only one x value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            if (x - self.xld) < 0.0:
                pot = -0.5 * np.exp((x - self.xld) / aq.lab)
            elif (x - self.xld) >= 0.0:
                pot = 0.5 * np.exp(-(x - self.xld) / aq.lab)
            else:
                raise ValueError("Something wrong with passed x value.")
            rv[:] = self.term * pot
        return rv

    def disvecinf(self, x, y=0, aq=None):
        """Can be called with only one x value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        rvy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            if (x - self.xld) < 0.0:
                qx = 0.5 * np.exp((x - self.xld) / aq.lab) / aq.lab
            elif (x - self.xld) >= 0.0:
                qx = 0.5 * np.exp(-(x - self.xld) / aq.lab) / aq.lab
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
        aq = self.model.aq.find_aquifer_data(self.xld, 0.0)
        for ilay in self.layers:
            ax.plot(
                [self.xld, self.xld],
                [aq.zaqtop[ilay], aq.zaqbot[ilay]],
                "k-",
            )


class LeakyWall1D(LineDoublet1DBase, LeakyWallEquation):
    r"""Leaky line doublet with specified resistance.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xld : float
        x-coordinate of the line doublet
    res : float
        resistance of the line doublet
    layers : int, array or list
        layer (int) or layers (list or array) in which line doublet is located
    label : string or None (default: None)
        label of the element
    """

    def __init__(self, model, xld=0, res="imp", layers=0, label=None):
        super().__init__(
            model,
            xld,
            tsandbc=[(0, 0)],
            res=res,
            layers=layers,
            type="z",
            name="LeakyWall1D",
            label=label,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )


class LeakyLineDoublet1D(LeakyWall1D):
    """Deprecated alias for :class:`.LeakyWall1D`.

    .. deprecated::
        Use :class:`.LeakyWall1D` instead. This alias will be removed in a
        future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LeakyLineDoublet1D is deprecated. Use LeakyWall1D instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
