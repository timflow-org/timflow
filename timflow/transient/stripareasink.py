"""Cross-section area-sink element for transient flow.

Models uniform infiltration over along a strip (cross-section) in
transient simulations.

Example::

    AreaSinkXsection(ml, x1=-50, x2=50, tsandN=[(0, 0.001)], layers=0)
"""

import matplotlib.pyplot as plt
import numpy as np

from timflow.transient.element import Element


class AreaSinkXsection(Element):
    def __init__(
        self,
        model,
        x1,
        x2,
        tsandN=[(0.0, 1.0)],
        layers=0,
        name="AreaSinkXsection",
        label=None,
    ):
        super().__init__(
            model,
            nparam=1,
            nunknowns=0,
            layers=layers,
            tsandbc=tsandN,
            type="g",
            name=name,
            label=label,
            inhomelement=True,
        )
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.model.addelement(self)

    def __repr__(self):
        return f"{self.__class__.__name__}: " + str([self.x1, self.x2])

    def initialize(self):
        self.xc = (self.x1 + self.x2) / 2.0
        self.L = np.abs(self.x2 - self.x1)
        self.aq = self.model.aq.find_aquifer_data(self.xc, 0.0)
        self.setbc()
        self.setflowcoef()
        self.term = self.flowcoef * self.aq.lab**2 * self.aq.coef[self.layers]
        self.dischargeinf = self.aq.coef[0, :] * self.flowcoef
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = 1.0 / self.model.p  # Step function

    def potinf(self, x, y, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            rv[:] = self.term
        return rv

    def disvecinf(self, x, y=0, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        qy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        return qx, qy

    def plot(self, ax, n_arrows=10, **kwargs):
        Ly = self.model.aq.z[0] - self.model.aq.z[-1]
        Lx = self.x2 - self.x1

        for i in np.linspace(self.x1, self.x2, n_arrows):
            xtail = i
            ytail = self.model.aq.z[0] + Ly / 20.0
            dx = 0
            dy = -0.9 * Ly / 20.0
            ax.arrow(
                xtail,
                ytail,
                dx,
                dy,
                width=kwargs.pop("width", Lx / 300.0),
                length_includes_head=kwargs.pop("length_includes_head", True),
                head_width=kwargs.pop("head_width", 4 * Lx / 300.0),
                head_length=kwargs.pop("head_length", 0.4 * Ly / 20.0),
                color=kwargs.pop("color", "k"),
                joinstyle=kwargs.pop("joinstyle", "miter"),
                capstyle=kwargs.pop("capstyle", "projecting"),
            )


class HstarXsection(Element):
    def __init__(
        self,
        model,
        x1,
        x2,
        tsandhstar=[(0.0, 1.0)],
        layers=0,
        name="HstarXsection",
        label=None,
    ):
        super().__init__(
            model,
            nparam=1,
            nunknowns=0,
            layers=layers,
            tsandbc=tsandhstar,
            type="g",
            name=name,
            label=label,
            inhomelement=True,
        )
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.model.addelement(self)

    def __repr__(self):
        return f"{self.__class__.__name__}: " + str([self.x1, self.x2])

    def initialize(self):
        if not np.isfinite(self.x1):
            self.xc = self.x2 - 1e-5
        elif not np.isfinite(self.x2):
            self.xc = self.x1 + 1e-5
        else:
            self.xc = (self.x1 + self.x2) / 2.0
        self.L = np.abs(self.x2 - self.x1)
        self.aq = self.model.aq.find_aquifer_data(self.xc, 0.0)
        self.setbc()
        self.setflowcoef()
        self.resfac = 1.0 / self.aq.c[0]
        self.term = (
            self.resfac * self.flowcoef * self.aq.lab**2 * self.aq.coef[self.layers]
        )
        self.dischargeinf = self.aq.coef[0, :] * self.flowcoef * self.resfac
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )

    def setflowcoef(self):
        self.flowcoef = 1.0 / self.model.p

    def potinf(self, x, y=0.0, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            rv[:] = self.term
        return rv

    def disvecinf(self, x, y=0, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        qy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        return qx, qy

    def plot(self, ax=None, hstar=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        aq = self.model.aq.find_aquifer_data(self.xc, 0.0)
        ztop = aq.z[0]
        Ly = aq.z[0] - aq.z[-1]
        if hstar is None:
            dy = Ly / 20.0
            zdivider = ztop + 1.1 * dy
        else:
            dy = hstar - ztop
            zdivider = hstar + 1

        if np.isfinite(self.x1):
            x1 = self.x1
            ax.plot(
                [x1, x1],
                [ztop, ztop + 1.5 * dy],
                color="k",
                lw=1.0,
                ls="dotted",
            )
        else:
            x1 = ax.get_xlim()[0]

        if np.isfinite(self.x2):
            x2 = self.x2
            ax.plot(
                [x2, x2],
                [zdivider, aq.z[-1]],
                color="k",
                lw=1.0,
                ls="dotted",
            )
        else:
            x2 = ax.get_xlim()[1]

        # water level
        c = kwargs.pop("color", "b")
        lw = kwargs.pop("lw", 1.0)
        ax.plot([x1, x2], [ztop + dy, ztop + dy], lw=lw, color=c, **kwargs)
