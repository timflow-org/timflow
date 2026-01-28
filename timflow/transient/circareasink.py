import inspect  # Used for storing the input

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import iv, kv

from .element import Element


class CircAreaSink(Element):
    """Create a circular area-sink with uniform infiltration rate in aquifer layer 0.

    Infiltration rate in length / time, positive for water entering the aquifer.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xc : float
        x-coordinate of center of area-sink
    yc : float
        y-coordinate of center of area-sink
    R : radius of area-sink
    tsandN : list of tuples
        tuples of starting time and infiltration rate after starting time
    label : string or None (default: None)
        label of the area-sink
    """

    def __init__(
        self, model, xc=0, yc=0, R=0.1, tsandN=[(0, 1)], name="CircAreaSink", label=None
    ):
        self.storeinput(inspect.currentframe())
        Element.__init__(
            self,
            model,
            nparam=1,
            nunknowns=0,
            layers=0,
            tsandbc=tsandN,
            type="g",
            name=name,
            label=label,
        )
        self.xc = float(xc)
        self.yc = float(yc)
        self.R = float(R)
        self.model.addelement(self)

    def __repr__(self):
        return self.name + " at " + str((self.xc, self.yc))

    def initialize(self):
        self.aq = self.model.aq.find_aquifer_data(self.xc, self.yc)
        self.setbc()
        self.setflowcoef()
        # Since recharge is in layer 0, and RHS is -N
        self.an = self.aq.coef[0, :] * self.flowcoef
        self.an.shape = (self.aq.naq, self.model.nint, self.model.npint)
        self.termin = self.aq.lab2 * self.R * self.an
        self.termin2 = self.aq.lab2**2 * self.an
        self.terminq = self.R * self.an
        self.termout = self.aq.lab2 * self.R * self.an
        self.i1R = iv(1, self.R / self.aq.lab2)
        self.k1R = kv(1, self.R / self.aq.lab2)
        self.termoutq = self.R * self.an
        self.dischargeinf = self.aq.coef[0, :] * self.flowcoef
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
            r = np.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2)
            # pot = np.zeros(self.model.npint, dtype=complex)
            if r < self.R:
                for i in range(self.aq.naq):
                    for j in range(self.model.nint):
                        # if r / abs(self.aq.lab2[i,j,0]) < self.rzero:
                        rv[0, i, j] = (
                            -self.termin[i, j] * self.K1RI0r(r, i, j) + self.termin2[i, j]
                        )
            else:
                for i in range(self.aq.naq):
                    for j in range(self.model.nint):
                        if (r - self.R) / abs(self.aq.lab2[i, j, 0]) < self.rzero:
                            rv[0, i, j, :] = self.termout[i, j, :] * self.I1RK0r(r, i, j)
        rv.shape = (self.nparam, aq.naq, self.model.npval)
        return rv

    def disvecinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        qy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        if aq == self.aq:
            qr = np.zeros(
                (self.nparam, aq.naq, self.model.nint, self.model.npint), dtype=complex
            )
            r = np.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2)
            if r < self.R:
                for i in range(self.aq.naq):
                    for j in range(self.model.nint):
                        # if r / abs(self.aq.lab2[i,j,0]) < self.rzero:
                        qr[0, i, j] = self.terminq[i, j] * self.K1RI1r(r, i, j)
            else:
                for i in range(self.aq.naq):
                    for j in range(self.model.nint):
                        if (r - self.R) / abs(self.aq.lab2[i, j, 0]) < self.rzero:
                            qr[0, i, j] = self.termoutq[i, j, :] * self.I1RK1r(r, i, j)
            qr.shape = (self.nparam, aq.naq, self.model.npval)
            qx[:] = qr * (x - self.xc) / r
            qy[:] = qr * (y - self.yc) / r
        return qx, qy

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(
            self.xc + self.R * np.cos(np.linspace(0, 2 * np.pi, 100)),
            self.yc + self.R * np.sin(np.linspace(0, 2 * np.pi, 100)),
            "k",
        )

    def K1RI0r(self, rin, iaq, ipint):
        r = rin / self.aq.lab2[iaq, ipint]
        R = self.R / self.aq.lab2[iaq, ipint]
        if np.isinf(self.i1R[iaq, ipint]).any():
            rv = (
                np.sqrt(1 / (4 * r * R))
                * np.exp(r - R)
                * (1 + 3 / (8 * R) - 15 / (128 * R**2) + 315 / (3072 * R**3))
                * (1 + 1 / (8 * r) + 9 / (128 * r**2) + 225 / (3072 * r**3))
            )
        else:
            rv = self.k1R[iaq, ipint] * iv(0, r)
        return rv

    def I1RK0r(self, rin, iaq, ipint):
        r = rin / self.aq.lab2[iaq, ipint]
        R = self.R / self.aq.lab2[iaq, ipint]
        if np.isinf(self.i1R[iaq, ipint]).any():
            rv = (
                np.sqrt(1 / (4 * r * R))
                * np.exp(R - r)
                * (1 - 3 / (8 * R) - 15 / (128 * R**2) - 315 / (3072 * R**3))
                * (1 - 1 / (8 * r) + 9 / (128 * r**2) - 225 / (3072 * r**3))
            )
        else:
            rv = self.i1R[iaq, ipint] * kv(0, r)
        return rv

    def K1RI1r(self, rin, iaq, ipint):
        r = rin / self.aq.lab2[iaq, ipint]
        R = self.R / self.aq.lab2[iaq, ipint]
        if np.isinf(self.i1R[iaq, ipint]).any():
            rv = (
                np.sqrt(1 / (4 * r * R))
                * np.exp(r - R)
                * (1 + 3 / (8 * R) - 15 / (128 * R**2) + 315 / (3072 * R**3))
                * (1 - 3 / (8 * r) - 15 / (128 * r**2) - 315 / (3072 * r**3))
            )
        else:
            rv = self.k1R[iaq, ipint] * iv(1, r)
        return rv

    def I1RK1r(self, rin, iaq, ipint):
        r = rin / self.aq.lab2[iaq, ipint]
        R = self.R / self.aq.lab2[iaq, ipint]
        if np.isinf(self.i1R[iaq, ipint]).any():
            rv = (
                np.sqrt(1 / (4 * r * R))
                * np.exp(R - r)
                * (1 - 3 / (8 * R) - 15 / (128 * R**2) - 315 / (3072 * R**3))
                * (1 + 3 / (8 * r) - 15 / (128 * r**2) + 315 / (3072 * r**3))
            )
        else:
            rv = self.i1R[iaq, ipint] * kv(1, r)
        return rv
