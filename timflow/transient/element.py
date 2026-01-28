import inspect  # Used for storing the input
from abc import ABC, abstractmethod

import numpy as np

from .invlapnumba import invlapcomp


class Element(ABC):
    def __init__(
        self,
        model,
        nparam=1,
        nunknowns=0,
        layers=0,
        tsandbc=[(0, 0)],
        type="z",
        name="",
        label=None,
        inhomelement=False,
    ):
        """Element base class.

        Types of elements:
        * 'g': strength is given through time
        * 'v': boundary condition is variable through time
        * 'z': boundary condition is zero through time

        Definition of nlayers, Ncp, Npar, nunknowns:
        * nlayers: Number of layers that the element is screened in,
          as set in Element
        * Ncp: Number of control points along the element
        * nparam: Number of parameters, commonly nlayers * Ncp
        * nunknowns: Number of unknown parameters, commonly zero or Npar
        """
        self.model = model
        self.aq = None  # Set in the initialization function
        self.nparam = nparam  # Number of parameters
        self.nunknowns = nunknowns
        self.layers = np.atleast_1d(layers)
        self.nlayers = len(self.layers)
        self.inhomelement = inhomelement  # tag inhomelements
        #
        tsandbc = np.atleast_2d(tsandbc).astype(float)
        tsandbc_error = (
            "tsandQ or tsandh need to be 2D lists"
            + " or arrays, like [(0, 1), (2, 5), (8, 0)] "
        )
        assert tsandbc.shape[1] == 2, tsandbc_error
        self.tstart, self.bcin = tsandbc[:, 0] - self.model.tstart, tsandbc[:, 1]
        if self.tstart[0] > 0:
            self.tstart = np.hstack((np.zeros(1), self.tstart))
            self.bcin = np.hstack((np.zeros(1), self.bcin))
        #
        # 'z' boundary condition thru time or 'v' boundary condition thru time
        self.type = type
        self.name = name
        self.label = label
        if self.label is not None:
            assert self.label not in self.model.elementdict.keys(), (
                "TTim error: label " + self.label + " already exists"
            )
        self.rzero = 30

    def setbc(self):
        if len(self.tstart) > 1:
            self.bc = np.zeros_like(self.bcin)
            self.bc[0] = self.bcin[0]
            self.bc[1:] = self.bcin[1:] - self.bcin[:-1]
        else:
            self.bc = self.bcin.copy()
        self.ntstart = len(self.tstart)

    @abstractmethod
    def initialize(self):
        """Initialize the element.

        Initialization of terms that cannot be initialized before other elements or
        the aquifer is defined.

        As we don't want to require a certain order of entering elements, these terms
        are initialized when Model.solve is called The initialization class needs to be
        overloaded by all derived classes
        """

    @abstractmethod
    def potinf(self, x, y, aq=None):
        """Returns complex array of size (nparam, naq, npval)."""

    def potential(self, x, y, aq=None):
        """Returns complex array of size (ngvbc, naq, npval)."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        return np.sum(
            self.parameters[:, :, np.newaxis, :] * self.potinf(x, y, aq), axis=1
        )

    def unitpotential(self, x, y, aq=None):
        """Returns complex array of size (naq, npval).

        Can be more efficient for given elements.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        return np.sum(self.potinf(x, y, aq), 0)

    def unitpotentialone(self, x, y, jtime, aq=None):
        """Returns complex array of size (naq, npval).

        Can be more efficient for given elements.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        return np.sum(self.potinfone(x, y, jtime, aq), 0)

    @abstractmethod
    def disvecinf(self, x, y, aq=None):
        """Returns 2 complex arrays of size (nparam, naq, npval)."""

    def disvec(self, x, y, aq=None):
        """Returns 2 complex arrays of size (ngvbc, naq, npval)."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx, qy = self.disvecinf(x, y, aq)
        return np.sum(self.parameters[:, :, np.newaxis, :] * qx, 1), np.sum(
            self.parameters[:, :, np.newaxis, :] * qy, 1
        )

    def unitdisvec(self, x, y, aq=None):
        """Returns 2 complex arrays of size (naq, npval).

        Can be more efficient for given elements.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx, qy = self.disvecinf(x, y, aq)
        return np.sum(qx, 0), np.sum(qy, 0)

    # Functions used to build equations
    def potinflayers(self, x, y, layers=0, aq=None):
        """Layers can be scalar, list, or array.

        returns array of size (len(layers),nparam,npval) only used in building equations
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        pot = self.potinf(x, y, aq)
        rv = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec, 2)
        # first axis needs to be the number of layers
        rv = rv.swapaxes(0, 1)
        return rv[layers, :]

    def potentiallayers(self, x, y, layers=0, aq=None):
        """Returns complex array of size (ngvbc, len(layers),npval).

        Only used in building equations.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        pot = self.potential(x, y, aq)
        phi = np.sum(pot[:, np.newaxis, :, :] * aq.eigvec, 2)
        return phi[:, layers, :]

    def unitpotentiallayers(self, x, y, layers=0, aq=None):
        """Returns complex array of size (len(layers), npval).

        Only used in building equations.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        pot = self.unitpotential(x, y, aq)
        phi = np.sum(pot[np.newaxis, :, :] * aq.eigvec, 1)
        return phi[layers, :]

    def disvecinflayers(self, x, y, layers=0, aq=None):
        """Layers can be scalar, list, or array.

        returns 2 arrays of size (len(layers),nparam,npval) only used in building
        equations
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx, qy = self.disvecinf(x, y, aq)
        rvx = np.sum(qx[:, np.newaxis, :, :] * aq.eigvec, 2)
        rvy = np.sum(qy[:, np.newaxis, :, :] * aq.eigvec, 2)
        # first axis needs to be the number of layers
        rvx = rvx.swapaxes(0, 1)
        rvy = rvy.swapaxes(0, 1)
        return rvx[layers, :], rvy[layers, :]

    def disveclayers(self, x, y, layers=0, aq=None):
        """Returns 2 complex array of size (ngvbc, len(layers), npval).

        Only used in building equations.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx, qy = self.disvec(x, y, aq)
        rvx = np.sum(qx[:, np.newaxis, :, :] * aq.eigvec, 2)
        rvy = np.sum(qy[:, np.newaxis, :, :] * aq.eigvec, 2)
        return rvx[:, layers, :], rvy[:, layers, :]

    def unitdisveclayers(self, x, y, layers=0, aq=None):
        """Returns complex array of size (len(layers), npval).

        Only used in building equations.
        """
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        qx, qy = self.unitdisvec(x, y, aq)
        rvx = np.sum(qx[np.newaxis, :, :] * aq.eigvec, 1)
        rvy = np.sum(qy[np.newaxis, :, :] * aq.eigvec, 1)
        return rvx[layers, :], rvy[layers, :]

    def discharge(self, t, derivative=0):
        """The discharge in each layer.

        Parameters
        ----------
        t : scalar, list or array
            times at which discharge is computed.
            t must be ordered and tmin <= t <= tmax

        Returns
        -------
        array of discharges (nlayers,len(t))
            Discharge in each screen with zeros for layers that are not
            screened
        """
        # Could potentially be more efficient if s is pre-computed for
        # all elements, but may not be worthwhile to store as it is quick now
        time = np.atleast_1d(t).astype(float)
        rv = np.zeros((self.nlayers, len(time)))
        if self.type == "g":
            s = self.dischargeinflayers * self.model.p**derivative
            rv = invlapcomp(
                time,
                s[np.newaxis, :],
                self.model.npint,
                self.model.M,
                self.model.tintervals,
                np.zeros(self.ntstart, dtype="int"),
                self.tstart,
                self.bc,
                self.nlayers,
            )
        else:
            s = np.sum(self.parameters[:, :, np.newaxis, :] * self.dischargeinf, 1)
            s = np.sum(s[:, np.newaxis, :, :] * self.aq.eigvec, 2)
            s = s[:, self.layers, :] * self.model.p**derivative
            rv = invlapcomp(
                time,
                s,
                self.model.npint,
                self.model.M,
                self.model.tintervals,
                self.model.enumber,
                self.model.etstart,
                self.model.ebc,
                self.nlayers,
            )
        return rv

    # this function is kept for testing of version 0.6.6
    def dischargeold(self, t, derivative=0):
        """The discharge in each layer.

        Parameters
        ----------
        t : scalar, list or array
            times at which discharge is computed.
            t must be ordered and tmin <= t <= tmax

        Returns
        -------
        array of discharges (nlayers,len(t))
            Discharge in each screen with zeros for layers that are not
            screened
        """
        # Could potentially be more efficient if s is pre-computed for
        # all elements, but may not be worthwhile to store as it is quick now
        time = np.atleast_1d(t).astype(float)
        if (time[0] < self.model.tmin) or (time[-1] > self.model.tmax):
            print(
                "Warning, some of the times are smaller than tmin or"
                + "larger than tmax; zeros are substituted"
            )
        rv = np.zeros((self.nlayers, np.size(time)))
        if self.type == "g":
            s = self.dischargeinflayers * self.model.p**derivative
            for itime in range(self.ntstart):
                t = time - self.tstart[itime]
                for i in range(self.nlayers):
                    rv[i] += self.bc[itime] * self.model.inverseLapTran(s[i], t)
        else:
            s = np.sum(self.parameters[:, :, np.newaxis, :] * self.dischargeinf, 1)
            s = np.sum(s[:, np.newaxis, :, :] * self.aq.eigvec, 2)
            s = s[:, self.layers, :] * self.model.p**derivative
            for k in range(self.model.ngvbc):
                e = self.model.gvbclist[k]
                for itime in range(e.ntstart):
                    t = time - e.tstart[itime]
                    if t[-1] >= self.model.tmin:  # Otherwise all zero
                        for i in range(self.nlayers):
                            rv[i] += e.bc[itime] * self.model.inverseLapTran(s[k, i], t)
        return rv

    def headinside(self, t):
        print("This function not implemented for this element")
        return

    def storeinput(self, frame):
        self.inputargs, _, _, self.inputvalues = inspect.getargvalues(frame)

    def write(self):
        rv = self.name + "(" + self.model.modelname + ",\n"
        for key in self.inputargs[2:]:  # The first two are ignored
            if isinstance(self.inputvalues[key], np.ndarray):
                rv += (
                    key
                    + " = "
                    + np.array2string(self.inputvalues[key], separator=",")
                    + ",\n"
                )
            elif isinstance(self.inputvalues[key], str):
                rv += key + " = '" + self.inputvalues[key] + "',\n"
            else:
                rv += key + " = " + str(self.inputvalues[key]) + ",\n"
        rv += ")\n"
        return rv

    def run_after_solve(self):  # noqa: B027
        """Function to run after a solution is completed.

        For most elements nothing needs to be done, but for strings of elements some
        arrays may need to be filled.
        """

    @abstractmethod
    def plot(self, ax=None):
        """Plot the element."""
