"""Well elements for TTim.

Provides classes to model wells with specified discharge or head in
transient simulations.

Example::

    Well(ml, xw=100, yw=200, tsandbc=[(0, 1000)], layers=[0])
"""

import inspect  # Used for storing the input

import matplotlib.pyplot as plt
import numpy as np

# from scipy.special import iv  # Needed for K1 in Well class, and in CircInhom
from scipy.special import kv

from timflow.transient.element import Element
from timflow.transient.equation import HeadEquation, WellBoreStorageEquation
from timflow.transient.invlapnumba import invlapcomp


class WellBase(Element):
    """Well Base Class.

    All Well elements are derived from this class
    """

    def __init__(
        self,
        model,
        xw=0,
        yw=0,
        rw=0.1,
        tsandbc=[(0, 1)],
        res=0,
        layers=0,
        type="",
        name="WellBase",
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
        # Defined here and not in Element as other elements can have multiple
        # parameters per layers
        self.nparam = len(self.layers)
        self.xw = float(xw)
        self.yw = float(yw)
        self.rw = float(rw)
        self.res = np.atleast_1d(res).astype(np.float64)
        self.model.addelement(self)

    def __repr__(self):
        return self.name + " at " + str((self.xw, self.yw))

    def initialize(self):
        # Control point to make sure the point is always the same for
        # all elements
        self.xc = np.array([self.xw + self.rw])
        self.yc = np.array([self.yw])
        self.ncp = 1
        self.aq = self.model.aq.find_aquifer_data(self.xw, self.yw)
        self.setbc()
        coef = self.aq.coef[self.layers, :]
        laboverrwk1 = self.aq.lab / (self.rw * kv(1, self.rw / self.aq.lab))
        self.setflowcoef()
        # term is shape (self.nparam,self.aq.naq,self.model.npval)
        self.term = -1.0 / (2 * np.pi) * laboverrwk1 * self.flowcoef * coef
        self.term2 = self.term.reshape(
            self.nparam, self.aq.naq, self.model.nint, self.model.npint
        )
        self.dischargeinf = self.flowcoef * coef
        self.dischargeinflayers = np.sum(
            self.dischargeinf * self.aq.eigvec[self.layers, :, :], 1
        )
        # Q = (h - hw) / resfach
        self.resfach = self.res / (2 * np.pi * self.rw * self.aq.Haq[self.layers])
        # Q = (Phi - Phiw) / resfacp
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
            r = np.sqrt((x - self.xw) ** 2 + (y - self.yw) ** 2)
            pot = np.zeros(self.model.npint, dtype=complex)
            if r < self.rw:
                r = self.rw  # If at well, set to at radius
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    if r / abs(self.aq.lab2[i, j, 0]) < self.rzero:
                        pot[:] = kv(0, r / self.aq.lab2[i, j, :])
                        # quicker?
                        # bessel.k0besselv( r / self.aq.lab2[i,j,:], pot )
                        rv[:, i, j, :] = self.term2[:, i, j, :] * pot
        return rv.reshape((self.nparam, aq.naq, self.model.npval))

    def potinfone(self, x, y, jtime, aq=None):
        """Can be called with only one x,y value for time interval jtime."""
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npint), dtype=complex)
        if aq == self.aq:
            r = np.sqrt((x - self.xw) ** 2 + (y - self.yw) ** 2)
            pot = np.zeros(self.model.npint, dtype=complex)
            if r < self.rw:
                r = self.rw  # If at well, set to at radius
            for i in range(self.aq.naq):
                if r / abs(self.aq.lab2[i, jtime, 0]) < self.rzero:
                    pot[:] = kv(0, r / self.aq.lab2[i, jtime, :])
                    rv[:, i, :] = self.term2[:, i, jtime, :] * pot
        # rv = rv.reshape((self.nparam, aq.naq, self.model.npval))
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
            r = np.sqrt((x - self.xw) ** 2 + (y - self.yw) ** 2)
            # pot = np.zeros(self.model.npint, dtype=complex)
            if r < self.rw:
                r = self.rw  # If at well, set to at radius
            for i in range(self.aq.naq):
                for j in range(self.model.nint):
                    if r / abs(self.aq.lab2[i, j, 0]) < self.rzero:
                        qr[:, i, j, :] = (
                            self.term2[:, i, j, :]
                            * kv(1, r / self.aq.lab2[i, j, :])
                            / self.aq.lab2[i, j, :]
                        )
            qr = qr.reshape((self.nparam, aq.naq, self.model.npval))
            qx[:] = qr * (x - self.xw) / r
            qy[:] = qr * (y - self.yw) / r
        return qx, qy

    def headinside(self, t, derivative=0):
        """Returns head inside the well for the layers that the well is screened in.

        Parameters
        ----------
        t : float, list or array
            time for which head is computed

        Returns
        -------
        Q : array of size `nscreens, ntimes`
            nsreens is the number of layers with a well screen
        """
        return self.model.head(self.xc[0], self.yc[0], t, derivative=derivative)[
            self.layers
        ] - self.resfach[:, np.newaxis] * self.discharge(t, derivative=derivative)

    def plot(self, ax=None, layer=None):
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="datalim")
        if layer is None or np.isin(self.layers, layer).any():
            ax.plot(self.xw, self.yw, "k.")

    def changetrace(
        self, xyzt1, xyzt2, aq, layer, ltype, modellayer, direction, hstepmax
    ):
        changed = False
        terminate = False
        xyztnew = 0
        message = None
        hdistance = np.sqrt((xyzt1[0] - self.xw) ** 2 + (xyzt1[1] - self.yw) ** 2)
        if hdistance < hstepmax:
            if ltype == "a":
                if (layer == self.layers).any():  # in a layer where well is screened
                    layernumber = np.where(self.layers == layer)[0][0]
                    dis = self.discharge(xyzt1[3])[layernumber, 0]
                    if (dis > 0 and direction > 0) or (dis < 0 and direction < 0):
                        vx, vy, vz = self.model.velocomp(*xyzt1)
                        tstep = np.sqrt(
                            (xyzt1[0] - self.xw) ** 2 + (xyzt1[1] - self.yw) ** 2
                        ) / np.sqrt(vx**2 + vy**2)
                        xnew = self.xw
                        ynew = self.yw
                        znew = xyzt1[2] + tstep * vz * direction
                        tnew = xyzt1[3] + tstep
                        xyztnew = np.array([xnew, ynew, znew, tnew])
                        changed = True
                        terminate = True
        if terminate:
            if self.label:
                message = "reached well element with label: " + self.label
            else:
                message = "reached element of type well: " + str(self)
        return changed, terminate, xyztnew, message


class DischargeWell(WellBase):
    r"""Well with a specified discharge for each layer that the well is screened in.

    This is not very common and is likely only used for testing and comparison with
    other codes. The discharge must be specified for each screened layer. The resistance
    of the screen may be specified. The head is computed such that the discharge
    :math:`Q_i` in layer :math:`i` is computed as

    .. math::
        Q_i = 2\pi r_wH_i(h_i - h_w)/c

    where :math:`c` is the resistance of the well screen and :math:`h_w` is
    the head inside the well.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xw : float
        x-coordinate of the well
    yw : float
        y-coordinate of the well
    tsandQ : list of tuples
        tuples of starting time and discharge after starting time
    rw : float
        radius of the well
    res : float
        resistance of the well screen
    layers : int, array or list
        layer (int) or layers (list or array) where well is screened
    label : string or None (default: None)
        label of the well

    Examples
    --------
    Example of a well that pumps with a discharge of 100 between times
    10 and 50, with a discharge of 20 between times 50 and 200, and zero
    discharge after time 200.

    >>> Well(ml, tsandQ=[(10, 100), (50, 20), (200, 0)])
    """

    def __init__(
        self, model, xw=0, yw=0, tsandQ=[(0, 1)], rw=0.1, res=0, layers=0, label=None
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            xw,
            yw,
            rw,
            tsandbc=tsandQ,
            res=res,
            layers=layers,
            type="g",
            name="DischargeWell",
            label=label,
        )


class Well(WellBase, WellBoreStorageEquation):
    r"""Create a well with a specified discharge.

    The well may be screened in multiple layers. The discharge is distributed across
    the layers such that the head inside the well is the same in all screened layers.
    Wellbore storage and skin effect may be taken into account. The head is computed
    such that the discharge :math:`Q_i` in layer :math:`i` is computed as

    .. math::
        Q_i = 2\pi r_wH_i(h_i - h_w)/c

    where :math:`c` is the resistance of the well screen and :math:`h_w` is
    the head inside the well.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xw : float
        x-coordinate of the well
    yw : float
        y-coordinate of the well
    rw : float
        radius of the well
    tsandQ : list of tuples
        tuples of starting time and discharge after starting time
    res : float
        resistance of the well screen
    rc : float
        radius of the caisson, the pipe where the water table inside
        the well flucuates, which accounts for the wellbore storage
    layers : int, array or list
        layer (int) or layers (list or array) where well is screened
    wbstype : string
        'pumping': Q is the discharge of the well
        'slug': volume of water instantaneously taken out of the well
    label : string (default: None)
        label of the well
    """

    def __init__(
        self,
        model,
        xw=0,
        yw=0,
        rw=0.1,
        tsandQ=[(0, 1)],
        res=0,
        rc=None,
        layers=0,
        wbstype="pumping",
        label=None,
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            xw,
            yw,
            rw,
            tsandbc=tsandQ,
            res=res,
            layers=layers,
            type="v",
            name="Well",
            label=label,
        )
        if (rc is None) or (rc <= 0):
            self.rc = np.zeros(1)
        else:
            self.rc = np.atleast_1d(rc).astype("float")
        # hdiff is not used right now, but may be used in the future
        self.hdiff = None
        # if hdiff is not None:
        #    self.hdiff = np.atleast_1d(hdiff)
        #    assert len(self.hdiff) == self.nlayers - 1, 'hdiff needs to
        # have length len(layers) -1'
        # else:
        #    self.hdiff = hdiff
        self.nunknowns = self.nparam
        self.wbstype = wbstype

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        if self.wbstype == "pumping":
            self.flowcoef = 1.0 / self.model.p  # Step function
        elif self.wbstype == "slug":
            self.flowcoef = 1.0  # Delta function


class HeadWell(WellBase, HeadEquation):
    r"""Create a well with a specified head inside the well.

    The well may be screened in multiple layers. The resistance of the screen may be
    specified. The head is computed such that the discharge :math:`Q_i` in layer
    :math:`i` is computed as

    .. math::
        Q_i = 2\pi r_wH_i(h_i - h_w)/c

    where :math:`c` is the resistance of the well screen and :math:`h_w` is
    the head inside the well.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xw : float
        x-coordinate of the well
    yw : float
        y-coordinate of the well
    rw : float
        radius of the well
    tsandh : list of tuples
        tuples of starting time and discharge after starting time
    res : float
        resistance of the well screen
    layers : int, array or list
        layer (int) or layers (list or array) where well is screened
    label : string (default: None)
        label of the well
    """

    def __init__(
        self, model, xw=0, yw=0, rw=0.1, tsandh=[(0, 1)], res=0, layers=0, label=None
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            xw,
            yw,
            rw,
            tsandbc=tsandh,
            res=res,
            layers=layers,
            type="v",
            name="HeadWell",
            label=label,
        )
        self.nunknowns = self.nparam

    def initialize(self):
        super().initialize()
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        # Needed in solving for a unit head
        self.pc = self.aq.T[self.layers]


class WellTest(WellBase):
    def __init__(
        self,
        model,
        xw=0,
        yw=0,
        tsandQ=[(0, 1)],
        rw=0.1,
        res=0,
        layers=0,
        label=None,
        fp=None,
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            xw,
            yw,
            rw,
            tsandbc=tsandQ,
            res=res,
            layers=layers,
            type="g",
            name="DischargeWell",
            label=label,
        )
        self.fp = fp

    def setflowcoef(self):
        """Separate function so that this can be overloaded for other types."""
        self.flowcoef = self.fp


class WellStringBase(Element):
    """Base class for multiple connected wells in transient flow."""

    def __init__(
        self,
        model,
        xy,
        tsandbc=[(0, 1)],
        layers=0,
        type="",
        name="WellStringBase",
        label=None,
    ):
        super().__init__(
            model,
            nparam=1,
            nunknowns=0,
            layers=0,
            tsandbc=tsandbc,
            type=type,
            name=name,
            label=label,
        )
        self.xy = np.atleast_2d(xy).astype(float)
        self.xw = self.xy[:, 0]
        self.yw = self.xy[:, 1]
        self.nw = len(self.xw)

        if isinstance(layers, (int, np.integer)):
            layers = layers * np.ones((self.nw, 1), dtype=int)
        elif isinstance(layers, (list, tuple)):
            try:
                nlayers = max(len(layers[i]) for i in range(self.nw))
                layers = np.array(layers) * np.ones((self.nw, nlayers), dtype=int)
            except ValueError:
                pass
            except TypeError:
                layers = np.atleast_1d(layers) * np.ones((self.nw, 1), dtype=int)
        elif isinstance(layers, np.ndarray):
            if layers.ndim == 1:
                layers = layers * np.ones((self.nw, layers.shape[0]), dtype=int)
            else:
                assert layers.shape[0] == self.nw, (
                    "layers array must be shape (nwells, nlayers)"
                )

        self.layers = layers
        self.nlayers_well = []
        for i in range(self.nw):
            self.nlayers_well.append(len(np.atleast_1d(self.layers[i])))
        self.wlist = []

    def __repr__(self):
        return self.name + " with nodes " + str(self.xy)

    def initialize(self):
        for w in self.wlist:
            w.initialize()

        self.aq = []
        for w in self.wlist:
            if w.aq not in self.aq:
                self.aq.append(w.aq)

        self.nparam = sum(w.nparam for w in self.wlist)
        self.nunknowns = self.nparam
        self.parameters = np.zeros(
            (self.model.ngvbc, self.nparam, self.model.npval), dtype=complex
        )
        self.setbc()

        self.resfach = []
        self.resfacp = []
        self.dischargeinf = np.zeros(
            (self.nparam, self.model.aq.naq, self.model.npval), dtype=complex
        )
        self.dischargeinflayers = np.zeros((self.nparam, self.model.npval), dtype=complex)
        self.xc = np.zeros(self.nw)
        self.yc = np.zeros(self.nw)

        j = 0
        for i, w in enumerate(self.wlist):
            nwparam = w.nparam
            self.resfach.extend(w.resfach.tolist())
            self.resfacp.extend(w.resfacp.tolist())
            self.dischargeinf[j : j + nwparam] = w.dischargeinf
            self.dischargeinflayers[j : j + nwparam] = w.dischargeinflayers
            self.xc[i] = w.xc[0]
            self.yc[i] = w.yc[0]
            j += nwparam
        self.resfach = np.array(self.resfach)
        self.resfacp = np.array(self.resfacp)

    def potinf(self, x, y, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rv = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        j = 0
        for w in self.wlist:
            rv[j : j + w.nparam] = w.potinf(x, y, aq)
            j += w.nparam
        return rv

    def disvecinf(self, x, y, aq=None):
        if aq is None:
            aq = self.model.aq.find_aquifer_data(x, y)
        rvx = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        rvy = np.zeros((self.nparam, aq.naq, self.model.npval), dtype=complex)
        j = 0
        for w in self.wlist:
            qx, qy = w.disvecinf(x, y, aq)
            rvx[j : j + w.nparam] = qx
            rvy[j : j + w.nparam] = qy
            j += w.nparam
        return rvx, rvy

    def equation(self):
        pass

    # mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
    # rhs = np.zeros(
    #     (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
    # )
    # irow = 0
    # jcol_self = int(
    #     np.sum(
    #         [
    #             e.nunknowns
    #             for e in self.model.elementlist[: self.model.elementlist.index(self)]
    #         ]
    #     )
    # )
    # jcol_self_well = jcol_self
    # iself = self.model.elementlist.index(self)
    # for w in self.wlist:
    #     ieq = 0
    #     for e in self.model.elementlist:
    #         if e.nunknowns > 0:
    #             mat[irow : irow + w.nlayers, ieq : ieq + e.nunknowns, :] = (
    #                 e.potinflayers(w.xc[0], w.yc[0], w.layers)
    #             )
    #             ieq += e.nunknowns
    #     for i in range(self.model.ngbc):
    #         rhs[irow : irow + w.nlayers, i, :] -= self.model.gbclist[
    #             i
    #         ].unitpotentiallayers(w.xc[0], w.yc[0], w.layers)
    #     for i in range(w.nlayers):
    #         mat[irow + i, jcol_self_well + i, :] -= (
    #             w.resfacp[i] * w.dischargeinflayers[i]
    #         )
    #     irow += w.nlayers
    #     jcol_self_well += w.nunknowns
    # return mat, rhs

    def run_after_solve(self):
        i = 0
        for w in self.wlist:
            w.parameters[:] = self.parameters[:, i : i + w.nparam, :]
            i += w.nparam

    def discharge(self, t, derivative=0):
        time = np.atleast_1d(t).astype(float)
        s = (
            self.parameters * self.dischargeinflayers[np.newaxis, :, :]
        ) * self.model.p**derivative
        qparam = invlapcomp(
            time,
            s,
            self.model.npint,
            self.model.M,
            self.model.tintervals,
            self.model.enumber,
            self.model.etstart,
            self.model.ebc,
            self.nparam,
        )
        q = np.zeros((self.model.aq.naq, len(time)))
        j = 0
        for w in self.wlist:
            wlayers = np.atleast_1d(w.layers)
            q[wlayers, :] += qparam[j : j + w.nparam, :]
            j += w.nparam
        return q

    def discharge_per_well(self, t, derivative=0):
        time = np.atleast_1d(t).astype(float)
        s = (
            self.parameters * self.dischargeinflayers[np.newaxis, :, :]
        ) * self.model.p**derivative
        qparam = invlapcomp(
            time,
            s,
            self.model.npint,
            self.model.M,
            self.model.tintervals,
            self.model.enumber,
            self.model.etstart,
            self.model.ebc,
            self.nparam,
        )
        q = np.zeros((self.model.aq.naq, self.nw, len(time)))
        j = 0
        for i, w in enumerate(self.wlist):
            wlayers = np.atleast_1d(w.layers)
            q[wlayers, i, :] = qparam[j : j + w.nparam, :]
            j += w.nparam
        return q

    def headinside(self, t, derivative=0):
        return self.wlist[0].headinside(t, derivative=derivative)[0]

    def plot(self, ax=None, layer=None):
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="datalim")
        for iw, w in enumerate(self.wlist):
            if (layer is None) or np.isin(layer, np.atleast_1d(self.layers[iw])).any():
                ax.plot(w.xw, w.yw, "k.")


class WellString(WellStringBase):
    """String of wells with specified total transient discharge."""

    def __init__(
        self,
        model,
        xy,
        tsandQ=[(0, 1)],
        rw=0.1,
        res=0.0,
        layers=0,
        rc=None,
        label=None,
    ):
        self.storeinput(inspect.currentframe())
        super().__init__(
            model,
            xy,
            tsandbc=tsandQ,
            layers=layers,
            type="v",
            name="WellString",
            label=label,
        )
        self.rw = rw
        self.res = res
        self.rc = rc
        self.tsandQ = tsandQ
        self.model.addelement(self)

    def initialize(self):
        self.wlist = []
        for i in range(self.nw):
            w = Well(
                self.model,
                xw=self.xw[i],
                yw=self.yw[i],
                rw=self.rw,
                tsandQ=self.tsandQ,
                res=self.res,
                rc=self.rc,
                layers=self.layers[i],
                wbstype="pumping",
                label=None,
            )
            self.model.removeelement(w)
            self.wlist.append(w)
        self.flowcoef = 1.0 / self.model.p
        super().initialize()

    def equation(self):
        mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )

        xcp = []
        ycp = []
        lcp = []
        for w in self.wlist:
            wlayers = np.atleast_1d(w.layers)
            for ilay in wlayers:
                xcp.append(w.xc[0])
                ycp.append(w.yc[0])
                lcp.append(int(ilay))

        iself = self.model.elementlist.index(self)
        jself = int(np.sum([e.nunknowns for e in self.model.elementlist[:iself]]))

        # Equations 0..nunknowns-2: all inside heads are equal.
        for irow in range(self.nunknowns - 1):
            layer0 = np.atleast_1d(lcp[irow])
            layer1 = np.atleast_1d(lcp[irow + 1])
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    head0 = (
                        e.potinflayers(xcp[irow], ycp[irow], layer0)
                        / self.model.aq.find_aquifer_data(xcp[irow], ycp[irow]).T[layer0][
                            :, np.newaxis, np.newaxis
                        ]
                    )
                    head1 = (
                        e.potinflayers(xcp[irow + 1], ycp[irow + 1], layer1)
                        / self.model.aq.find_aquifer_data(xcp[irow + 1], ycp[irow + 1]).T[
                            layer1
                        ][:, np.newaxis, np.newaxis]
                    )
                    mat[irow, ieq : ieq + e.nunknowns, :] = head0[0] - head1[0]
                    ieq += e.nunknowns
            for ig in range(self.model.ngbc):
                gh0 = (
                    self.model.gbclist[ig].unitpotentiallayers(
                        xcp[irow], ycp[irow], layer0
                    )
                    / self.model.aq.find_aquifer_data(xcp[irow], ycp[irow]).T[layer0][
                        :, np.newaxis
                    ]
                )
                gh1 = (
                    self.model.gbclist[ig].unitpotentiallayers(
                        xcp[irow + 1], ycp[irow + 1], layer1
                    )
                    / self.model.aq.find_aquifer_data(xcp[irow + 1], ycp[irow + 1]).T[
                        layer1
                    ][:, np.newaxis]
                )
                rhs[irow, ig, :] -= gh0[0] - gh1[0]

            # Correct for screen resistance on the two compared unknowns.
            mat[irow, jself + irow, :] -= (
                self.resfach[irow] * self.dischargeinflayers[irow]
            )
            mat[irow, jself + irow + 1, :] += (
                self.resfach[irow + 1] * self.dischargeinflayers[irow + 1]
            )

        # Last equation: sum of all screen discharges equals specified well-string Q.
        mat[-1, jself : jself + self.nunknowns, :] = 1.0
        if self.type == "v":
            ivbc = self.model.vbclist.index(self)
            rhs[-1, self.model.ngbc + ivbc, :] = 1.0
        return mat, rhs
