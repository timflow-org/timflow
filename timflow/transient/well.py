import inspect  # Used for storing the input

import matplotlib.pyplot as plt
import numpy as np

# from scipy.special import iv  # Needed for K1 in Well class, and in CircInhom
from scipy.special import kv

from timflow.transient.element import Element
from timflow.transient.equation import HeadEquation, WellBoreStorageEquation


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
        rv.shape = (self.nparam, aq.naq, self.model.npval)
        return rv

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
        # rv.shape = (self.nparam, aq.naq, self.model.npval)
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
            qr.shape = (self.nparam, aq.naq, self.model.npval)
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
