"""Aquifer data structures and helpers.

Defines `AquiferData` and `Aquifer` used to store aquifer properties
and derived parameters used for computations throughout timflow.

Example::

    aq = SimpleAquifer(naq=2)

"""

import inspect  # Used for storing the input

import numpy as np
import pandas as pd

from timflow.steady.constant import ConstantStar

__all__ = ["Aquifer", "SimpleAquifer"]


class AquiferData:
    def __init__(self, model, kaq, c, z, npor, ltype, model3d=False):
        """Initialize aquifer data.

        Parameters
        ----------
        model : Model
            The model to which the aquifer belongs.
        kaq : float or array of floats
            Hydraulic conductivity of the aquifer(s).
        c : float or array of floats
            Resistance of the leaky layers.
        z : float or array of floats
            Elevations of the tops of the layers.
        npor : float or array of floats
            Porosity of the aquifer(s).
        ltype : string or array of strings
            Type of the layers: 'a' for aquifer, 'l' for leaky layer.
        model3d : bool, optional
            Whether the model is Model3D (default is False).
        """
        # All input variables except model should be numpy arrays
        # That should be checked outside this function
        self.model = model
        # Needed for heads
        self.kaq = np.atleast_1d(kaq)
        self.naq = len(kaq)
        self.c = np.atleast_1d(c)
        self.hstar = None
        # Needed for tracing
        self.z = np.atleast_1d(z)
        self.Hlayer = self.z[:-1] - self.z[1:]  # thickness of all layers
        self.nlayers = len(self.z) - 1
        self.npor = np.atleast_1d(npor)
        self.ltype = np.atleast_1d(ltype)
        # tag indicating whether an aquifer is Laplace (confined on top)
        if self.ltype[0] == "a":
            self.ilap = 1
        else:
            self.ilap = 0
        #
        self.area = 1e200  # Smaller than default of ml.aq so that inhom is found
        self.layernumber = np.zeros(self.nlayers, dtype="int")
        self.layernumber[self.ltype == "a"] = np.arange(self.naq)
        self.layernumber[self.ltype == "l"] = np.arange(self.nlayers - self.naq)
        if self.ltype[0] == "a":
            self.layernumber[self.ltype == "l"] += (
                1  # first leaky layer below first aquifer layer
            )
        self.zaqtop = self.z[:-1][self.ltype == "a"]
        self.zaqbot = self.z[1:][self.ltype == "a"]
        self.Haq = self.zaqtop - self.zaqbot
        self.T = self.kaq * self.Haq
        self.Tcol = self.T[:, np.newaxis]
        self.zlltop = self.z[:-1][self.ltype == "l"]
        self.zllbot = self.z[1:][self.ltype == "l"]
        if self.ltype[0] == "a":
            self.zlltop = np.hstack((self.z[0], self.zlltop))
            self.zllbot = np.hstack((self.z[0], self.zllbot))
        self.Hll = self.zlltop - self.zllbot
        self.nporaq = self.npor[self.ltype == "a"]
        if self.ltype[0] == "a":
            self.nporll = np.ones(len(self.npor[self.ltype == "l"]) + 1)
            self.nporll[1:] = self.npor[self.ltype == "l"]
        else:
            self.nporll = self.npor[self.ltype == "l"]
        self.model3d = model3d

    def initialize(self):
        self.elementlist = []  # Elementlist of aquifer
        # Recompute T for when kaq is changed
        self.T = self.kaq * self.Haq
        self.Tcol = self.T.reshape(self.naq, 1)
        d0 = 1.0 / (self.c * self.T)
        d0[:-1] += 1.0 / (self.c[1:] * self.T[:-1])
        dp1 = -1.0 / (self.c[1:] * self.T[1:])
        dm1 = -1.0 / (self.c[1:] * self.T[:-1])
        A = np.diag(dm1, -1) + np.diag(d0, 0) + np.diag(dp1, 1)
        w, v = np.linalg.eig(A)
        # take the real part for the rare occasion that the eig
        # function returns a complex answer with very small
        # imaginary part
        w = w.real
        v = v.real
        # sort lab in decending order, hence w in ascending order
        index = np.argsort(abs(w))
        w = w[index]
        v = v[:, index]
        if self.ilap:
            self.lab = np.zeros(self.naq)
            self.lab[1:] = 1.0 / np.sqrt(w[1:])
            self.zeropluslab = (
                self.lab
            )  # to be deprecated when new lambda is fully implemented
            v[:, 0] = self.T / np.sum(self.T)  # first column is normalized T
        else:
            self.lab = 1.0 / np.sqrt(w)
        self.eigvec = v
        self.coef = np.linalg.solve(v, np.diag(np.ones(self.naq))).T

    def add_element(self, e):
        self.elementlist.append(e)
        if isinstance(e, ConstantStar):
            self.hstar = e.hstar

    def isinside(self, x, y):
        raise Exception("Must overload AquiferData.isinside()")

    def storeinput(self, frame):
        self.inputargs, _, _, self.inputvalues = inspect.getargvalues(frame)

    def findlayer(self, z):
        """Returns layer-number, layer-type and model-layer-number."""
        if z > self.z[0]:
            modellayer, ltype = -1, "above"
            layernumber = None
        elif z < self.z[-1]:
            modellayer, ltype = len(self.layernumber), "below"
            layernumber = None
        else:
            modellayer = np.argwhere((z <= self.z[:-1]) & (z >= self.z[1:]))[0, 0]
            layernumber = self.layernumber[modellayer]
            ltype = self.ltype[modellayer]
        return layernumber, ltype, modellayer

    def summary(self):
        """Get summary of aquifer parameters.

        Returns
        -------
        pd.DataFrame
            Summary of aquifer parameters including layer type, thickness,
            hydraulic conductivity, and resistance.
        """
        if self.model3d:
            add_cols = ["kzoverkh"]
        else:
            add_cols = []
        summary = pd.DataFrame(
            index=range(self.nlayers),
            columns=["layer", "layer_type", "H", "k_h"] + add_cols + ["c"],
        )
        summary.index.name = "#"
        layertype = {"a": "aquifer", "l": "leaky layer"}
        summary["layer_type"] = [layertype[lt] for lt in self.ltype]
        maskaq = self.ltype == "a"
        if self.ilap == 1:  # confined on top
            summary.loc[maskaq, "H"] = self.Haq
            summary.loc[maskaq, "k_h"] = self.kaq
            if self.model3d:
                summary.loc[maskaq, "c"] = self.c
                summary.loc[0, "c"] = np.nan  # reset confined resistance to nan
                summary.loc[maskaq, "kzoverkh"] = self.kzoverkh
            else:
                summary.loc[~maskaq, "H"] = self.Hll[1:]
                summary.loc[~maskaq, "c"] = self.c[1:]
        else:
            summary.loc[maskaq, "H"] = self.Haq
            summary.loc[maskaq, "k_h"] = self.kaq
            if self.model3d:
                summary.loc[~maskaq, "H"] = self.Hll[0]
                summary.loc[maskaq, "c"] = self.c
                summary.loc[maskaq, "kzoverkh"] = self.kzoverkh
            else:
                summary.loc[~maskaq, "H"] = self.Hll
                summary.loc[~maskaq, "c"] = self.c
        summary.loc[:, "layer"] = self.layernumber
        return summary  # .set_index("layer")


class Aquifer(AquiferData):
    """Background aquifer for models.

    Extends AquiferData and supports inhomogeneities within a model.
    """

    def __init__(self, model, kaq, c, z, npor, ltype, model3d=False):
        super().__init__(model, kaq, c, z, npor, ltype, model3d=model3d)
        self.inhomdict = {}
        self.area = 1e300  # Needed to find smallest inhom

    def initialize(self):
        super().initialize()
        # 2 passes to ensure all data is present prior to creating elements
        for inhom in self.inhomdict.values():
            inhom.initialize()
        for inhom in self.inhomdict.values():
            inhom.create_elements()

    def add_inhom(self, inhom):
        inhom_number = len(self.inhomdict)
        if not hasattr(inhom, "name") or inhom.name is None:
            inhom.name = f"inhom{inhom_number:02g}"
        if inhom.name in self.inhomdict:
            raise ValueError(f"Inhomogeneity name '{inhom.name}' already exists.")
        self.inhomdict[inhom.name] = inhom
        return inhom_number

    def find_aquifer_data(self, x, y):
        rv = self
        for inhom in self.inhomdict.values():
            if inhom.isinside(x, y):
                if inhom.area < rv.area:
                    rv = inhom
        return rv


class SimpleAquifer(Aquifer):
    """Simple aquifer for cross-section models.

    Only the number of aquifers has to be specified.

    Parameters
    ----------
    naq : int
        Number of aquifers.
    """

    def __init__(self, naq):
        self.naq = naq
        self.inhomdict = {}
        self.area = 1e300  # Needed to find smallest inhomogeneity
        self.elementlist = []

    def __repr__(self):
        return f"Simple Aquifer: {self.naq} aquifer(s)"

    def initialize(self):
        # 2 passes to ensure all data is present prior to creating elements
        for inhom in self.inhomdict.values():
            inhom.initialize()
        for inhom in self.inhomdict.values():
            inhom.create_elements()
