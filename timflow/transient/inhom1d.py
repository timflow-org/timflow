import matplotlib.pyplot as plt
import numpy as np

from ttim.aquifer import AquiferData
from ttim.aquifer_parameters import param_3d, param_maq
from ttim.linesink1d import FluxDiffLineSink1D, HeadDiffLineSink1D
from ttim.stripareasink import AreaSinkXsection, HstarXsection


class Xsection(AquiferData):
    """Base class for a cross-section inhomogeneity.

    Parameters
    ----------
    model : Model
        Model to add the cross-section to, usually an instance of ModelXsection.
    x1 : float
        x-coordinate of the left boundary of the cross-section.
    x2 : float
        x-coordinate of the right boundary of the cross-section.
    kaq : array
        Hydraulic conductivities of the aquifers.
    z : array
        Elevations of the tops and bottoms of the layers.
    Haq : array
        Thicknesses of the aquifers.
    Hll : array
        Thicknesses of the leaky layers.
    c : array
        Resistance of the leaky layers.
    Saq : array
        Specific storage of the aquifers.
    Sll : array
        Specific storage of the leaky layers.
    poraq : array
        Porosities of the aquifers.
    porll : array
        Porosities of the leaky layers.
    ltype : array
        Type of each layer. 'a' for aquifer, 'l' for leaky layer.
    topboundary : str
        Type of top boundary. Can be 'conf' for confined, 'semi' for semi-confined
        or "leaky" for a leaky top boundary.
    phreatictop : bool
        If true, interpret the first specific storage coefficient as specific
        yield., i.e. it is not multiplied by aquifer thickness.
    tsandhstar : list of tuples
        list containing time and water level pairs for the hstar boundary condition.
    tsandN : list of tuples
        list containing time and infiltration pairs for the infiltration boundary
        condition.
    kzoverkh: float, optional,
        anisotropy factor for vertical resistance, kzoverkh = kz / kh. Default is 1.
    model3d : bool, optional
        Boolean indicating whether model is Model3D-type.
    name : str
        Name of the cross-section inhomogeneity.
    """

    tiny = 1e-12

    def __init__(
        self,
        model,
        x1,
        x2,
        kaq,
        z,
        Haq,
        Hll,
        c,
        Saq,
        Sll,
        poraq,
        porll,
        ltype,
        topboundary,
        phreatictop,
        tsandhstar,
        tsandN,
        kzoverkh=None,
        model3d=False,
        name=None,
    ):
        super().__init__(
            model,
            kaq,
            z,
            Haq,
            Hll,
            c,
            Saq,
            Sll,
            poraq,
            porll,
            ltype,
            topboundary,
            phreatictop,
            kzoverkh=kzoverkh,
            model3d=model3d,
            name=name,
        )
        self.x1 = x1
        self.x2 = x2
        self.tsandhstar = tsandhstar
        self.tsandN = tsandN
        self.inhom_number = self.model.aq.add_inhom(self)
        self.addlinesinks = True  # Set to False not to add line-sinks

    def __repr__(self):
        if self.tsandhstar is not None:
            hstar = " with h*(t)"
        else:
            hstar = ""
        if self.tsandN is not None:
            inf = " with N(t)"
        else:
            inf = ""

        return (
            f"{self.name}: {self.__class__.__name__} "
            + str([self.x1, self.x2])
            + hstar
            + inf
        )

    def is_inside(self, x, _):
        """Check if a point is inside the cross-section.

        Parameters
        ----------
        x : float
            x-coordinate of the point.

        Returns
        -------
        bool
            True if the point is inside the cross-section, False otherwise.
        """
        return (x >= self.x1) and (x < self.x2)

    def initialize(self):
        super().initialize()
        self.create_elements()

    def create_elements(self):
        """Create linesinks to meet the continuity conditions the at the boundaries."""
        if (self.x1 == -np.inf) and (self.x2 == np.inf):
            # no reason to add elements
            pass
        # HeadDiff on right side, FluxDiff on left side
        elif self.x1 == -np.inf:
            xin = self.x2 - self.tiny
            # xoutright = self.x2 + self.tiny
            aqin = self.model.aq.find_aquifer_data(xin, 0)
            # aqoutright = self.model.aq.find_aquifer_data(xoutright, 0)
            if self.addlinesinks:
                HeadDiffLineSink1D(
                    self.model,
                    self.x2,
                    layers=range(self.naq),
                    aq=aqin,
                    label=None,
                )
        elif self.x2 == np.inf:
            xin = self.x1 + self.tiny
            # xoutleft = self.x1 - self.tiny
            aqin = self.model.aq.find_aquifer_data(xin, 0)
            # aqoutleft = self.model.aq.find_aquifer_data(xoutleft, 0)
            if self.addlinesinks:
                FluxDiffLineSink1D(
                    self.model, self.x1, range(self.naq), aq=aqin, label=None
                )
        else:
            xin = 0.5 * (self.x1 + self.x2)
            # xoutleft = self.x1 - self.tiny
            # xoutright = self.x2 + self.tiny
            aqin = self.model.aq.find_aquifer_data(xin, 0)
            # aqleft = self.model.aq.find_aquifer_data(xoutleft, 0)
            # aqright = self.model.aq.find_aquifer_data(xoutright, 0)
            if self.addlinesinks:
                HeadDiffLineSink1D(
                    self.model, self.x2, range(self.naq), label=None, aq=aqin
                )
                FluxDiffLineSink1D(
                    self.model, self.x1, range(self.naq), label=None, aq=aqin
                )
        if self.tsandN is not None:
            assert self.topboundary == "con", Exception(
                "Infiltration can only be applied to a confined aquifer."
            )
            AreaSinkXsection(self.model, self.x1, self.x2, tsandN=self.tsandN)
        if self.tsandhstar is not None:
            assert self.topboundary == "sem", Exception(
                "hstar can only be implemented on top of a semi-confined aquifer."
            )
            HstarXsection(self.model, self.x1, self.x2, tsandhstar=self.tsandhstar)

    def plot(self, ax=None, labels=False, params=False, names=False, fmt=None, **kwargs):
        """Plot the cross-section.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axis to plot the cross-section on. If None, a new axis will be created.
        labels : bool, optional
            If True, add layer-name labels.
        params : bool, optional
            If True, add parameter labels.
        names : bool, optional
            If True, add inhomogeneity names.
        fmt : str, optional
            format string for parameter values, e.g. '.2f' for 2 decimals.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        if "x1" in kwargs:
            x1 = kwargs.pop("x1")
            if np.isfinite(self.x1):
                x1 = max(x1, self.x1)
        elif np.isfinite(self.x1):
            x1 = self.x1
        else:
            x1 = self.x2 - 100.0
        if "x2" in kwargs:
            x2 = kwargs.pop("x2")
            if np.isfinite(self.x2):
                x2 = min(x2, self.x2)
        elif np.isfinite(self.x2):
            x2 = self.x2
        else:
            x2 = self.x1 + 100.0

        if self.x1 > x2 or self.x2 < x1:
            # do nothing, inhom is outside the window
            return ax

        if fmt is None:
            fmt = ""
        ssfmt = ".2e"

        r = x2 - x1
        r0 = x1

        if labels or params:
            lli = 1 if self.topboundary == "con" else 0
            aqi = 0
        else:
            lli = None
            aqi = None

        if names:
            ax.text(
                r0 + 0.5 * r,
                0.95,
                self.name,
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.get_xaxis_transform(),
            )

        for i in range(self.nlayers):
            if self.ltype[i] == "l":
                ax.fill_between(
                    x=[r0, r0 + r],
                    y1=self.z[i + 1],
                    y2=self.z[i],
                    color=[0.8, 0.8, 0.8],
                )
                if labels:
                    ax.text(
                        r0 + 0.5 * r if not params else r0 + 0.25 * r,
                        np.mean(self.z[i : i + 2]),
                        f"leaky layer {lli}",
                        ha="center",
                        va="center",
                    )
                if params:
                    paramtxt = (
                        f"$c$ = {self.c[lli]:{fmt}}, $S_s$ = {self.Sll[lli]:{ssfmt}}"
                    )
                    ax.text(
                        r0 + 0.75 * r if labels else r0 + 0.5 * r,
                        np.mean(self.z[i : i + 2]),
                        paramtxt,
                        ha="center",
                        va="center",
                    )
                if labels or params:
                    lli += 1

            if labels and self.ltype[i] == "a":
                ax.text(
                    r0 + 0.5 * r if not params else r0 + 0.25 * r,
                    np.mean(self.z[i : i + 2]),
                    f"aquifer {aqi}",
                    ha="center",
                    va="center",
                )
            if params and self.ltype[i] == "a":
                if aqi == 0 and i == 0 and self.phreatictop:
                    paramtxt = (
                        f"$k_h$ = {self.kaq[aqi]:{fmt}}, $S$ = {self.Saq[aqi]:{fmt}}"
                    )
                else:
                    paramtxt = (
                        f"$k_h$ = {self.kaq[aqi]:{fmt}}, $S_s$ = {self.Saq[aqi]:{ssfmt}}"
                    )
                ax.text(
                    r0 + 0.75 * r if labels else r0 + 0.5 * r,
                    np.mean(self.z[i : i + 2]),
                    paramtxt,
                    ha="center",
                    va="center",
                )
            if (labels or params) and self.ltype[i] == "a":
                aqi += 1

        for i in range(1, self.nlayers):
            if self.ltype[i] == "a" and self.ltype[i - 1] == "a":
                ax.fill_between(
                    x=[r0, r0 + r],
                    y1=self.z[i],
                    y2=self.z[i],
                    color=[0.8, 0.8, 0.8],
                )

        ax.hlines(self.z[0], xmin=r0, xmax=r0 + r, color="k", lw=0.75)
        ax.hlines(self.z[-1], xmin=r0, xmax=r0 + r, color="k", lw=3.0)
        ax.set_ylabel("elevation")
        return ax


class XsectionMaq(Xsection):
    """Cross-section inhomogeneity consisting of stacked aquifer layers.

    Parameters
    ----------
    model : Model
        Model to add the cross-section to, usually an instance of ModelXsection.
    x1 : float
        x-coordinate of the left boundary of the cross-section.
    x2 : float
        x-coordinate of the right boundary of the cross-section.
    kaq : array
        Hydraulic conductivities of the aquifers.
    z : array
        Elevations of the tops and bottoms of the layers.
    c : array
        Resistance of the leaky layers.
    Saq : array
        Specific storage of the aquifers.
    Sll : array
        Specific storage of the leaky layers.
    poraq : array
        Porosities of the aquifers.
    porll : array
        Porosities of the leaky layers.
    topboundary : str
        Type of top boundary. Can be 'conf' for confined, 'semi' for semi-confined
        or "leaky" for a leaky top boundary.
    phreatictop : bool
        If true, interpret the first specific storage coefficient as specific
        yield., i.e. it is not multiplied by aquifer thickness.
    tsandhstar : list of tuples
        list containing time and water level pairs for the hstar boundary condition.
    tsandN : list of tuples
        list containing time and infiltration pairs for the infiltration boundary
        condition.
    name : str
        Name of the cross-section.
    """

    def __init__(
        self,
        model,
        x1,
        x2,
        kaq=1,
        z=(1, 0),
        c=(),
        Saq=0.001,
        Sll=0,
        poraq=0.3,
        porll=0.3,
        topboundary="conf",
        phreatictop=False,
        tsandhstar=None,
        tsandN=None,
        name=None,
    ):
        kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype = param_maq(
            kaq, z, c, Saq, Sll, poraq, porll, topboundary, phreatictop
        )
        super().__init__(
            model,
            x1,
            x2,
            kaq,
            z,
            Haq,
            Hll,
            c,
            Saq,
            Sll,
            poraq,
            porll,
            ltype,
            topboundary,
            phreatictop,
            tsandhstar,
            tsandN,
            name=name,
        )


class Xsection3D(Xsection):
    """Cross-section inhomogeneity consisting of stacked aquifer layers.

    Vertical resistance is computed from vertical hydraulic conductivity and the
    anisotropy factor.

    Parameters
    ----------
    model : Model
        Model to add the cross-section to, usually an instance of ModelXsection.
    x1 : scalar
        x-coordinate of the left boundary of the cross-section.
    x2 : scalar
        x-coordinate of the right boundary of the cross-section.
    kaq : array
        Hydraulic conductivities of the aquifers.
    z : array
        Elevations of the tops and bottoms of the layers.
    Saq : array
        Specific storage of the aquifers.
    kzoverkh : scalar
        Ratio of vertical hydraulic conductivity to horizontal hydraulic
        conductivity.
    poraq : array
        Porosities of the aquifers.
    topboundary : str
        Type of top boundary. Can be 'conf' for confined, 'semi' for semi-confined
        or "leaky" for a leaky top boundary.
    phreatictop : bool
        If true, interpret the first specific storage coefficient as specific
        yield., i.e. it is not multiplied by aquifer thickness.
    topres : scalar
        Resistance of the top boundary. Only used if topboundary is 'leaky'.
    topthick : scalar
        Thickness of the top boundary. Only used if topboundary is 'leaky'.
    topSll : scalar
        Specific storage of the top boundary. Only used if topboundary is 'leaky'.
    toppor : scalar
        Porosity of the top boundary. Only used if topboundary is 'leaky'.
    tsandhstar : list of tuples
        list containing time and water level pairs for the hstar boundary condition.
    tsandN : list of tuples
        list containing time and infiltration pairs for the infiltration boundary
        condition.
    name : str
        Name of the cross-section.
    """

    def __init__(
        self,
        model,
        x1,
        x2,
        kaq=1,
        z=(4, 3, 2, 1),
        Saq=0.001,
        kzoverkh=0.1,
        poraq=0.3,
        topboundary="conf",
        phreatictop=False,
        topres=0,
        topthick=0,
        topSll=0,
        toppor=0.3,
        tsandhstar=None,
        tsandN=None,
        name=None,
    ):
        kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype, z = param_3d(
            kaq,
            z,
            Saq,
            kzoverkh,
            poraq,
            phreatictop,
            topboundary,
            topres,
            topthick,
            topSll,
            toppor,
        )
        super().__init__(
            model,
            x1,
            x2,
            kaq,
            z,
            Haq,
            Hll,
            c,
            Saq,
            Sll,
            poraq,
            porll,
            ltype,
            topboundary,
            phreatictop,
            tsandhstar,
            tsandN,
            name=name,
        )
