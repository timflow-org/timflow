"""Plot helpers for TTim.

Provides top-view, contours, and tracing visualization functions
for transient flow simulations.

Example::

    ml.plots.topview()
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from timflow.plots.plots import PlotBase


class PlotTransient(PlotBase):
    """Plotting functionality for timflow.transient models.

    Provides methods for visualizing model layouts, contours, and other
    transient model results.
    """

    def __repr__(self):
        """Return string representation of Plots submodule."""
        methods = "".join(
            [f"\n - {meth}" for meth in dir(self) if not meth.startswith("_")]
        )
        return "timflow.transient plots, available methods:" + methods

    def head_along_line(
        self,
        x1=0,
        x2=1,
        y1=0,
        y2=0,
        npoints=100,
        t=1.0,
        layers=0,
        sstart=0,
        color=None,
        lw=1,
        figsize=None,
        ax=None,
        legend=True,
        grid=True,
    ):
        """Plot head along line.

        Parameters
        ----------
        x1, x2, y1, y2 : float
            start and end coordinates of line
        npoints : int
            number of points along line
        t : scalar or array
            times at which to plot heads
        layers :
            layers for which to plot heads
        sstart : float
            starting distance for cross-section
        color : str
            color of line
        lw : float
            line width
        figsize : tuple of 2 values
            size of figure
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        legend : bool
            add legend to plot
        grid : bool
            add grid to plot

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        layers = np.atleast_1d(layers)
        t = np.atleast_1d(t)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        s = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2) + sstart
        h = self._ml.headalongline(x, y, t, layers)
        nlayers, ntime, npoints = h.shape
        for i in range(nlayers):
            for j in range(ntime):
                lbl = f"head (t={t[j]}, layer={layers[i]})"
                if color is None:
                    ax.plot(s, h[i, j, :], lw=lw, label=lbl)
                else:
                    ax.plot(s, h[i, j, :], color, lw=lw, label=lbl)
        if legend:
            ax.legend(loc=(0, 1), ncol=3, frameon=False)
        if grid:
            ax.grid(True)
        return ax

    def contour(
        self,
        win,
        ngr=20,
        t=1,
        layers=0,
        levels=20,
        layout=True,
        labels=True,
        decimals=1,
        color=None,
        ax=None,
        figsize=None,
        legend=True,
        parallel=False,
    ):
        """Contour plot.

        Parameters
        ----------
        win : list or tuple
            [x1, x2, y1, y2]
        ngr : scalar, tuple or list
            if scalar: number of grid points in x and y direction
            if tuple or list: nx, ny, number of grid points in x and y direction
        t : scalar
            time
        layers : integer, list or array
            layers for which grid is returned
        levels : integer or array (default 20)
            levels that are contoured
        layout : boolean (default True_)
            plot layout of elements
        labels : boolean (default True)
            print labels along contours
        decimals : integer (default 1)
            number of decimals of labels along contours
        color : str or list of strings
            color of contour lines
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        legend : list or boolean (default True)
            add legend to figure
            if list of strings: use strings as names in legend
        parallel : bool, optional
            if True, compute headgrid in parallel using multiple threads,
            default is False

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        x1, x2, y1, y2 = win
        if np.isscalar(ngr):
            nx = ny = ngr
        else:
            nx, ny = ngr
        layers = np.atleast_1d(layers)
        xg = np.linspace(x1, x2, nx)
        yg = np.linspace(y1, y2, ny)
        h = self._ml.headgrid(xg, yg, t, layers, parallel=parallel)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        # color
        if color is None:
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        elif isinstance(color, str):
            c = len(layers) * [color]
        elif isinstance(color, list):
            c = color
        if len(c) < len(layers):
            n = np.ceil(self._ml.aq.naq / len(c))
            c = n * c

        # contour
        cslist = []
        cshandlelist = []
        for i in range(len(layers)):
            cs = ax.contour(
                xg, yg, h[i, 0], levels, colors=c[i], negative_linestyles="solid"
            )
            cslist.append(cs)
            handles, _ = cs.legend_elements()
            cshandlelist.append(handles[0])
            if labels:
                fmt = f"%1.{decimals}f"
                ax.clabel(cs, fmt=fmt)
        if isinstance(legend, list):
            ax.legend(cshandlelist, legend, loc=(0, 1), ncol=3, frameon=False)
        elif legend:
            legendlist = ["layer " + str(i) for i in layers]
            ax.legend(cshandlelist, legendlist, loc=(0, 1), ncol=3, frameon=False)

        if layout:
            self.topview(win=[x1, x2, y1, y2], ax=ax)
        return ax

    def vcontour(
        self,
        win,
        n,
        t,
        levels=20,
        labels=True,
        decimals=0,
        color=None,
        vinterp=True,
        nudge=1e-6,
        newfig=True,
        figsize=None,
        layout=True,
        horizontal_axis: Literal["x", "y", "s"] = "s",
    ):
        """Head contour plot in vertical cross-section.

        Parameters
        ----------
        win : list or tuple
            [xmin, xmax, ymin, ymax]
        n : integer
            number of grid points along cross-section
        t : scalar
            time
        levels : integer or array (default 20)
            levels that are contoured
        labels : boolean (default True)
            print labels along contours
        decimals : integer (default 0)
            number of decimals of labels along contours
        color : str or list of strings
            color of contour lines
        vinterp : boolean
            when True, interpolate between centers of layers
            when False, constant value vertically in each layer
        nudge : float
            first value is computed nudge from the specified window
        newfig : boolean (default True)
            create new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        layout : boolean
            plot layout if True
        horizontal_axis : str, optional
            's' for distance along cross-section on x-axis (default)
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis

        Returns
        -------
        cs : contour set
        """
        x1, x2, y1, y2 = win
        xg = np.linspace(x1 + nudge, x2 - nudge, n)
        yg = np.linspace(y1 + nudge, y2 - nudge, n)
        h = self._ml.headalongline(xg, yg, t)[:, 0, :]
        if horizontal_axis == "x":
            xg = xg
        elif horizontal_axis == "y":
            xg = yg
        elif horizontal_axis == "s":
            L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            xg = np.linspace(0, L, n)
        else:
            raise ValueError("horizontal_axis must be 'x', 'y', or 's'")
        if vinterp:
            zg = 0.5 * (self._ml.aq.zaqbot + self._ml.aq.zaqtop)
            zg = np.hstack((self._ml.aq.zaqtop[0], zg, self._ml.aq.zaqbot[-1]))
            h = np.vstack((h[0], h, h[-1]))
        else:
            zg = np.empty(2 * self._ml.aq.naq)
            for i in range(self._ml.aq.naq):
                zg[2 * i] = self._ml.aq.zaqtop[i]
                zg[2 * i + 1] = self._ml.aq.zaqbot[i]
            h = np.repeat(h, 2, 0)
        if newfig:
            _, ax = plt.subplots(figsize=figsize)
        if layout:
            self.xsection(
                xy=[(x1, y1), (x2, y2)],
                labels=False,
                ax=ax,
                horizontal_axis=horizontal_axis,
            )
        cs = ax.contour(xg, zg, h, levels, colors=color)
        if labels:
            fmt = "%1." + str(decimals) + "f"
            ax.clabel(cs, fmt=fmt)

        return cs
