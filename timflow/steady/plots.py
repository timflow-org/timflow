"""Plot helpers for timflow.

Provides top-view, contours, and tracing visualization functions.

Example::

    ml.plots.topview()
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from timflow.plots.plots import PlotBase
from timflow.steady.trace import timtraceline

__all__ = ["PlotSteady"]


class PlotSteady(PlotBase):
    """Plotting functionality for timflow.steady models.

    Provides methods for visualizing model layouts, contours, pathlines,
    and other model results.
    """

    def __repr__(self):
        """Return string representation of Plots submodule."""
        methods = "".join(
            [f"\n - {meth}" for meth in dir(self) if not meth.startswith("_")]
        )
        return f"timflow {self._ml.model_type} plots, available methods:" + methods

    def contour(
        self,
        win,
        ngr=20,
        layers=0,
        levels=20,
        layout=True,
        labels=True,
        decimals=0,
        color=None,
        ax=None,
        figsize=None,
        legend=True,
        return_contours=False,
        **kwargs,
    ):
        """Head contour plot.

        Parameters
        ----------
        win : list or tuple
            [xmin, xmax, ymin, ymax]
        ngr : scalar, tuple or list
            if scalar: number of grid points in x and y direction
            if tuple or list: nx, ny, number of grid points in x and y
            directions
        layers : integer, list or array
            layers for which grid is returned
        levels : integer or array (default 20)
            levels that are contoured
        layout : boolean (default True)
            plot layout of elements
        labels : boolean (default True)
            print labels along contours
        decimals : integer (default 0)
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
        return_contours : bool, optional
            if True, return list of contour sets for each contoured layer


        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        cs : list
            of contour sets for each contoured layer, only if return_contours=True
        """
        x1, x2, y1, y2 = win
        if np.isscalar(ngr):
            nx = ny = ngr
        else:
            nx, ny = ngr
        layers = np.atleast_1d(layers)
        xg = np.linspace(x1, x2, nx)
        yg = np.linspace(y1, y2, ny)
        h = self._ml.headgrid(xg, yg, layers)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
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
            cs = plt.contour(xg, yg, h[i], levels, colors=c[i], **kwargs)
            cslist.append(cs)
            handles, _ = cs.legend_elements()
            cshandlelist.append(handles[0])
            if labels:
                fmt = "%1." + str(decimals) + "f"
                plt.clabel(cs, fmt=fmt)
        if isinstance(legend, list):
            plt.legend(cshandlelist, legend)
        elif legend:
            legendlist = ["layer " + str(i) for i in layers]
            plt.legend(cshandlelist, legendlist)
        plt.axis("scaled")
        if layout:
            self.topview(win=[x1, x2, y1, y2], layers=layers, ax=ax)
        if return_contours:
            return ax, cslist
        return ax

    def vcontour(
        self,
        win,
        n,
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
        h = self._ml.headalongline(xg, yg)
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

    def tracelines(
        self,
        xstart,
        ystart,
        zstart,
        hstepmax,
        vstepfrac=0.2,
        tmax=1e12,
        nstepmax=100,
        silent=".",
        color=None,
        orientation: Literal["hor", "ver", "both"] = "hor",
        win=None,
        axes=None,
        figsize=None,
        *,
        return_traces=False,
        metadata=False,
    ):
        """Function to trace multiple pathlines.

        Parameters
        ----------
        ml : Model object
            model to which the element is added
        xstart : array
            x-coordinates of starting locations
        ystart : array
            y-coordinates of starting locations
        zstart : array
            z-coordinates of starting locations
        hstepmax : scalar
            maximum horizontal step size [L]
        vstepfrac : scalar
            maximum vertical step as fraction of layer thickness
        tmax : scalar
            maximum travel time
        nstepmax : int
            maximum number of steps
        silent : string
            if '.', prints dot upon completion of each traceline
        color : string
            matplotlib color of traceline
        orientation : ('hor', 'ver', 'both')
            'hor' for horizontal, 'ver' for vertical
            'both' for horizontal above vertical
        win : list
            list with [xmin, xmax, ymin, ymax]
        axes : matplotlib.Axes or array of Axes
            axes to plot on, default is None which creates a new figure
        return_traces : boolean
            return traces if True
        metadata: boolean
            if False, return list of xyzt arrays
            if True, return list of result dictionaries

        Returns
        -------
        traces : result
            only if return_traces = True
        """
        if win is None:
            win = [-1e30, 1e30, -1e30, 1e30]

        if color is None:
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        elif isinstance(color, str):
            c = self._ml.aq.naq * [color]
        elif isinstance(color, list):
            c = color

        if len(c) < self._ml.aq.naq:
            n = int(np.ceil(self._ml.aq.naq / len(c)))
            c = n * c

        axes_dict = {}
        if orientation == "both":
            if axes is None:
                axes = self.topview_and_xsection(win=win, figsize=figsize)
            axes_dict["hor"] = axes[0]
            axes_dict["ver"] = axes[1]
        elif orientation[:3] == "hor":
            if axes is None:
                axes_dict["hor"] = self.topview(win=win, figsize=figsize)
            else:
                axes_dict["hor"] = axes[0]
        elif orientation[:3] == "ver":
            if axes is None:
                axes_dict["ver"] = self.xsection(
                    xy=[(win[0], np.mean(win[2:])), (win[1], np.mean(win[2:]))],
                    figsize=figsize,
                )
            else:
                axes_dict["ver"] = axes[-1]

        if return_traces:
            traces = []
        else:
            metadata = True  # suppress future warning from timtraceline
        for i, _ in enumerate(xstart):
            trace = timtraceline(
                self._ml,
                xstart[i],
                ystart[i],
                zstart[i],
                hstepmax=hstepmax,
                vstepfrac=vstepfrac,
                tmax=tmax,
                nstepmax=nstepmax,
                silent=silent,
                win=win,
                returnlayers=True,
                metadata=metadata,
            )
            if return_traces:
                traces.append(trace)
            if metadata:
                xyzt, layerlist = trace["trace"], trace["layers"]
            else:
                xyzt, layerlist = trace
            if silent == ".":
                print(".", end="", flush=True)
            if "hor" in axes_dict:
                color = []
                for ixyzt, ilayer in zip(xyzt, layerlist, strict=False):
                    aq = self._ml.aq.find_aquifer_data(ixyzt[0], ixyzt[1])
                    color.append(
                        c[aq.layernumber[ilayer]] if aq.ltype[ilayer] == "a" else "k"
                    )
                points = np.array([xyzt[:, 0], xyzt[:, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=color)
                axes_dict["hor"].add_collection(lc)
                # ax1.plot(xyzt[:, 0], xyzt[:, 1], color=color)
            if "ver" in axes_dict:
                color = []
                for ixyzt, ilayer in zip(xyzt, layerlist, strict=False):
                    aq = self._ml.aq.find_aquifer_data(ixyzt[0], ixyzt[1])
                    color.append(
                        c[aq.layernumber[ilayer]] if aq.ltype[ilayer] == "a" else "k"
                    )
                points = np.array([xyzt[:, 0], xyzt[:, 2]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=color)
                axes_dict["ver"].add_collection(lc)
                axes_dict["ver"].set_ylim(aq.z[-1], aq.z[0])
        if silent == ".":
            print("")  # Print the final newline after the dots
        if return_traces:
            return traces

    def vcontoursf1D(
        self,
        x1,
        x2,
        nx,
        levels,
        labels=False,
        decimals=0,
        color=None,
        nudge=1e-6,
        figsize=None,
        layout=True,
        ax=None,
        horizontal_axis: Literal["x", "y", "s"] = "s",
    ):
        """Contour plot in vertical cross-section of 1D model.

        Parameters
        ----------
        x1 : scalar
            left edge of contour domain
        x2 : scalar
            right edge of contour domain
        nx : integer
            number of grid points along cross-section
        levels : integer or array (default 20)
            levels that are contoured
        labels : boolean (default True)
            print labels along contours
        decimals : integer (default 0)
            number of decimals of labels along contours
        color : str or list of strings
            color of contour lines
        nudge : float
            first value is computed nudge from the specified x1 and x2
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        layout : boolean
            plot layout if True
        ax : matplotlib axis
            add plot to specified axis
        horizontal_axis : str, optional
            's' for distance along cross-section on x-axis (default)
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis

        Returns
        -------
        ax : axis
        """
        naq = self._ml.aq.naq
        xflow = np.linspace(x1 + nudge, x2 - nudge, nx)
        Qx = np.empty((naq, nx))
        for i in range(nx):
            Qx[:, i], _ = self._ml.disvec(xflow[i], 0)
        zflow = np.empty(2 * naq)
        for i in range(self._ml.aq.naq):
            aq = self._ml.aq.find_aquifer_data(xflow[0], 0)  # use first x as reference
            zflow[2 * i] = aq.zaqtop[i]
            zflow[2 * i + 1] = aq.zaqbot[i]
        Qx = Qx[::-1]  # set upside down
        Qxgrid = np.empty((2 * naq, nx))
        Qxgrid[0] = 0
        for i in range(naq - 1):
            Qxgrid[2 * i + 1] = Qxgrid[2 * i] - Qx[i]
            Qxgrid[2 * i + 2] = Qxgrid[2 * i + 1]
        Qxgrid[-1] = Qxgrid[-2] - Qx[-1]
        Qxgrid = Qxgrid[::-1]  # index 0 at top

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        cs = ax.contour(xflow, zflow, Qxgrid, levels, colors=color)
        if labels:
            fmt = "%1." + str(decimals) + "f"
            plt.clabel(cs, fmt=fmt)
        if layout:
            self.xsection(
                xy=[(x1, 0), (x2, 0)],
                labels=False,
                ax=ax,
                horizontal_axis=horizontal_axis,
            )
        return ax
