"""Plot helpers for timflow.steady.

Provides contours, vcontours and particle tracking visualization functions.

Example::

    ml.plots.contour()
"""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import timflow.steady.trace as tst
from timflow.plots.plots import PlotBase
from timflow.steady.well import WellBase


def _pop_deprecated_metadata_kwarg(kwargs: dict, *, fname: str) -> None:
    """Warn on legacy ``metadata=False``; strip ``metadata`` from ``kwargs``."""
    try:
        meta = kwargs.pop("metadata")
    except KeyError:
        return
    if meta is False:
        warnings.warn(
            f"{fname}: metadata=False is deprecated; full trace result dicts are "
            "always returned (omit the metadata argument).",
            DeprecationWarning,
            stacklevel=3,
        )
    elif meta is not True:
        msg = "metadata must be True or False"
        raise TypeError(msg)


__all__ = ["PlotSteady"]


class PlotSteady(PlotBase):
    """Plotting functionality for timflow.steady models.

    Provides methods for visualizing model layouts, contours, pathlines,
    and other model results.
    """

    def contour(
        self,
        win,
        ngr=20,
        layers=0,
        levels=20,
        layout=True,
        labels=True,
        decimals=1,
        color=None,
        cmap=None,
        ax=None,
        figsize=None,
        legend=True,
        return_contours=False,
        parallel=False,
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
        cmap : str or matplotlib colormap
            colormap for contour lines, only used if color is None
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        legend : list or boolean (default True)
            add legend to figure
            if list of strings: use strings as names in legend
        return_contours : bool, optional
            if True, return list of contour sets for each contoured layer
        parallel : bool, optional
            if True, compute headgrid in parallel using multiple threads,
            default is False
        **kwargs
            additional keyword arguments passed to ax.contour()


        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        cs : list
            of contour sets for each contoured layer, only if return_contours=True
        """
        xg, yg = self._get_xy_arrays(win, ngr)
        h = self._ml.headgrid(xg, yg, np.atleast_1d(layers), parallel=parallel)
        return self.contour_array(
            xg,
            yg,
            h,
            layers,
            levels,
            color=color,
            cmap=cmap,
            figsize=figsize,
            ax=ax,
            labels=labels,
            decimals=decimals,
            legend=legend,
            layout=layout,
            return_contours=return_contours,
            **kwargs,
        )

    def headalongline(self, x, y, layers=None, ax=None, sstart=0, **kwargs):
        """Plot head along the line provided by x and y coordinates.

        .. deprecated:: 0.3.0
            Use :meth:`head_along_line` instead.

        Parameters
        ----------
        x : array
            x-coordinates of the line
        y : array
            y-coordinates of the line
        layers : integer, list or array
            layers for which head is plotted, default is all layers
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        **kwargs
            additional keyword arguments passed to ax.plot()

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        warnings.warn(
            "The 'ml.plots.headalongline' method is deprecated. Use "
            "'ml.plots.head_along_line' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ax is None:
            _, ax = plt.subplots()
        head = self._ml.headalongline(x, y, layers=layers)
        r = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2) + sstart
        if layers is None:
            layers = np.arange(self._ml.aq.naq)
        for ilay in layers:
            ax.plot(r, head[ilay], label=kwargs.pop("label", f"Layer {ilay}"), **kwargs)
        return ax

    def head_along_line(
        self,
        x1=0,
        x2=1,
        y1=0,
        y2=0,
        npoints=100,
        layers=None,
        sstart=0,
        color=None,
        lw=1,
        figsize=None,
        ax=None,
        legend=True,
        grid=True,
        **kwargs,
    ):
        """Plot head along line.

        Parameters
        ----------
        x1, x2, y1, y2 : float
            start and end coordinates of line
        npoints : int
            number of points along line
        layers : int, list or array
            layers for which head is plotted, default is all layers
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        **kwargs
            additional keyword arguments passed to ax.plot()

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        s = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2) + sstart
        head = self._ml.headalongline(x, y, layers=layers)
        if layers is None:
            plotlayers = np.arange(self._ml.aq.naq)
        else:
            plotlayers = np.atleast_1d(layers)
        for ilay in plotlayers:
            ax.plot(
                s,
                head[ilay],
                label=kwargs.pop("label", f"head (layer={ilay}"),
                c=color,
                lw=lw,
                **kwargs,
            )
        if legend:
            ax.legend(loc=(0, 1), ncol=3, frameon=False)
        if grid:
            ax.grid(True)
        return ax

    def vcontour(
        self,
        win,
        n,
        levels=20,
        labels=True,
        decimals=0,
        color=None,
        cmap=None,
        vinterp=True,
        nudge=1e-6,
        ax=None,
        figsize=None,
        layout=True,
        horizontal_axis: Literal["x", "y", "s"] = "s",
        return_contours=False,
        **kwargs,
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
        cmap : str or matplotlib colormap
            colormap for contour lines, only used if color is None
        vinterp : boolean
            when True, interpolate between centers of layers
            when False, constant value vertically in each layer
        nudge : float
            small value to nudge grid points away from boundaries
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        layout : boolean
            plot layout if True
        horizontal_axis : str, optional
            's' for distance along cross-section on x-axis (default)
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis
        return_contours : bool
            if True, return contour set, default is False
        **kwargs
            additional keyword arguments passed to ax.contour()

        Returns
        -------
        cs : contour set
        """
        xg, yg = self._get_xy_arrays(win, n, nudge=nudge)
        h = self._ml.headalongline(xg, yg)
        return self.vcontour_array(
            xg,
            yg,
            h,
            levels=levels,
            labels=labels,
            decimals=decimals,
            color=color,
            cmap=cmap,
            vinterp=vinterp,
            ax=ax,
            figsize=figsize,
            layout=layout,
            horizontal_axis=horizontal_axis,
            return_contours=return_contours,
            **kwargs,
        )

    def vcontour_stream_function(
        self,
        x1,
        x2,
        nx,
        levels,
        labels=False,
        decimals=0,
        color=None,
        cmap=None,
        ax=None,
        figsize=None,
        layout=True,
        nudge=1e-6,
        radial=False,
        horizontal_axis: Literal["x", "y", "s"] = "s",
        **kwargs,
    ):
        """Contour stream_function.

        Only applicable to models where flow is 1D (along a line), e.g. flow in a
        vertical cross-section or flow in a radially symmetric model.

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
        cmap : str or matplotlib colormap
            colormap for contour lines, only used if color is None
        ax : matplotlib axis
            add plot to specified axis
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        layout : boolean
            plot layout if True
        nudge : float
            first value is computed nudge from the specified x1 and x2
        radial : bool
            if True, compute stream function for radially symmetric flow. Default is
            False.
        horizontal_axis : str, optional
            's' for distance along cross-section on x-axis (default)
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis

        Returns
        -------
        ax : axis
        """
        xflow = np.linspace(x1 + nudge, x2 - nudge, nx)
        Qxgrid, zflow = self._ml.stream_function(xflow, radial=radial)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        if color is not None and cmap is not None:
            cmap = None
        cs = ax.contour(xflow, zflow, Qxgrid, levels, colors=color, cmap=cmap, **kwargs)
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
        ax=None,
        figsize=None,
        *,
        return_traces=False,
        **kwargs,
    ):
        """Plot or compute multiple pathlines.

        Parameters
        ----------
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
        ax : matplotlib.Axes or list of Axes
            axes to plot on, default is None which creates a new figure
        return_traces : boolean
            if True, also return one trace result dict per pathline
        **kwargs
            kwargs are passed on to LineCollections for plotting.
            For backward compatibility a deprecated ``metadata`` keyword
            is accepted and removed. ``metadata=False`` emits a
            ``DeprecationWarning``; results always use the dict form from
            :func:`~timflow.steady.trace.traceline`.

        Returns
        -------
        ax : matplotlib.Axes or list of Axes
            axes with plot
        traces : list of dict
            only if ``return_traces`` is True; each dict matches
            :func:`~timflow.steady.trace.traceline` output.
        """
        _pop_deprecated_metadata_kwarg(kwargs, fname="ml.plots.tracelines")
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

        # Check if ax is iterable; if not, make it a single entry list
        if ax is not None:
            try:
                iter(ax)
            except TypeError:
                ax = [ax]

        if orientation == "both":
            if ax is None:
                ax = self.topview_and_xsection(win=win, figsize=figsize)
            axes_dict["hor"] = ax[0]
            axes_dict["ver"] = ax[1]
        elif orientation[:3] == "hor":
            if ax is None:
                axes_dict["hor"] = self.topview(win=win, figsize=figsize)
            else:
                axes_dict["hor"] = ax[0]
        elif orientation[:3] == "ver":
            if ax is None:
                axes_dict["ver"] = self.xsection(
                    xy=[(win[0], np.mean(win[2:])), (win[1], np.mean(win[2:]))],
                    figsize=figsize,
                )
            else:
                axes_dict["ver"] = ax[-1]

        if return_traces:
            traces = []
        for i, _ in enumerate(xstart):
            trace_result = tst.traceline(
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
            )
            if return_traces:
                traces.append(trace_result)
            xyzt, layerlist = trace_result["trace"], trace_result["layers"]
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
                lc = LineCollection(segments, colors=color, **kwargs)
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
                lc = LineCollection(segments, colors=color, **kwargs)
                axes_dict["ver"].add_collection(lc)
                axes_dict["ver"].set_ylim(aq.z[-1], aq.z[0])
        if silent == ".":
            print("")  # Print the final newline after the dots
        if return_traces:
            return ax, traces
        return ax

    def plotcapzone(
        self,
        well,
        nt=10,
        zstart=None,
        hstepmax=20,
        vstepfrac=0.2,
        tmax=365,
        nstepmax=100,
        silent=".",
        color=None,
        orientation="hor",
        win=None,
        ax=None,
        figsize=None,
        *,
        return_traces=False,
        **kwargs,
    ):
        """Plot a capture zone.

        Parameters
        ----------
        well : timflow.steady.Well, list of wells or list of str
            well element from which capture zone is started. Accepts a well object,
            a list of wells, or a list of well names.
        nt : int
            number of path lines
        zstart : scalar
            starting elevation of the path lines. Halfway aquifer thickness if None
        hstepmax : scalar
            maximum step in horizontal space
        vstepfrac : float
            maximum fraction of aquifer layer thickness during one step
        tmax : scalar
            maximum time
        nstepmax : scalar(int)
            maximum number of steps
        silent : boolean or string
            True (no messages), False (all messages), or '.'
            (print dot for each path line)
        color : color
        orientation : string
            'hor' for horizontal, 'ver' for vertical, or 'both' for both
        win : array_like (length 4)
            [xmin, xmax, ymin, ymax]
        axes : matplotlib.Axes, tuple of 2 matplotlib.Axes, or None
            axes to plot on, default is None which creates a new figure
        figsize : tuple of integers, optional, default: None
            width, height in inches.
        return_traces : boolean (default False)
            return the traces instead of plotting
        **kwargs
            For backward compatibility only: deprecated ``metadata`` keyword
            (see :meth:`tracelines`).

        Returns
        -------
        ax : matplotlib.Axes or list of Axes
            axes with plot
        traces : list of list of dict
            only if return_traces is True; outer list is per well, inner list
            matches :meth:`tracelines` with ``return_traces=True``
        """
        _pop_deprecated_metadata_kwarg(kwargs, fname="ml.plots.plotcapzone")
        if win is None:
            win = [-1e30, 1e30, -1e30, 1e30]
        # make well a list
        if isinstance(well, (str, WellBase)):
            well = [well]
        # loop over wells
        traces = []
        for w in well:
            if isinstance(w, str):
                w = self._ml.elementdict[w]
            xstart, ystart, zstart = w.capzonestart(nt, zstart)
            traces.append(
                self.tracelines(
                    xstart,
                    ystart,
                    zstart,
                    hstepmax=-abs(hstepmax),
                    vstepfrac=vstepfrac,
                    tmax=tmax,
                    nstepmax=nstepmax,
                    silent=silent,
                    color=color,
                    orientation=orientation,
                    win=win,
                    ax=ax,
                    figsize=figsize,
                    return_traces=return_traces,
                    **kwargs,
                )
            )
        if return_traces:
            return ax, traces
        return ax

    def vcontoursf1D(self, *args, **kwargs):
        warnings.warn(
            "The 'ml.plots.vcontoursf1D' method is deprecated. "
            "Use 'ml.plots.vcontour_stream_function' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vcontour_stream_function(*args, **kwargs)

    def quiver_xy(
        self, x, y, z, normalize=False, ax=None, figsize=None, parallel=False, **kwargs
    ):
        """Quiver plot of velocity field in xy-plane.

        Parameters
        ----------
        x, y : 1D arrays
            coordinates of grid points in x and y directions
        z : scalar
            z-coordinate of plane in which to plot velocity field
        normalize : bool
            if True, normalize velocity vectors to have length 1
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values
            size of figure
        parallel : bool
            if True, compute velocity grid in parallel using multiple threads,
            default is False
        **kwargs
            additional keyword arguments passed to ax.quiver()

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        # v has shape (3, nz, ny, nx) ordered as (vx, vy, vz)
        v = self._ml.velocity_grid(x, y, np.atleast_1d(z), parallel=parallel)
        U = v[0, 0]
        V = v[1, 0]
        return super().quiver_xy(
            x, y, U, V, normalize=normalize, ax=ax, figsize=figsize, **kwargs
        )

    def quiver_z(
        self, x, y, z, normalize=False, ax=None, figsize=None, parallel=False, **kwargs
    ):
        """Quiver plot of velocity field in xz- or yz-plane.

        Parameters
        ----------
        x, y : 1D arrays
            coordinates of grid points in x and y directions
        z : scalar
            z-coordinate of plane in which to plot velocity field
        normalize : bool
            if True, normalize velocity vectors to have length 1
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values
            size of figure
        parallel : bool
            if True, compute velocity grid in parallel using multiple threads,
            default is False
        **kwargs
            additional keyword arguments passed to ax.quiver()

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if len(x) > 1 and len(y) > 1:
            raise ValueError(
                "quiver_z is only implemented along the x-, or y-axis. "
                "Either x, or y has to have length 1."
            )
        # v has shape (3, nz, ny, nx) ordered as (vx, vy, vz)
        v = self._ml.velocity_grid(x, y, np.atleast_1d(z), parallel=parallel)
        U = v[0].squeeze() if len(y) == 1 else v[1].squeeze()
        V = v[2, 0]  # vz
        return super().quiver_z(
            x, y, z, U, V, normalize=normalize, ax=ax, figsize=figsize, **kwargs
        )
