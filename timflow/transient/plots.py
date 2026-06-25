"""Plot helpers for timflow.transient.

Provides contours, vcontours head along line plotting utilities for
transient flow simulations.

Example::

    ml.plots.contour()
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

    def head_along_line(
        self,
        x1=0,
        x2=1,
        y1=0,
        y2=0,
        npoints=100,
        t=1.0,
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
        t : scalar or array
            times at which to plot heads
        layers : int, list of ints, optional
            layers for which to plot heads, default is None, which plots all layers
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
        t = np.atleast_1d(t)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        s = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2) + sstart
        h = self._ml.headalongline(x, y, t, layers=layers)
        if layers is None:
            plotlayers = np.arange(self._ml.aq.naq)
        else:
            plotlayers = np.atleast_1d(layers)
        nlayers, ntime, npoints = h.shape
        for i in range(nlayers):
            for j in range(ntime):
                lbl = f"head (t={t[j]}, layer={plotlayers[i]})"
                ax.plot(s, h[i, j, :], c=color, lw=lw, label=lbl, **kwargs)
        if legend:
            ax.legend(loc=(0, 1), ncol=3, frameon=False)
        if grid:
            ax.grid(True)
        return ax

    def discharge_along_line(
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
        """Plot discharge along line.

        Parameters
        ----------
        x1, x2, y1, y2 : float
            start and end coordinates of line
        npoints : int
            number of points along line
        t : scalar or array
            times at which to plot discharge
        layers :
            layers for which to plot discharge
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
        qx, qy = self._ml.disvecalongline(x, y, t, layers)
        if x1 == x2:
            direction = "y"
            qs = qy
        elif y1 == y2:
            direction = "x"
            qs = qx
        else:
            direction = "s"
            qs = np.sqrt(qx**2 + qy**2)
        nlayers, ntime, npoints = qx.shape
        for i in range(nlayers):
            for j in range(ntime):
                lbl = f"q$_{direction}$ (t={t[j]}, layer={layers[i]})"
                if color is None:
                    ax.plot(s, qs[i, j, :], lw=lw, label=lbl)
                else:
                    ax.plot(s, qs[i, j, :], color, lw=lw, label=lbl)
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
        cmap=None,
        ax=None,
        figsize=None,
        legend=True,
        return_contours=False,
        parallel=False,
        show_progress=False,
        **kwargs,
    ):
        """Head contour plot.

        Parameters
        ----------
        win : list or tuple
            [x1, x2, y1, y2]
        ngr : scalar, tuple or list
            if scalar: number of grid points in x and y directions
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
        show_progress : bool, optional
            if True, show progress bar when computing headgrid in parallel,
            default is False.
        **kwargs
            additional keyword arguments passed to ax.contour()

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        xg, yg = self._get_xy_arrays(win, ngr)
        h = self._ml.headgrid(
            xg,
            yg,
            t=t,
            layers=np.atleast_1d(layers),
            parallel=parallel,
            show_progress=show_progress,
        )[:, 0, ...]  # squeeze time dimension
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

    def vcontour(
        self,
        win,
        n,
        t,
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
        h = self._ml.headalongline(xg, yg, t=t)[:, 0, :]
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

    def quiver_xy(
        self, x, y, z, t, normalize=False, ax=None, figsize=None, parallel=False, **kwargs
    ):
        """Quiver plot of velocity field in xy-plane.

        Parameters
        ----------
        x, y : 1D arrays
            coordinates of grid points in x and y directions
        z : scalar
            z-coordinate of plane in which to plot velocity field
        t : scalar
            time at which to plot velocity field
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
        v = self._ml.velocity_grid(x, y, np.atleast_1d(z), t, parallel=parallel)
        U = v[0, 0]
        V = v[1, 0]
        return super().quiver_xy(
            x, y, U, V, normalize=normalize, ax=ax, figsize=figsize, **kwargs
        )

    def quiver_z(
        self, x, y, z, t, normalize=False, ax=None, figsize=None, parallel=False, **kwargs
    ):
        """Quiver plot of velocity field in xz- or yz-plane.

        Parameters
        ----------
        x, y : 1D arrays
            coordinates of grid points in x and y directions
        z : scalar
            z-coordinate of plane in which to plot velocity field
        t : scalar
            time at which to plot velocity field
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
                "Either x or y array has to have length 1."
            )
        # v has shape (3, nz, ny, nx) ordered as (vx, vy, vz)
        v = self._ml.velocity_grid(x, y, np.atleast_1d(z), t=t, parallel=parallel)
        U = v[0].squeeze() if len(y) == 1 else v[1].squeeze()
        W = v[2].squeeze()  # vz
        return super().quiver_z(
            x, y, z, U, W, normalize=normalize, ax=ax, figsize=figsize, **kwargs
        )
