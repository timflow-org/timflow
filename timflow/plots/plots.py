"""Base plotting functionality for timflow models.

Provides shared plotting methods that are inherited by both steady-state
and transient model plotting classes.
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["contour.negative_linestyle"] = "solid"

__all__ = ["PlotBase"]


class PlotBase:
    """Base class for plotting functionality in timflow models.

    This class contains shared plotting methods that are used by both steady-state
    (timflow.steady) and transient (timflow.transient) models. Model-specific plotting
    classes should inherit from this base class.

    Attributes
    ----------
    _ml : Model
        Reference to the model instance
    """

    def __init__(self, ml):
        """Initialize the base plotting class.

        Parameters
        ----------
        ml : Model
            The model instance to plot
        """
        self._ml = ml

    def __repr__(self):
        """Return string representation of Plots submodule."""
        methods = "".join(
            [f"\n - {meth}" for meth in dir(self) if not meth.startswith("_")]
        )
        model_type = getattr(self._ml, "model_type", "unknown")
        return f"timflow {model_type} plots, available methods:" + methods

    def topview(self, win=None, ax=None, figsize=None, layers=None):
        """Plot top-view.

        This method plots all elements (in specified layers).

        Parameters
        ----------
        win : list or tuple
            [x1, x2, y1, y2]
        ax : matplotlib.Axes, optional
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values
            size of figure
        layers : int or list of ints, optional
            layers to plot, default is None which plots elements in all layers

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        if layers is not None and isinstance(layers, int):
            layers = [layers]
        for e in self._ml.elementlist:
            if layers is None or np.isin(e.layers, layers).any():
                e.plot(ax=ax, layer=layers)
        if win is not None:
            ax.axis(win)
        return ax

    def xsection(
        self,
        xy: Optional[list[tuple[float]]] = None,
        labels=True,
        params=False,
        names=False,
        ax=None,
        fmt=None,
        units=None,
        hstar=None,
        boundaries=True,
        horizontal_axis: Literal["x", "y", "s"] = "s",
        sep: Literal[", ", "\n"] = ", ",
        ha: str = "center",
        **kwargs,
    ):
        r"""Plot cross-section of model.

        This is a shared method that handles cross-section plotting for both
        steady and transient models. The method automatically adapts the parameter
        labels based on the model type.

        Parameters
        ----------
        xy : list of tuples, optional
            list of tuples with coordinates of the form [(x0, y0), (x1, y1)]. If not
            provided, a cross section with length 1 is plotted for 3D models.
            For cross-section models the left and right are derived from the elements.
        labels : bool, optional
            add layer numbering labels to plot
        params : bool, optional
            add parameter values to plot
        names : bool, optional
            add inhomogeneity names to plot, only supported for cross-section models.
        ax : matplotlib.Axes, optional
            axes to plot on, default is None which creates a new figure
        fmt : str, optional
            format string for parameter values, e.g. '.2f' for 2 decimals
        units : dict, optional
            dictionary with units keyed by timflow parameter names,
            e.g. {'kaq': 'm/d', 'c': 'd', 'Saq': 'm$^{-1}$', 'Sll': 'm$^{-1}$'}
        horizontal_axis : str
            's' for distance along cross-section on x-axis (default)
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis
        hstar : float, optional
            override hstar value for plotting water level in transient
            1D inhomogeneities that use hstar, useful for plotting pretty
            cross-sections when reference level is not equal to 0.
        boundaries : bool, optional
            whether to plot aquifer boundaries for cross-section models,
            default is True
        sep : str
            Separator between parameters, either ", " or "\n"
        ha : str, optional
            Horizontal alignment for parameter labels. Defaults to "center".
        **kwargs
            passed on to all ax.plot calls

        Returns
        -------
        ax : matplotlib.Axes
            axes with plot
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        # Check if SimpleAquifer, import here to avoid circular imports
        from timflow.steady.aquifer import SimpleAquifer as SteadySimpleAquifer
        from timflow.transient.aquifer import SimpleAquifer as TransientSimpleAquifer

        if isinstance(self._ml.aq, (SteadySimpleAquifer, TransientSimpleAquifer)):
            return self._xsection_simple_aquifer(
                xy=xy,
                labels=labels,
                params=params,
                names=names,
                ax=ax,
                fmt=fmt,
                units=units,
                hstar=hstar,
                boundaries=boundaries,
                sep=sep,
                ha=ha,
            )

        # Standard cross-section for multi-layer models
        if fmt is None:
            fmt = ""

        # Get cross-section line parameters
        r0, r, _, _, _, _ = self._get_xsection_line_params(xy, ax, horizontal_axis)

        # Get layer and aquifer numbering indices
        lli, aqi = self._get_layer_indices()

        # Plot layers
        self._xection_plot_layers(
            r0, r, labels, params, fmt, units, lli, aqi, ax, sep=sep, ha=ha, **kwargs
        )

        # Plot aquifer-aquifer boundaries
        self._xsection_aquifer_boundaries(ax, **kwargs)

        # Add top and bottom lines
        ax.axhline(self._ml.aq.z[0], color="k", lw=0.75, **kwargs)
        ax.axhline(self._ml.aq.z[-1], color="k", lw=3.0, **kwargs)

        # Set ylabel
        ax.set_ylabel("elevation")

        # Remove x-ticks if no coordinates provided
        if xy is None:
            ax.xaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])

        return ax

    def topview_and_xsection(
        self,
        win=None,
        axes=None,
        figsize=None,
        topfigfrac=0.8,
        layers=None,
        horizontal_axis: Literal["x", "y"] = "x",
        **xsection_kwargs,
    ):
        """Plot top-view above and cross-section below.

        This method plots the top-view and cross-section side by side
        in a single figure.

        Parameters
        ----------
        win : list or tuple
            [x1, x2, y1, y2]
        axes : list of matplotlib.Axes, optional
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values
            size of figure
        topfigfrac : float
            fraction of figure width for top-view plot
        layers : int or list of ints, optional
            layers to plot, default is None which plots elements in all layers
        horizontal_axis : str
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis
        **xsection_kwargs : dict
            additional keyword arguments passed to xsection()

        Returns
        -------
        axes : list of matplotlib.Axes
            axes with plots [topview_ax, xsection_ax]
        """
        if axes is None:
            _, axes = plt.subplots(
                2,
                1,
                figsize=figsize,
                gridspec_kw={"height_ratios": [topfigfrac, 1 - topfigfrac]},
                sharex=True if horizontal_axis == "x" else False,
            )

            axes[0].set_aspect("equal", adjustable="box")
        self.topview(win=win, ax=axes[0], layers=layers)
        # Match bottom axes width to top axes after aspect adjustments
        top_pos = axes[0].get_position()
        bottom_pos = axes[1].get_position()
        axes[1].set_position(
            [top_pos.x0, bottom_pos.y0, top_pos.width, bottom_pos.height]
        )
        if win is None:
            win = axes[0].axis()  # Get limits from topview
        if horizontal_axis == "x":
            # along x, at the middle of y-range
            xy = [(win[0], np.mean(win[2:])), (win[1], np.mean(win[2:]))]
        elif horizontal_axis == "y":
            # along x, at the middle of y-range
            xy = [(np.mean(win[:2]), win[2]), (np.mean(win[:2]), win[3])]
        else:
            raise ValueError("horizontal_axis must be 'x' or 'y'.")
        self.xsection(
            xy=xy,
            ax=axes[1],
            horizontal_axis=horizontal_axis,
            labels=xsection_kwargs.pop("labels", False),
            **xsection_kwargs,
        )
        return axes

    def _xsection_simple_aquifer(
        self,
        xy,
        labels,
        params,
        names,
        ax,
        fmt,
        units=None,
        hstar=None,
        boundaries=True,
        sep: Literal[", ", "\n"] = ", ",
        ha: str = "center",
    ):
        """Handle cross-section plotting for SimpleAquifer models."""
        # Default implementation - can be overridden
        # Plot elements

        x_min = np.inf
        x_max = -np.inf
        for e in self._ml.elementlist:
            x_min = min(
                [
                    getattr(e, "xls", np.inf),
                    getattr(e, "xld", np.inf),
                    x_min,
                ]
            )
            x_max = max(
                [
                    getattr(e, "xls", -np.inf),
                    getattr(e, "xld", -np.inf),
                    x_max,
                ]
            )

        if xy is not None:
            (x1, _), (x2, _) = xy
        else:
            dx = x_max - x_min
            x1 = x_min - 0.25 * dx
            x2 = x_max + 0.25 * dx

        # Plot inhoms (implementation differs between steady/transient)
        self._xsection_plot_inhoms(
            ax=ax,
            labels=labels,
            params=params,
            names=names,
            x1=x1,
            x2=x2,
            fmt=fmt,
            units=units,
            sep=sep,
            ha=ha,
        )
        ax.set_xlim(x1, x2)
        ax.set_ylabel("elevation")
        ax.set_xlabel("x")

        # import here to avoid circular imports
        from timflow.transient.stripareasink import HstarXsection

        for e in self._ml.elementlist:
            if isinstance(e, HstarXsection):
                e.plot(ax=ax, hstar=hstar)
            else:
                if not e.inhomelement or boundaries:
                    e.plot(ax=ax)

        return ax

    def _xsection_plot_inhoms(
        self,
        ax,
        labels,
        params,
        names,
        x1,
        x2,
        fmt,
        units,
        sep: Literal[", ", "\n"] = ", ",
        ha: str = "center",
    ):
        r"""Plot inhomogeneities for SimpleAquifer models.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes to plot on
        labels : bool
            Whether to add labels
        params : bool
            Whether to add parameter values
        names : bool
            Whether to add inhomogeneity names
        x1, x2 : float
            Extent of the plot
        fmt : str
            Format string for parameter values
        units : dict or None
            Dictionary of units keyed by timflow parameter names
            e.g. {'kaq': 'm/d', 'c': 'd', 'Saq': 'm$^{-1}$', 'Sll': 'm$^{-1}$'}.
        sep : str, optional
            Separator between parameters, either ", " or "\n"
        ha : str, optional
            Horizontal alignment for parameter labels. Defaults to "center".
        """
        for inhom in self._ml.aq.inhomdict.values():
            inhom.plot(
                ax=ax,
                labels=labels,
                params=params,
                names=names,
                x1=x1,
                x2=x2,
                fmt=fmt,
                units=units,
                sep=sep,
                ha=ha,
            )

    def _get_xsection_line_params(self, xy, ax, horizontal_axis):
        """Get parameters for cross-section line.

        Parameters
        ----------
        xy : list of tuples or None
            Coordinates defining the cross-section
        ax : matplotlib.Axes
            Axes to plot on
        horizontal_axis : str
            Which axis to use for horizontal ('x', 'y', or 's')

        Returns
        -------
        r0 : float
            Starting position for horizontal axis
        r : float
            Length of cross-section
        x0, y0, x1, y1 : float
            Coordinates of start and end points
        """
        if xy is not None:
            (x0, y0), (x1, y1) = xy
            r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if horizontal_axis == "s":
                ax.set_xlim(0, r)
                r0 = 0.0
            elif horizontal_axis == "x":
                ax.set_xlim(np.min([x0, x1]), np.max([x0, x1]))
                r0 = np.min([x0, x1])
            elif horizontal_axis == "y":
                ax.set_xlim(np.min([y0, y1]), np.max([y0, y1]))
                r0 = np.min([y0, y1])
            else:
                raise ValueError("horizontal_axis must be 'x', 'y', or 's'")
        else:
            r0 = 0.0
            r = 1.0
            x0, y0, x1, y1 = 0.0, 0.0, 1.0, 0.0
            ax.set_xticks([])

        return r0, r, x0, y0, x1, y1

    def _get_layer_indices(self):
        """Get starting indices for layer and aquifer numbering.

        Returns
        -------
        lli : int or None
            Starting index for leaky layers
        aqi : int or None
            Starting index for aquifers
        """
        lli = 1 if self._ml.aq.ltype[0] == "a" else 0
        aqi = 0
        return lli, aqi

    def _xection_plot_layers(
        self,
        r0,
        r,
        labels,
        params,
        fmt,
        units,
        lli,
        aqi,
        ax,
        sep: Literal[", ", "\n"] = ", ",
        ha: str = "center",
        **kwargs,
    ):
        r"""Plot individual layers in the cross-section.

        Parameters
        ----------
        r0 : float
            Starting position for horizontal axis
        r : float
            Length of cross-section
        labels : bool
            Whether to add layer labels
        params : bool
            Whether to add parameter values
        fmt : str
            Format string for parameter values
        units : dict or None
            Dictionary of units keyed by timflow parameter names
            e.g. {'kaq': 'm/d', 'c': 'd', 'Saq': 'm$^{-1}$', 'Sll': 'm$^{-1}$'}
        lli : int or None
            Current leaky layer index
        aqi : int or None
            Current aquifer index
        ax : matplotlib.Axes
            Axes to plot on
        sep : str
            Separator between parameters, either ", " or "\n"
        ha : str, optional
            Horizontal alignment for parameter labels. Defaults to "center".
        **kwargs
            passed on to all ax.plot calls
        """
        for i in range(self._ml.aq.nlayers):
            # Plot leaky layers
            if self._ml.aq.ltype[i] == "l":
                ax.axhspan(
                    ymin=self._ml.aq.z[i + 1],
                    ymax=self._ml.aq.z[i],
                    color=[0.8, 0.8, 0.8],
                    **kwargs,
                )
                if labels:
                    ax.text(
                        r0 + 0.5 * r if not params else r0 + 0.25 * r,
                        np.mean(self._ml.aq.z[i : i + 2]),
                        f"leaky layer {lli}",
                        ha="center",
                        va="center",
                    )
                if params:
                    self._xsection_leaky_layer_params(
                        ax, r0, r, labels, fmt, units, lli, i, sep=sep, ha=ha
                    )
                if labels or params:
                    lli += 1

            # Plot aquifers
            if self._ml.aq.ltype[i] == "a":
                if labels:
                    ax.text(
                        r0 + 0.5 * r if not params else r0 + 0.25 * r,
                        np.mean(self._ml.aq.z[i : i + 2]),
                        f"aquifer {aqi}",
                        ha="center",
                        va="center",
                        **kwargs,
                    )
                if params:
                    self._xsection_aquifer_params(
                        ax, r0, r, labels, fmt, units, aqi, i, sep=sep, ha=ha
                    )
                if labels or params:
                    aqi += 1

    def _xsection_leaky_layer_params(
        self,
        ax,
        r0,
        r,
        labels,
        fmt,
        units,
        lli,
        layer_idx,
        sep: Literal[", ", "\n"] = ", ",
        ha: str = "center",
    ):
        r"""Add parameter text for leaky layers.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes to plot on
        r0 : float
            Starting position for horizontal axis
        r : float
            Length of cross-section
        labels : bool
            Whether labels are being plotted
        fmt : str
            Format string for parameter values
        units : dict or None
            Dictionary of units keyed by timflow parameter names
            e.g. {'kaq': 'm/d', 'c': 'd', 'Saq': 'm$^{-1}$', 'Sll': 'm$^{-1}$'}
        lli : int
            Leaky layer index
        layer_idx : int
            Layer index in the model
        sep : str
            Separator between parameters, either ", " or "\n"
        ha : str, optional
            Horizontal alignment for parameter labels. Defaults to "center".
        """
        if self._ml.model_type == "steady":
            # Steady state: only resistance c
            if units is not None:
                unitstr = f" {units['c']}" if "c" in units else ""
            else:
                unitstr = ""
            paramtxt = f"$c$ = {self._ml.aq.c[lli]:{fmt}}" + unitstr
        else:
            # Transient: resistance c and storage Sll
            ssfmt = ".2e"
            cstr = f"$c$ = {self._ml.aq.c[lli]:{fmt}}"
            sstr = f"$S_s$ = {self._ml.aq.Sll[lli]:{ssfmt}}"
            if units is not None:
                c_unitstr = f" {units['c']}" if "c" in units else ""
                # Prefer Sll unit; fall back to Saq for compatibility.
                ss_unitstr = f" {units['Sll']}" if "Sll" in units else ""
            else:
                c_unitstr = ""
                ss_unitstr = ""
            paramtxt = cstr + c_unitstr + sep + sstr + ss_unitstr

        ax.text(
            r0 + 0.75 * r if labels else r0 + 0.5 * r,
            np.mean(self._ml.aq.z[layer_idx : layer_idx + 2]),
            paramtxt,
            ha=ha,
            va="center",
        )

    def _xsection_aquifer_params(
        self,
        ax,
        r0,
        r,
        labels,
        fmt,
        units,
        aqi,
        layer_idx,
        sep: Literal[", ", "\n"] = ", ",
        ha: str = "center",
    ):
        r"""Add parameter text for aquifers.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes to plot on
        r0 : float
            Starting position for horizontal axis
        r : float
            Length of cross-section
        labels : bool
            Whether labels are being plotted
        fmt : str
            Format string for parameter values
        units : dict or None
            Dictionary of units keyed by timflow parameter names
            e.g.{'kaq': 'm/d', 'c': 'd', 'Saq': 'm$^{-1}$', 'Sll': 'm$^{-1}$'}
        aqi : int
            Aquifer index
        layer_idx : int
            Layer index in the model
        sep : str
            Separator between parameters, either ", " or "\n"
        ha : str, optional
            Horizontal alignment for parameter labels. Defaults to "center".
        """
        # Steady state: only hydraulic conductivity
        if units is not None:
            kh_unitstr = f" {units['kaq']}" if "kaq" in units else ""
            ss_unitstr = f" {units['Saq']}" if "Saq" in units else ""
        else:
            kh_unitstr = ""
            ss_unitstr = ""
        paramtxt = f"$k_h$ = {self._ml.aq.kaq[aqi]:{fmt}}" + kh_unitstr

        # Model3D adds vertical anisotropy
        if self._ml.name == "Model3D":
            paramtxt += f"{sep}$k_z/k_h$ = {self._ml.aq.kzoverkh[aqi]:{fmt}}"

        # Transient: add (specific) storage
        if self._ml.model_type == "transient":
            # Transient: hydraulic conductivity and storage
            ssfmt = ".2e"
            if aqi == 0 and self._ml.aq.phreatictop:
                # Top phreatic aquifer uses S instead of Ss
                paramtxt += f"{sep}$S$ = {self._ml.aq.Saq[aqi]:{fmt}}"
            else:
                paramtxt += f"{sep}$S_s$ = {self._ml.aq.Saq[aqi]:{ssfmt}}" + ss_unitstr

        ax.text(
            r0 + 0.75 * r if labels else r0 + 0.5 * r,
            np.mean(self._ml.aq.z[layer_idx : layer_idx + 2]),
            paramtxt,
            ha=ha,
            va="center",
        )

    def _xsection_aquifer_boundaries(self, ax, **kwargs):
        """Plot boundaries between aquifer-aquifer interfaces.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes to plot on
        """
        for i in range(1, self._ml.aq.nlayers):
            if self._ml.aq.ltype[i] == "a" and self._ml.aq.ltype[i - 1] == "a":
                ax.axhspan(
                    ymin=self._ml.aq.z[i],
                    ymax=self._ml.aq.z[i],
                    color=[0.8, 0.8, 0.8],
                    **kwargs,
                )

    def contour(self, **kwargs):
        """Create head contour plot.

        This method should be implemented by subclasses to provide
        model-specific contour calls.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
        raise NotImplementedError("contour() must be implemented in subclass")

    @staticmethod
    def _get_xy_arrays(win, ngr, nudge=0.0):
        """Helper to create x and y arrays for contouring.

        Parameters
        ----------
        win : list or tuple
            [x1, x2, y1, y2]
        ngr : scalar, tuple or list
            if scalar: number of grid points in x and y direction
            if tuple or list: nx, ny, number of grid points in x and y
            directions
        nudge : float
            small value to nudge grid points away from boundaries, default is 0

        Returns
        -------
        xg, yg : 1D arrays
            x and y coordinates of grid points for contouring
        """
        x1, x2, y1, y2 = win
        if np.isscalar(ngr):
            nx = ny = ngr
        else:
            nx, ny = ngr
        xg = np.linspace(x1 + nudge, x2 - nudge, nx)
        yg = np.linspace(y1 + nudge, y2 - nudge, ny)
        return xg, yg

    def contour_array(
        self,
        x,
        y,
        arr,
        layers=0,
        levels=20,
        color=None,
        cmap=None,
        figsize=None,
        ax=None,
        labels=True,
        decimals=0,
        legend=True,
        layout=True,
        return_contours=False,
        **kwargs,
    ):
        layers = np.atleast_1d(layers)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        # color
        per_level_colors = False
        if color is None and cmap is None:
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        elif isinstance(color, str):
            c = len(layers) * [color]
        elif isinstance(color, list):
            c = color
            if len(c) > 0 and not isinstance(c[0], str):
                per_level_colors = True  # list of RGBA tuples, one per contour level
        else:
            c = None

        # contour
        cslist = []
        cshandlelist = []
        for i in range(len(layers)):
            _colors = c if per_level_colors else c[i]
            iarr = arr[i] if arr.ndim == 3 else arr
            cs = ax.contour(x, y, iarr, levels, colors=_colors, cmap=cmap, **kwargs)
            cslist.append(cs)
            handles, _ = cs.legend_elements()
            cshandlelist.append(handles[0])
            if labels:
                fmt = "%1." + str(decimals) + "f"
                ax.clabel(cs, fmt=fmt)
        if isinstance(legend, list):
            ax.legend(cshandlelist, legend, loc=(0, 1), frameon=False, ncol=3)
        elif legend:
            legendlist = ["layer " + str(i) for i in layers]
            ax.legend(cshandlelist, legendlist, loc=(0, 1), frameon=False, ncol=3)
        if layout:
            self.topview(win=[x.min(), x.max(), y.min(), y.max()], layers=layers, ax=ax)
        if return_contours:
            return ax, cslist
        return ax

    def vcontour_array(
        self,
        x,
        y,
        arr,
        levels=20,
        labels=True,
        decimals=0,
        color=None,
        cmap=None,
        vinterp=True,
        ax=None,
        figsize=None,
        layout=True,
        horizontal_axis: Literal["x", "y", "s"] = "s",
        return_contours=False,
        **kwargs,
    ):
        """Contour array in vertical cross-section.

        This method derives the vertical coordinates based on the model layers.
        It assumes that the input array has shape (layers, len(x)). Use vinterp
        to control whether to interpolate between layer centers or use constant
        values within each layer.

        Parameters
        ----------
        x : 1D array
            horizontal coordinates of grid points
        y : 1D array
            horizontal coordinates of grid points
        arr : 2D array
            array to contour, shape (naq, len(x))
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
        if horizontal_axis == "x":
            x = x
        elif horizontal_axis == "y":
            x = y
        elif horizontal_axis == "s":
            x = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
        else:
            raise ValueError("horizontal_axis must be 'x', 'y', or 's'")
        if vinterp:
            z = 0.5 * (self._ml.aq.zaqbot + self._ml.aq.zaqtop)
            z = np.hstack((self._ml.aq.zaqtop[0], z, self._ml.aq.zaqbot[-1]))
            arr = np.vstack((arr[0], arr, arr[-1]))
        else:
            z = np.empty(2 * self._ml.aq.naq)
            for i in range(self._ml.aq.naq):
                z[2 * i] = self._ml.aq.zaqtop[i]
                z[2 * i + 1] = self._ml.aq.zaqbot[i]
            arr = np.repeat(arr, 2, 0)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if layout:
            self.xsection(
                xy=[(x[0], y[0]), (x[-1], y[-1])],
                labels=False,
                ax=ax,
                horizontal_axis=horizontal_axis,
            )
        if color is not None and cmap is not None:
            cmap = None
        cs = ax.contour(x, z, arr, levels, colors=color, cmap=cmap, **kwargs)
        if labels:
            fmt = "%1." + str(decimals) + "f"
            ax.clabel(cs, fmt=fmt)
        if return_contours:
            return ax, cs
        return ax

    def headalongline(self, *args, **kwargs):
        """Plot head along a line.

        This method should be implemented by subclasses to provide
        model-specific head plotting functionality.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
        raise NotImplementedError("headalongline() must be implemented in subclass")

    def quiver_xy(self, x, y, U, V, normalize=False, ax=None, figsize=None, **kwargs):
        """Plot quiver of flow vectors.

        This method should be implemented by subclasses to provide
        model-specific quiver plotting functionality.

        Parameters
        ----------
        x : 2D array
            x coordinates of grid points
        y : 2D array
            y coordinates of grid points
        z : float
            z coordinate of grid points
        normalize : bool
            whether to normalize flow vectors for plotting
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        **kwargs
            additional keyword arguments passed to ax.quiver()

        Returns
        -------
        ax : matplotlib.Axes
            axes with quiver plot
        """
        if normalize:
            speed = np.sqrt(U**2 + V**2)
            U = U / speed
            V = V / speed
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        ax.quiver(x, y, U, V, **kwargs)
        return ax

    def quiver_z(
        self,
        x,
        y,
        z,
        U,
        V,
        normalize=False,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """Plot quiver of flow vectors in 3D.

        This method should be implemented by subclasses to provide
        model-specific quiver plotting functionality.

        Parameters
        ----------
        x : 1D array
            x coordinates of grid points
        y : 1D array
            y coordinates of grid points
        z : 1D array
            z coordinates of grid points
        U : 2D array
            x component of flow vectors
        V : 2D array
            y component of flow vectors
        normalize : bool
            whether to normalize flow vectors for plotting
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        **kwargs
            additional keyword arguments passed to ax.quiver()

        Returns
        -------
        ax : matplotlib.Axes
            axes with quiver plot
        """
        s = x if len(y) == 1 else y
        if normalize:
            speed = np.sqrt(U**2 + V**2)
            U = U / speed
            V = V / speed
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.quiver(s, z, U, V, **kwargs)
        return ax
