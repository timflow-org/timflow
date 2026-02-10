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
        horizontal_axis: Literal["x", "y", "s"] = "s",
        sep: Literal[", ", "\n"] = ", ",
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
            dictionary with units for parameters, e.g. {'k': 'm/d', 'c': 'd'}
        horizontal_axis : str
            's' for distance along cross-section on x-axis (default)
            'x' for using x-coordinates on x-axis
            'y' for using y-coordinates on x-axis
        hstar : float, optional
            override hstar value for plotting water level in transient
            1D inhomogeneities that use hstar, useful for plotting pretty
            cross-sections when reference level is not equal to 0.
        sep : str
            Separator between parameters, either ", " or "\n"
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
                sep=sep,
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
            r0, r, labels, params, fmt, units, lli, aqi, ax, sep=sep, **kwargs
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
        sep: Literal[", ", "\n"] = ", ",
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
    ):
        """Plot inhomogeneities for SimpleAquifer models.

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
            Dictionary of units for parameters (unused in transient, kept for
            compatibility)
        """
        if self._ml.model_type == "steady":
            for inhom in self._ml.aq.inhomlist:
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
                )
        elif self._ml.model_type == "transient":
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
            Dictionary of units for parameters
        lli : int or None
            Current leaky layer index
        aqi : int or None
            Current aquifer index
        ax : matplotlib.Axes
            Axes to plot on
        sep : str
            Separator between parameters, either ", " or "\n"
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
                        ax, r0, r, labels, fmt, units, lli, i, sep=sep
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
                        ax, r0, r, labels, fmt, units, aqi, i, sep=sep
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
            Dictionary of units for parameters
        lli : int
            Leaky layer index
        layer_idx : int
            Layer index in the model
        sep : str
            Separator between parameters, either ", " or "\n"
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
            if sep == "\n":
                nspaces = max(len(sstr) - len(cstr), 1)
                paramtxt = cstr + " " * nspaces + sep + sstr
            else:
                paramtxt = cstr + sep + sstr

        ax.text(
            r0 + 0.75 * r if labels else r0 + 0.5 * r,
            np.mean(self._ml.aq.z[layer_idx : layer_idx + 2]),
            paramtxt,
            ha="center",
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
            Dictionary of units for parameters
        aqi : int
            Aquifer index
        layer_idx : int
            Layer index in the model
        sep : str
            Separator between parameters, either ", " or "\n"
        """
        # Steady state: only hydraulic conductivity
        if units is not None:
            unitstr = f" {units['k']}" if "k" in units else ""
        else:
            unitstr = ""
        paramtxt = f"$k_h$ = {self._ml.aq.kaq[aqi]:{fmt}}" + unitstr

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
                paramtxt += f"{sep}$S_s$ = {self._ml.aq.Saq[aqi]:{ssfmt}}"

        ax.text(
            r0 + 0.75 * r if labels else r0 + 0.5 * r,
            np.mean(self._ml.aq.z[layer_idx : layer_idx + 2]),
            paramtxt,
            ha="center",
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
        """Create contour plot.

        This method should be implemented by subclasses to provide
        model-specific contouring functionality.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
        raise NotImplementedError("contour() must be implemented in subclass")

    def headalongline(self, **kwargs):
        """Plot head along a line.

        This method should be implemented by subclasses to provide
        model-specific head plotting functionality.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass
        """
        raise NotImplementedError("headalongline() must be implemented in subclass")
