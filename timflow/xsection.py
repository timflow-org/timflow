"""Unified cross-section models combining steady and transient flow in one model.

A single ``ModelXsection`` drives a steady and a transient submodel, so each
inhomogeneity and element is added only once. Boundary-condition time series
are split automatically into a steady and a transient component. This module
contains the unified versions of ``XsectionMaq`` and ``Xsection3D``, and of the
elements ``River1D``, ``LineSink1D``, ``LeakyWall1D``, ``ImpermeableWall1D``,
``Constant``, and ``Uflow``.

Examples
--------
Build a model from two semi-confined sections, a river on the left and a
polder on the right, and solve it once::

    import numpy as np
    import timflow

    ml = timflow.ModelXsection(naq=1, tmin=1, tmax=10)
    river = timflow.XsectionMaq(
        ml, x1=-np.inf, x2=0.0, z=[0, -1, -10], c=[10], topboundary="semi",
        tsandhstar=[(0, 1), (1, 2), (2, 1)], hsteady="first",
    )
    polder = timflow.XsectionMaq(
        ml, x1=0.0, x2=np.inf, z=[0, -1, -10], c=[10], topboundary="semi",
        hsteady=-1.0,
    )
    ml.solve()

Query the head at any time: no ``t`` returns the steady head, ``t="final"``
the final steady state::

    ml.head(100)             # steady head at x = 100
    ml.head(100, t=5)        # head at x = 100 after 5 days
    ml.head(100, t=np.inf)   # final steady state at x = 100

Or along a line of x-values, for multiple times at once::

    x = np.linspace(-250, 250, 101)
    h = ml.headalongline(x, t=[1, 5, 10])  # shape (nlayers, ntimes, nx)

Elements are added to the model in the same way, e.g. a river with a
transient head on a single section::

    ml = timflow.ModelXsection(naq=1, tmin=1, tmax=10)
    timflow.XsectionMaq(
        ml, x1=-np.inf, x2=np.inf, z=[0, -1, -10], c=[10], hsteady=0.0,
        topboundary="semi",
    )
    river = timflow.River1D(ml, xls=0, tsandh=[(0, 1), (5, 3)], hsteady="first", res=10)
    ml.solve()

The submodels are exposed as ``ml.steady`` and ``ml.transient``; the steady and
transient elements are similarly accessed through `river.steady`` and
`river.transient``.

Plotting is available through ``ml.plots`` (which uses the transient model plots)::

    ml.plots.xsection(x1=-100, x2=100)
"""

import inspect
from typing import Literal

import numpy as np

import timflow.steady as tfs
import timflow.transient as tft


def is_unified_modelxsection(model):
    """Check if the model is a unified ModelXsection.

    Parameters
    ----------
    model : ModelXsection
        The model to check.

    Raises
    ------
    ValueError
        If the model is not an instance of timflow.ModelXsection.
    """
    if not isinstance(model, ModelXsection):
        raise ValueError(
            "model must be an instance of timflow.ModelXsection. "
            "(not timflow.steady.ModelXsection or timflow.transient.ModelXsection)"
        )


class ModelXsection:
    r"""Unified model class for steady and transient flow in cross-sections.

    Parameters
    ----------
    naq : integer
        number of aquifers
    tmin : float
        the minimum time for which heads can be computed after any change
        in boundary condition.
    tmax : float
        the maximum time for which heads can be computed.
    tstart : float, optional
        time at start of simulation (default 0)
    M : integer, optional
        the number of terms to be used in the numerical inversion algorithm.
        10 is usually sufficient.

    Notes
    -----
    A `ModelXsection` may consist of an arbitrary number of `Xsection3D` or
    `XsectionMaq` sections. Currently, the combined domain of all sections must span
    from $x = -\infty$ to $x = +\infty$, with no gaps.

    Boundary-condition time series are split into steady and transient components.
    A steady value of ``"first"``, ``"mean"``, or ``"last"`` selects a baseline
    from the series; a numeric steady value is used directly. The baseline is
    subtracted from the series to obtain the transient component. With a time series,
    a steady value of ``None`` uses a zero baseline. Without a time series, ``None``
    means no boundary condition. Element steady values default to ``"first"``;
    cross-section steady values default to ``None`` so either boundary may be omitted.
    """

    def __init__(
        self,
        naq: int = 1,
        tmin: float = 1.0,
        tmax: float = 10.0,
        tstart: float = 0.0,
        M: int = 10,
    ):
        """Initialize the unified cross-section model."""
        self.naq = naq
        self.tmin = tmin
        self.tmax = tmax
        self.tstart = tstart
        self.M = M

        # set up submodels for steady and transient flow
        self.steady = tfs.ModelXsection(naq=naq)
        self.transient = tft.ModelXsection(
            naq=naq, tmin=tmin, tmax=tmax, tstart=tstart, M=M
        )

        self.elementlist = []
        self.elementdict = {}
        self.inhomdict = {}

        # defer to transient model for overviews (transient contains all information)
        self.plots = self.transient.plots
        self.aquifer_summary = self.transient.aquifer_summary

    def __repr__(self):
        s = "ModelXsection (steady + transient):"
        if len(self.inhomdict) > 0:
            s += "\n  Xsections:"
            for inhom in self.inhomdict.values():
                s += "\n   -" + str(inhom)
        if len(self.elementlist) > 0:
            s += "\n  Elements:"
            for e in self.elementlist:
                s += "\n   -" + str(e)
        return s

    def add_element(self, element: "UnifiedElement1D"):
        """Add element to the model.

        Parameters
        ----------
        element : UnifiedElement1D
            element to add to the model

        Raises
        ------
        ValueError
            If the element is not an instance of UnifiedElement1D.
        """
        if not isinstance(element, UnifiedElement1D):
            raise ValueError("element must be an instance of UnifiedElement1D.")
        self.elementlist.append(element)
        if element.label is not None:
            self.elementdict[element.label] = element

    def remove_element(self, element: "UnifiedElement1D"):
        """Remove element from the model.

        Parameters
        ----------
        element : UnifiedElement1D
            element to remove from the model

        Raises
        ------
        ValueError
            If the element is not in the model.
        """
        if element in self.elementlist:
            self.elementlist.remove(element)
            if element.label is not None and element.label in self.elementdict:
                del self.elementdict[element.label]
        else:
            raise ValueError("Element not found in the model.")

    def add_inhom(self, inhom: "UnifiedInhom1D"):
        """Add inhomogeneity to the model.

        Parameters
        ----------
        inhom : UnifiedInhom1D
            inhomogeneity to add to the model. If it has no label, a label is
            generated automatically.
        """
        if inhom.label is None:
            label = f"inhom{len(self.inhomdict) + 1:02g}"
        else:
            label = inhom.label
        self.inhomdict[label] = inhom

    def remove_inhom(self, label: str):
        """Remove inhomogeneity with the given label from the model.

        Parameters
        ----------
        label : str
            label of the inhomogeneity to remove

        Raises
        ------
        ValueError
            If no inhomogeneity with the given label is in the model.
        """
        if label in self.inhomdict:
            del self.inhomdict[label]
        else:
            raise ValueError(f"Inhomogeneity with label '{label}' not found.")

    @staticmethod
    def _is_final(t) -> bool:
        """Return True if t requests the final steady state.

        True for the string "final" (case-insensitive) or when all values of
        t are infinite. Any other string raises a ValueError.
        """
        if isinstance(t, str):
            if t.lower() == "final":
                return True
            raise ValueError(
                f"Unknown time specifier {t!r}; expected 'final' or a numeric time."
            )
        return bool(np.isinf(t).all())

    def head(self, x: float, t: float | np.ndarray | None = None, layers=None):
        """Head at x, t where t can be multiple times.

        Parameters
        ----------
        x : float
            x-coordinate at which the head is computed
        t : float, array, "final", or None, optional
            time(s) for which the head is returned. If None, only the
            steady-state component is returned. If "final" or infinite, the
            final steady state is returned.
        layers : integer, list or array, optional
            layers for which head is returned
            if None: all layers are returned

        Returns
        -------
        h : array size `nlayers` or `nlayers, ntimes`
            head in all `layers` (if not `None`), or all layers of the aquifer
            (otherwise). Size is `nlayers` if `t` is None, otherwise
            `nlayers, ntimes`.
        """
        if t is None:
            return self.steady.head(x, 0.0, layers=layers)
        if self._is_final(t):
            return self.final_steady_model().head(x, layers=layers)
        return self.steady.head(x, 0.0, layers=layers)[
            :, np.newaxis
        ] + self.transient.head(x, 0.0, t, layers=layers)

    def headalongline(
        self,
        x: np.ndarray,
        t: float | np.ndarray | None = None,
        layers: int | list | None = None,
    ):
        """Head along line (the y = 0 plane of the cross-section).

        Parameters
        ----------
        x : 1D array or list
            x values of line
        t : float, 1D array, "final", or None, optional
            time(s) for which the head is returned. If None, only the
            steady-state component is returned. If "final" or infinite, the
            final steady state is returned.
        layers : integer, list or array, optional
            layers for which head is returned
            if None: all layers are returned

        Returns
        -------
        h : array size `nlayers, nx` or `nlayers, ntimes, nx`
            head along the line. Size is `nlayers, nx` if `t` is None, otherwise
            `nlayers, ntimes, nx`.
        """
        if t is None:
            return self.steady.headalongline(x, np.zeros_like(x), layers=layers)
        if self._is_final(t):
            return self.final_steady_model().headalongline(x, layers=layers)
        return self.steady.headalongline(x, np.zeros_like(x), layers=layers)[
            :, np.newaxis, :
        ] + self.transient.headalongline(x, np.zeros_like(x), t, layers=layers)

    def head_array(
        self,
        x: np.ndarray,
        t: float | np.ndarray | None = None,
        layers: int | list | None = None,
        show_progress: bool = False,
        parallel: bool = False,
    ):
        """Head for array of points in the cross-section (the y = 0 plane).

        Parameters
        ----------
        x : 1D array or list
            x values of points
        t : float, 1D array, "final", or None, optional
            time(s) for which the head is returned. If None, only the
            steady-state component is returned. If "final" or infinite, the
            final steady state is returned.
        layers : integer, list or array, optional
            layers for which head is returned
            if None: all layers are returned
        show_progress : bool
            show computation progress, by default False
        parallel : bool or int, optional
            if `True`, computes head_array in parallel using multiprocessing,
            by default `False`. If an integer is provided, it is interpreted as
            the number of processes to use.

        Returns
        -------
        h : array size `nlayers, npoints` or `nlayers, ntimes, npoints`
            heads array. Size is `nlayers, npoints` if `t` is None, otherwise
            `nlayers, ntimes, npoints`.
        """
        if t is None:
            return self.steady.head_array(
                x,
                np.zeros_like(x),
                layers=layers,
                show_progress=show_progress,
                parallel=parallel,
            )
        if self._is_final(t):
            return self.final_steady_model().head_array(
                x, layers=layers, show_progress=show_progress, parallel=parallel
            )
        return self.steady.head_array(
            x,
            np.zeros_like(x),
            layers=layers,
            show_progress=show_progress,
            parallel=parallel,
        )[:, np.newaxis, :] + self.transient.head_array(
            x,
            np.zeros_like(x),
            t,
            layers=layers,
            show_progress=show_progress,
            parallel=parallel,
        )

    def disvec(self, x: float, t: float | np.ndarray | None = None, layers=None):
        """Compute discharge vector at x, t.

        Parameters
        ----------
        x : float
            x-coordinate at which the discharge vector is computed
        t : float, array, "final", or None, optional
            time(s) for which the discharge vector is returned. If None, only
            the steady-state component is returned. If "final" or infinite, the
            final steady state is returned.
        layers : integer, list or array, optional
            layers for which the discharge vector is returned
            if None: all layers are returned

        Returns
        -------
        qx : array size `nlayers` or `nlayers, ntimes`
            x-component of the discharge vector
        qy : array size `nlayers` or `nlayers, ntimes`
            y-component of the discharge vector (zero in the y = 0 plane)
        """
        if t is None:
            return self.steady.disvec(x, 0.0, layers=layers)
        if self._is_final(t):
            return self.final_steady_model().disvec(x, layers=layers)
        qx, qy = self.steady.disvec(x, 0.0, layers=layers)
        qxt, qyt = self.transient.disvec(x, 0.0, t, layers=layers)
        return qx[:, np.newaxis] + qxt, qy[:, np.newaxis] + qyt

    def disvecalongline(
        self,
        x: np.ndarray,
        t: float | np.ndarray | None = None,
        layers: int | list | None = None,
    ):
        """Compute discharge vector along line (the y = 0 plane of the cross-section).

        Parameters
        ----------
        x : 1D array or list
            x values of line
        t : float, 1D array, "final", or None, optional
            time(s) for which the discharge vector is returned. If None, only
            the steady-state component is returned. If "final" or infinite, the
            final steady state is returned.
        layers : integer, list or array, optional
            layers for which the discharge vector is returned
            if None: all layers are returned

        Returns
        -------
        qx : array size `nlayers, nx` or `nlayers, ntimes, nx`
            x-component of the discharge vector along the line
        qy : array size `nlayers, nx` or `nlayers, ntimes, nx`
            y-component of the discharge vector along the line (zero in the
            y = 0 plane)
        """
        if t is None:
            return self.steady.disvecalongline(x, np.zeros_like(x), layers=layers)
        if self._is_final(t):
            return self.final_steady_model().disvecalongline(x, layers=layers)
        qx_steady, qy_steady = self.steady.disvecalongline(
            x, np.zeros_like(x), layers=layers
        )
        qx_transient, qy_transient = self.transient.disvecalongline(
            x, np.zeros_like(x), t, layers=layers
        )
        return (
            qx_steady[:, np.newaxis, :] + qx_transient,
            qy_steady[:, np.newaxis, :] + qy_transient,
        )

    def velocity_grid(
        self,
        x: np.ndarray,
        z: np.ndarray,
        t: float | None = None,
        show_progress: bool = False,
        parallel: bool = False,
    ):
        """Compute velocity grid in the cross-section (the y = 0 plane).

        Parameters
        ----------
        x : 1d-array
            x values of grid
        z : 1d-array
            z values of grid
        t : float, optional
            time for which the grid is returned. If None, only the
            steady-state component is returned.
        show_progress : bool
            show computation progress, by default False
        parallel : bool or int, optional
            if True, compute in parallel using multiprocessing,
            by default False. If an int is provided, it is interpreted as the
            number of processes to use.

        Returns
        -------
        velocity : array
            velocity vector (vx, vy, vz) at each point in grid,
            size (3, len(z), 1, len(x))
        """
        x = np.atleast_1d(x)
        z = np.atleast_1d(z)
        y = np.zeros(1)  # cross-section lies in the y = 0 plane
        if t is None:
            return self.steady.velocity_grid(
                x, y, z, show_progress=show_progress, parallel=parallel
            )
        if self._is_final(t):
            return self.final_steady_model().velocity_grid(
                x, z, show_progress=show_progress, parallel=parallel
            )
        v_steady = self.steady.velocity_grid(
            x, y, z, show_progress=show_progress, parallel=parallel
        )
        return v_steady + self.transient.velocity_grid(
            x, y, z, t, show_progress=show_progress, parallel=parallel
        )

    def solve(self, printmat: int = 0, sendback: int = 0, silent: bool = False):
        """Compute solution for both the steady and the transient submodel.

        Parameters
        ----------
        printmat : int, optional
            if nonzero, return the matrix and right-hand side instead of the
            solution, by default 0
        sendback : int, optional
            if nonzero, return the solution vector, by default 0
        silent : bool, optional
            if True, suppress printed output, by default False

        Returns
        -------
        sol_steady, sol_transient : tuple
            solutions of the steady and transient submodels, only returned if
            `printmat` or `sendback` is nonzero
        """
        if not silent:
            print("Solving for steady and transient flow ...")

        sol_steady = self.steady.solve(printmat=printmat, sendback=sendback, silent=True)
        if not silent:
            print(f" - steady model   : {self.steady.neq} equations")
        sol_transient = self.transient.solve(
            printmat=printmat, sendback=sendback, silent=True
        )
        if not silent:
            print(f" - transient model: {self.transient.neq} equations")
            print("Solution complete.")
        if printmat or sendback:
            return sol_steady, sol_transient

    def handle_bc(
        self,
        tsandbc: list[tuple[float, float]] | str | None = None,
        bcsteady: float | Literal["first", "mean", "last"] | None = "first",
    ) -> tuple[float | None, list[tuple[float, float]] | str | None]:
        """Split boundary condition into steady and transient components.

        Parameters
        ----------
        tsandbc: list of tuples or str
            List of (t, bc) tuples describing the total boundary-condition history.
            If "fixed", the transient component is fixed at zero.
        bcsteady: float, {"first", "mean", "last"}, or None, optional
            Steady component or a selector for a value from ``tsandbc``. Defaults to
            "first". If None and a series is provided, use a zero steady component;
            if both inputs are None, no boundary condition is set.

        Returns
        -------
        bcsteady: float or None
            Steady boundary condition.
        tsandbc_transient: list of tuples, str, or None
            List of (t, bc_transient) tuples, "fixed", or None.

        Notes
        -----
        A selector requires a time series. To compute transient-only effects, pass
        ``bcsteady=None`` or ``bcsteady=0`` so the full time series is transient. To
        compute steady-only effects, omit ``tsandbc`` and provide a numeric steady
        value. If both inputs are None, this method returns ``(None, None)``.

        For example, ``[(0, 3), (1, 5)]`` defaults to a steady value of 3 and a
        transient series of ``[(0, 0), (1, 2)]``.
        """
        if isinstance(bcsteady, str) and bcsteady not in ("first", "mean", "last"):
            raise ValueError(
                "bcsteady must be a number, None, 'first', 'mean', or 'last'."
            )
        if tsandbc is None:
            if isinstance(bcsteady, str):
                raise ValueError("A time series is required to determine bcsteady.")
            if bcsteady is None:
                return None, None
            return bcsteady, "fixed"
        if isinstance(tsandbc, str):
            if tsandbc != "fixed":
                raise ValueError("tsandbc must be a time series, 'fixed', or None.")
            tsandbc = [(0.0, 0.0)]
            fixed = True
        else:
            fixed = False
        ts = np.asarray(tsandbc)
        if isinstance(bcsteady, str):
            if bcsteady == "mean":
                bcsteady = float(np.mean(ts[:, 1]))
            elif bcsteady == "first":
                bcsteady = tsandbc[0][-1]
            else:
                bcsteady = tsandbc[-1][-1]
        elif bcsteady is None:
            bcsteady = 0.0

        if fixed:
            return bcsteady, "fixed"

        bctransient = ts[:, 1] - bcsteady

        return bcsteady, list(zip(ts[:, 0], bctransient, strict=True))

    def _final_bc(self, bcsteady, tsandbc):
        """Final BC = steady + last transient value."""
        if tsandbc is None or tsandbc == "fixed":
            return bcsteady
        return (bcsteady or 0.0) + tsandbc[-1][1]

    def final_steady_model(self):
        """Build and solve a steady-only model representing the final state.

        The final boundary condition of each element and inhomogeneity is its
        steady component plus the last value of its transient time series.

        Returns
        -------
        mlss : ModelXsection
            solved steady-only model equivalent to the final state of this model
        """
        mlss = ModelXsection(naq=self.naq)
        for inhom in self.inhomdict.values():
            hfinal = self._final_bc(inhom.hsteady, inhom.tsandhstar)
            Nfinal = self._final_bc(inhom.Nsteady, inhom.tsandN)
            inhom_type = type(inhom)
            sig = inspect.signature(inhom_type)
            kwargs = {"model": mlss}
            for param in sig.parameters.keys():
                if param == "model":
                    continue
                if param == "hsteady":
                    kwargs[param] = hfinal
                elif param == "Nsteady":
                    kwargs[param] = Nfinal
                elif hasattr(inhom.steady, param):
                    kwargs[param] = getattr(inhom.steady, param)
                else:
                    kwargs[param] = getattr(inhom.transient, param)
            inhom_type(**kwargs)
        for e in self.elementlist:
            if hasattr(e, "tsandbc"):
                bcfinal = self._final_bc(e.bcsteady, e.tsandbc)
                e_type = type(e)
                sig = inspect.signature(e_type)
                kwargs = {"model": mlss}
                for param in sig.parameters.keys():
                    if param == "model":
                        continue
                    if param.startswith("tsand"):
                        # the final state is steady-only, no transient series
                        kwargs[param] = None
                    elif param in ["qsteady", "hsteady"]:
                        kwargs[param] = bcfinal
                    elif hasattr(e, param):
                        kwargs[param] = getattr(e, param)
                    elif e.steady is not None and hasattr(e.steady, param):
                        # element geometry (xls, xld, res, wh, layers, ...) is
                        # stored on the sub-elements, not on the unified element
                        kwargs[param] = getattr(e.steady, param)
                    elif e.transient is not None and hasattr(e.transient, param):
                        kwargs[param] = getattr(e.transient, param)
                e_type(**kwargs)
        mlss.solve(silent=True)
        return mlss


class UnifiedInhom1D:
    """Base class for unified cross-section inhomogeneities.

    Splits the boundary-condition time series ``tsandhstar`` and ``tsandN`` into
    steady and transient components and adds the corresponding steady and
    transient inhomogeneities to the submodels of a unified `ModelXsection`.

    Parameters
    ----------
    model : ModelXsection
        unified model to which the inhomogeneity is added
    tsandhstar : list of tuples or str, optional
        list containing time and water level pairs for the total hstar boundary
        condition, or "fixed" for a fixed transient value of zero
    hsteady : float, {"first", "mean", "last"}, or None, optional
        steady hstar component, defaults to None
    tsandN : list of tuples or str, optional
        list containing time and infiltration pairs for the total infiltration
        boundary condition, or "fixed" for a fixed transient value of zero
    Nsteady : float, {"first", "mean", "last"}, or None, optional
        steady infiltration component, defaults to None
    label : str or None, optional
        label of the inhomogeneity
    """

    def __init__(
        self,
        model: ModelXsection,
        tsandhstar: list[tuple[float, float]] | str | None = None,
        hsteady: float | Literal["first", "mean", "last"] | None = None,
        tsandN: list[tuple[float, float]] | str | None = None,
        Nsteady: float | Literal["first", "mean", "last"] | None = None,
        label: str | None = None,
    ):
        """Initialize the unified inhomogeneity and add it to the model."""
        # check if model is a unified ModelXsection
        is_unified_modelxsection(model)
        self.hsteady, self.tsandhstar = model.handle_bc(tsandhstar, hsteady)
        self.Nsteady, self.tsandN = model.handle_bc(tsandN, Nsteady)
        self.model = model
        self.label = label
        self.model.add_inhom(self)

    def __repr__(self):
        """Return a string combining the steady and transient representations."""
        steady_label = self.steady.__repr__()
        transient_label = (
            self.transient.__repr__() if self.transient is not None else "fixed"
        )
        if "fixed" in transient_label:
            transient_label = ""
        else:
            transient_label = " + " + transient_label.split(" with ")[-1]
        return steady_label + transient_label


class UnifiedElement1D:
    """Base class for unified 1D elements of a cross-section model.

    Splits the boundary-condition time series ``tsandbc`` into steady and
    transient components and adds the corresponding steady and transient
    elements to the submodels of a unified `ModelXsection`.

    Parameters
    ----------
    model : ModelXsection
        unified model to which the element is added
    tsandbc : list of tuples or str, optional
        list of (t, bc) tuples describing the total boundary-condition history,
        or "fixed" for a fixed transient value of zero
    bcsteady : float, {"first", "mean", "last"}, or None, optional
        steady component or a selector for a value from ``tsandbc``,
        defaults to "first"
    label : str or None, optional
        label of the element
    require_bc : bool, optional
        if True, raise a ValueError when both ``tsandbc`` and ``bcsteady`` are
        None, by default False

    Raises
    ------
    ValueError
        If `require_bc` is True and no boundary condition is provided.
    """

    def __init__(
        self,
        model: ModelXsection,
        tsandbc: list[tuple[float, float]] | str | None = None,
        bcsteady: float | Literal["first", "mean", "last"] | None = None,
        label: str | None = None,
        require_bc: bool = False,
    ):
        """Initialize the unified element and add it to the model."""
        # check if model is a unified ModelXsection
        is_unified_modelxsection(model)
        self.model = model
        self.label = label
        self.bcsteady, self.tsandbc = self.model.handle_bc(tsandbc, bcsteady)
        if require_bc and self.bcsteady is None and self.tsandbc is None:
            raise ValueError("Either tsandbc or bcsteady must be provided.")
        self.model.add_element(self)

    def __repr__(self):
        """Return the string representation of the transient element."""
        return self.transient.__repr__()

    def plot(self, *args, **kwargs):
        """Plot the element.

        Delegates to the transient element if it exists, otherwise to the steady
        element. All arguments are passed through.

        Parameters
        ----------
        *args
            passed on to the element's plot method
        **kwargs
            passed on to the element's plot method
        """
        return (
            self.transient.plot(*args, **kwargs)
            if self.transient is not None
            else self.steady.plot(*args, **kwargs)
        )


class XsectionMaq(UnifiedInhom1D):
    """Unified cross-section inhomogeneity for a multi-aquifer sequence.

    Boundary-condition time series ``tsandhstar`` and ``tsandN`` describe the total
    boundary history. The corresponding steady values default to None. With a time
    series, None uses a zero steady component; use ``"first"``, ``"mean"``, or
    ``"last"`` to derive the steady component from the series, or provide a numeric
    value. The steady component is subtracted from the series. For steady-only
    effects, provide a numeric steady value without the corresponding time series.
    If both inputs for a boundary are None, that boundary is omitted.

    Parameters
    ----------
    model : ModelXsection
        unified model to which the cross-section is added
    x1 : float
        left boundary of the cross-section (may be -np.inf if it extends to infinity)
    x2 : float
        right boundary of the cross-section (may be np.inf if it extends to infinity)
    kaq : float, array or list
        hydraulic conductivity of each aquifer from the top down
        if float, hydraulic conductivity is the same in all aquifers
    z : array or list
        elevation tops and bottoms of the aquifers from the top down
        leaky layers may have zero thickness
        if topboundary='conf': length is 2 * number of aquifers
        if topboundary='semi': length is 2 * number of aquifers + 1 as top
        of leaky layer on top of system needs to be specified
    c : float, array or list
        resistance of leaky layers from the top down
        if float, resistance is the same for all leaky layers
        if topboundary='conf': length is number of aquifers - 1
        if topboundary='semi': length is number of aquifers
    Saq : float, array or list
        specific storage of the aquifers
    Sll : float, array or list
        specific storage of the leaky layers
    leffaq : float, array or list
        loading efficiency of the aquifer
        only used when topboundary='semi' and hstar varies with time
    leffll : float, array or list
        loading efficiency of the leaky layer
        only used when topboundary='semi' and hstar varies with time
    poraq : float, array or list
        porosity of all aquifers from the top down
    porll : float, array or list
        porosity of all leaky layers from the top down
    topboundary : string, 'conf', 'semi', 'phreatic', or 'leaky' (default is 'conf')
        indicating whether the top is confined ('con' is enough), phreatic ('phr' is
        enough), semi-confined ('sem' is enough), or a leaky layer ('lea' is enough).
        When phreatic, the storage coefficient (Saq) of the top model layer is
        treated as phreatic storage (and not multiplied with the aquifer thickness)
    tsandhstar : list of tuples or str, optional
        list containing time and water level pairs for the total hstar boundary
        condition, or "fixed" for a fixed transient value of zero
    hsteady : float, {"first", "mean", "last"}, or None, optional
        steady hstar component. A selector requires the corresponding time series.
        Defaults to None.
    tsandN : list of tuples or str, optional
        list containing time and infiltration pairs for the total infiltration
        boundary condition, or "fixed" for a fixed transient value of zero
    Nsteady : float, {"first", "mean", "last"}, or None, optional
        steady infiltration component. A selector requires the corresponding time
        series. Defaults to None.
    name : string, optional
        name of the cross-section
    """

    def __init__(
        self,
        model: ModelXsection,
        x1: float,
        x2: float,
        kaq=1,
        z=(1, 0),
        c=(),
        Saq=0.001,
        Sll=0,
        leffaq=0,
        leffll=0,
        poraq=0.3,
        porll=0.3,
        topboundary: Literal["conf", "semi", "phreatic", "leaky"] = "conf",
        tsandhstar=None,
        hsteady=None,
        tsandN=None,
        Nsteady=None,
        name=None,
    ):
        """Initialize the unified XsectionMaq and add it to the model."""
        super().__init__(
            model,
            tsandhstar=tsandhstar,
            hsteady=hsteady,
            tsandN=tsandN,
            Nsteady=Nsteady,
            label=name,
        )

        # steady model
        steady_topboundary = (
            "conf"
            if topboundary.startswith("con") or topboundary.startswith("phre")
            else "semi"
        )
        self.steady = tfs.XsectionMaq(
            model.steady,
            x1,
            x2,
            kaq=kaq,
            z=z,
            c=c,
            npor=poraq,
            topboundary=steady_topboundary,
            hstar=self.hsteady,
            N=self.Nsteady,
            name=name,
        )
        # transient model
        self.transient = tft.XsectionMaq(
            model.transient,
            x1,
            x2,
            kaq=kaq,
            z=z,
            c=c,
            Saq=Saq,
            Sll=Sll,
            leffaq=leffaq,
            leffll=leffll,
            poraq=poraq,
            porll=porll,
            topboundary=topboundary,
            tsandhstar=self.tsandhstar,
            tsandN=self.tsandN,
            name=name,
        )


class Xsection3D(UnifiedInhom1D):
    """Unified cross-section inhomogeneity consisting of stacked aquifer layers.

    Vertical resistance is computed from the vertical hydraulic conductivity and
    the anisotropy factor.

    Boundary-condition time series ``tsandhstar`` and ``tsandN`` describe the total
    boundary history. The corresponding steady values default to None. With a time
    series, None uses a zero steady component; use ``"first"``, ``"mean"``, or
    ``"last"`` to derive the steady component from the series, or provide a numeric
    value. The steady component is subtracted from the series. For steady-only
    effects, provide a numeric steady value without the corresponding time series.
    If both inputs for a boundary are None, that boundary is omitted.

    Parameters
    ----------
    model : ModelXsection
        unified model to which the cross-section is added
    x1 : float
        left boundary of the cross-section (may be -np.inf if it extends to infinity)
    x2 : float
        right boundary of the cross-section (may be np.inf if it extends to infinity)
    kaq : float, array or list
        hydraulic conductivity of each layer from the top down
        if float, hydraulic conductivity is the same in all layers
    z : array or list
        elevation of top of system followed by bottoms of all layers from the top
        down; bottom of layer is automatically equal to top of layer below it
        if topboundary='conf': length is number of layers + 1
        if topboundary='semi': length is number of layers + 2 as top
        of leaky layer on top of system needs to be specified
    Saq : float, array or list
        specific storage of the aquifers
    kzoverkh : float
        vertical anisotropy ratio kz divided by kh
        if float, value is the same for all layers
    leffaq : float, array or list
        loading efficiency of the aquifer
        only used when topboundary='semi' and hstar varies with time
    leffll : float, array or list
        loading efficiency of the leaky layer
        only used when topboundary='semi' and hstar varies with time
    poraq : float, array or list
        porosity of all aquifer layers from the top down
    topboundary : string, 'conf', 'semi', 'phreatic', or 'leaky' (default is 'conf')
        indicating whether the top is confined ('con' is enough), phreatic ('phr' is
        enough), semi-confined ('sem' is enough), or a leaky layer ('lea' is enough).
        When phreatic, the storage coefficient (Saq) of the top model layer is
        treated as phreatic storage (and not multiplied with the aquifer thickness).
        When 'semi' or 'leaky', topres and topthick must be specified.
    topres : float
        resistance of the top boundary, only read if topboundary='semi' or 'leaky'
    topthick : float
        thickness of the top boundary, only read if topboundary='semi' or 'leaky'
    topSll : float
        specific storage of the top boundary, only read if topboundary='leaky'
    toppor : float
        porosity of the top boundary, only read if topboundary='leaky'
    tsandhstar : list of tuples or str, optional
        list containing time and water level pairs for the total hstar boundary
        condition, or "fixed" for a fixed transient value of zero
    hsteady : float, {"first", "mean", "last"}, or None, optional
        steady hstar component. A selector requires the corresponding time series.
        Defaults to None.
    tsandN : list of tuples or str, optional
        list containing time and infiltration pairs for the total infiltration
        boundary condition, or "fixed" for a fixed transient value of zero
    Nsteady : float, {"first", "mean", "last"}, or None, optional
        steady infiltration component. A selector requires the corresponding time
        series. Defaults to None.
    name : string, optional
        name of the cross-section
    """

    def __init__(
        self,
        model: ModelXsection,
        x1: float,
        x2: float,
        kaq=1,
        z=(4, 3, 2, 1),
        Saq=0.001,
        kzoverkh=0.1,
        leffaq=0,
        leffll=0,
        poraq=0.3,
        topboundary="conf",
        topres=0,
        topthick=0,
        topSll=0,
        toppor=0.3,
        tsandhstar=None,
        hsteady=None,
        tsandN=None,
        Nsteady=None,
        name=None,
    ):
        """Initialize the unified Xsection3D and add it to the model."""
        super().__init__(
            model,
            tsandhstar=tsandhstar,
            hsteady=hsteady,
            tsandN=tsandN,
            Nsteady=Nsteady,
            label=name,
        )
        # steady model
        self.steady = tfs.Xsection3D(
            model.steady,
            x1,
            x2,
            kaq,
            z=z,
            kzoverkh=kzoverkh,
            npor=poraq,
            topboundary=topboundary,
            hstar=self.hsteady,
            topres=topres,
            topthick=topthick,
            N=self.Nsteady,
            name=name,
        )
        # transient model
        self.transient = tft.Xsection3D(
            model.transient,
            x1,
            x2,
            kaq=kaq,
            z=z,
            Saq=Saq,
            kzoverkh=kzoverkh,
            leffaq=leffaq,
            leffll=leffll,
            poraq=poraq,
            topboundary=topboundary,
            topres=topres,
            topthick=topthick,
            topSll=topSll,
            toppor=toppor,
            tsandhstar=self.tsandhstar,
            tsandN=self.tsandN,
            name=name,
        )


class LineSink1D(UnifiedElement1D):
    r"""Linesink1D with a specified discharge.

    Parameters
    ----------
    model : ModelXsection
        unified model to which the element is added
    xls : float
        x-coordinate of the linesink
    tsandq : list of tuples or str
        Time and specific-discharge pairs describing the total discharge history.
        ``"fixed"`` specifies zero transient discharge.
    qsteady : float, {"first", "mean", "last"}, or None, optional
        Steady discharge component. Defaults to "first"; selectors require a time
        series. The steady value is subtracted from ``tsandq`` to obtain transient
        discharge. Set ``qsteady=None`` or ``qsteady=0`` for transient-only effects.
        For steady-only effects, provide a numeric value and set ``tsandq=None``.
        At least one of ``tsandq`` or ``qsteady`` must be provided.
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string or None (default: None)
        label of the linesink
    """

    def __init__(
        self,
        model: ModelXsection,
        xls: float = 0,
        tsandq: list[tuple[float, float]] | str | None = [(0, 1)],
        qsteady: float | Literal["first", "mean", "last"] | None = "first",
        # res=0,  TODO: resistance not implemented in steady?
        # wh="H",
        # vres=0.0,  TODO: vertical resistance in linesink not implemented in steady?
        # wv=1.0,
        layers=0,
        label=None,
    ):
        """Initialize a unified 1D line-sink with a specified discharge."""
        super().__init__(
            model,
            tsandbc=tsandq,
            bcsteady=qsteady,
            label=label,
            require_bc=True,
        )
        self.steady = tfs.LineSink1D(
            model.steady,
            xls=xls,
            sigls=self.bcsteady,
            layers=layers,
            label=label,
        )
        tsandq = self.tsandbc
        if isinstance(tsandq, str) and tsandq == "fixed":
            tsandq = [(0.0, 0.0)]
        self.transient = tft.LineSink1D(
            model.transient,
            xls=xls,
            tsandq=tsandq,
            layers=layers,
            label=label,
        )


class River1D(UnifiedElement1D):
    """Head-specified element for cross-section models.

    Parameters
    ----------
    model : Model object
        Model to which the element is added
    xls : float
        x-coordinate of the river
    tsandh : str or list of tuples
        Time and head pairs describing the total head history, or ``"fixed"`` for zero
        transient head.
    hsteady : float, {"first", "mean", "last"}, or None, optional
        Steady head component. Defaults to "first"; selectors require a time series.
        The steady value is subtracted from ``tsandh`` to obtain the transient head.
        Set ``hsteady=None`` or ``hsteady=0`` for transient-only effects. For
        steady-only effects, provide a numeric value and set ``tsandh=None``. At least
        one of ``tsandh`` or ``hsteady`` must be provided.
    res : float
        resistance to flow into or out of the river, default is 0.0 (no resistance)
    wh : string or float
        wetted perimeter of the linesink, "H" for aquifer height,
        "2H" for 2x aquifer height (two-sided flow) or specify any float value
    layers : int, array or list
        layer (int) or layers (list or array) in which linesink is located
    label : string, optional
        label of the element
    """

    def __init__(
        self,
        model: ModelXsection,
        xls: float = 0,
        tsandh: list[tuple[float, float]] | str | None = [(0, 1)],
        hsteady: float | Literal["first", "mean", "last"] | None = "first",
        res: float = 0,
        wh: str | float = "H",
        layers=0,
        label=None,
    ):
        """Initialize a unified 1D head-specified line-sink (river)."""
        super().__init__(
            model,
            tsandbc=tsandh,
            bcsteady=hsteady,
            label=label,
            require_bc=True,
        )
        self.steady = tfs.River1D(
            model.steady,
            xls=xls,
            hls=self.bcsteady,
            res=res,
            wh=wh,
            layers=layers,
            label=label,
        )
        self.transient = tft.River1D(
            model.transient,
            xls=xls,
            tsandh=self.tsandbc,
            res=res,
            wh=wh,
            layers=layers,
            label=label,
        )


class ImpermeableWall1D(UnifiedElement1D):
    """Impermeable wall for cross-section models.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xld : float
        x-coordinate of the impermeable wall
    layers : int, array or list
        layer (int) or layers (list or array) in which impermeable wall is located
    label : string or None (default: None)
        label of the element
    """

    def __init__(
        self,
        model: ModelXsection,
        xld: float = 0,
        layers: int | list | np.ndarray = 0,
        label: str | None = None,
    ):
        """Initialize a unified impermeable wall."""
        super().__init__(model, label=label, bcsteady=None)
        self.steady = tfs.ImpermeableWall1D(
            model.steady,
            xld=xld,
            layers=layers,
            label=label,
        )
        self.transient = tft.ImpermeableWall1D(
            model.transient,
            xld=xld,
            layers=layers,
            label=label,
        )


class LeakyWall1D(UnifiedElement1D):
    r"""Leaky wall with specified resistance for cross-section models.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xld : float
        x-coordinate of the leaky wall
    res : float or str
        resistance of the leaky wall, default is np.inf which models an impermeable
        wall, specify a float to model a finite resistance.
    layers : int, array or list
        layer (int) or layers (list or array) in which leaky wall is located
    label : string or None (default: None)
        label of the element
    """

    def __init__(
        self,
        model: ModelXsection,
        xld: float = 0,
        res: float | str = np.inf,
        layers: int | list | np.ndarray = 0,
        label: str | None = None,
    ):
        """Initialize a unified leaky wall with a specified resistance."""
        super().__init__(model, label=label, bcsteady=None)
        self.steady = tfs.LeakyWall1D(
            model.steady,
            xld=xld,
            res=res,
            layers=layers,
            label=label,
        )
        self.transient = tft.LeakyWall1D(
            model.transient,
            xld=xld,
            res=res,
            layers=layers,
            label=label,
        )


class Constant(UnifiedElement1D):
    """Constant head element for cross-section models.

    Thin wrapper around timflow.steady.Constant.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    xr : float
        x-coordinate of the constant head element
    hr : float
        head value of the constant head element
    label : string or None (default: None)
        label of the element
    layer : int (default: 0)
        layer in which the constant head element is located
    """

    def __init__(self, model, xr=0, hr=0, layer=0, label=None):
        """Initialize a unified constant head element."""
        super().__init__(model, label=label, bcsteady=hr)
        self.steady = tfs.Constant(
            model.steady,
            xr=xr,
            yr=0.0,
            hr=hr,
            layer=layer,
            label=label,
        )
        self.transient = None


class Uflow(UnifiedElement1D):
    """Uflow element for cross-section models.

    Thin wrapper around timflow.steady.Uflow.

    Parameters
    ----------
    model : Model object
        model to which the element is added
    """

    def __init__(self, model, slope=0.0, label=None):
        """Initialize a unified uniform flow element."""
        super().__init__(model, label=label, bcsteady=None)
        self.steady = tfs.Uflow(model.steady, slope=slope, angle=0.0, label=label)
        self.transient = None
