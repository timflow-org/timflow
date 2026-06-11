"""Unified calibration framework for transient and steady-state models.

Supports independent and joint calibration where models share parameters.

Example::

    import timflow as tf

    # setup joint calibration
    cal = tf.Calibrate(transient_model=ml_t, steady_model=ml_s)

    # add observations, either a steady head observation or a time series of heads
    cal.add_head_time_series(name='obs1', x=0, y=0, layer=0, t=t, h=h)
    cal.add_steady_head(name='obs2', x=0, y=0, layer=0, h=h_steady)

    # set calibration parameters
    cal.set_aquifer_parameter('kaq', layers=0, initial=10, pmin=1, pmax=100, model="both")

    # calibrate model
    cal.fit()
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, get_lapack_funcs, svd
from scipy.optimize import least_squares

from timflow.steady.element import Element as SteadyElement
from timflow.steady.model import Model as SteadyModel
from timflow.transient.element import Element as TransientElement
from timflow.transient.model import Model as TransientModel


@dataclass
class ParameterTarget:
    """A single array slice in a model that a Parameter controls."""

    array: np.ndarray
    key: slice  # the slice into `array`

    def set(self, value: float) -> None:
        self.array[self.key] = value

    def get(self) -> np.ndarray:
        return self.array[self.key]


@dataclass
class Parameter:
    """A single optimizable parameter that may affect multiple model arrays.

    Parameters can span both transient and steady-state models, enabling
    joint calibration with shared parameters.
    """

    name: str
    initial: float | None
    pmin: float = -np.inf
    pmax: float = np.inf
    targets: list[ParameterTarget] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    inhoms: list[str] = field(default_factory=list)
    log_scale: bool = False

    # filled after fitting
    optimal: float = np.nan
    std: Optional[float] = None

    @property
    def effective_initial(self) -> float:
        """Effective starting value for optimization.

        Returns ``initial`` if set explicitly, otherwise reads the current
        value from the model parameter arrays. This allows calibration to
        start from the current model state when ``initial=None``.
        """
        if self.initial is not None:
            return self.initial
        return self.get()

    def set(self, value: float) -> None:
        """Push value to all model arrays this parameter controls."""
        if self.initial is None:
            initial = float(self.targets[0].get()[0])  # read current model value
        else:
            initial = self.initial
        if value is None:
            value = initial
        if self.log_scale:
            value = np.sign(initial) * 10**value
        for target in self.targets:
            target.set(value)

    def get(self) -> float:
        """Read back current value from first target."""
        if self.targets or self.initial is None:
            value = float(self.targets[0].get()[0])
        else:
            value = self.initial
        return value

    def add_target(
        self, array: np.ndarray, key: slice, model: Any = None, inhom: Any = None
    ) -> None:
        self.targets.append(ParameterTarget(array, key))
        if model is not None:
            self.models.append(model.model_type)
        if inhom is not None:
            self.inhoms.append(inhom.name)


@dataclass
class SteadyHead:
    """A single steady-state head observation at a point in the aquifer.

    Attributes
    ----------
    x, y : float
        Observation coordinates.
    layer : int
        Aquifer layer index (0-based).
    h : float
        Observed head.
    weight : float
        Weight applied to the residual in the objective function.
    """

    x: float
    y: float
    layer: int
    h: float
    weight: float = 1.0
    model_key: str = field(default="steady", init=False)  # 'steady'


@dataclass
class SteadyHeadInWell:
    """A single steady-state head observation in a well.

    Attributes
    ----------
    element : SteadyElement
        Well element in which the head is observed.
    x, y : float
        Location of the element.
    layer : int
        Aquifer layer index (0-based).
    h : float
        Observed head.
    weight : float
        Weight applied to the residual in the objective function.
    """

    element: SteadyElement
    x: float
    y: float
    layer: int
    h: float
    weight: float = 1.0
    model_key: str = field(default="steady", init=False)  # 'steady'


@dataclass
class HeadSeries:
    """A transient head observation time series at a point in the aquifer.

    Attributes
    ----------
    x, y : float
        Observation coordinates.
    layer : int
        Aquifer layer index (0-based).
    t : np.ndarray
        Observation times.
    h : np.ndarray
        Observed heads.
    weights : float, np.ndarray, optional
        Per time-series (float) or per-timestep weights (array). Defaults to uniform
        weight of 1.0 if ``None``.
    constant : float or (float, float, float), optional
        If not ``None``, a constant offset is added as a calibration
        parameter. Supply a float for the initial value (unbounded), or a
        ``(initial, pmin, pmax)`` tuple to set bounds.
    time_shift : float or (float, float, float), optional
        If not ``None``, a time shift is added as a calibration parameter.
        Supply a float for the initial value (unbounded), or a
        ``(initial, pmin, pmax)`` tuple to set bounds.
    """

    x: float
    y: float
    layer: int
    t: np.ndarray
    h: np.ndarray
    weights: Optional[np.ndarray] = None
    normalized: bool = False
    model_key: str = field(default="transient", init=False)  # 'transient'
    constant: float | tuple[float, float, float] | None = (
        None  # constant parameter and optional bounds
    )
    time_shift: float | tuple[float, float, float] | None = (
        None  #  time shift and optional bounds
    )
    # placeholder for constant and time_shift parameters
    _constant: np.ndarray = field(default_factory=lambda: np.zeros(1), init=False)
    _time_shift: np.ndarray = field(default_factory=lambda: np.zeros(1), init=False)

    def plot(
        self,
        ax: plt.Axes | None = None,
        apply_time_shift: bool = True,
        apply_constant: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot the observation time series.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. A new figure is created if ``None``.
        apply_time_shift : bool
            Subtract the fitted time shift from ``t`` before plotting.
            Has no effect if no time-shift parameter was added.
        apply_constant : bool
            Subtract the fitted constant from ``h`` before plotting.
            Has no effect if no constant parameter was added.
        **kwargs
            Passed to :func:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots()
        t = (
            self.t - self._time_shift
            if (self.time_shift is not None) and apply_time_shift
            else self.t
        )
        h = (
            self.h - self._constant
            if (self.constant is not None) and apply_constant
            else self.h
        )
        ax.plot(t, h, **kwargs)
        return ax


@dataclass
class HeadSeriesInWell:
    """A transient head observation time series in a well.

    Attributes
    ----------
    element : TransientElement
        Well element at which heads are observed.
    t : np.ndarray
        Observation times.
    h : np.ndarray
        Observed heads.
    weights : float, np.ndarray, optional
        Per time-series (float) or per-timestep weights (array). Defaults to uniform
        weight of 1.0 if ``None``.
    constant : float or (float, float, float), optional
        If not ``None``, a constant offset is added as a calibration
        parameter. Supply a float for the initial value (unbounded), or a
        ``(initial, pmin, pmax)`` tuple to set bounds.
    time_shift : float or (float, float, float), optional
        If not ``None``, a time shift is added as a calibration parameter.
        Supply a float for the initial value (unbounded), or a
        ``(initial, pmin, pmax)`` tuple to set bounds.
    """

    element: TransientElement
    t: np.ndarray
    h: np.ndarray
    weights: Optional[np.ndarray] = None
    model_key: str = field(default="transient", init=False)  # 'transient'
    constant: float | tuple[float, float, float] | None = (
        None  # constant parameter and optional bounds
    )
    time_shift: float | tuple[float, float, float] | None = (
        None  #  time shift and optional bounds
    )
    # placeholder for constant and time_shift parameters
    _constant: np.ndarray = field(default_factory=lambda: np.zeros(1), init=False)
    _time_shift: np.ndarray = field(default_factory=lambda: np.zeros(1), init=False)


class Calibrate:
    """Unified calibration class for transient and/or steady-state models.

    Supports independent and joint calibration where both model types share
    parameters (e.g., hydraulic conductivity, aquitard resistance).

    Parameters
    ----------
    transient_model : optional
        Transient model instance.
    steady_model : optional
        Steady-state model instance.
    reference_time : float, optional
        Specify reference time to compute head changes relative to the head at that
        time. The residuals are then computed with the following formula::

            res = ((sim - sim(t_ref)) - (obs - obs(t_ref))) * w

        The default is None, which uses the following formula for the residuals::

            res = (sim  - obs) * w

        This can be useful for noisy aquifer test data, for example.

    Notes
    -----
    When both models are provided, parameters registered via
    :meth:`set_aquifer_parameter` default to ``model='both'``, linking them
    across models so the optimizer updates both simultaneously. Pass
    ``model='transient'`` or ``model='steady'`` to restrict a parameter to
    one model.

    Examples
    --------
    Transient-only calibration::

        cal = tf.Calibrate(transient_model=ml_tr)
        cal.add_head_time_series("obs1", x=100.0, y=0.0, layer=0, t=t, h=h)
        cal.set_aquifer_parameter("kaq", layers=[0], initial=5.0, inhoms=["polder"])
        cal.fit()

    Joint steady/transient calibration with shared parameters::

        cal = tf.Calibrate(transient_model=ml_tr, steady_model=ml_ss)
        cal.add_steady_head("h_mean", x=100.0, y=0.0, layer=0, h=1.2)
        cal.add_head_time_series("h_obs", x=100.0, y=0.0, layer=0, t=t, h=h)
        cal.set_aquifer_parameter("kaq", layers=[0], initial=5.0, model="both")
        cal.fit()
    """

    def __init__(
        self,
        transient_model: TransientModel | None = None,
        steady_model: SteadyModel | None = None,
        reference_time: float | None = None,
    ):
        if transient_model is None and steady_model is None:
            raise ValueError("At least one model must be provided.")
        self.transient_model = transient_model
        self.reference_time = reference_time
        self._parameters: dict[str, Parameter] = {}
        self.observations_dict: dict[str, HeadSeries | SteadyHead] = {}
        self.observations_in_well_dict: dict[
            str, (SteadyHeadInWell | HeadSeriesInWell)
        ] = {}

        embedded_steady = getattr(transient_model, "steady", None)
        if embedded_steady is not None:
            if steady_model is not None and steady_model is not embedded_steady:
                warnings.warn(
                    "The transient model already has a steady model embedded "
                    "(transient_model.steady). The steady_model argument will be "
                    "ignored and the embedded steady model will be used instead.",
                    stacklevel=2,
                )
            elif steady_model is None:
                warnings.warn(
                    "The transient model has an embedded steady model "
                    "(transient_model.steady). Using it as the steady model for "
                    "this calibration.",
                    stacklevel=2,
                )
            self.steady_model = embedded_steady
        else:
            self.steady_model = steady_model

    def set_aquifer_parameter(
        self,
        name: str,
        layers: int | Iterable[int],
        initial: float | None = None,
        pmin: float = -np.inf,
        pmax: float = np.inf,
        log_scale: bool = False,
        inhoms: str | list[str] | None = None,
        model: str = "both",
    ) -> None:
        """Register an aquifer parameter for optimization.

        Parameters
        ----------
        name : str
            Parameter type: 'kaq', 'Saq', 'c', 'Sll', or 'kzoverkh'.
        layers : int or iterable of int
            Layer(s) affected. Consecutive layers are grouped under one
            scalar parameter.
        initial : float
            Starting value (in linear space). if None, uses current model parameter
            value. In case of coupled parameters between inhoms or models, uses
            first value it encounters.
        pmin, pmax : float
            Bounds for the optimizer.
        log_scale : bool
            Whether to optimize in log10 space (recommended for parameters like
            hydraulic conductivity that can span orders of magnitude).
        inhoms : str or list, optional
            Inhomogeneity name(s) to target. ``None`` targets the background
            aquifer.
        model : {'both', 'transient', 'steady'}
            Which model(s) this parameter applies to.

        Examples
        --------
        Set calibration parameter for hydraulic conductivity::

            cal.set_aquifer_parameter(
                "kaq", layers=[0], initial=1.0, pmin=1.0, pmax=100.0
            )

        See Also
        --------
        set_parameter_by_reference : Register a parameter by supplying the array
            reference directly, useful for non-aquifer parameters like well
            resistance.
        """
        assert isinstance(name, str), "name must be a string"
        assert model in ("both", "transient", "steady"), (
            "model must be 'both', 'transient', or 'steady'"
        )

        from_lay, to_lay = self._parse_layers(name, layers)
        models = self._resolve_models(model)
        pname = self._make_pname(name, from_lay, to_lay, inhoms, models)

        # set up parameter and add target arrays
        param = Parameter(
            name=pname, initial=initial, pmin=pmin, pmax=pmax, log_scale=log_scale
        )
        for ml in models:
            for iaq in self._resolve_inhoms(ml, inhoms):
                arr = self._get_aquifer_parameter_array(ml, iaq, name)
                slc = slice(from_lay, to_lay + 1)
                param.add_target(arr, slc, model=ml, inhom=iaq)

        if initial is not None:
            if log_scale:
                param.set(np.log10(np.abs(initial)))  # initialise arrays immediately
            else:
                param.set(initial)  # initialise arrays immediately
        self._parameters[pname] = param

    def set_parameter_by_reference(
        self,
        name: str,
        parameter: np.ndarray,
        initial: float = 0.0,
        pmin: float = -np.inf,
        pmax: float = np.inf,
        log_scale: bool = False,
    ) -> None:
        """Register a parameter by supplying the array reference directly.

        Useful for element-level parameters (e.g., well resistance) that are
        not part of the standard aquifer parameter arrays.

        Parameters
        ----------
        name : str
            Unique name for the parameter.
        parameter : np.ndarray
            Array whose values will be updated during optimization.
        initial : float
            Initial parameter value.
        pmin, pmax : float
            Lower and upper bounds for the optimizer.
        log_scale : bool
            Optimize in log10 space.

        Examples
        --------
        Set a calibration parameter by supplying the array reference directly::

            cal.set_parameter_by_reference(
                "well_resistance", well.res, initial=1.0, pmin=0.1, pmax=10.0
            )

        See Also
        --------
        set_aquifer_parameter : Register an aquifer parameter by specifying the type and
            layer(s) it applies to.
        """
        assert isinstance(parameter, np.ndarray), "parameter must be a numpy array"
        if initial is None:
            initial = parameter[0]
        param = Parameter(
            name=name, initial=initial, pmin=pmin, pmax=pmax, log_scale=log_scale
        )
        param.add_target(parameter, slice(None))
        if log_scale:
            param.set(np.log10(np.abs(initial)))
        else:
            param.set(initial)
        self._parameters[name] = param

    def add_steady_head(
        self, name: str, x: float, y: float, layer: int, h: float, weight: float = 1.0
    ) -> None:
        """Add a steady-state head observation.

        Parameters
        ----------
        name : str
            Unique observation name.
        x, y : float
            Observation coordinates.
        layer : int
            Aquifer layer index (0-based).
        h : float
            Observed head.
        weight : float, optional
            Weight in the objective function. Default is 1.

        Examples
        --------
        Add steady head observation at (x=100, y=0) in layer 0 with observed head of 1.5::

            cal.add_steady_head("piezometer1", x=100.0, y=0.0, layer=0, h=1.5)
        """
        if self.steady_model is None:
            raise ValueError(
                "Steady model must be provided to add steady head observations."
            )
        self.observations_dict[name] = SteadyHead(
            x=x, y=y, layer=layer, h=h, weight=weight
        )

    def add_steady_head_in_well(
        self, name: str, well_element: SteadyElement, h: float, weight: float = 1.0
    ) -> None:
        """Add a steady-state head observation inside a well.

        Parameters
        ----------
        name : str
            Unique observation name.
        well_element : SteadyElement
            Well element at which the head is observed.
        h : float
            Observed head.
        weight : float, optional
            Weight in the objective function. Default is 1.

        Examples
        --------
        Add a steady head observation inside a well::

            pump_well = timflow.steady.Well(...)
            cal.add_steady_head_in_well("well1", well_element=pump_well, h=0.8)
        """
        if self.steady_model is None:
            raise ValueError(
                "Steady model must be provided to add steady head observations."
            )
        self.observations_in_well_dict[name] = SteadyHeadInWell(
            element=well_element,
            x=well_element.x,
            y=well_element.y,
            layer=well_element.layer,
            h=h,
            weight=weight,
        )

    @staticmethod
    def _parse_optional_param(
        arg: float | tuple[float, float, float] | None,
        default_initial: float,
    ) -> tuple[float, float, float] | None:
        """Parse None | float | (initial, pmin, pmax) -> (initial, pmin, pmax) or None."""
        if arg is None:
            return None
        if isinstance(arg, bool):
            return (default_initial, -np.inf, np.inf)
        if isinstance(arg, (int, float)):
            return (arg, -np.inf, np.inf)
        return tuple(arg)  # (initial, pmin, pmax)

    def _add_series_constant(
        self,
        name: str,
        obs: Any,
        initial: float,
        pmin: float = -np.inf,
        pmax: float = np.inf,
    ) -> None:
        constant = Parameter(name + "_constant", initial=initial, pmin=pmin, pmax=pmax)
        constant.add_target(obs._constant, slice(None))
        self._parameters[name + "_constant"] = constant

    def _add_series_time_shift(
        self,
        name: str,
        obs: Any,
        initial: float,
        pmin: float = -np.inf,
        pmax: float = np.inf,
    ) -> None:
        time_shift = Parameter(
            name + "_time_shift", initial=initial, pmin=pmin, pmax=pmax
        )
        time_shift.add_target(obs._time_shift, slice(None))
        self._parameters[name + "_time_shift"] = time_shift

    def add_head_time_series(
        self,
        name: str,
        x: float,
        y: float,
        layer: int,
        t: np.ndarray,
        h: np.ndarray,
        weights: np.ndarray | None = None,
        normalized: bool = False,
        constant: float | tuple[float, float, float] | None = None,
        time_shift: float | tuple[float, float, float] | None = None,
    ) -> None:
        """Add a transient head observation time series.

        Parameters
        ----------
        name : str
            Unique observation name.
        x, y : float
            Observation coordinates.
        layer : int
            Aquifer layer index (0-based).
        t : array_like
            Observation times.
        h : array_like
            Observed heads.
        weights : float, np.ndarray, optional
            Per time-series (float) or per-timestep weights (array). Defaults to
            uniform weight of 1.0 if ``None``.
        normalized : bool
            Indicates whether head observations were normalized relative to some
            reference level, e.g. heads fluctuate around 0, or whether heads were
            provided in absolute values.
        constant : float or (float, float, float), optional
            Add a calibrated constant offset to this series. Supply a float
            for the initial value (unbounded), or a ``(initial, pmin, pmax)``
            tuple to set bounds. ``None`` (default) disables this parameter.
        time_shift : float or (float, float, float), optional
            Add a calibrated time shift to this series. Supply a float for
            the initial value (unbounded), or a ``(initial, pmin, pmax)``
            tuple to set bounds. ``None`` (default) disables this parameter.

        Examples
        --------
        Basic usage::

            cal.add_head_time_series("obs1", x=50.0, y=0.0, layer=0, t=t, h=h)

        With a bounded constant offset::

            cal.add_head_time_series(
                "obs1", x=50.0, y=0.0, layer=0, t=t, h=h, constant=(0.1, -1.0, 1.0)
            )
        """
        if self.transient_model is None:
            raise ValueError("Transient model must be provided to add head time series.")
        obs = HeadSeries(
            x=x,
            y=y,
            layer=layer,
            t=t,
            h=h,
            weights=weights,
            normalized=normalized,
            constant=constant,
            time_shift=time_shift,
        )
        self.observations_dict[name] = obs
        if constant is not None:
            initial, pmin, pmax = self._parse_optional_param(
                constant, default_initial=np.mean(h)
            )
            self._add_series_constant(name, obs, initial=initial, pmin=pmin, pmax=pmax)
        if time_shift is not None:
            initial, pmin, pmax = self._parse_optional_param(
                time_shift, default_initial=1 / 24.0
            )
            self._add_series_time_shift(name, obs, initial=initial, pmin=pmin, pmax=pmax)

    def add_head_time_series_in_well(
        self,
        name: str,
        well_element: TransientElement,
        t: np.ndarray,
        h: np.ndarray,
        constant: float | tuple[float, float, float] | None = None,
        time_shift: float | tuple[float, float, float] | None = None,
    ) -> None:
        """Add a transient head observation time series inside a well.

        Parameters
        ----------
        name : str
            Unique observation name.
        well_element : TransientElement
            Well element in which heads are observed.
        t : array_like
            Observation times.
        h : array_like
            Observed heads.
        constant : float or (float, float, float), optional
            Add a calibrated constant offset to this series. Supply a float
            for the initial value (unbounded), or a ``(initial, pmin, pmax)``
            tuple to set bounds. ``None`` (default) disables this parameter.
        time_shift : float or (float, float, float), optional
            Add a calibrated time shift to this series. Supply a float for
            the initial value (unbounded), or a ``(initial, pmin, pmax)``
            tuple to set bounds. ``None`` (default) disables this parameter.

        Examples
        --------
        Add observed head time series inside a well::

            pump_well = timflow.transient.Well(...)
            cal.add_head_time_series_in_well("well1", well_element=pump_well, t=t, h=h)
        """
        if self.transient_model is None:
            raise ValueError("Transient model must be provided to add head time series.")
        obs = HeadSeriesInWell(
            element=well_element,
            t=t,
            h=h,
            constant=constant,
            time_shift=time_shift,
        )
        self.observations_in_well_dict[name] = obs
        if constant is not None:
            initial, pmin, pmax = self._parse_optional_param(
                constant, default_initial=np.mean(h)
            )
            self._add_series_constant(name, obs, initial=initial, pmin=pmin, pmax=pmax)
        if time_shift is not None:
            initial, pmin, pmax = self._parse_optional_param(
                time_shift, default_initial=1 / 24.0
            )
            self._add_series_time_shift(name, obs, initial=initial, pmin=pmin, pmax=pmax)

    def residuals(self, p: np.ndarray, printdot: bool = False) -> np.ndarray:
        """Compute residuals for parameter vector ``p``.

        Parameters
        ----------
        p : np.ndarray
            Current parameter values in the same order as
            ``self.parameters``.
        printdot : bool
            Print a dot to stdout on each call, useful for tracking progress.

        Returns
        -------
        np.ndarray
            1-D array of weighted residuals (observed minus simulated).
        """
        if printdot:
            print(".", end="", flush=True)

        # 1. Push parameter values into model arrays
        for value, param in zip(p, self._parameters.values(), strict=True):
            param.set(value)

        # 2. Solve whichever models are registered
        needs_steady = any(
            s.model_key == "steady" for s in self.observations_dict.values()
        ) or any(s.model_key == "steady" for s in self.observations_in_well_dict.values())
        needs_transient = any(
            s.model_key == "transient" for s in self.observations_dict.values()
        ) or any(
            s.model_key == "transient" for s in self.observations_in_well_dict.values()
        )

        if needs_steady and self.steady_model is not None:
            self.steady_model.solve(silent=True)
        if needs_transient and self.transient_model is not None:
            # Solve the steady model first so transient_model.head() returns
            # up-to-date h_transient + h_steady values.
            if self.steady_model is not None:
                self.steady_model.solve(silent=True)
            self.transient_model.solve(silent=True)

        # 3. Accumulate residuals
        rv = np.empty(0)
        for obs in self.observations_dict.values():
            if obs.model_key == "transient":
                dt = obs._time_shift if obs.time_shift is not None else 0.0
                _tr_has_steady = getattr(self.transient_model, "steady", None) is not None
                if obs.normalized or _tr_has_steady:
                    # Normalized obs need no steady component; and when the
                    # transient model already embeds a steady model, its head()
                    # call already returns h_transient + h_steady, so there is
                    # nothing to add manually.
                    hsteady = 0.0
                elif self.steady_model is not None:
                    hsteady = self.steady_model.head(obs.x, obs.y, layers=obs.layer)
                else:
                    warnings.warn(
                        f"Observation '{obs.x},{obs.y}' is not marked as normalized "
                        "but no steady model is provided and the transient model has "
                        "no embedded steady model. Residuals will be computed against "
                        "transient heads only.",
                        stacklevel=2,
                    )
                    hsteady = 0.0
                h = (
                    self.transient_model.head(obs.x, obs.y, obs.t - dt, layers=obs.layer)
                    + hsteady
                )
                w = obs.weights if obs.weights is not None else np.ones_like(h)
                c = obs._constant if obs.constant is not None else 0.0
                if self.reference_time is not None:
                    # get closest observation to reference time
                    tref_idx = np.abs(obs.t - self.reference_time).argmin()
                    closest_ref_time = obs.t[tref_idx]
                    htref = self.transient_model.head(
                        obs.x, obs.y, closest_ref_time, layers=obs.layer
                    ).squeeze()
                    res = ((obs.h - obs.h[tref_idx]) - (h - htref) - c) * w
                else:
                    res = (obs.h - h - c) * w
                rv = np.append(rv, res)
            elif obs.model_key == "steady":
                h = self.steady_model.head(obs.x, obs.y, layers=obs.layer)
                w = obs.weight if obs.weight is not None else 1.0
                rv = np.append(rv, np.atleast_1d((obs.h - h) * w))

        for obs in self.observations_in_well_dict.values():
            if obs.model_key == "transient":
                dt = obs._time_shift if obs.time_shift is not None else 0.0
                t = obs.t - dt
                h = obs.element.headinside(t)[0]
                w = obs.weights if obs.weights is not None else np.ones_like(h)
                c = obs._constant if obs.constant is not None else 0.0
                if self.reference_time is not None:
                    # get closest observation to reference time
                    tref_idx = np.abs(obs.t - self.reference_time).argmin()
                    closest_ref_time = obs.t[tref_idx]
                    htref = obs.element.headinside(closest_ref_time)[0]
                    res = ((obs.h - obs.h[tref_idx]) - (h - htref) - c) * w
                else:
                    res = (obs.h - h - c) * w
                rv = np.append(rv, res)
                # fix for nans, when tmin is larger than timestep after change in bc
                # not ideal but better than crashing the optimizer. Warnings are already
                # printed by the model when this happens.
                nan_mask = np.isnan(rv)
                if nan_mask.any():
                    rv[nan_mask] = np.interp(t[nan_mask], t[~nan_mask], rv[~nan_mask])
            elif obs.model_key == "steady":
                h = obs.element.headinside()
                w = obs.weight if obs.weight is not None else 1.0
                rv = np.append(rv, np.atleast_1d((obs.h - h) * w))
        return rv

    def residuals_lmfit(self, lmfitparams: Any, printdot: bool = False) -> np.ndarray:
        """Compute residuals from an ``lmfit.Parameters`` object.

        Wrapper around :meth:`residuals` that extracts the parameter vector
        from an ``lmfit.Parameters`` object, for use with
        :func:`lmfit.minimize`.

        Parameters
        ----------
        lmfitparams : lmfit.Parameters
            Current parameter values.
        printdot : bool
            Print a dot on each call.

        Returns
        -------
        np.ndarray
            1-D array of weighted residuals.
        """
        p = np.array(list(lmfitparams.valuesdict().values()))
        return self.residuals(p, printdot=printdot)

    def lmfit(
        self,
        printdot: bool = True,
        report: bool = True,
        initial: bool = True,
        **kwargs,
    ) -> None:
        """Run the least-squares fit using ``lmfit``.

        Uses the Levenberg-Marquardt algorithm via :func:`lmfit.minimize`.
        Results are stored in ``self.result`` and optimal values are written
        back to each registered :class:`Parameter`.

        Parameters
        ----------
        printdot : bool
            Print a dot per function evaluation to indicate progress.
        report : bool
            Print a fit summary (message, parameter table, RMSE) on completion.
        initial : bool
            When ``True`` (default) the optimizer starts from the registered
            ``initial`` values.  Set to ``False`` to warm-start from the
            previously fitted optimal values instead.  If no optimal values
            are available yet the method silently falls back to the initial
            values.
        **kwargs
            Forwarded to :func:`lmfit.minimize`.

        Examples
        --------
        Basic usage::

            cal.lmfit()

        Warm-start from a previous fit::

            cal.lmfit()              # phase 1
            cal.lmfit(initial=False) # phase 2: start from phase-1 optimal

        See Also
        --------
        :func:`Calibrate.fit` for fitting with ``scipy.optimize.least_squares``.
        ``lmfit.minimize`` for more optimization options.
        """
        import lmfit

        lmfitparams = lmfit.Parameters()
        for name, p in self._parameters.items():
            sv = self._start_value(p, initial)
            if p.log_scale:
                lb, ub = self._log_scale_bounds(p.pmin, p.pmax, np.sign(sv))
                lmfitparams.add(
                    name,
                    value=np.log10(np.abs(sv)),
                    min=lb if np.isfinite(lb) else None,
                    max=ub if np.isfinite(ub) else None,
                )
            else:
                lmfitparams.add(name, value=sv, min=p.pmin, max=p.pmax)
        fit_kws = {"epsfcn": 1e-4}  # this is essential to specify step for the Jacobian
        self.result = lmfit.minimize(
            self.residuals_lmfit,
            lmfitparams,
            method="leastsq",
            kws={"printdot": printdot},
            **fit_kws,
            **kwargs,
        )
        self.result.method = "lm"
        print("", flush=True)

        if self.result.success:
            for (_, popt), param in zip(
                self.result.params.items(), self._parameters.values(), strict=True
            ):
                if param.log_scale:
                    sv = self._start_value(param, initial)
                    param.optimal = np.sign(sv) * 10**popt.value
                else:
                    param.optimal = popt.value

        if report:
            res = self.residuals_lmfit(self.result.params)
            print(self.result.message)
            print(self.parameters)
            print(f"RMSE: {np.sqrt(np.mean(res**2)):.3e}")

    @staticmethod
    def _start_value(p: "Parameter", use_initial: bool) -> float:
        """Return the linear-space starting value for parameter ``p``.

        When *use_initial* is ``True`` (the default) the registered
        ``initial`` value (or the current model-array value when
        ``initial=None``) is used.  When ``False`` the previously fitted
        optimal is used instead; if no optimal is available yet the method
        silently falls back to the initial value.
        """
        if not use_initial and not np.isnan(p.optimal):
            return p.optimal
        return p.effective_initial

    def fit(
        self,
        method: str = "trf",
        diff_step: float = 1e-4,
        xtol: float = 1e-8,
        report: bool = True,
        printdot: bool = True,
        initial: bool = True,
        **kwargs,
    ) -> None:
        """Run the least-squares fit using :func:`scipy.optimize.least_squares`.

        Parameters
        ----------
        method : {'lm', 'trf', 'dogbox'}
            Optimization algorithm. ``'lm'`` (Levenberg-Marquardt) ignores
            bounds; use ``'trf'`` or ``'dogbox'`` when ``pmin``/``pmax``
            constraints are set. Default is ``'trf'``.
        diff_step : float
            Relative step size used to compute the numerical Jacobian.
        xtol : float
            Convergence tolerance on the change in parameters.
        report : bool
            Print a fit summary (parameter table, RMSE) on completion.
        printdot : bool
            Print a dot per function evaluation to indicate progress.
        initial : bool
            When ``True`` (default) the optimizer starts from the registered
            ``initial`` values (or the current model-array values when
            ``initial=None`` was passed to :meth:`set_aquifer_parameter`).
            Set to ``False`` to warm-start from the previously fitted optimal
            values instead — useful for sequential calibration workflows (e.g.
            calibrate on steady observations first, then continue with
            transient observations). If no optimal values are available yet
            the method silently falls back to the initial values.
        **kwargs
            Forwarded to :func:`scipy.optimize.least_squares`.

        Examples
        --------
        Fit with method="trf" which supports bounded parameters::

            cal.fit()

        Levenberg-Marquardt with looser tolerance (does not support bounds)::

            cal.fit(method="lm", xtol=1e-6)

        Warm-start from a previous fit (e.g. after calibrating steady-state
        observations first)::

            cal.fit()                    # phase 1: steady calibration
            cal.add_head_time_series(...)  # add transient observations
            cal.fit(initial=False)       # phase 2: start from steady optimal

        See Also
        --------
        :func:` Calibrate.lmfit` for fitting with ``lmfit``.
        :func:`scipy.optimize.least_squares` for more optimization options.
        """
        p0 = np.array(
            [
                self._start_value(p, initial)
                if not p.log_scale
                else np.log10(np.abs(self._start_value(p, initial)))
                for p in self._parameters.values()
            ]
        )
        lb = np.array(
            [
                self._log_scale_bounds(
                    p.pmin, p.pmax, np.sign(self._start_value(p, initial))
                )[0]
                if p.log_scale
                else p.pmin
                for p in self._parameters.values()
            ]
        )
        ub = np.array(
            [
                self._log_scale_bounds(
                    p.pmin, p.pmax, np.sign(self._start_value(p, initial))
                )[1]
                if p.log_scale
                else p.pmax
                for p in self._parameters.values()
            ]
        )
        bounds = (lb, ub)

        self.result = least_squares(
            self.residuals,
            p0,
            args=(printdot,),
            bounds=bounds,
            method=method,
            diff_step=diff_step,
            xtol=xtol,
            x_scale="jac",
            **kwargs,
        )
        self.result.method = method
        print("", flush=True)

        # Push optimal values into models and record results
        res = self.residuals(self.result.x)
        for value, param in zip(self.result.x, self._parameters.values(), strict=True):
            if param.log_scale:
                param.optimal = np.sign(self._start_value(param, initial)) * 10**value
            else:
                param.optimal = value

        if report:
            print(self.parameters)
            print(f"RMSE: {np.sqrt(np.mean(res**2)):.3e}")

    def rmse(self) -> float:
        """Return the root-mean-square error at the current optimal parameters.

        Returns
        -------
        float
            RMSE of the weighted residuals.
        """
        result = getattr(self, "result", None)
        if result is not None and getattr(result, "x", None) is not None:
            params_vec = result.x
        else:
            # Fall back to reconstructing optimization-space parameters
            values = []
            for p in self._parameters.values():
                if getattr(p, "log_scale", False):
                    values.append(
                        np.log10(
                            np.abs(p.effective_initial)
                            if np.isnan(p.optimal)
                            else np.abs(p.optimal)
                        )
                    )
                else:
                    values.append(
                        p.effective_initial if np.isnan(p.optimal) else p.optimal
                    )
            params_vec = np.array(values, dtype=float)

        r = self.residuals(params_vec)
        return float(np.sqrt(np.mean(r**2)))

    @staticmethod
    def get_covariances(
        jacobian: np.ndarray,
        cost: float,
        method: Literal["trf", "dogbox", "lm"] = "trf",
        absolute_sigma: bool = False,
    ) -> np.ndarray:
        """
        Method to get the covariance matrix from the jacobian.

        Parameters
        ----------
        jacobian : ArrayLike
            The jacobian matrix with dimensions nobs, npar.
        cost : float
            The cost value of the scipy.optimize.OptimizeResult which is half
            the sum of squares. That's why the cost is multiplied by a factor
            of two internally to get the sum of squares.
        method : Literal["trf", "dogbox", "lm"], optional
            Algorithm with which the minimization is performed. Default is "trf".
        absolute_sigma : bool, optional
            If True, `sigma` is used in an absolute sense and the estimated
            parameter covariance `pcov` reflects these absolute values. If
            False (default), only the relative magnitudes of the `sigma` values
            matter. The returned parameter covariance matrix `pcov` is based on
            scaling `sigma` by a constant factor. This constant is set by
            demanding that the reduced `chisq` for the optimal parameters
            `popt` when using the *scaled* `sigma` equals unity. In other
            words, `sigma` is scaled to match the sample variance of the
            residuals after the fit. Default is False.
            Mathematically, ``pcov(absolute_sigma=False) =
            pcov(absolute_sigma=True) * chisq(popt)/(M-N)``

        Returns
        -------
        pcov: array_like
            numpy array with the covariance matrix.

        Notes
        -----
        This method is copied from Scipy:
        https://github.com/scipy/scipy/blob/92d2a8592782ee19a1161d0bf3fc2241ba78bb63/scipy/optimize/_minpack_py.py
        Please refer to the SciPy optimization module::
        https://docs.scipy.org/doc/scipy/reference/optimize.html
        """
        nobs, npar = jacobian.shape
        cost = 2 * cost  # res.cost is half sum of squares!
        s_sq = cost / (nobs - npar)  # variance of the residuals

        if method == "lm":
            # https://github.com/scipy/scipy/blob/92d2a8592782ee19a1161d0bf3fc2241ba78bb63/scipy/optimize/_minpack_py.py#L480C9-L499C38
            # fjac A permutation of the R matrix of a QR factorization of the
            # final approximate Jacobian matrix.
            _, fjac = np.linalg.qr(jacobian)
            # leastsq expects the fjacobian to be in fortran order (npar, nobs)
            # that why it is transposed in the original code

            ipvt = np.arange(1, npar + 1, dtype=int)
            n = len(ipvt)
            r = np.triu(fjac[:n, :])

            # old method deprecated in scipy 1.10.0 since
            # the explicit dot product was not necessary and sometimes
            # the result was not symmetric positive definite.
            # See https://github.com/scipy/scipy/issues/4555.
            # old method
            # perm = np.take(np.eye(n), ipvt - 1, 0)
            # R = np.dot(r, perm)
            # cov_x = np.linalg.inv(np.dot(np.transpose(R), R))

            # new method:
            perm = ipvt - 1
            inv_triu = get_lapack_funcs("trtri", (r,))
            try:
                # inverse of permuted matrix is a permutation of matrix inverse
                invR, trtri_info = inv_triu(r)  # default: upper, non-unit diag
                if trtri_info != 0:  # explicit comparison for readability
                    raise LinAlgError
                invR[perm] = invR.copy()
                pcov = invR @ invR.T  # cov_x in the original code
            except (LinAlgError, ValueError):
                pcov = None
        else:
            # https://github.com/scipy/scipy/blob/92d2a8592782ee19a1161d0bf3fc2241ba78bb63/scipy/optimize/_minpack_py.py#L1029-L1055
            # Do Moore-Penrose inverse discarding zero singular values.
            _, s, VT = svd(jacobian, full_matrices=False)
            threshold = np.finfo(float).eps * max(jacobian.shape) * s[0]
            s = s[s > threshold]
            VT = VT[: s.size]
            pcov = np.dot(VT.T / s**2, VT)

        if pcov is None or np.isnan(pcov).any():
            # indeterminate covariance
            pcov = np.full(shape=(npar, npar), fill_value=np.inf, dtype=float)
        elif not absolute_sigma:
            if nobs > npar:
                pcov = pcov * s_sq
            else:
                pcov = np.full(shape=(npar, npar), fill_value=np.inf, dtype=float)

        return pcov

    def get_correlations(self) -> np.ndarray:
        """Return the parameter correlation matrix.

        Uses the covariance matrix from the fit result. For lmfit results the
        covariance is read directly from ``self.result.covar``; for scipy
        results it is computed from the Jacobian via :meth:`get_covariances`.

        Returns
        -------
        np.ndarray
            Correlation matrix of shape ``(n_params, n_params)``.
        """
        if hasattr(self.result, "covar") and self.result.covar is not None:
            pcov = self.result.covar
        else:
            pcov = self.get_covariances(
                jacobian=self.result.jac,
                cost=self.result.cost,
                method=self.result.method,
                absolute_sigma=False,
            )
        v = np.sqrt(np.diag(pcov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = pcov / np.outer(v, v)
        corr[pcov == 0] = 0
        return corr

    @property
    def parameters(self) -> pd.DataFrame:
        """Summary of all registered calibration parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by parameter name with columns: ``initial``,
            ``optimal``, ``pmin``, ``pmax``, ``log_scaled``, ``n_targets``,
            ``n_models``, ``n_inhoms``.
        """
        rows = []
        for p in self._parameters.values():
            rows.append(
                {
                    "name": p.name,
                    "initial": p.initial,
                    "optimal": p.optimal,
                    "pmin": p.pmin,
                    "pmax": p.pmax,
                    "log_scaled": p.log_scale,
                    "n_targets": len(p.targets),
                    "n_models": len(set(p.models)),
                    "n_inhoms": len(set(p.inhoms)),
                }
            )
        return pd.DataFrame(rows).set_index("name")

    def _parse_layers(self, name: str, layers: int | Iterable[int]) -> tuple[int, int]:
        """Parse layer specification and return (from_layer, to_layer)."""
        if isinstance(layers, Iterable) and not isinstance(layers, str):
            layers = list(layers)
            from_lay, to_lay = min(layers), max(layers)
            if (np.diff(layers) > 1).any():
                warnings.warn(
                    f"Non-consecutive layers; setting '{name}' for "
                    f"layers {from_lay}–{to_lay}.",
                    stacklevel=3,
                )
        elif isinstance(layers, int):
            from_lay = to_lay = layers
        else:
            raise TypeError(f"layers must be int or iterable of int, got {type(layers)}")
        return from_lay, to_lay

    def _resolve_models(self, model_key: str) -> list:
        """Return list of model objects corresponding to model_key."""
        mapping = {
            "transient": [self.transient_model],
            "steady": [self.steady_model],
            "both": [self.transient_model, self.steady_model],
        }
        return [ml for ml in mapping[model_key] if ml is not None]

    def _resolve_inhoms(self, model: Any, inhoms: str | list[str] | None) -> list:
        """Return list of aquifer objects for given inhoms spec."""
        if inhoms is None:
            return [model.aq]
        if isinstance(inhoms, str):
            inhoms = [inhoms]
        elif not isinstance(inhoms, list):
            inhoms = list(inhoms)
        return [model.aq.inhomdict[i] if isinstance(i, str) else i for i in inhoms]

    @staticmethod
    def _log_scale_bounds(pmin: float, pmax: float, sign: float) -> tuple[float, float]:
        """Convert linear-space bounds to log10(abs) optimizer space.

        For a positive parameter (sign >= 0):
            pmin > 0  →  lb = log10(pmin),  ub = log10(pmax)
        For a negative parameter (sign < 0):
            pmin ≤ pmax ≤ 0, so abs(pmax) ≤ abs(value) ≤ abs(pmin)
            lb = log10(abs(pmax)),  ub = log10(abs(pmin))
        Infinite or incompatible bounds are passed through as ±inf.
        """
        if sign >= 0:
            lb = np.log10(pmin) if (np.isfinite(pmin) and pmin > 0) else -np.inf
            ub = np.log10(pmax) if (np.isfinite(pmax) and pmax > 0) else np.inf
        else:
            lb = np.log10(-pmax) if (np.isfinite(pmax) and pmax < 0) else -np.inf
            ub = np.log10(-pmin) if (np.isfinite(pmin) and pmin < 0) else np.inf
        return lb, ub

    @staticmethod
    def _get_aquifer_parameter_array(model, aq, name: str) -> np.ndarray:
        """Return reference to the appropriate parameter array in the aquifer object."""
        lookup = {"kaq": aq.kaq, "c": aq.c}
        if hasattr(aq, "Saq"):
            lookup["Saq"] = aq.Saq
        if hasattr(aq, "Sll"):
            lookup["Sll"] = aq.Sll
        if hasattr(aq, "kzoverkh"):
            lookup["kzoverkh"] = aq.kzoverkh
        assert aq.model == model, "Aquifer does not belong to the given model"
        for prefix, arr in lookup.items():
            if name.startswith(prefix):
                return arr
        raise ValueError(
            f"Parameter '{name}' not recognised. Supported: {list(lookup.keys())}"
        )

    @staticmethod
    def _make_pname(
        name: str,
        from_lay: int,
        to_lay: int,
        inhoms: str | list[str] | None,
        models: list,
    ) -> str:
        """Construct a unique parameter name based on the specification."""
        base = f"{name}_{from_lay}_{to_lay}"
        if inhoms is not None:
            inhom_names = (
                [inhoms]
                if isinstance(inhoms, str)
                else [i if isinstance(i, str) else i.name for i in inhoms]
            )
            base += "_" + "_".join(inhom_names)
        return base

    @staticmethod
    def _nse(h_obs: np.ndarray, h_mod: np.ndarray) -> float:
        """Compute the Nash-Sutcliffe Efficiency (NSE).

        NSE = 1 - Σ(h_obs - h_mod)² / Σ(h_obs - mean(h_obs))²

        Parameters
        ----------
        h_obs : np.ndarray
            Observed heads.
        h_mod : np.ndarray
            Modeled heads.

        Returns
        -------
        nse : float
            NSE value, where 1 is a perfect fit, 0 means the model is as good as using
            the mean of the observations, and negative values indicate a worse fit.
            Returns NaN when all observations are identical (zero variance).
        """
        denom = float(np.sum((h_obs - np.mean(h_obs)) ** 2))
        if denom == 0.0:
            return np.nan
        return float(1.0 - np.sum((h_obs - h_mod) ** 2) / denom)

    def plot_transient_results(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        figsize: tuple[float, float] = (10, 8),
        obs_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        sharey: bool = False,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot modeled vs observed transient head time series.

        Creates one subplot per transient observation with a shared x-axis,
        comparing observed heads to the current model response. Call this
        method before calibration to inspect the initial fit, or after
        calibration to inspect the calibrated fit.

        Parameters
        ----------
        tmin : float, optional
            Start of the plotted time window. Defaults to the earliest
            observation time across all series.
        tmax : float, optional
            End of the plotted time window. Defaults to the latest
            observation time across all series.
        figsize : tuple of float, optional
            Figure size ``(width, height)`` in inches. Default is ``(10, 8)``.
        obs_kwargs : dict, optional
            Keyword arguments passed to :func:`matplotlib.axes.Axes.plot` for
            the observed data. Default: black dots.
        model_kwargs : dict, optional
            Keyword arguments passed to :func:`matplotlib.axes.Axes.plot` for
            the modeled response. Default: blue solid line.
        sharey : bool, optional
            If ``True``, all subplots share the same y-axis limits.
            Default is ``False``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray of matplotlib.axes.Axes
            One Axes per transient observation.
        """
        # Collect transient observations in insertion order
        transient_items = [
            (name, obs)
            for name, obs in self.observations_dict.items()
            if obs.model_key == "transient"
        ]
        transient_well_items = [
            (name, obs)
            for name, obs in self.observations_in_well_dict.items()
            if obs.model_key == "transient"
        ]
        all_items = transient_items + transient_well_items
        n_obs = len(all_items)

        if n_obs == 0:
            raise ValueError("No transient observations to plot.")

        # Default styling
        obs_kw: dict = {"color": "k", "marker": ".", "linestyle": "none"}
        obs_kw.update(obs_kwargs or {})

        model_kw: dict = {"color": "tab:blue", "label": "model"}
        model_kw.update(model_kwargs or {})

        _tr_has_steady = getattr(self.transient_model, "steady", None) is not None

        # Solve the current model state, just in case, but should already be done usually.
        if self.steady_model is not None:
            self.steady_model.solve(silent=True)
        if self.transient_model is not None:
            self.transient_model.solve(silent=True)

        # Create subplots
        fig, ax_array = plt.subplots(
            n_obs, 1, sharex=True, sharey=sharey, figsize=figsize
        )
        axes = np.atleast_1d(ax_array)

        i = 0
        for ax, (name, obs) in zip(axes, all_items, strict=True):
            dt = obs._time_shift if obs.time_shift is not None else 0.0
            t_plot = obs.t - dt  # shared time axis for observed and modeled

            # Build observed label, annotating any corrections
            obs_label_parts = []
            if obs.time_shift is not None:
                obs_label_parts.append(f"\u0394t={float(obs._time_shift[0]):.2f}")
            if obs.constant is not None:
                obs_label_parts.append(f"\u0394h={float(obs._constant[0]):.2f}")
            obs_suffix = f" ({', '.join(obs_label_parts)})" if obs_label_parts else ""
            obs_label = f"{name}{obs_suffix}"

            # Compute observed heads with corrections applied
            if self.reference_time is not None:
                tref_idx = np.abs(obs.t - self.reference_time).argmin()
                h_obs_plot = obs.h - obs.h[tref_idx]
            else:
                c = float(obs._constant[0]) if obs.constant is not None else 0.0
                h_obs_plot = obs.h - c

            # Compute modeled heads
            if name in dict(transient_items):
                if obs.normalized or _tr_has_steady:
                    hsteady = 0.0
                elif self.steady_model is not None:
                    hsteady = float(
                        np.squeeze(self.steady_model.head(obs.x, obs.y, layers=obs.layer))
                    )
                else:
                    hsteady = 0.0
                h_mod = (
                    self.transient_model.head(obs.x, obs.y, obs.t - dt, layers=obs.layer)
                    + hsteady
                ).squeeze()
            else:
                h_mod = obs.element.headinside(obs.t - dt)[0]

            if self.reference_time is not None:
                h_mod = h_mod - h_mod[tref_idx]

            # Apply tmin/tmax window
            mask = np.ones(len(t_plot), dtype=bool)
            if tmin is not None:
                mask &= t_plot >= tmin
            if tmax is not None:
                mask &= t_plot <= tmax

            nse = self._nse(h_obs_plot[mask], h_mod[mask])
            nse_str = f"NSE={nse:.2f}" if np.isfinite(nse) else "NSE=n/a"
            model_label = f"{model_kw.get('label', 'model')} ({nse_str})"

            model_kw["color"] = f"C{i}"  # cycle through colors for each subplot
            ax.plot(t_plot[mask], h_obs_plot[mask], label=obs_label, **obs_kw)
            ax.plot(t_plot[mask], h_mod[mask], **{**model_kw, "label": model_label})
            ax.set_ylabel("head")
            ax.grid(True)
            ax.legend(loc=(0, 1), frameon=False, ncol=2)
            ax.set_xlim(left=t_plot[mask][0], right=t_plot[mask][-1])
            i += 1
        axes[-1].set_xlabel("time")
        fig.tight_layout()
        return fig, axes
