"""Calibration and parameter estimation tools.

Provides `Calibrate` class for fitting transient model parameters to
observed data.

Example::

    cal = Calibrate(ml)
    cal.series(name, x, y, layer, t, h)
    cal.set_parameter(name='kaq', layers=0, initial=10, pmin=1, pmax=100)
    cal.fit()
"""

import re
import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, leastsq


class Calibrate:
    def __init__(self, model):
        """Initialize Calibration class.

        Parameters
        ----------
        model : timflow.transient.Model
            model to calibrate
        """
        self.model = model
        self.parameters = pd.DataFrame(
            columns=[
                "layers",
                "optimal",
                "std",
                "perc_std",
                "pmin",
                "pmax",
                "initial",
                "inhoms",
                "parray",
            ]
        )
        self.seriesdict = {}
        self.seriesinwelldict = {}

    def set_parameter(
        self, name=None, layers=None, initial=0, pmin=-np.inf, pmax=np.inf, inhoms=None
    ):
        """Set parameter to be optimized.

        Parameters
        ----------
        name : str
            name can be 'kaq', 'Saq', 'c', 'Sll' or 'kzoverkh'.
        layers : int or list of ints
            layer number(s) for which the parameter is set. If an integer is passed,
            parameter is associated with a single layer. If a list of layers is passed,
            layers must be consecutive and parameter is set for each layer from
            min(layers) up to and including max(layers).
        initial : float, optional
            initial value for the parameter (the default is 0)
        pmin : float, optional
            lower bound for parameter value (the default is -np.inf)
        pmax : float, optional
            upper bound for paramater value (the default is np.inf)
        inhoms : str, list
            string with name of inhomogeneity, list with string names of inhomogeneities
            or list of inhomogeneities
            inhomogeneity(ies) for which the parameter is set. If a string is passed,
            parameter is associated with a single inhomogeneity. If a list of strings
            of inhomogeneities (or list of inhomogeneities) is passed, parameter is set
            for each inhomogeneity in the list. This allows linking of parameters across
            inhomogeneities.
        """
        assert isinstance(name, str), "Error: name must be string"

        if isinstance(layers, Iterable):
            from_lay = min(layers)
            to_lay = max(layers)
            if (np.diff(layers) > 1).any():
                warnings.warn(
                    "Non-consecutive layers are not supported. "
                    f"Setting parameter '{name}' for layers {from_lay} - {to_lay}.",
                    stacklevel=1,
                )
        elif isinstance(layers, int):
            from_lay = layers
            to_lay = layers
        else:
            warnings.warn(
                "Setting layers in the parameter name is deprecated. "
                f"Set the layers= keyword argument for parameter '{name}' to silence "
                "this warning. The parameter name can still include layer info, but "
                "this will be ignored in a future version of timflow.",
                DeprecationWarning,
                stacklevel=2,
            )
            # find numbers in name str for support layer ranges
            layers_from_name = re.findall(r"\d+", name)
            if len(layers_from_name) == 0:
                raise ValueError(
                    "No layer information found in parameter name. "
                    "Please specify layers explicitly."
                )
            elif len(layers_from_name) == 1:
                from_lay = int(layers_from_name[0])
                to_lay = from_lay
            elif len(layers_from_name) == 2:
                from_lay, to_lay = map(int, layers_from_name)

        # get aquifer information and create list if necessary
        if inhoms is None:
            aq = [self.model.aq]
        elif isinstance(inhoms, tuple):
            aq = list(inhoms)
        elif not isinstance(inhoms, list):
            aq = [inhoms]
        else:
            aq = inhoms

        # convert aquifer names to aquifer objects
        for i, iaq in enumerate(aq):
            if isinstance(iaq, str):
                aq[i] = self.model.aq.inhomdict[iaq]

        plist = []
        for iaq in aq:
            if name[:3] == "kaq":
                p = iaq.kaq[from_lay : to_lay + 1]
            elif name[:3] == "Saq":
                p = iaq.Saq[from_lay : to_lay + 1]
            elif name[0] == "c":
                p = iaq.c[from_lay : to_lay + 1]
            elif name[:3] == "Sll":
                p = iaq.Sll[from_lay : to_lay + 1]
            elif name[0:8] == "kzoverkh":
                p = iaq.kzoverkh[from_lay : to_lay + 1]
            else:
                raise ValueError(
                    f"Parameter name '{name}' not recognized. "
                    "Supported parameters are 'kaq', 'Saq', 'c', 'Sll' or 'kzoverkh'."
                )
            plist.append(p[:])

        if p is None:  # no parameter set
            print("parameter name not recognized or no parameter ref supplied")
            return

        if inhoms is None:
            pname = f"{name}_{from_lay}_{to_lay}"
        else:
            pname = f"{name}_{from_lay}_{to_lay}_{'_'.join([iaq.name for iaq in aq])}"
        self.parameters.loc[pname] = {
            "layers": layers,
            "optimal": float(initial),
            "std": None,
            "perc_std": None,
            "pmin": float(pmin),
            "pmax": float(pmax),
            "initial": float(initial),
            "inhoms": aq if inhoms is not None else None,
            "parray": plist,
        }

    def set_parameter_by_reference(
        self, name=None, parameter=None, initial=0, pmin=-np.inf, pmax=np.inf
    ):
        """Set parameter to be optimized.

        Parameters
        ----------
        name : str
            parameter name
        parameter : np.array
            array reference containing the parameter to be optimized. must be
            specified as reference, i.e. w.rc[0:]
        initial : float, optional
            initial value for the parameter (the default is 0)
        pmin : float, optional
            lower bound for parameter value (the default is -np.inf)
        pmax : float, optional
            upper bound for paramater value (the default is np.inf)
        """
        assert isinstance(name, str), "Error: name must be string"
        if parameter is not None:
            assert isinstance(parameter, np.ndarray), (
                "Error: parameter needs to be numpy array"
            )
            p = parameter
        self.parameters.loc[name] = {
            "optimal": float(initial),
            "std": None,
            "perc_std": None,
            "pmin": float(pmin),
            "pmax": float(pmax),
            "initial": float(initial),
            "parray": [p[:]],
        }

    def series(self, name, x, y, layer, t, h, weights=None):
        """Method to add observations to Calibration object.

        Parameters
        ----------
        name : str
            name of series
        x : float
            x-coordinate
        y : float
            y-coordinate
        layer : int
            layer number, 0-indexed
        t : np.array
            array containing timestamps of timeseries
        h : np.array
            array containing timeseries values, i.e. head observations
        """
        s = Series(x, y, layer, t, h, weights=weights)
        self.seriesdict[name] = s

    def seriesinwell(self, name, element, t, h):
        """Method to add observations to Calibration object.

        Parameters
        ----------
        name : str
            name of series
        element: element object with headinside function
        t : np.array
            array containing timestamps of timeseries
        h : np.array
            array containing timeseries values, i.e. head observations
        """
        e = SeriesInWell(element, t, h)
        self.seriesinwelldict[name] = e

    def residuals(self, p, printdot=False, weighted=True, layers=None, series=None):
        """Method to calculate residuals given certain parameters.

        Parameters
        ----------
        p : np.array
            array containing parameter values
        printdot : bool, optional
            print dot for each function call

        Returns
        -------
        np.array
            array containing all residuals
        """
        if printdot:
            print(".", end="")
        # set the values of the variables

        if printdot == 7:
            print(p)

        if layers is None:
            layers = range(self.model.aq.naq)

        for i, k in enumerate(self.parameters.index):
            parraylist = self.parameters.loc[k, "parray"]
            for parray in parraylist:
                # [:] needed to do set value in array
                parray[:] = p[i]
        self.model.solve(silent=True)

        rv = np.empty(0)
        cal_series = self.seriesdict.keys() if series is None else series
        for key in cal_series:
            s = self.seriesdict[key]
            if s.layer not in layers:
                continue
            h = self.model.head(s.x, s.y, s.t, layers=s.layer)
            w = s.weights if ((s.weights is not None) and weighted) else np.ones_like(h)
            rv = np.append(rv, (s.h - h) * w)
        for key in self.seriesinwelldict:
            s = self.seriesinwelldict[key]
            h = s.element.headinside(s.t)[0]
            rv = np.append(rv, s.h - h)
        return rv

    def residuals_lmfit(self, lmfitparams, printdot=False):
        vals = lmfitparams.valuesdict()
        p = np.array([vals[k] for k in self.parameters.index])
        return self.residuals(p, printdot)

    def fit_least_squares(self, report=True, diff_step=1e-4, xtol=1e-8, method="lm"):
        self.fitresult = least_squares(
            self.residuals,
            self.parameters.initial.values,
            args=(True,),
            bounds=(self.parameters.pmin.values, self.parameters.pmax.values),
            method=method,
            diff_step=diff_step,
            xtol=xtol,
            x_scale="jac",
        )
        print("", flush=True)
        # Call residuals to specify optimal values for model
        res = self.residuals(self.fitresult.x)
        for ipar in self.parameters.index:
            self.parameters.loc[ipar, "optimal"] = self.parameters.loc[ipar, "parray"][0]
        nparam = len(self.fitresult.x)
        H = self.fitresult.jac.T @ self.fitresult.jac
        sigsq = np.var(res, ddof=nparam)
        self.covmat = np.linalg.inv(H) * sigsq
        self.sig = np.sqrt(np.diag(self.covmat))
        D = np.diag(1 / self.sig)
        self.cormat = D @ self.covmat @ D
        self.parameters["std"] = self.sig
        self.parameters["perc_std"] = self.sig / self.parameters["optimal"] * 100
        if report:
            print(self.parameters)
            print(self.sig)
            print(self.covmat)
            print(self.cormat)

    def fit_lmfit(self, report=False, printdot=True, **kwargs):
        import lmfit

        self.lmfitparams = lmfit.Parameters()
        for name in self.parameters.index:
            p = self.parameters.loc[name]
            self.lmfitparams.add(name, value=p["initial"], min=p["pmin"], max=p["pmax"])
        fit_kws = {"epsfcn": 1e-4}  # this is essential to specify step for the Jacobian
        self.fitresult = lmfit.minimize(
            self.residuals_lmfit,
            self.lmfitparams,
            method="leastsq",
            kws={"printdot": printdot},
            **fit_kws,
            **kwargs,
        )
        print("", flush=True)
        print(self.fitresult.message)
        if self.fitresult.success:
            for name in self.parameters.index:
                self.parameters.loc[name, "optimal"] = self.fitresult.params.valuesdict()[
                    name
                ]
            if hasattr(self.fitresult, "covar"):
                self.parameters["std"] = np.sqrt(np.diag(self.fitresult.covar))
                self.parameters["perc_std"] = (
                    100 * self.parameters["std"] / np.abs(self.parameters["optimal"])
                )
            else:
                self.parameters["std"] = np.nan
                self.parameters["perc_std"] = np.nan
        if report:
            print(lmfit.fit_report(self.fitresult))

    def residuals_leastsq(self, logparams, printdot=False):
        params = 10**logparams
        print("params ", params)
        return self.residuals(params, printdot)

    def fit_leastsq(self, report=True, diff_step=1e-4, xtol=1e-8):
        params_initial = np.log10(self.parameters.initial.values)
        print("params_initial ", params_initial)
        plog, mes = leastsq(self.residuals_leastsq, params_initial, epsfcn=1e-3)
        print("", flush=True)
        params = 10**plog
        # Call residuals to specify optimal values for model
        self.residuals(params)
        for ipar in self.parameters.index:
            self.parameters.loc[ipar, "optimal"] = self.parameters.loc[ipar, "parray"][0]
        # nparam = len(self.fitresult.x)
        # H = self.fitresult.jac.T @ self.fitresult.jac
        # sigsq = np.var(res, ddof=nparam)
        # self.covmat = np.linalg.inv(H) * sigsq
        # self.sig = np.sqrt(np.diag(self.covmat))
        # D = np.diag(1 / self.sig)
        # self.cormat = D @ self.covmat @ D
        # self.parameters["std"] = self.sig
        # self.parameters["perc_std"] = self.sig / self.parameters["optimal"] * 100
        # if report:
        #     print(self.parameters)
        #     print(self.sig)
        #     print(self.covmat)
        #     print(self.cormat)

    def fit(self, report=False, printdot=True, **kwargs):
        # current default fitting routine is lmfit
        # return self.fit_least_squares(report) # does not support bounds by default
        return self.fit_lmfit(report, printdot, **kwargs)

    def rmse(self, weighted=True, layers=None):
        """Calculate root-mean-squared-error.

        Returns
        -------
        float
            return rmse value
        """
        r = self.residuals(
            self.parameters["optimal"].values, weighted=weighted, layers=layers
        )
        return np.sqrt(np.mean(r**2))

    def topview(self, ax=None, layers=None, labels=True):
        """Plot topview of model with calibration points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            axes to plot on (the default is None, which creates a new figure)
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="datalim")
            # self.model.plots.topview(ax=ax)
        for key, s in self.seriesdict.items():
            if layers is None or s.layer in layers:
                ax.plot(s.x, s.y, "ko")
                if labels:
                    ax.text(s.x, s.y, key, ha="left", va="bottom")
        return ax

    def xsection(self, ax=None, labels=True):
        """Plot cross-section of model with calibration points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            axes to plot on (the default is None, which creates a new figure)
        """
        if ax is None:
            _, ax = plt.subplots()
        for key, s in self.seriesdict.items():
            aq = self.model.aq.find_aquifer_data(s.x, s.y)
            ztop = aq.z[0]
            zpb_top = aq.zaqtop[s.layer]
            zpb_bot = aq.zaqbot[s.layer]
            ax.plot([s.x, s.x], [zpb_top, zpb_bot], c="k", ls="dotted")
            ax.plot([s.x, s.x], [ztop + 1, zpb_top], c="k", ls="solid", lw=1.0)
            if labels:
                ax.text(s.x, ztop + 1, key, ha="left", va="bottom", rotation=45)
        return ax


class Series:
    def __init__(self, x, y, layer, t, h, weights=None):
        self.x = x
        self.y = y
        self.layer = layer
        self.t = t
        self.h = h
        self.weights = weights


class SeriesInWell:
    def __init__(self, element, t, h):
        self.element = element
        self.t = t
        self.h = h
