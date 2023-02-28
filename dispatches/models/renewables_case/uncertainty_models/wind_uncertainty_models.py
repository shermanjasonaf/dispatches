#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-21 Thu 16:59:08

@author: jasherma

Uncertainty models for wind-battery model.
"""

import abc
import numpy as np
# import matplotlib.pyplot as plt

from pyomo.contrib.pyros import uncertainty_sets

from dispatches.models.renewables_case.uncertainty_models.uncertainty_models \
        import UncertaintySet


class WindDiscreteSet(UncertaintySet):
    """
    Set of discrete wind production scenarios.
    """
    def __init__(self, scenarios, nom_idx, wind_capacity):
        """Construct discrete set of LMP scenarios."""
        self.scenarios = np.array(scenarios)

        if not isinstance(nom_idx, int):
            raise TypeError("Argument `nom_idx` must be of type `int`")
        elif nom_idx < 0 or nom_idx > self.scenarios.shape[0] - 1:
            raise ValueError(
                "Argument `nom_idx` must be between 0 and "
                f"{self.scenarios.shape[0] - 1}, (provided value {nom_idx})"
            )
        self._nom_idx = nom_idx
        self._wind_capacity = wind_capacity

    @property
    def sig_nom(self, capacity_factors=False):
        """Nominal LMP signal."""
        if capacity_factors:
            return self.scenarios[self._nom_idx] / self._wind_capacity
        else:
            return self.scenarios[self._nom_idx]

    def pyros_set(self, include_fixed_dims=True, capacity_factors=True):
        """Construct PyROS Uncertainty set."""
        if capacity_factors:
            scenarios = self.scenarios / self._wind_capacity
        else:
            scenarios = self.scenarios
        return uncertainty_sets.DiscreteScenarioSet(scenarios)

    def determine_fixed_dims(self, tol=0):
        """
        Determine dimensions of the uncertainty set
        for which the values are the same across all scenarios.
        """
        return [
            idx for idx, scenario_col in enumerate(self.scenarios.T)
            if np.allclose(scenario_col, scenario_col[0], atol=tol)
        ]

    def plot_set(self, ax, offset=0):
        """Plot discrete LMP scenarios for given axis."""

        periods = np.array(range(self.scenarios.shape[1])) + offset
        seen_off_nominal = False

        # plot the scenarios, distinguishing nominal from other
        # scenarios
        for idx, scenario in enumerate(self.scenarios):
            if idx == self._nom_idx:
                linestyle = "-"
                color = "black"
                label = "nominal"
            elif not seen_off_nominal:
                seen_off_nominal = True
                linestyle = "--"
                color = "green"
                label = "other scenarios"
            else:
                linestyle = "--"
                color = "green"
                label = None

            ax.step(periods, scenario, linestyle, color=color, linewidth=1.5,
                    where="post", label=label)


class WindBoxSet(UncertaintySet):
    def __init__(
            self,
            wind_data,
            include_peak_effects=False,
            start_day_hour=0
            ):
        self.wind_sig_nom = wind_data
        self.n_time_points = len(self.wind_sig_nom)

        assert self.bounds_valid()

    @property
    def sig_nom(self):
        return self.wind_sig_nom

    @abc.abstractmethod
    def bounds(self):
        ...

    def bounds_valid(self):
        bounds = self.bounds()
        for time in range(len(bounds)):
            lb, ub = bounds[time]
            if lb > self.sig_nom[time] or ub < self.sig_nom[time]:
                return False
        return True

    def write_bounds(self, filename):
        """Write box set bounds to file."""
        bounds = self.bounds()

        # resolve bounds into lower and upper
        lower_bds = np.array([bounds[idx][0] for idx in range(len(bounds))])
        upper_bds = np.array([bounds[idx][1] for idx in range(len(bounds))])

        # write all to file
        periods = np.arange(self.n_time_points)
        uncertainty_data = np.array([periods, lower_bds, self.sig_nom,
                                     upper_bds]).T
        np.savetxt('wind_uncertainty_data.dat',
                   uncertainty_data, delimiter='\t',
                   header='time\tlower\tnominal\tupper',
                   comments='', fmt=['%d', '%.3f', '%.3f', '%.3f'])

    def determine_fixed_dims(self, tol=0):
        """
        Determine dimensions of the uncertainty set
        for which the bounds are the same.
        """
        return [
            idx for idx, bd in enumerate(self.bounds())
            if bd[1] <= bd[0] + tol
        ]

    def pyros_set(self, include_fixed_dims=True):
        """
        Obtain corresponding PyROS BoxSet. Optionally,
        include only the dimensions for which the bounds are
        unequal.
        """
        if include_fixed_dims:
            final_bounds = self.bounds()
        else:
            bounds = self.bounds()
            fixed_dims = self.determine_fixed_dims()
            final_bounds = [
                bd for idx, bd in enumerate(bounds)
                if idx not in fixed_dims
            ]
        return uncertainty_sets.BoxSet(bounds=final_bounds)

    def plot_bounds(self, ax, highlight_peak_effects=False, offset=0):
        """Plot wind bounds against planning period."""
        bounds = self.bounds()

        # resolve bounds into lower and upper
        periods = np.array(list(range(len(bounds)))) + offset
        lower_bds = np.array([bounds[idx][0] for idx in range(len(bounds))])
        upper_bds = np.array([bounds[idx][1] for idx in range(len(bounds))])

        # generate the plots
        ax.step(periods, upper_bds, "--", color="purple", linewidth=1.0,
                where="post")
        ax.step(periods, lower_bds, "--", color="purple", label="bounds",
                linewidth=1.0, where="post")
        ax.fill_between(periods, lower_bds, upper_bds, step="post",
                        color="purple", alpha=0.1, label="uncertainty"
                        )

        # plot nominal realization
        ax.step(periods, self.sig_nom, color="black", label='nominal',
                linewidth=1.8, where="post")

    def plot_set(self, ax, offset=0):
        """Plot uncertainty set."""
        return self.plot_bounds(
            ax,
            offset=offset,
        )


class CustomBoundsWindBoxSet(WindBoxSet):
    """
    LMP box set with custom bounds.
    """
    def __init__(self, wind_data, bounds):
        assert len(wind_data) == len(bounds)

        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (len(wind_data), 2)

        self.wind_sig_nom = wind_data
        self.bounds_arr = bounds_arr.tolist()

    def bounds(self):
        return [tuple(bd) for bd in self.bounds_arr]


class ConstantUncertaintyNonnegBoxSet(WindBoxSet):
    def __init__(
            self,
            wind_data,
            uncertainty,
            first_period_certain=False,
            min_val=0,
            max_val=None,
            ):
        """Initialize self.

        """
        self.wind_sig_nom = wind_data
        self.n_time_points = len(wind_data)
        self.uncertainty = uncertainty
        self.first_period_certain = first_period_certain

        # verify nominal signal is in set
        if min_val is not None:
            assert np.all(self.sig_nom >= min_val)
        if max_val is not None:
            assert np.all(self.sig_nom <= max_val)

        self.min_val = min_val
        self.max_val = max_val

        # verify uncertainty is valid
        assert self.uncertainty >= 0

    @property
    def sig_nom(self):
        return self.wind_sig_nom

    def bounds(self):
        bounds = []
        for time in range(self.n_time_points):
            lower_bound = max(
                self.min_val,
                self.sig_nom[time] - self.uncertainty,
            )
            upper_bound = max(
                self.min_val,
                min(self.sig_nom[time] + self.uncertainty, self.max_val),
            )
            bounds.append((lower_bound, upper_bound))

        if self.first_period_certain:
            bounds[0] = (self.sig_nom[0], self.sig_nom[0])

        return bounds
