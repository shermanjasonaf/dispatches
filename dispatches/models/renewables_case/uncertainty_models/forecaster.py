#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-18 Mon 11:34:46

@author: jasherma

This module defines classes used for forecasting LMP signals
and windpower capacity levels and their associated uncertainties.
"""

import abc
import itertools

import numpy as np
import pandas as pd

import pyomo.contrib.pyros as pyros

from dispatches.models.renewables_case.uncertainty_models \
        .lmp_uncertainty_models import LMPDiscreteSet, CustomBoundsLMPBoxSet
from dispatches.models.renewables_case.uncertainty_models \
        .wind_uncertainty_models import WindDiscreteSet, CustomBoundsWindBoxSet


class NoForecast(Exception):
    """
    Exception raised in the event there is no forecast available
    for wind production or energy prices.
    """
    pass


class Forecaster(abc.ABC):
    """
    Class for forecasting LMP signal and wind production capacities.
    """
    @abc.abstractmethod
    def forecast_energy_prices(self, start, number_periods):
        """
        Forecast (nominal) energy prices (LMPs) for the next
        `number_periods` time intervals, including the
        current period.
        """
        ...

    @abc.abstractmethod
    def forecast_wind_production(self, start, number_periods):
        """
        Forecast (nominal) wind generation capacities for the
        next `number_periods` time intervals, including
        the current period.
        """
        ...

    @abc.abstractmethod
    def forecast_price_uncertainty(self, start, number_periods):
        """
        Forecast uncertainty in the LMPs for the next `number_periods`
        time intervals, including the current period.
        """
        ...

    @abc.abstractmethod
    def forecast_wind_uncertainty(self, start, number_periods):
        """
        Forecast uncertainty in the wind prodution levels for the
        next `number_periods` time intervals, including the current
        period.
        """
        ...

    @abc.abstractmethod
    def get_uncertain_params(self, lmp_params, wind_params, start, **kwds):
        """
        Given LMP and wind uncertain parameters, group uncertain
        parameters by decision making stage (in a multistage
        optimization under uncertainty context) and/or remove uncertain
        parameters.
        """
        ...

    @abc.abstractmethod
    def get_joint_lmp_wind_pyros_set(
            self,
            start,
            num_intervals,
            **kwds,
            ):
        """
        Obtain a PyROS `UncertaintySet` object modelling the LMP
        and wind production uncertainty.
        """
        ...


class Bus309Backcaster(Forecaster):
    """
    Backcaster for wind-battery models in which only the LMPs
    are taken to be uncertain.

    Parameters
    ----------
    data_file : pandas.DataFrame
        CSV file of LMP and wind production data.
    n_prev_days : int
        Number of prior 24-hr periods on which to base backcasting.
    lmp_set_class : class
        An `UncertaintySet` subclass used for predicting uncertainty
        in the LMPs.
    wind_set_class : class
        An `UncertaintySet` subclass used for predicting uncertainty
        in the wind production levels.
    wind_capacity : float or int, optional
        Nameplate wind production capacity.
    lmp_set_class_params : dict or None, optional
        Optional parameters for the LMP uncertainty prediction.
        Default is `None`.
    wind_set_class_params : dict or None, optional
        Optional parameters for the wind production uncertainty
        prediction. Default is `None`.
    start : int, optional
        Ordinal position of the interval considered to be the
        start of the planning horizon. The default is 0.
    """
    def __init__(
            self,
            data_file,
            n_prev_days,
            lmp_set_class,
            wind_set_class,
            wind_capacity,
            lmp_set_class_params=None,
            wind_set_class_params=None,
            ):
        """Construct LMP forecaster.

        """
        # initialize the dataframe. Account for misspellings in
        # the column names
        self.df = pd.read_csv(data_file)
        self.fix_df_column_names(self.df)

        # change index to integer counters
        self.df["period_index"] = range(len(self.df.index))
        self.df.set_index(["period_index"], drop=False)

        if not isinstance(n_prev_days, int):
            raise TypeError(
                "Argument `n_prev_days` must be of type `int` "
                f"({type(n_prev_days)} provided)"
            )
        assert n_prev_days > 0, "`n_prev_days` must be int greater than 0"
        self.n_prev_days = n_prev_days

        # LMP uncertainty set
        self.lmp_set_class = lmp_set_class
        if lmp_set_class_params is None:
            self.lmp_set_class_params = dict()
        else:
            self.lmp_set_class_params = lmp_set_class_params

        # wind production level uncertainty set
        self.wind_set_class = wind_set_class
        if wind_set_class_params is None:
            self.wind_set_class_params = dict()
        else:
            self.wind_set_class_params = wind_set_class_params

        self._wind_capacity = wind_capacity

        outputs_within_bounds = all(
            (self.df["Output DA Forecast"] <= self._wind_capacity)
        )
        if not outputs_within_bounds:
            raise ValueError(
                "Output wind production forecasts exceed wind capacity "
                f"of {wind_capacity}"
            )

    @classmethod
    def fix_df_column_names(cls, df):
        """
        Fix DataFrame column name misspellings.
        """
        col_changes_dict = {}
        for col_name in df.columns:
            col_changes_dict[col_name] = col_name.replace("Ouput", "Output")

        df.rename(columns=col_changes_dict, inplace=True)

    @property
    def wind_capacity(self):
        return self._wind_capacity

    @wind_capacity.setter
    def wind_capacity(self, val):
        raise AttributeError("Cannot set attribute `wind_capacity`")

    def _forecast(self, column_name, start, num_intervals):
        """Backcast energy prices.

        Parameters
        ----------
        column_name : str
            Name of column of the DataFrame for which to perform
            backcasting.
        start : int
            Starting interval (i.e. corresponding dataframe
            index value).
        num_intervals : int
            Number of intervals for which to backcast prices.

        Returns
        -------
        samples : pandas.DataFrame
            Index is `range(self.n_prev_days)`.
            Columns are `range(start, start+num_intervals)`.
        """
        samples = pd.DataFrame(
            index=range(self.n_prev_days),
            columns=range(start, start + num_intervals),
        )
        samples.index.name = "day"
        for day_idx in range(self.n_prev_days):
            hour_range = np.array([
                start - 24 * (day_idx + 1) + n
                for n in range(num_intervals)
            ])

            # make the prices for the year 'cyclic'
            while hour_range[hour_range < 0].size > 0:
                hour_range[hour_range < 0] += len(self.df.index)

            samples.loc[day_idx] = (
                self.df.loc[hour_range, column_name].values
            )

        return samples

    def forecast_energy_prices(self, start, num_intervals):
        """
        Forecast energy prices, by backcasting each corresponding
        hour of previous day.

        Parameters
        ----------
        start : int
            Index for starting period.
        num_intervals : int
            Number of time intervals for which to forecast prices.
        capacity_factors : bool, optional
            Return capacity factors instead of wind production levels.
            The default is False.

        Returns
        -------
        production_forecast : (`num_intervals`,) numpy.ndarray
            Wind production forecast for the next `num_intervals`
            hours.
        """
        prices = self._forecast("LMP DA", start, num_intervals)

        return prices.loc[0].values

    def forecast_wind_production(
            self,
            start,
            num_intervals,
            capacity_factors=False,
            ):
        """
        Forecast wind production levels.

        Parameters
        ----------
        start : int
            Index for starting period.
        num_intervals : int
            Number of time intervals for which to
        capacity_factors : bool, optional
            Return capacity factors instead of wind production levels.
            The default is False.

        Returns
        -------
        production_forecast : (`num_intervals`,) numpy.ndarray
            Wind production forecast for the next `num_intervals`
            hours.
        """
        production_levels = self._forecast(
            "Output DA Forecast",
            start,
            num_intervals,
        )
        production_forecast = production_levels.loc[0].values

        if capacity_factors:
            return production_forecast / self._wind_capacity
        else:
            return production_forecast

    def forecast_price_uncertainty(self, start, num_intervals):
        """Forecast uncertainty in LMPs.

        """
        # multiple scenarios
        prices = self._forecast("LMP DA", start, num_intervals)
        nom_prices = self.forecast_energy_prices(start, num_intervals)

        # now generate the uncertainty set
        if self.lmp_set_class is None:
            raise NoForecast
        elif self.lmp_set_class is LMPDiscreteSet:
            return self.lmp_set_class(scenarios=prices.values, nom_idx=0)
        elif self.lmp_set_class is CustomBoundsLMPBoxSet:
            min_prices = prices.min(axis=0)
            max_prices = prices.max(axis=0)
            bounds = [[p1, p2] for p1, p2 in zip(min_prices, max_prices)]
            return self.lmp_set_class(nom_prices, bounds)
        else:
            return self.lmp_set_class(nom_prices, **self.lmp_set_params)

    def forecast_wind_uncertainty(
            self,
            start,
            num_intervals,
            capacity_factors=False,
            ):
        """Forecast uncertainty in wind production levels.

        """
        wind_production = self._forecast(
            "Output DA Forecast",
            start=start,
            num_intervals=num_intervals,
        )
        if capacity_factors:
            wind_production /= self._wind_capacity

        sig_nom = self.forecast_wind_production(
            start,
            num_intervals,
            capacity_factors=capacity_factors,
        )

        if self.wind_set_class is None:
            raise NoForecast
        elif self.wind_set_class is WindDiscreteSet:
            return self.wind_set_class(
                scenarios=wind_production.values,
                nom_idx=0,
                wind_capacity=self._wind_capacity,
            )
        elif self.wind_set_class is CustomBoundsWindBoxSet:
            lower_bounds = wind_production.min(axis=0).values
            upper_bounds = wind_production.max(axis=0).values
            bounds = [[lb, ub] for lb, ub in zip(lower_bounds, upper_bounds)]
            return self.wind_set_class(wind_data=sig_nom, bounds=bounds)
        else:
            return self.wind_set_class(sig_nom, **self.wind_set_class_params)

    def get_uncertain_params(
            self,
            lmp_params,
            wind_params,
            start,
            include_fixed_dims=True,
            nested=False,
            ):
        """
        Get uncertain parameters and organize by problem stage
        (if returning nested list).
        """
        num_lmp_params = len(lmp_params)
        num_wind_params = len(wind_params)

        assert num_lmp_params == num_wind_params

        if self.lmp_set_class is not None:
            lmp_set = self.forecast_price_uncertainty(
                start=start,
                num_intervals=num_lmp_params,
            )
            lmp_params = lmp_set.get_uncertain_params(
                lmp_params,
                include_fixed_dims=include_fixed_dims,
                nested=True,
            )
        else:
            lmp_params = [[] for _ in range(num_lmp_params)]

        if self.wind_set_class is not None:
            wind_set = self.forecast_wind_uncertainty(
                start=start,
                num_intervals=num_wind_params,
            )
            wind_params = wind_set.get_uncertain_params(
                wind_params,
                include_fixed_dims=include_fixed_dims,
                nested=True,
            )
        else:
            wind_params = list([] for _ in range(num_wind_params))

        uncertain_params = list([] for _ in range(num_lmp_params))
        for i, (lmp_lst, wind_lst) in enumerate(zip(lmp_params, wind_params)):
            uncertain_params[i].extend(lmp_lst + wind_lst)

        if nested:
            uncertain_params[0] += uncertain_params[1]
            uncertain_params.remove(uncertain_params[1])
            return uncertain_params
        else:
            return list(itertools.chain.from_iterable(uncertain_params))

    def get_joint_lmp_wind_pyros_set(
            self,
            start,
            num_intervals,
            include_fixed_dims=True,
            capacity_factors=True,
            ):
        """Get joint LMP and wind PyROS uncertainty set.

        """
        if self.lmp_set_class is None:
            wind_set = self.forecast_wind_uncertainty(
                start,
                num_intervals,
            )
            return wind_set.pyros_set()
        elif self.wind_set_class is None:
            lmp_set = self.forecast_price_uncertainty(
                start,
                num_intervals,
            )
            return lmp_set.pyros_set()
        else:
            lmp_set = self.forecast_price_uncertainty(
                start,
                num_intervals,
            )
            wind_set = self.forecast_wind_uncertainty(
                start,
                num_intervals,
                capacity_factors=capacity_factors,
            )

            lmp_pyros_set = lmp_set.pyros_set(include_fixed_dims=True)
            wind_pyros_set = wind_set.pyros_set(include_fixed_dims=True)

            valid_pyros_types = (pyros.BoxSet, pyros.DiscreteScenarioSet)

            # verify both PyROS sets are of the same type
            assert type(lmp_pyros_set) is type(wind_pyros_set)

            for set_type in valid_pyros_types:
                set_type_matched = isinstance(lmp_pyros_set, set_type)

                if set_type_matched and set_type is pyros.BoxSet:
                    lmp_bounds = lmp_set.bounds()
                    wind_bounds = wind_set.bounds()
                    joint_bounds = self.get_uncertain_params(
                        lmp_bounds,
                        wind_bounds,
                        start=start,
                        include_fixed_dims=include_fixed_dims,
                        nested=False,
                    )
                    joint_set = pyros.BoxSet(bounds=joint_bounds)
                elif set_type is pyros.DiscreteScenarioSet:
                    scenario_product = itertools.product(
                        lmp_set.scenarios,
                        wind_set.scenarios,
                    )
                    joint_scenarios = list()
                    for lmp_scenario, wind_scenario in scenario_product:
                        joint_scenario = self.get_uncertain_params(
                            lmp_scenario,
                            wind_scenario,
                            start=start,
                            include_fixed_dims=include_fixed_dims,
                            nested=False,
                        )
                        joint_scenarios.append(joint_scenario)
                    joint_set = pyros.DiscreteScenarioSet(joint_scenarios)

                return joint_set

            raise TypeError(
                "Only `BoxSet` and `DiscreteScenarioSet` PyROS set types "
                "are currently supported"
            )


if __name__ == "__main__":
    forecaster = Bus309Backcaster(
        "../../../../../results/wind_profile_data/309_wind_1_profiles.csv",
        n_prev_days=7,
        lmp_set_class=CustomBoundsLMPBoxSet,
        wind_set_class=None,
        wind_capacity=148.3,
        lmp_set_class_params=None,
        wind_set_class_params=None,
    )
    lmp_set = forecaster.forecast_price_uncertainty(3990, 12)
    # wind_set = forecaster.forecast_wind_uncertainty(4000, 12)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    print(lmp_set.bounds())
    lmp_set.plot_set(ax, highlight_peak_effects=False, offset=0)
    plt.savefig("test.png")
    plt.close()

    lmp_params = [f"p[{idx}]" for idx in range(12)]
    wind_params = [f"w[{idx}]" for idx in range(12)]
    params = forecaster.get_uncertain_params(
        lmp_params,
        wind_params,
        4000,
        nested=True,
    )
    joint_set = forecaster.get_joint_lmp_wind_pyros_set(
        start=100,
        num_intervals=24,
    )

    import pdb
    pdb.set_trace()
