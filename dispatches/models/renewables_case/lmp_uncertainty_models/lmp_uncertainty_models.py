#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-12-01 Wed 16:29:50

@author: jasherma

Uncertainty models for multiperiod rankine model LMP's.
"""

import copy
from itertools import chain

import numpy as np
# import matplotlib.pyplot as plt

from pyomo.contrib.pyros import uncertainty_sets


def get_lmp_data(filename, n_time_points, start=0):
    """
    Extract LMP signal data from file for first `n_time_points`
    time periods, beginning with period `start`.
    """
    assert type(n_time_points) is int
    assert type(start) is int

    # read lmp signal (all 8736 hrs) data from file
    path = filename
    with open(path, 'rb') as f:
        _ = np.load(f)  # dispatch levels
        prices = np.load(f)  # lmp signal data

    # max out prices at $200 / MWH
    prices_used = copy.copy(prices)
    prices_used[prices_used > 200] = 200

    # plt.bar(np.arange(prices_used.size), prices_used)
    # plt.show()

    # return appropriate portion of LMP data.
    return prices_used[start:n_time_points + start]


class LMPBoxSet():
    def __init__(self, lmp_data, include_peak_effects=False, start_day_hour=0):
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)

    def bounds(self):
        return [(self.lmp_sig_nom[time], self.lmp_sig_nom[time])
                for time in range(self.n_time_points)]

    def bounds_valid(self,):
        bounds = self.bounds()
        for time in range(len(bounds)):
            lb, ub = bounds[time]
            if lb > self.lmp_sig_nom[time] or ub < self.lmp_sig_nom[time]:
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
        uncertainty_data = np.array([periods, lower_bds, self.lmp_sig_nom,
                                     upper_bds]).T
        np.savetxt('lmp_uncertainty_data.dat',
                   uncertainty_data, delimiter='\t',
                   header='time\tlower\tnominal\tupper',
                   comments='', fmt=['%d', '%.3f', '%.3f', '%.3f'])

    def determine_dims_fixed_by_bounds(self, tol=0):
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
            fixed_dims = self.determine_dims_fixed_by_bounds()
            final_bounds = [
                bd for idx, bd in enumerate(bounds)
                if idx not in fixed_dims
            ]
        return uncertainty_sets.BoxSet(bounds=final_bounds)

    def get_uncertain_params(
            self,
            uncertain_params,
            include_fixed_dims=True,
            nested=False,
            ):
        """
        Given a list of model components of length equal to the
        dimensionality of the set, determine the components
        corresponding to the dimensions not fixed by the
        set's bounds (if so desired); otherwise, return all components.
        """
        assert len(uncertain_params) == len(self.lmp_sig_nom)

        if not include_fixed_dims:
            fixed_dims = self.determine_dims_fixed_by_bounds()
        else:
            fixed_dims = []

        nested_param_list = list()
        for idx, param in enumerate(uncertain_params):
            if idx not in fixed_dims:
                nested_param_list.append([param])
            else:
                nested_param_list.append([])

        if nested:
            return nested_param_list
        else:
            return list(chain.from_iterable(nested_param_list))

    def lift_uncertain_params(self, reduced_params, lifting_values):
        """
        'Lift' the reduced dimensions of the set (i.e the dimensions
        for which the bounds are equal) back into the original set
        dimensions.

        Parameters
        ----------
        reduced_params : list(object)
            List of parameter objects corresponding to reduced
            dimensions.
        lifting_values : dict, optional
            Values to insert into each fixed dimension. Keys are
            the indices of the fixed dimensions. Values are the
            actual objects to be placed into the list to obtain
            the 'lifted' object.

        Returns
        -------
        lifted_params : list(object)
            List of parameter objects in the original, full
            uncertainty space.
        """
        fixed_dims = self.determine_dims_fixed_by_bounds()

        if fixed_dims != list(lifting_values.keys()):
            raise ValueError(
                "Fixed dimensions mismatch. Provided insertion into "
                f"dimensions {list(lifting_values.keys())}, but fixed "
                f"dimensions are actually {fixed_dims}"
            )

        lifted_params = list()
        reduced_dim_count = 0
        for idx in range(len(self.lmp_sig_nom)):
            if idx in fixed_dims:
                val = lifting_values[idx]
            else:
                val = reduced_params[reduced_dim_count]
                reduced_dim_count += 1
            lifted_params.append(val)

        return lifted_params

    def plot_bounds(self, ax, highlight_peak_effects=False, filename=None,
                    offset=0):
        """Plot LMP bounds against planning period."""
        bounds = self.bounds()

        # resolve bounds into lower and upper
        periods = np.array(list(range(len(bounds)))) + offset
        lower_bds = np.array([bounds[idx][0] for idx in range(len(bounds))])
        upper_bds = np.array([bounds[idx][1] for idx in range(len(bounds))])

        color = "black"

        # generate the plots
        ax.plot(periods, upper_bds, "--", color="green", linewidth=1.0)
        ax.plot(periods, lower_bds, "--", color="green", label="bounds",
                linewidth=1.0)
        ax.fill_between(periods, lower_bds, upper_bds,
                        color="green", alpha=0.1, label="uncertainty")

        # highlight peak effects if desired
        times = np.arange(len(self.lmp_sig_nom))
        if highlight_peak_effects:
            hours_of_day = (times + self.start_day_hour) % 24
            at_sunrise = np.logical_and(hours_of_day >= 6, hours_of_day <= 8)
            at_sunset = np.logical_and(hours_of_day >= 18, hours_of_day <= 20)

            for cond in [at_sunrise, at_sunset]:
                peak_times = cond
                # plot bounds and LMP signal
                peak_hrs = times[peak_times] + offset
                ax.plot(peak_hrs, upper_bds[peak_times], "--", color="red",
                        linewidth=1.0)
                ax.plot(peak_hrs, lower_bds[peak_times], "--", color="red",
                        label='peak times', linewidth=1.0)
                ax.fill_between(peak_hrs, lower_bds[peak_times],
                                upper_bds[peak_times],
                                color="red", alpha=0.1)

        # plot nominal LMP
        ax.plot(periods, self.lmp_sig_nom, color=color, label='nominal',
                linewidth=1.8)

        # labels
        # ax.set_ylabel('LMP signal ($/MWh)')
        # plt.legend()

        # if filename is not None:
        # plt.savefig(filename, bbox_inches="tight", dpi=300)

        # plt.show()
        # plt.close()


class ConstantUncertaintyBoxSet(LMPBoxSet):
    def __init__(self, lmp_data, uncertainty):
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)
        self.uncertainty = uncertainty

    def bounds(self):
        bounds = [(self.lmp_sig_nom[time] - self.uncertainty,
                   self.lmp_sig_nom[time] + self.uncertainty)
                  for time in range(self.n_time_points)]
        bounds[0] = (self.lmp_sig_nom[0], self.lmp_sig_nom[0])
        return bounds


class SimpleLMPBoxSet(LMPBoxSet):
    def __init__(self, lmp_data, n_recent, growth_rate, avg_multiplier):
        """Initialize simple LMP box set."""
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)
        self.n_recent = n_recent
        self.growth_rate = growth_rate
        self.mov_avg_multiplier = avg_multiplier

    def bounds(self):
        """Evaluate LMP uncertainty set bounds."""
        multipliers = [self.growth_rate * t for t in range(self.n_time_points)]

        moving_average = [self.mov_avg_multiplier
                          * sum([self.lmp_sig_nom[tprime]
                                / (t - max(0, t - self.n_recent))
                                for tprime in range(max(0, t - self.n_recent),
                                                    t - 1)])
                          for t in range(self.n_time_points)]
        lmp_bounds = [(max(0, self.lmp_sig_nom[time]
                           * (1 - multipliers[time])),
                      max(moving_average[time], self.lmp_sig_nom[time] *
                          (1 + multipliers[time])))
                      for time in range(self.n_time_points)]

        return lmp_bounds


class HysterLMPBoxSet(SimpleLMPBoxSet):
    def __init__(self, lmp_data, n_recent, growth_rate, avg_multiplier,
                 latency, start_day_hour=0, include_peak_effects=False):
        """Initialize hysteresis LMP box set."""
        super().__init__(lmp_data, n_recent, growth_rate, avg_multiplier)
        self.latency = latency
        self.start_day_hour = start_day_hour
        self.include_peak_effects = include_peak_effects

    def bounds(self):
        """Evaluate LMP uncertainty set bounds."""
        multipliers = [self.growth_rate * t for t in range(self.n_time_points)]

        # calculate moving averages
        moving_avgs = [self.mov_avg_multiplier
                       * sum([self.lmp_sig_nom[tprime]
                              / (t - max(0, t - self.n_recent))
                             for tprime in range(max(0, t - self.n_recent),
                                                 t - 1)])
                       for t in range(self.n_time_points)]

        # evaluate the bounds
        lmp_bounds = []
        drop_times = [time for time in range(self.n_time_points) if
                      self.lmp_sig_nom[time] <= 0 and
                      self.lmp_sig_nom[max(0, time - 1)] > 0]
        spike_times = [time for time in range(self.n_time_points) if
                       self.lmp_sig_nom[time] > 0 and
                       self.lmp_sig_nom[max(0, time - 1)] <= 0]
        spike_diffs = [self.lmp_sig_nom[time] - self.lmp_sig_nom[time - 1]
                       for time in spike_times]
        for time in range(self.n_time_points):
            # calculate lower bound
            time_low = max(0, time - self.latency)
            time_high = min(self.n_time_points, time + self.latency + 1)
            lb = max(0, self.lmp_sig_nom[time] * (1 - multipliers[time]))
            if np.any(self.lmp_sig_nom[time_low:time_high] <= 0):
                periods_since_drop = range(time_low, time)
                if len(periods_since_drop) != 0:
                    avg_lb = (sum(lmp_bounds[t][0] for t in periods_since_drop)
                              / len(periods_since_drop)) * 0.7
                else:
                    avg_lb = 0
                lb = min(0, -moving_avgs[time], avg_lb)
                # lb = min(0, -moving_avgs[time])

            ub = max(moving_avgs[time],
                     self.lmp_sig_nom[time] * (1 + multipliers[time]))
            if time in drop_times:
                diff = self.lmp_sig_nom[time - 1] - self.lmp_sig_nom[time - 2]
                ub = max(self.lmp_sig_nom[time] + abs(diff * 0.1),
                         lmp_bounds[time - 1][1] + diff)
            elif time + 1 in spike_times:
                diff = spike_diffs[spike_times == time]
                # print(diff)
                ub = diff

            # augment uncertainty during peak hours
            if self.include_peak_effects:
                time_of_day = (time + self.start_day_hour) % 24
                at_sunrise = time_of_day >= 6 and time_of_day <= 8
                at_sunset = time_of_day >= 18 and time_of_day <= 20
                if at_sunrise or at_sunset:
                    lb -= (self.lmp_sig_nom[time] - lb) * 0.5
                    lb = max(0, lb) if self.lmp_sig_nom[time] > 0 else lb
                    ub -= (self.lmp_sig_nom[time] - ub) * 0.5

            lmp_bounds.append((lb, ub))

        return lmp_bounds


class ExpandDiffSet(LMPBoxSet):
    def __init__(self, lmp_data, growth_rate):
        self.lmp_sig_nom = lmp_data
        self.growth_rate = growth_rate
        self.n_time_points = len(self.lmp_sig_nom)

    def bounds(self):
        lower_bounds = [self.lmp_sig_nom[0]]
        upper_bounds = [self.lmp_sig_nom[0]]
        for time in range(1, self.n_time_points):
            diff = self.lmp_sig_nom[time] - self.lmp_sig_nom[time - 1]
            growth_factor = ((self.growth_rate * time)
                             if diff <= 0
                             else (- self.growth_rate * time))
            lb = lower_bounds[time - 1] + diff * (1 + growth_factor)
            ub = upper_bounds[time - 1] + diff * (1 - growth_factor)

            lower_bounds.append(lb)
            upper_bounds.append(ub)

        return [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]


class TimeFcnDiffUncertaintySet(LMPBoxSet):
    def __init__(self, lmp_data, lower_bound_func, upper_bound_func,
                 include_peak_effects=True, start_day_hour=0):
        self.lmp_sig_nom = lmp_data
        self.lower_bound_func = lower_bound_func
        self.upper_bound_func = upper_bound_func
        self.include_peak_effects = include_peak_effects
        self.start_day_hour = start_day_hour

    def bounds(self):
        lmp_sig = np.array(self.lmp_sig_nom)
        times = np.arange(len(self.lmp_sig_nom))
        lower_bounds = lmp_sig - self.lower_bound_func(times)
        upper_bounds = lmp_sig + self.upper_bound_func(times)

        if self.include_peak_effects:
            hours_of_day = (times + self.start_day_hour) % 24
            at_sunrise = np.logical_and(hours_of_day >= 6, hours_of_day <= 8)
            at_sunset = np.logical_and(hours_of_day >= 18, hours_of_day <= 20)
            peak_times = np.logical_or(at_sunrise, at_sunset)
            lower_bounds[peak_times] -= (self.lmp_sig_nom[peak_times]
                                         - lower_bounds[peak_times]) * 0.5
            upper_bounds[peak_times] -= (self.lmp_sig_nom[peak_times]
                                         - upper_bounds[peak_times]) * 0.5
            # lb = max(0, lb) if self.lmp_sig_nom[time] > 0 else lb

        return [(lower_bounds[time], upper_bounds[time]) for time in times]


if __name__ == "__main__":
    # decide set parameters
    n_periods = 24
    start = 4000
    n_recent = 4
    growth_rate = 0.03
    mov_avg_mult = 0.3
    hyster_latency = 1

    lmp_data = get_lmp_data(n_periods, start=start)
    lmp_set = SimpleLMPBoxSet(lmp_data, n_recent, growth_rate, mov_avg_mult)
    # lmp_set.plot_bounds()

    hyster_set = HysterLMPBoxSet(lmp_data, n_recent, growth_rate, mov_avg_mult,
                                 hyster_latency, start_day_hour=0,
                                 include_peak_effects=False)
    print("Bounds valid:", hyster_set.bounds_valid())
    hyster_set.plot_bounds(highlight_peak_effects=False)

    # constant_set = ConstantUncertaintyBoxSet(lmp_data, 10)
    # constant_set.plot_bounds()
    # expand_set = ExpandDiffSet(lmp_data, growth_rate)
    # expand_set.plot_bounds()

    # time-dependent box set
    def lb_func(time):
        return 1.0 * time

    def ub_func(time):
        return lb_func(time)

    time_set = TimeFcnDiffUncertaintySet(lmp_data, lb_func, ub_func,
                                         include_peak_effects=False)
    time_set.plot_bounds(highlight_peak_effects=True)
