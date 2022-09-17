#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-12-01 Wed 16:29:50

@author: jasherma

Uncertainty models for LMPs.
"""

import abc

import numpy as np
# import matplotlib.pyplot as plt

from pyomo.contrib.pyros import uncertainty_sets

from dispatches.models.renewables_case.uncertainty_models.uncertainty_models \
        import UncertaintySet


class LMPDiscreteSet(UncertaintySet):
    """
    Set of discrete LMP scenarios.
    """
    def __init__(self, scenarios, nom_idx):
        """Construct discrete set of LMP scenarios."""
        self.scenarios = np.array(scenarios)

        assert len(self.scenarios.shape) == 2

        if not isinstance(nom_idx, int):
            raise TypeError("Argument `nom_idx` must be of type `int`")
        elif nom_idx < 0 or nom_idx > self.scenarios.shape[0] - 1:
            raise ValueError(
                "Argument `nom_idx` must be between 0 and "
                f"{scenarios.shape[0] - 1}, (provided value {nom_idx})"
            )
        self._nom_idx = nom_idx

    @property
    def sig_nom(self):
        """Nominal LMP signal."""
        return self.scenarios[self._nom_idx]

    def pyros_set(self, include_fixed_dims=True):
        """Construct PyROS Uncertainty set."""
        return uncertainty_sets.DiscreteScenarioSet(scenarios=self.scenarios)

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


class LMPBoxSet(UncertaintySet):
    def __init__(self, lmp_data, include_peak_effects=False, start_day_hour=0):
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)

        assert self.bounds_valid()

    @abc.abstractmethod
    def bounds(self):
        ...

    def bounds_valid(self):
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
        """Plot LMP bounds against planning period."""
        bounds = self.bounds()

        # resolve bounds into lower and upper
        periods = np.array(list(range(len(bounds)))) + offset
        lower_bds = np.array([bounds[idx][0] for idx in range(len(bounds))])
        upper_bds = np.array([bounds[idx][1] for idx in range(len(bounds))])

        color = "black"

        # generate the plots
        ax.step(periods, upper_bds, "--", color="green", linewidth=1.0,
                where="post")
        ax.step(periods, lower_bds, "--", color="green", label="bounds",
                linewidth=1.0, where="post")
        ax.fill_between(periods, lower_bds, upper_bds, step="post",
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
                ax.step(peak_hrs, upper_bds[peak_times], "--", color="red",
                        linewidth=1.0, where="post")
                ax.step(peak_hrs, lower_bds[peak_times], "--", color="red",
                        label='peak times', linewidth=1.0, where="post")
                ax.fill_between(peak_hrs, lower_bds[peak_times],
                                upper_bds[peak_times],
                                color="red", alpha=0.1, step="post")

        # plot nominal LMP
        ax.step(periods, self.lmp_sig_nom, color=color, label='nominal',
                linewidth=1.8, where="post")

    def plot_set(self, ax, highlight_peak_effects=False, offset=0):
        """Plot uncertainty set."""
        return self.plot_bounds(
            ax,
            highlight_peak_effects=highlight_peak_effects,
            offset=offset,
        )


class CustomBoundsLMPBoxSet(LMPBoxSet):
    """
    LMP box set with custom bounds.
    """
    def __init__(self, lmp_data, bounds, first_period_certain=False):
        assert len(lmp_data) == len(bounds)

        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (len(lmp_data), 2)

        self.lmp_sig_nom = lmp_data
        self.bounds_arr = bounds_arr.tolist()
        self.first_period_certain = first_period_certain

    def bounds(self):
        bounds = [tuple(bd) for bd in self.bounds_arr]

        if self.first_period_certain:
            bounds[0] = (self.lmp_sig_nom[0], self.lmp_sig_nom[0])

        return bounds

    @property
    def sig_nom(self):
        return self.lmp_sig_nom


class ConstantUncertaintyBoxSet(LMPBoxSet):
    def __init__(self, lmp_data, uncertainty, first_period_certain=False):
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)
        self.uncertainty = uncertainty
        self.first_period_certain = first_period_certain

    @property
    def sig_nom(self):
        return self.lmp_sig_nom

    def bounds(self):
        bounds = [(self.lmp_sig_nom[time] - self.uncertainty,
                   self.lmp_sig_nom[time] + self.uncertainty)
                  for time in range(self.n_time_points)]

        if self.first_period_certain:
            bounds[0] = (self.lmp_sig_nom[0], self.lmp_sig_nom[0])

        return bounds


class ConstantFractionalUncertaintyBoxSet(LMPBoxSet):
    def __init__(
            self,
            lmp_data,
            fractional_uncertainty,
            first_period_certain=False,
            ):
        assert fractional_uncertainty >= 0
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)
        self.fractional_uncertainty = fractional_uncertainty
        self.first_period_certain = first_period_certain

    @property
    def sig_nom(self):
        return self.lmp_sig_nom

    def bounds(self):
        bounds = []
        for time in range(self.n_time_points):
            nom_val = self.lmp_sig_nom[time]
            lb = nom_val - abs(nom_val) * self.fractional_uncertainty
            ub = nom_val + abs(nom_val) * self.fractional_uncertainty
            bounds.append((lb, ub))

        if self.first_period_certain:
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

    @property
    def sig_nom(self):
        return self.lmp_sig_nom

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

    @property
    def sig_nom(self):
        return self.lmp_sig_nom

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

    @property
    def sig_nom(self):
        return self.lmp_sig_nom

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

    @property
    def sig_nom(self):
        return self.lmp_sig_nom

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
    box_set = ExpandDiffSet(np.random.rand(10), 0.1)

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    box_set.plot_set(ax, offset=0)
    plt.legend()
    plt.savefig("example.png")
