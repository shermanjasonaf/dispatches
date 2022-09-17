#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-01 Wed 17:50:23

@author: jasherma

This file contains an implementation of the multi-stage
robust optimization extension of the multi-period wind-battery model.
"""


import os
import pdb
import logging

import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt
import pandas as pd

import pyomo.environ as pyo
import pyomo.contrib.pyros as pyros
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.current import identify_variables
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

from dispatches.models.renewables_case.wind_battery_double_loop import (
    create_multiperiod_wind_battery_model,
    transform_design_model_to_operation_model,
)


def plot_soc_results(
        mp_model,
        lmp_set=None,
        plot_lmp=True,
        plot_uncertainty=True,
        filename=None,
        start=None,
        stop=None,
        highlight_active_periods=False,
        label_inactive_periods=False,
        custom_lmp_vals=None,
        lmp_bounds=None,
        xmin=None,
        xmax=None,
        ):
    """
    Plot battery state-of-charge values (in MWh).

    Parameters
    ----------
    mp_model : MultiPeriodModel
        Model of interest.
    lmp_set : LMPBoxSet, optional
        LMP uncertainty set. The default is None.
        If an LMP set is provided, the band of uncertainty
        is plotted provied `plot_lmp` is set to `True`.
        The dimension of the set must be equal to `mp_model`'s prediction
        horizon length (i.e. number of active process blocks).
    plot_lmp : bool, optional
        Plot LMP signal values along with the state of charge values.
        The default is `True`.
    plot_uncertainty : bool, optional
        Plot LMP uncertainty set bounds. The default is `False`.
    filename : path-like, optional
        Output file path. The default is `None`, in which
        case the plot is not exported to file.
    start : int, optional
        First period to include in the plot.
        The default is `None`, in which case `mp_model.current_time`
        is used.
    stop : int, optional
        Last period to include in the plot.
        The default is `None`, in which case the model's last
        active time is used.
    highlight_active_periods : bool, optional
        Highlight periods corresponding to the model's active blocks,
        (or more precisely, dim periods corresponding to inactive
        process blocks).
        The default is `False`.
    custom_lmp_vals : dict, optional
        Custom LMP signal values to include in the plot;
        these values are added to the plot only if `plot_lmp`.
        is set to `True`. The default is `None`.
        If a dictionary is provided, then the keys are the plot legend
        labels, and the values are dictionaries whose:
        - keys are integers denoting periods for which to plot the LMPs
        - values are floats of the LMPs ($/MWh) for the corresponding
          periods.
    lmp_bounds : tuple, optional
        A 2-tuple of bounds to be used for the LMP signal plot axis.
        The default is `None`, in which case the greatest upper bound
        and smallest lower bound from the LMP uncertainty set
        are used for the axis bounds
        (if an uncertainty set is provided).
    xmin : float, optional
        Time axis (x-axis) lower bound. The default is `None`,
        in which caset `start` is used.
    xmax : float, optional
        Time axis (x-axis) upper bound. The default is `None`,
        in which case `stop` is used.
    """
    active_blks = mp_model.get_active_process_blocks()

    plt.rcParams.update({'font.size': 20})

    fig, ax1 = plt.subplots(figsize=(9, 8))
    ax1.grid(False)
    ax1.set_facecolor("#EAEAF2")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # get model blocks of interest
    start = mp_model.current_time if start is None else start
    stop = (
        mp_model.current_time + len(active_blks) - 1
        if stop is None else stop
    )

    assert start >= 0
    assert stop < len(mp_model.pyomo_model.blocks)

    if lmp_bounds is not None:
        assert lmp_bounds[0] < lmp_bounds[1]

    periods = np.array(range(start, stop + 1), dtype=int)
    active_start = mp_model.current_time
    blocks = list(
        blk.process
        for idx, blk in enumerate(mp_model.pyomo_model.blocks.values())
        if idx >= start and idx <= stop
    )

    # initialize containers
    charges = list()
    lmp_values = list()
    charge_bounds = list()

    for t, blk in zip(periods, blocks):
        charges.append(pyo.value(blk.fs.battery.initial_state_of_charge))
        lmp_values.append(pyo.value(mp_model.pyomo_model.LMP[t]))

        charge_lim = pyo.value(
            blk.fs.battery.nameplate_energy
            - blk.fs.battery.degradation_rate
            * blk.fs.battery.energy_throughput[0.0]
        )
        charge_bounds.append(charge_lim)

    # add final state of charge
    charges.append(pyo.value(blk.fs.battery.state_of_charge[0.0]))
    app_periods = np.append(periods, [periods[-1] + 1])

    # plot results for active periods
    if periods[periods >= active_start].size > 0:
        ax1.plot(
            app_periods[app_periods >= active_start],
            np.array(charges)[app_periods >= active_start] / 1e3,
            label="state of charge",
            linewidth=1.8,
            color="blue"
        )
        ax1.plot(
            periods[periods >= active_start],
            np.array(charge_bounds)[periods >= active_start] / 1e3,
            label="charging limit",
            linewidth=1.8,
            color="blue",
            linestyle="dashed",
        )

    # plot results for inactive periods
    if periods[periods < active_start].size > 0:
        alpha = 0.3 if highlight_active_periods else 1
        ax1.plot(
            app_periods[app_periods <= active_start],
            np.array(charges)[app_periods <= active_start] / 1e3,
            label=(
                "state of charge (prev)"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            linewidth=1.8,
            color="blue",
            alpha=alpha,
        )
        ax1.plot(
            periods[periods <= active_start],
            np.array(charge_bounds)[periods <= active_start] / 1e3,
            label=(
                "charging limit (prev)"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            linewidth=1.8,
            color="blue",
            linestyle="dashed",
            alpha=alpha,
        )

    ax1.set_xlabel("Period (hr)")
    ax1.set_ylabel("State of charge (MWh)")
    # ax1.set_xticks(model._pyomo_model.Horizon)
    ax1.legend(bbox_to_anchor=(0, -0.15), loc="upper left", ncol=1)

    if plot_lmp:
        ax2 = ax1.twinx()
        ax2.grid(False)
        _plot_lmp(ax2, periods, lmp_values, lmp_set, active_start,
                  lmp_bounds=lmp_bounds, plot_uncertainty=plot_uncertainty,
                  highlight_active_periods=highlight_active_periods,
                  custom_lmp_vals=custom_lmp_vals)

    # set time interval axis limits
    xmin = start if xmin is None else xmin
    xmax = stop if xmin is None else xmax
    ax1.set_xlim([xmin, xmax])

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close()


def _plot_lmp(ax, periods, lmp_values, lmp_set,
              active_start, lmp_bounds=None, plot_uncertainty=True,
              highlight_active_periods=True, custom_lmp_vals=None):
    """
    Plot LMP signal.
    """
    ax.set_ylabel("LMP Signal ($/MWh)", color="black")
    if plot_uncertainty and lmp_set is not None:
        # plot LMP uncertainty set bounds.
        # NOTE: the LMPs are scaled to MWh before plotting
        #       since the values provided are in kWh
        # TODO: enforce LMP units at model declaration
        lmp_set.plot_set(ax, offset=active_start)
    else:
        ax.step(
            periods[periods >= active_start],
            np.array(lmp_values)[periods >= active_start],
            label="LMP",
            linewidth=1.8,
            color="black",
            where="post",
        )
    if periods[periods < active_start].size > 0:
        alpha = 0.3 if highlight_active_periods else 1
        ax.step(
            periods[periods <= active_start],
            np.array(lmp_values)[periods <= active_start],
            label="LMP (prev)",
            linewidth=1.8,
            color="black",
            alpha=alpha,
            where="post",
        )

    # plot custom LMP values
    custom_lmp_vals = {} if custom_lmp_vals is None else custom_lmp_vals
    for lmp_label, lmp_dict in custom_lmp_vals.items():
        per_arr = np.array(list(lmp_dict.keys()))
        assert np.all(per_arr == np.arange(per_arr[0], per_arr[-1] + 1))
        lmp_arr = np.array(list(lmp_dict.values()))
        ax.step(per_arr, lmp_arr, label=lmp_label, linewidth=1.8,
                color="green", where="post")

    ax.legend(bbox_to_anchor=(1, -0.15), loc="upper right", ncol=1)

    # set LMP axis bounds
    if lmp_bounds is None:
        if lmp_set is not None:
            lmp_bounds = lmp_set.bounds()
            y_min = min(bound[0] for bound in lmp_bounds)
            y_max = max(bound[1] for bound in lmp_bounds)
            ax.set_ylim([y_min, y_max])
    else:
        y_min = lmp_bounds[0]
        y_max = lmp_bounds[1]
        ax.set_ylim([y_min, y_max])


def plot_power_output_results(
        mp_model,
        lmp_set=None,
        plot_lmp=True,
        plot_uncertainty=True,
        filename=None,
        start=None,
        stop=None,
        highlight_active_periods=False,
        label_inactive_periods=False,
        custom_lmp_vals=None,
        lmp_bounds=None,
        xmin=None,
        xmax=None,
        ):
    """
    Plot model power output results
    (wind production, battery-to-grid discharge,
    and wind-to-grid power).

    Arguments
    ---------
    Same as those used for `plot_soc_results`.
    """
    active_blks = mp_model.get_active_process_blocks()

    plt.rcParams.update({'font.size': 20})

    fig, ax1 = plt.subplots(figsize=(9, 8))
    ax1.grid(False)

    # configure axis appearance
    ax1.set_facecolor("#EAEAF2")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # get model blocks of interest
    start = mp_model.current_time if start is None else start
    stop = (
        mp_model.current_time + len(active_blks) - 1 if stop is None else stop
    )

    assert start >= 0
    assert stop < len(mp_model.pyomo_model.blocks)

    if lmp_bounds is not None:
        assert lmp_bounds[0] < lmp_bounds[1]

    periods = np.array(range(start, stop + 2))
    active_start = mp_model.current_time
    blocks = list(
        blk.process
        for idx, blk in enumerate(mp_model.pyomo_model.blocks.values())
        if idx >= start and idx <= stop
    )

    # initialize containers
    battery_elecs = list()
    nameplate_powers = list()
    grid_elecs = list()
    wind_outputs = list()
    lmp_values = list()
    wind_capacities = list()

    # set power output axis limits
    windpower_design_capacity = pyo.value(
        pyo.units.convert(
            blocks[0].fs.windpower.system_capacity,
            pyo.units.MW,
        )
    )
    ax1.set_ylim(
        [-0.05 * windpower_design_capacity, windpower_design_capacity * 1.05],
    )

    for t, blk in zip(periods, blocks):
        battery_elecs.append(pyo.value(blk.fs.battery.elec_out[0]))
        grid_elecs.append(pyo.value(blk.fs.splitter.grid_elec[0]))
        wind_outputs.append(pyo.value(blk.fs.windpower.electricity[0]))
        lmp_values.append(pyo.value(mp_model._pyomo_model.LMP[t]))
        nameplate_powers.append(pyo.value(blk.fs.battery.nameplate_power))
        wind_capacities.append(
            pyo.value(
                blk.fs.windpower.system_capacity
                * blk.fs.windpower.capacity_factor[0.0]
            )
        )

    battery_elecs.append(battery_elecs[-1])
    grid_elecs.append(grid_elecs[-1])
    wind_outputs.append(wind_outputs[-1])
    nameplate_powers.append(nameplate_powers[-1])
    wind_capacities.append(wind_capacities[-1])

    if periods[periods >= active_start].size > 0:
        ax1.step(
            periods[periods >= active_start],
            np.array(grid_elecs)[periods >= active_start] / 1e3,
            label="wind-to-grid",
            where="post",
            linewidth=1.8,
            color="red",
        )
        ax1.step(
            periods[periods >= active_start],
            np.array(battery_elecs)[periods >= active_start] / 1e3,
            label="battery-to-grid",
            linewidth=1.8,
            where="post",
            color="blue",
        )
        ax1.step(
            periods[periods >= active_start],
            np.array(wind_outputs)[periods >= active_start] / 1e3,
            where="post",
            label="wind production",
            linewidth=1.8,
            color="purple",
        )
        ax1.step(
            periods[periods >= active_start],
            np.array(nameplate_powers)[periods >= active_start] / 1e3,
            where="post",
            label="nameplate power",
            linewidth=1.5,
            color="blue",
            linestyle="dotted",
        )
        ax1.step(
            periods[periods >= active_start],
            np.array(wind_capacities)[periods >= active_start] / 1e3,
            where="post",
            label="wind capacity",
            linewidth=1.5,
            color="purple",
            linestyle="dotted",
        )

    if periods[periods < active_start].size > 0:
        alpha = 0.3 if highlight_active_periods else 1
        ax1.step(
            periods[periods <= active_start],
            np.array(grid_elecs)[periods <= active_start] / 1e3,
            label=(
                "wind-to-grid (prev)"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            where="post",
            linewidth=1.8,
            color="red",
            alpha=alpha,
        )
        ax1.step(
            periods[periods <= active_start],
            np.array(battery_elecs)[periods <= active_start] / 1e3,
            label=(
                "battery-to-grid (prev)"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            linewidth=1.8,
            where="post",
            color="blue",
            alpha=alpha,
        )
        ax1.step(
            periods[periods <= active_start],
            np.array(wind_outputs)[periods <= active_start] / 1e3,
            where="post",
            label=(
                "wind-to-all"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            linewidth=1.8,
            color="purple",
            alpha=alpha,
        )
        ax1.step(
            periods[periods <= active_start],
            np.array(nameplate_powers)[periods <= active_start] / 1e3,
            where="post",
            label=(
                "nameplate power (prev)"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            linewidth=1.5,
            color="blue",
            linestyle="dotted",
            alpha=alpha,
        )
        ax1.step(
            periods[periods <= active_start],
            np.array(wind_capacities)[periods <= active_start] / 1e3,
            where="post",
            label=(
                "wind capacity (prev)"
                if highlight_active_periods and label_inactive_periods
                else None
            ),
            linewidth=1.5,
            color="purple",
            linestyle="dotted",
            alpha=alpha,
        )

    ax1.set_xlabel("Period (hr)")
    ax1.set_ylabel("Power Output (MW)")
    # ax1.set_xticks(model._pyomo_model.Horizon)
    ax1.legend(bbox_to_anchor=(0, -0.15), loc="upper left", ncol=1)

    if plot_lmp:
        ax2 = ax1.twinx()
        ax2.grid(False)
        _plot_lmp(ax2, periods[:-1], lmp_values, lmp_set, active_start,
                  lmp_bounds=lmp_bounds, plot_uncertainty=plot_uncertainty,
                  highlight_active_periods=highlight_active_periods,
                  custom_lmp_vals=custom_lmp_vals)

    # set time interval axis limits
    xmin = start if xmin is None else xmin
    xmax = stop + 1 if xmin is None else xmax
    ax1.set_xlim([xmin, xmax])

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close()


def plot_results(
        mp_model,
        lmp_set=None,
        plot_lmp=True,
        plot_uncertainty=True,
        start=None,
        stop=None,
        highlight_active_periods=False,
        custom_lmp_vals=None,
        output_dir=None,
        lmp_bounds=None,
        xmin=None,
        xmax=None,
        ):
    """
    Plot model power output results
    (wind production, battery-to-grid discharge,
    and wind-to-grid power).

    Arguments
    ---------
    Same as those used for `plot_soc_results`.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        pwr_filename = os.path.join(output_dir, "power_output.png")
        soc_filename = os.path.join(output_dir, "soc.png")
    else:
        pwr_filename = None
        soc_filename = None

    plot_power_output_results(
        mp_model,
        lmp_set=lmp_set,
        plot_lmp=plot_lmp,
        plot_uncertainty=plot_uncertainty,
        custom_lmp_vals=custom_lmp_vals,
        filename=pwr_filename,
        start=start,
        stop=stop,
        highlight_active_periods=highlight_active_periods,
        lmp_bounds=lmp_bounds,
        xmin=xmin,
        xmax=xmax,
    )
    plot_soc_results(
        mp_model,
        lmp_set=lmp_set,
        plot_lmp=plot_lmp,
        plot_uncertainty=plot_uncertainty,
        custom_lmp_vals=custom_lmp_vals,
        filename=soc_filename,
        start=start,
        stop=stop,
        highlight_active_periods=highlight_active_periods,
        lmp_bounds=lmp_bounds,
        xmin=xmin,
        xmax=xmax,
    )


def construct_profit_obj(model, lmp_signal):
    """
    Construct model LMP profit objective.

    Parameters
    ----------
    model : MultiPeriodModel
        Model of interest.
    lmp_signal : (M,) array-like
        Energy prices (LMPs) in $/MWh. The number of entries `M`
        in the array must be equal to the number of process
        blocks in the model.
    """
    horizon = len(lmp_signal)
    pyomo_model = model.pyomo_model

    assert horizon == len(pyomo_model.blocks)

    # reconstruct objective
    attrs_to_del = [
        "profit_obj",
        "total_power_output",
        "operating_cost",
        "Horizon",
        "LMP",
    ]
    for attr in attrs_to_del:
        if hasattr(pyomo_model, attr):
            pyomo_model.del_component(attr)

    pyomo_model.Horizon = pyo.Set(initialize=range(len(lmp_signal)))

    from pyomo.environ import units as u

    # add LMPs
    pyomo_model.LMP = pyo.Param(
        pyomo_model.Horizon,
        initialize=lmp_signal,
        mutable=True,
        doc="Locational marginal prices, $/MWh",
        units=1/pyo.units.MWh,
    )

    @pyomo_model.Expression(pyomo_model.Horizon,
                            doc="Total power output to grid, MW")
    def total_power_output(m, t):
        b = m.blocks[t].process
        return u.convert(
            b.fs.splitter.grid_elec[0] + b.fs.battery.elec_out[0],
            u.MW,
        )

    @pyomo_model.Expression(pyomo_model.Horizon,
                            doc="Total operating cost, $/hr")
    def operating_cost(m, t):
        b = m.blocks[t].process
        return b.fs.windpower.op_total_cost / u.kW / u.h

    @pyomo_model.Objective(doc="Total profit ($)", sense=pyo.maximize)
    def profit_obj(m):
        profit = 0
        for t, blk in pyomo_model.blocks.items():
            dt = blk.process.fs.battery.dt
            revenue = m.LMP[t] * m.total_power_output[t]
            cost = m.operating_cost[t]
            profit += (revenue - cost) * dt
        return profit


def get_dof_vars(model, nested=True):
    """
    Obtain first-stage and second-stage degrees of freedom
    for all active model blocks.

    Optionally, compile the second-stage variables into a
    nested list of Vars, in which each inner list contains
    the variables corresponding to a corresponding stage of
    decision making (for multi-stage RO).
    """
    active_blks = model.get_active_process_blocks()

    first_stage_vars = [
        list(active_blks)[0].fs.splitter.grid_elec[0],
        list(active_blks)[0].fs.battery.elec_out[0],
        list(active_blks)[0].fs.windpower.electricity[0.0],
    ]

    grid_vars = list()
    battery_vars = list()
    wind_vars = list()

    second_stage_vars = list()
    for blk in list(active_blks)[1:]:
        grid_vars.append(blk.fs.splitter.grid_elec[0])
        battery_vars.append(blk.fs.battery.elec_out[0])
        wind_vars.append(blk.fs.windpower.electricity[0])

    if nested:
        second_stage_vars = list(
            [gvar, bvar, wvar]
            for gvar, bvar, wvar
            in zip(grid_vars, battery_vars, wind_vars)
        )
    else:
        second_stage_vars = grid_vars + battery_vars + wind_vars

    return first_stage_vars, second_stage_vars


def get_uncertain_params(model, lmp_set, include_fixed_dims=True, nested=True):
    """
    Obtain uncertain parameters for all active model blocks.
    Optionally, include dimensions which are implicitly 'fixed'
    by the uncertainty set parameter bounds.
    """
    start_time = model.current_time
    lmp_params = list(
        val for t, val in model.pyomo_model.LMP.items() if t >= start_time
    )
    uncertain_params = lmp_set.get_uncertain_params(
        lmp_params,
        include_fixed_dims=include_fixed_dims,
        nested=nested,
    )

    if nested:
        # lump the first and second lists into a single list,
        # as the uncertain parameters in these lists are realized
        # by the true second stage
        uncertain_params[0] += uncertain_params[1]
        uncertain_params.remove(uncertain_params[1])

    return uncertain_params


def evaluate_objective(
        mp_model,
        start=None,
        stop=None,
        lmp_signal=None,
        ):
    """
    Evaluate model objective and related quantities.
    In particular, we evaluate the operating revenue, cost, and
    profit.

    Parameters
    ----------
    mp_model : MultiPeriodModel
        Wind-battery model of interest.
    start : int, optional
        Index of first block to include from the list
        `list(model.pyomo_model.blocks.values())`.
        The default is `None`, in which case the index
        of the first active process block is used.
    stop : int, optional
        Index of last block to include block from the list
        `list(model.pyomo_model.blocks.values())`.
        The default is `None`, in which case
        the index of the last active process block is used.
    lmp_signal : array-like, optional
        LMP values ($/MWh) against which to evaluate the
        objective. The default is `None`, in which case the
        values are taken from `model.pyomo_model.LMP`.
        If an array is provided, then the array must contain
        `stop - start` entries.

    Returns
    -------
    : tuple
        A 3-tuple consisting of the total revenue, operating
        cost, and profit (revenue - cost), in that order, in
        dollars ($).
    """
    start = mp_model.current_time if start is None else start
    stop = (
        start + len(list(mp_model.get_active_process_blocks()))
        if stop is None else stop
    )
    assert start >= 0
    assert stop <= len(mp_model.pyomo_model.blocks)
    assert start <= stop

    blocks = list(
        blk.process
        for idx, blk in enumerate(mp_model.pyomo_model.blocks.values())
        if idx >= start and idx <= stop
    )

    if lmp_signal is None:
        lmp_signal = list(
            pyo.value(lmp)
            for lmp in
            list(mp_model.pyomo_model.LMP.values())[start:stop + 1]
        )

    assert len(lmp_signal) == len(blocks)

    revenue = 0
    cost = 0
    for lmp_val, pblk in zip(lmp_signal, blocks):
        total_power_out = pyo.units.convert(
            pblk.fs.battery.elec_out[0] + pblk.fs.splitter.grid_elec[0],
            pyo.units.MW,
        )
        interval_time = pyo.units.convert(
            pblk.fs.battery.dt,
            pyo.units.h,
        )
        revenue += pyo.value(total_power_out * lmp_val * interval_time)
        cost += pyo.value(pblk.fs.windpower.op_total_cost * interval_time)

    return (revenue, cost, revenue - cost)


def _exclude_energy_throughputs(model):
    """
    Exclude energy throughputs from a wind-battery model
    process block, by (1) fixing energy throughputs to 0;
    (2) deactivating constraints containing only throughput vars.
    """
    active_blks = model.get_active_process_blocks()
    throughput_vars = ComponentSet(
        blk.fs.battery.energy_throughput[0.0]
        for blk in active_blks
    )
    init_throughput_vars = ComponentSet(
        blk.fs.battery.initial_energy_throughput
        for blk in active_blks
    )

    for var in throughput_vars | init_throughput_vars:
        var.fix(0)

    # deactivate energy throughput update
    for blk in active_blks:
        blk.fs.battery.accumulate_energy_throughput.deactivate()

    # deactivate energy throughput link constraint
    for con in model.pyomo_model.component_data_objects(
            pyo.Constraint, active=True):
        vars_in_con = ComponentSet(identify_variables(con.body))
        is_subset = all(
            var in
            throughput_vars | init_throughput_vars
            for var in vars_in_con
        )

        if is_subset:
            con.deactivate()


def _simplify_battery_power_limits(model):
    """
    Simplify inequality constraints limiting ramping of battery
    charging levels and power flows through the battery.
    """
    active_blks = model.get_active_process_blocks()

    for blk in active_blks:
        # expressions for state of charge rampimg limits
        # inferred from the charging balance constraints
        inferred_ramp_lim_exprs = {
            "in": (
                blk.fs.battery.nameplate_power
                * blk.fs.battery.charging_eta
                * blk.fs.battery.dt
            ),
            "out": (
                blk.fs.battery.nameplate_power
                / blk.fs.battery.discharging_eta
                * blk.fs.battery.dt
            ),
        }

        # determine which expression is valid
        ramp_lim_key = max(
            inferred_ramp_lim_exprs,
            key=lambda x: pyo.value(inferred_ramp_lim_exprs[x]),
        )
        inferred_ramp_lim_expr = inferred_ramp_lim_exprs[ramp_lim_key]

        # tighten ramping limit constraints, as necessary
        ramp_cons = [
            blk.fs.battery.energy_up_ramp,
            blk.fs.battery.energy_down_ramp,
        ]
        for con in ramp_cons:
            # we assume constraint written as expression <= ub
            # and the constraint hasn't already been simplified
            # is_changing_con_ok = (
            #     con.lower is None
            #     and con.upper is not None
            #     and inferred_ramp_lim_expr not in ComponentSet(con.body.args)
            # )
            # assert is_changing_con_ok

            soc_term = ComponentSet()
            initial_soc_term = ComponentSet()
            additional_terms = ComponentSet()
            for arg in con.body.args:
                vars_in_arg = ComponentSet(identify_variables(arg))
                is_soc_term = (
                    blk.fs.battery.state_of_charge[0.0] in vars_in_arg
                    and len(vars_in_arg) == 1
                )
                if is_soc_term and not soc_term:
                    soc_term.add(arg)
                is_initial_soc_term = (
                    blk.fs.battery.initial_state_of_charge in vars_in_arg
                    and len(vars_in_arg) == 1
                )
                if is_initial_soc_term and not initial_soc_term:
                    initial_soc_term.add(arg)
                if not is_initial_soc_term and not is_soc_term:
                    additional_terms.add(arg)

            # these two terms must be in the ramping constraint.
            # constitute difference between current and previous
            # charging levels
            assert soc_term and initial_soc_term

            ramp_lim_ineq_ub = con.upper - sum(additional_terms)

            # tighten ramping limit to inferred value, if necessary
            # note: this may rewrite as expression - inferred_lim <= 0
            #       for this reason, use this expression only once
            #       (see above assertion)
            if pyo.value(ramp_lim_ineq_ub) > pyo.value(inferred_ramp_lim_expr):
                # con.set_value((con.lower, con.body, inferred_ramp_lim_expr))
                con.deactivate()


def create_two_stg_wind_battery_model(
        lmp_signal,
        wind_cfs,
        wind_capacity,
        battery_capacity,
        exclude_energy_throughputs=False,
        simplify_battery_power_limits=False,
        charging_eta=None,
        discharging_eta=None,
        ):
    """
    Create a multiperiod wind-battery model.

    Parameters
    ----------
    lmp_signal : array-like
        LMP signal, of which the number of entries is equal
        to the model's prediction horizon length.
    wind_cfs : dict
        Wind production capacity factors for each period.
        Maps period indices to wind capacity factors.
    wind_capacity : float
        Wind production system capacity (design capacity)
        in MW.
    battery_capacity : float
        Battery storage capacity in MW.
    exclude_energy_throughputs : bool, optional
        Exclude net energy throughputs from model.
        (By fixing values to zero and deactivating energy
        throughput update constraints).
    simplify_battery_power_limits : bool, optional
        Simplify constraints limiting power flows through
        the battery (this is meant to make RO with PyROS
        more efficient).
    charging_eta : float or None, optional
        Battery charging efficiency. Must be a value in (0, 1].
    discharging_eta : float or None, optional
        Battery discharging efficiency. Must be a value in (0, 1].

    Returns
    -------
    model : MultiPeriodModel
        Multi-period wind battery model, equipped with an
        economic objective.
    """
    horizon = len(lmp_signal)

    model = create_multiperiod_wind_battery_model(
        n_time_points=horizon,
        wind_cfs=wind_cfs,
        input_params={
            "wind_mw": wind_capacity,
            "wind_mw_ub": wind_capacity,
            "batt_mw": battery_capacity,
        },
    )
    transform_design_model_to_operation_model(
        model,
        wind_capacity=wind_capacity * 1e3,
        battery_power_capacity=battery_capacity * 1e3,
    )
    update_wind_capacity_factors(model, wind_cfs)

    # set battery charging efficiencies
    for blk in model.get_active_process_blocks():
        if charging_eta is not None:
            blk.fs.battery.charging_eta.set_value(charging_eta)
        if discharging_eta is not None:
            blk.fs.battery.discharging_eta.set_value(discharging_eta)

    if exclude_energy_throughputs:
        _exclude_energy_throughputs(model)
    if simplify_battery_power_limits:
        _simplify_battery_power_limits(model)

    # get the pyomo model
    pyomo_model = model.pyomo_model

    # deactivate all objectives
    for obj in pyomo_model.component_objects(pyo.Objective):
        obj.deactivate()

    construct_profit_obj(model, lmp_signal)

    # pyomo_model.blocks[0].process.fs.battery.state_of_charge[0.0].fix()
    pyomo_model.blocks[0].process.fs.battery.initial_energy_throughput.fix()

    # deactivate any constraints imposed on fixed variables
    for con in pyomo_model.component_data_objects(pyo.Constraint, active=True):
        if all(var.fixed for var in identify_variables(con.body)):
            con.deactivate()

    return model


def advance_time(
        model,
        forecaster,
        battery_power_capacity=25,
        battery_energy_capacity=100,
        exclude_energy_throughputs=False,
        simplify_battery_power_limits=False,
        charging_eta=None,
        discharging_eta=None,
        ):
    """
    Advance multi-period wind-battery model to next time period,
    and construct an updated uncertainty set.

    Parameters
    ----------
    model : MultiPeriodModel
        Wind-battery model of interest.
    new_lmp_sig : array-like
        LMP signal values for next prediction horizon.
        The number of entries must be equal to the
        model's prediction horizon length (number of active blocks).
    lmp_set_class : array-like, optional
        Constructor for LMP uncertainty set. This is used to update
        the uncertainty set in tandem with the active process
        blocks.
    lmp_set_class_params : dict, optional
        Keyword arguments to the LMP uncertainty set constructor.
    wind_capacity : float, optional
        Wind production capacity (in MWh).
    battery_power_capacity : float, optional
        Battery power capacity (in MW).
    battery_energy_capacity : float, optional
        Battery energy capacity (in MWh).
    """
    prediction_length = len(model.get_active_process_blocks())

    # advance the forecaster
    forecaster.advance_time()

    # LMP for current model time now known.
    # update to actual (most recent historical) value
    model.pyomo_model.LMP[model.current_time] = (
        forecaster.historical_energy_prices(forecaster.current_time - 1)
    )[0]

    # TODO: do we need to do same for wind capacity factor?
    # what if actual capacity is lower than forecast?
    # (solution may be infeasible ...?)

    # get new wind capacity factors
    wind_cfs = forecaster.forecast_wind_production(
        num_intervals=prediction_length,
        capacity_factors=True,
    )
    new_wind_resource = {
        "wind_resource_config": {"capacity_factor": {0.0: wind_cfs}}
    }

    # get initial block
    block_init = model.get_active_process_blocks()[0]

    # fix variables, deactivate constraints on this block
    for var in block_init.component_data_objects(pyo.Var):
        var.fix()
    for con in block_init.component_data_objects(pyo.Constraint, active=True):
        con.deactivate()

    # now advance time (to obtain new process block)
    # also updates wind capacity factors
    model.advance_time(**new_wind_resource)

    update_wind_capacity_factors(model, wind_cfs)

    # retrieve new block
    b = model.get_active_process_blocks()[-1]

    # transform latest block to operation mode
    # note: forecaster wind capacity converted from MW to kW
    b.fs.windpower.system_capacity.fix(forecaster.wind_capacity * 1e3)
    b.fs.battery.nameplate_power.fix(battery_power_capacity * 1e3)
    b.fs.battery.nameplate_energy.fix(battery_energy_capacity * 1e3)
    b.periodic_constraints[0].deactivate()

    # simplifications to constraints on battery power flows/levels
    if exclude_energy_throughputs:
        _exclude_energy_throughputs(model)
    if simplify_battery_power_limits:
        _simplify_battery_power_limits(model)

    if charging_eta is not None:
        b.fs.battery.charging_eta.set_value(charging_eta)
    if discharging_eta is not None:
        b.fs.battery.discharging_eta.set_value(discharging_eta)

    # fix initial state of charge and throughput to values
    # from previous period
    b_init = model.get_active_process_blocks()[0]
    b_init.fs.battery.initial_state_of_charge.fix()
    b_init.fs.battery.initial_energy_throughput.fix()

    prediction_length = len(model.get_active_process_blocks())

    # update LMP signal and model objective
    lmp_update_sig = forecaster.forecast_energy_prices(
        num_intervals=prediction_length,
    )
    lmp_sig = {
        t: pyo.value(model.pyomo_model.LMP[t])
        for t in range(model.current_time + 1)
    }
    lmp_sig.update({
        t + model.current_time: lmp_update_sig[t]
        for t in range(len(lmp_update_sig))
    })
    construct_profit_obj(model, lmp_sig)

    # deactivate constraints in fixed variables
    m = model.pyomo_model
    for con in m.component_data_objects(pyo.Constraint, active=True):
        if all(var.fixed for var in identify_variables(con.body)):
            con.deactivate()


def update_wind_capacity_factors(model, new_factors):
    """
    Update wind production capacity factor values for active
    model process blocks.

    Parameters
    ----------
    new_factors : list(float)
        Wind production capacity factors.
    """
    assert len(model.get_active_process_blocks()) == len(new_factors)
    for blk, cf in zip(model.get_active_process_blocks(), new_factors):
        blk.fs.windpower.capacity_factor[0].set_value(cf)


def solve_rolling_horizon(
        model,
        forecaster,
        solver,
        control_length,
        num_steps,
        output_dir=None,
        solver_kwargs=None,
        charging_eta=None,
        discharging_eta=None,
        exclude_energy_throughputs=False,
        simplify_battery_power_limits=False,
        simplify_uncertainty_set=False,
        ):
    """
    Solve a multi-period wind-battery model on a rolling horizon.

    Parameters
    ----------
    model : MultiPeriodModel
        Model of interest.
    forecaster : uncertainty_models.Forecaster
        Predictor for LMPs and wind production capacities
        and the uncertainty in these forecasts.
    solver : pyomo SolverFactory object
        Optimizer used for the Pyomo model.
        PyROS (`pyomo.environ.SolverFactory('pyros')`) may be
        provided as a two-stage RO solver, but in this case,
        an uncertainty set class must be provided (through the
        `lmp_set_class` argument) as well.
    control_length : int
        Control horizon length, i.e. number of periods
        for which the optimal settings in each period
        are actually used.
    num_steps : int
        Number of prediction horizons for which to solve the model.
        (I.e., the number of times to update the active
        time periods/blocks.)
    output_dir : path-like or None, optional
        Path to directory to which output plots and results
        produced for each time step.
        If `None` is provided, then no plots
        are produced. If a path is provided, plots are generated,
        and exported to this directory.
    solver_kwargs : dict, optional
        Keyword arguments to the method `solver.solve()`.
    charging_eta : float or None, optional
        Battery charging efficiency. The charging efficiency
        for every process block is set to this value (if a float is
        provided). If `None` is provided, then the charging
        efficiencies are not set.
    discharging_eta : float or None, optional
        Battery discharging efficiency. The charging efficiency
        for every process block is set to this value (if a float is
        provided). If `None` is provided, then the charging
        efficiencies are not set.
    exclude_energy_throughputs : bool, optional
        Exclude battery energy throughputs from the model,
        by fixing all to 0 and deactivating the throughput
        update constraints and constraints linking throughputs
        across periods. The default is `False`.
    simplify_battery_power_limits : bool, optional
        Simplify constraints limiting power flows through
        the battery.
    simplify_uncertainty_set : bool, optional
        Simplify LMP uncertainty set by including only the
        dimensions for which the bounds are unequal in the PyROS
        solver calls. The default is `False`.

    Returns
    -------
    accumul_results_df : pandas.DataFrame
        A `DataFrame` consisting of the realized energy price ($/MWh),
        along with the operating revenue, cost, and profit (all in $)
        for every time period adapted.
    """
    # cannot control beyond prediction horizon
    prediction_length = len(model.get_active_process_blocks())
    assert prediction_length >= control_length

    solver_kwargs = dict() if solver_kwargs is None else solver_kwargs

    # validate battery charging and discharging efficiencies
    if charging_eta is not None:
        assert charging_eta > 0 and charging_eta <= 1
    if discharging_eta is not None:
        assert discharging_eta > 0 and discharging_eta <= 1

    # solving RO model? if so, validate keyword args
    is_pyros = isinstance(solver, type(pyo.SolverFactory("pyros")))
    if is_pyros:
        prereq_args = {"local_solver", "global_solver"}
        antireq_args = {
            "model",
            "first_stage_variables",
            "second_stage_variables",
            "uncertain_params",
            "uncertainty_set",
        }
        kwargs_set = set(solver_kwargs.keys())

        assert prereq_args.issubset(kwargs_set)
        assert not kwargs_set.intersection(antireq_args)
    else:
        assert "load_solutions" not in solver_kwargs

    # get LMP axis bounds for plotting
    if output_dir is not None:
        if forecaster.lmp_set_class is not None:
            temp_forecaster = forecaster.copy()

            lmp_sets = []
            for idx in range(0, num_steps):
                lmp_sets.append(
                    temp_forecaster.forecast_price_uncertainty(
                        num_intervals=prediction_length,
                    )
                )
                for tm in range(control_length):
                    temp_forecaster.advance_time()

            max_ubs = list()
            min_lbs = list()
            for price_set in lmp_sets:
                lower_bounds = [
                    bd[0] for bd in price_set.pyros_set().parameter_bounds
                ]
                upper_bounds = [
                    bd[1] for bd in price_set.pyros_set().parameter_bounds
                ]
                max_ubs.append(max(upper_bounds))
                min_lbs.append(min(lower_bounds))
            lmp_bounds = (
                min(min_lbs) - 5,
                max(max_ubs) + 5,
            )

    m_idx = pd.MultiIndex.from_tuples(
        (step, forecaster.current_time + step * control_length + subperiod)
        for step in range(num_steps)
        for subperiod in range(control_length)
    )
    m_idx.names = ["step", "forecaster period no."]
    accumul_results_df = pd.DataFrame(
        index=m_idx,
        columns=[
            "Wind Production (MW)",
            "Wind-to-Grid (MW)",
            "Battery-to-Grid (MW)",
            "Battery storage (MWh)",
            "Energy Price ($/MWh)",
            "Revenue ($)",
            "Cost ($)",
            "Profit ($)",
        ],
    )

    for idx in range(num_steps):
        # solve the model
        if is_pyros:
            first_stage_vars, second_stage_vars = get_dof_vars(
                model,
                nested=True,
            )
            lmp_params = list(
                model.pyomo_model.LMP[t]
                for t in range(
                    model.current_time, model.current_time + prediction_length
                )
            )
            wind_params = list(
                blk.fs.windpower.capacity_factor[0]
                for blk in model.get_active_process_blocks()
            )
            uncertain_params = forecaster.get_uncertain_params(
                lmp_params,
                wind_params,
                include_fixed_dims=not simplify_uncertainty_set,
                nested=True,
            )
            uncertainty_set = forecaster.get_joint_lmp_wind_pyros_set(
                num_intervals=len(lmp_params),
                include_fixed_dims=not simplify_uncertainty_set,
                capacity_factors=True,
            )

            res = solver.solve(
                model.pyomo_model,
                first_stage_vars,
                second_stage_vars,
                uncertain_params,
                uncertainty_set,
                **solver_kwargs,
            )
        else:
            res = solver.solve(
                model.pyomo_model,
                load_solutions=False,
                **solver_kwargs,
            )

        if pyo.check_optimal_termination(res):
            model.pyomo_model.solutions.load_from(res)
        else:
            raise RuntimeError(
                f"Model at step {idx} was not successfully solved"
                f" to optimality. Solver results: {res.solver}"
                "\nTerminating rolling horizon optimization."
            )

        # EVALUATE AND DISPLAY revenue, cost, profit:
        # (1) accumulated up to + including current time
        # (2) obtained/incurred for next control period
        #     (starting from + including current time)
        #     for a control horizon of length 1, this is just
        #     current time
        # (3) projected for just solved prediction horizon,
        #     starting from and including current time
        controlled_periods = list(
            range(idx * control_length, (idx + 1) * control_length),
        )
        print("=" * 80)
        print(f"Step {idx} (decision for periods {controlled_periods})")
        accumulated_obj_tuple = evaluate_objective(
            model,
            start=0,
            stop=model.current_time,
        )
        next_ctrl_obj_tuple = evaluate_objective(
            model,
            start=model.current_time,
            stop=model.current_time + control_length - 1,
        )
        projected_obj_tuple = evaluate_objective(
            model,
            start=model.current_time,
            stop=model.current_time + prediction_length,
        )
        print(f"{'':13s}{'revenue':10s}{'cost':10s}{'profit':10s}")
        acc_val_str = "".join(f"{val:<10.2f}" for val in accumulated_obj_tuple)
        ctrl_val_str = "".join(f"{val:<10.2f}" for val in next_ctrl_obj_tuple)
        proj_val_str = "".join(f"{val:<10.2f}" for val in projected_obj_tuple)
        print(f"{'accumulated':13s}{acc_val_str}")
        print(f"{'next ctrl':13s}{ctrl_val_str}")
        print(f"{'projected':13s}{proj_val_str}")

        if output_dir is not None:
            # obtain worst-case LMP, plot as 'custom' LMP signal.
            # this is only provided in the event that PyROS is used
            worst_case_params = getattr(
                res.solver,
                "worst_case_param_realization",
                None,
            )
            if worst_case_params is not None:
                # if LMP uncertainty set was simplified, need
                # to restore the nominal LMP values for the fixed
                # dimensions
                worst_case_lmp = [
                    p for pname, p in worst_case_params.items()
                    if "LMP" in pname
                ]
                lmp_set = forecaster.forecast_price_uncertainty(
                    num_intervals=prediction_length,
                )
                if simplify_uncertainty_set:
                    worst_case_lmp = lmp_set.lift_uncertain_params(
                        worst_case_lmp,
                        {
                            idx: val
                            for idx, val in enumerate(lmp_set.sig_nom)
                            if idx in lmp_set.determine_fixed_dims()
                        }
                    )

                active_periods = range(
                    model.current_time, model.current_time + prediction_length
                )
                custom_lmp_vals = {
                    "worst case": {
                        per: val
                        for per, val in zip(active_periods, worst_case_lmp)
                    }
                }
            else:
                custom_lmp_vals = None

            plot_results(
                model,
                highlight_active_periods=True,
                lmp_set=forecaster.forecast_price_uncertainty(
                    num_intervals=prediction_length,
                ),
                output_dir=os.path.join(output_dir, f"step_{idx}"),
                lmp_bounds=lmp_bounds,
                start=0,
                stop=model.current_time + prediction_length - 1,
                xmax=(num_steps - 1) * control_length + prediction_length,
                plot_uncertainty=is_pyros,
                custom_lmp_vals=custom_lmp_vals,
            )

        # advance the model and forecaster in time,
        # extend LMP signal,
        # and update the uncertainty set
        for step in range(control_length):
            advance_time(
                model,
                forecaster,
                exclude_energy_throughputs=exclude_energy_throughputs,
                simplify_battery_power_limits=(
                    simplify_battery_power_limits
                ),
                charging_eta=charging_eta,
                discharging_eta=discharging_eta,
            )

            # get model solution components for period just passed
            prev_model_time = model.current_time - 1
            prev_blk = (
                model.pyomo_model.blocks[prev_model_time].process.fs
            )
            wind_production = pyo.value(
                pyo.units.convert(
                    prev_blk.windpower.electricity[0],
                    pyo.units.MW,
                )
            )
            wind_to_grid = pyo.value(
                pyo.units.convert(
                    prev_blk.splitter.grid_elec[0],
                    pyo.units.MW,
                )
            )
            batt_to_grid = pyo.value(
                pyo.units.convert(
                    prev_blk.battery.elec_out[0],
                    pyo.units.MW,
                )
            )
            batt_charge = pyo.value(
                pyo.units.convert(
                    prev_blk.battery.state_of_charge[0],
                    pyo.units.MWh,
                )
            )
            lmp_value = pyo.value(
                pyo.units.convert(
                    model.pyomo_model.LMP[prev_model_time],
                    1/pyo.units.MWh,
                )
            )
            revenue, cost, profit = evaluate_objective(
                model,
                start=prev_model_time,
                stop=prev_model_time,
            )

            # log solution results to dataframe
            accumul_results_df.loc[idx, forecaster.current_time - 1] = (
                wind_production,
                wind_to_grid,
                batt_to_grid,
                batt_charge,
                lmp_value,
                revenue,
                cost,
                profit,
            )

        if output_dir is not None:
            accumul_results_df.to_csv(os.path.join(output_dir, "revenues.csv"))

    return accumul_results_df


def perform_incidence_analysis(model):
    """
    Perform structural and numerical analysis of the model's equality
    constraints.
    """
    m = model.pyomo_model
    first_stage_vars, second_stage_vars = get_dof_vars(model)

    eqns = ComponentSet(
        con
        for con in m.component_data_objects(
            pyo.Constraint,
            active=True, descend_into=True)
        if con.equality
        and not all(var.fixed for var in identify_variables(con.body))
    )
    state_vars = ComponentSet(
        var
        for con in m.component_data_objects(
            pyo.Constraint,
            active=True, descend_into=True)
        for var in identify_variables(con.body)
        if var not in ComponentSet(first_stage_vars)
        and var not in ComponentSet(second_stage_vars)
        and not var.fixed
        and con.equality
        and con.active
    )
    all_vars = list(
        var for var in m.component_data_objects(pyo.Var)
        if not var.fixed
    )

    print("Number all vars", len(all_vars))
    print("Number first-stage vars", len(first_stage_vars))
    print("Number second-stage vars", len(second_stage_vars))
    print("Number state vars", len(state_vars))
    print("Number equations", len(eqns))
    print("All first-stage vars unfixed",
          all(not var.fixed for var in first_stage_vars))
    print("All second-stage vars unfixed",
          all(not var.fixed for var in second_stage_vars))

    igraph = IncidenceGraphInterface()

    matching = igraph.maximum_matching(
        variables=list(state_vars),
        constraints=list(eqns),
    )
    print("Maximum matching length", len(matching))
    nlp = PyomoNLP(m)
    nlp.n_primals()
    jac = nlp.extract_submatrix_jacobian(list(state_vars), list(eqns))
    # splu(jac.tocsc())
    jacarr = jac.toarray()
    print("Jacobian determinant", det(jacarr))


def main():
    """
    Script for running the present module if main.
    """
    import sys
    from dispatches.models.renewables_case.uncertainty_models.\
        lmp_uncertainty_models import (
            CustomBoundsLMPBoxSet,
            ConstantFractionalUncertaintyBoxSet,
        )
    from dispatches.models.renewables_case.uncertainty_models.\
        forecaster import Perfect309Forecaster, AvgSample309Backcaster

    # horizon lengths
    horizon = 12
    num_steps = 24
    control_length = 1

    # model settings
    charging_eff = 0.95
    excl_throughputs = True
    simplify_battery_power_limits = True
    simplify_uncertainty_set = True

    # settings for modifying dataset and forecasting
    start = 2000
    lmp_incr_frac = 0
    perfect_information = False
    first_period_certain = True
    use_fractional_box_set = False
    fractional_uncertainty = 0.2

    # pyros solver setting
    dr_order = 1

    # configure system
    logging.basicConfig(level=logging.INFO)
    sys.setrecursionlimit(15000)

    # determine forecaster class
    if perfect_information:
        forecaster_class = Perfect309Forecaster
    else:
        forecaster_class = AvgSample309Backcaster

    def frac_to_string(frac, prefix="", postfix=""):
        """
        Convert fractional floating point value to string
        of form '{prefix}{sign}_0pt{decimal_places}{postfix}', where
        {sign} is 'pl' or 'mn' (depending on sign of fraction).
        """
        assert -1 < frac
        assert frac < 1

        if frac == 0:
            increment_str = ""
        else:
            if frac > 0:
                sign_str = "pl_"
            else:
                sign_str = "mn_"

            # remove +/- sign. Replace decimal point with 'pt'
            num_str = (
                str(frac)
                .replace("-", "")
                .replace("+", "")
                .replace(".", "pt")
            )

            # now put everything together
            increment_str = prefix + sign_str + num_str + postfix

        return increment_str

    def create_wind_lmp_dataset(base_dataset_file_path, lmp_increment_frac):
        """
        Create new wind/data LMP dataset from file and return
        path to that file. If `lmp_increment_frac` is 0,
        then no new file is created, and `base_dataset_file_path`
        is returned.
        """

        if lmp_incr_frac == 0:
            # don't need to create new dataset
            new_df_path = base_dataset_file_path
        else:
            # create new dataset file path
            increment_str = frac_to_string(lmp_increment_frac, prefix="_")
            new_df_path = (
                base_dataset_file_path.split(".csv")[0]
                + increment_str
                + ".csv"
            )

            # open base dataset, copy, and create LMPs
            df = pd.read_csv(base_dataset_file_path)
            df["LMP DA"] += lmp_increment_frac * abs(df["LMP DA"])
            df.to_csv(new_df_path)

            logging.info(
                "Successfully wrote new dataset to spreadsheet "
                f"{new_df_path}"
            )

        return new_df_path

    dataset_path = create_wind_lmp_dataset(
        "../../../../results/wind_profile_data/309_wind_1_profiles.csv",
        lmp_incr_frac,
    )

    # set up backcaster for wind and LMP uncertainty
    lmp_set_class_params = {"first_period_certain": first_period_certain}
    if use_fractional_box_set:
        lmp_set_class = ConstantFractionalUncertaintyBoxSet
        lmp_set_class_params.update({
            "fractional_uncertainty": fractional_uncertainty,
        })
        lmp_set_frac_str = frac_to_string(fractional_uncertainty)
        lmp_set_qualifier = f"_frac_uncert_{lmp_set_frac_str}"
    else:
        lmp_set_class = CustomBoundsLMPBoxSet
        lmp_set_qualifier = ""

    wind_set_class = None
    backcaster = forecaster_class(
        dataset_path,
        n_prev_days=7,
        lmp_set_class=lmp_set_class,
        wind_set_class=wind_set_class,
        lmp_set_class_params=lmp_set_class_params,
        wind_capacity=148.3,
        start=start,
    )
    ro_backcaster = backcaster.copy()

    # make directory for storing results
    actual_offset_frac_str = frac_to_string(lmp_incr_frac, prefix="_")
    base_dir = (
        f"../../../../results/new_wind_lmp_results/"
        f"hor_{horizon}_start_{start}_steps_{num_steps}/"
        f"{backcaster.__class__.__name__}{actual_offset_frac_str}"
        f"{lmp_set_qualifier}"
    )
    os.makedirs(base_dir, exist_ok=True)

    # set up solvers
    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    gams_baron = pyo.SolverFactory("gams")
    gams_baron.options["solver"] = "baron"
    couenne = pyo.SolverFactory("couenne")

    # generate initial LMP and wind production level values
    # instantiate the model
    lmp_signal = backcaster.forecast_energy_prices(
        num_intervals=horizon,
    )
    wind_cfs = backcaster.forecast_wind_production(
        num_intervals=horizon,
        capacity_factors=True,
    )
    model = create_two_stg_wind_battery_model(
        lmp_signal=lmp_signal,
        wind_cfs=wind_cfs,
        wind_capacity=backcaster.wind_capacity,
        battery_capacity=25,
        exclude_energy_throughputs=excl_throughputs,
        simplify_battery_power_limits=simplify_battery_power_limits,
        charging_eta=charging_eff,
        discharging_eta=charging_eff,
    )

    # perform rolling horizon simulation of deterministic model
    solve_rolling_horizon(
        model=model,
        forecaster=backcaster,
        solver=solver,
        control_length=control_length,
        num_steps=num_steps,
        output_dir=os.path.join(
            base_dir,
            "rolling_horizon_deterministic",
        ),
        charging_eta=charging_eff,
        discharging_eta=charging_eff,
        exclude_energy_throughputs=excl_throughputs,
        simplify_battery_power_limits=simplify_battery_power_limits,
        simplify_uncertainty_set=simplify_uncertainty_set,
    )

    pdb.set_trace()

    # set up RO model
    mdl = create_two_stg_wind_battery_model(
        lmp_signal=lmp_signal,
        wind_cfs=wind_cfs,
        wind_capacity=ro_backcaster.wind_capacity,
        battery_capacity=25,
        exclude_energy_throughputs=excl_throughputs,
        simplify_battery_power_limits=simplify_battery_power_limits,
        charging_eta=charging_eff,
        discharging_eta=charging_eff,
    )

    # set up PyROS solver and solver options
    pyros_solver = pyo.SolverFactory("pyros")
    pyros_kwargs = dict(
        local_solver=solver,
        global_solver=solver,
        backup_local_solvers=[gams_baron, couenne],
        backup_global_solvers=[gams_baron, couenne],
        decision_rule_order=dr_order,
        bypass_local_separation=True,
        objective_focus=pyros.ObjectiveType.worst_case,
        solve_master_globally=True,
        keepfiles=True,
        load_solution=True,
        output_verbose_results=True,
        subproblem_file_directory=os.path.join(base_dir, "pyros_sublogs"),
    )

    # rolling horizon simulation of RO model
    solve_rolling_horizon(
        model=mdl,
        forecaster=ro_backcaster,
        solver=pyros_solver,
        control_length=control_length,
        num_steps=num_steps,
        output_dir=os.path.join(base_dir, f"rolling_horizon_ro_dr_{dr_order}"),
        charging_eta=charging_eff,
        discharging_eta=charging_eff,
        exclude_energy_throughputs=excl_throughputs,
        simplify_battery_power_limits=simplify_battery_power_limits,
        simplify_uncertainty_set=simplify_uncertainty_set,
        solver_kwargs=pyros_kwargs,
    )
    pdb.set_trace()


if __name__ == "__main__":
    main()
