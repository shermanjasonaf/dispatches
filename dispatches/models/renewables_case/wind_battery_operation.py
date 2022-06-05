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
from lmp_uncertainty_models.lmp_uncertainty_models import (
        get_lmp_data, HysterLMPBoxSet
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
        custom_lmp_vals=None,
        lmp_bounds=None,
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
        labels, the values are array-like objects whose entries are
        the LMP signal values (in $/MWh).
    lmp_bounds : tuple, optional
        A 2-tuple of bounds to be used for the LMP signal plot axis.
        The default is `None`, in which case the greatest upper bound
        and smallest lower bound from the LMP uncertainty set
        are used for the axis bounds
        (if an uncertainty set is provided).
    """
    active_blks = mp_model.get_active_process_blocks()

    fig, ax1 = plt.subplots(figsize=(9, 8))
    ax1.grid(False)

    # get model blocks of interest
    start = model.current_time if start is None else start
    stop = model.current_time + len(active_blks) - 1 if stop is None else stop

    assert start >= 0
    assert stop < len(mp_model.pyomo_model.blocks)

    if lmp_bounds is not None:
        assert lmp_bounds[0] < lmp_bounds[1]

    periods = np.array(range(start, stop + 1), dtype=int)
    active_start = model.current_time
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
        charges.append(pyo.value(blk.fs.battery.state_of_charge[0.0]))
        lmp_values.append(pyo.value(mp_model.pyomo_model.LMP[t]))

        charge_lim = pyo.value(
            blk.fs.battery.nameplate_energy
            - blk.fs.battery.degradation_rate
            * blk.fs.battery.energy_throughput[0.0]
        )
        charge_bounds.append(charge_lim)

    # plot results for active periods
    if periods[periods >= active_start].size > 0:
        active_qual = " (active)" if highlight_active_periods else ""
        ax1.plot(
            periods[periods >= active_start],
            np.array(charges)[periods >= active_start] / 1e3,
            label="state of charge" + active_qual,
            linewidth=1.8,
            color="blue"
        )
        ax1.plot(
            periods[periods >= active_start],
            np.array(charge_bounds)[periods >= active_start] / 1e3,
            label="charging limit" + active_qual,
            linewidth=1.8,
            color="red",
            linestyle="dashed",
        )

    # plot results for inactive periods
    if periods[periods < active_start].size > 0:
        alpha = 0.3 if highlight_active_periods else 1
        ax1.plot(
            periods[periods <= active_start],
            np.array(charges)[periods <= active_start] / 1e3,
            label="state of charge" if highlight_active_periods else None,
            linewidth=1.8,
            color="blue",
            alpha=alpha,
        )
        ax1.plot(
            periods[periods <= active_start],
            np.array(charge_bounds)[periods <= active_start] / 1e3,
            label="charging limit" if highlight_active_periods else None,
            linewidth=1.8,
            color="red",
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
        ax2.set_ylabel("LMP Signal ($/MWh)", color="black")
        if plot_uncertainty and lmp_set is not None:
            # plot LMP uncertainty set bounds.
            # NOTE: the LMPs are scaled to MWh before plotting
            #       since the values provided are in kWh
            # TODO: enforce LMP units at model declaration
            lmp_set.lmp_sig_nom *= 1e3
            lmp_set.plot_bounds(ax2, offset=start)
            lmp_set.lmp_sig_nom *= 1e-3
        else:
            ax2.plot(periods, np.array(lmp_values) * 1e3,
                     label="LMP", linewidth=1.8,
                     color="black")
        custom_lmp_vals = [] if custom_lmp_vals is None else custom_lmp_vals
        for val in custom_lmp_vals:
            lmp_arr = np.array(val) * 1e3
            ax2.plot(periods, lmp_arr, label="worst case", linewidth=1.8,
                     color="green")
        ax2.legend(bbox_to_anchor=(1, -0.15), loc="upper right", ncol=1)

        # set LMP axis bounds
        if lmp_bounds is None:
            if lmp_set is not None:
                lmp_bounds = lmp_set.bounds()
                y_min = min(bound[0] for bound in lmp_bounds) * 1e3
                y_max = max(bound[1] for bound in lmp_bounds) * 1e3
                ax2.set_ylim([y_min, y_max])
        else:
            y_min = lmp_bounds[0]
            y_max = lmp_bounds[1]
            ax2.set_ylim([y_min, y_max])

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close()


def plot_power_output_results(
        mp_model,
        lmp_set=None,
        plot_lmp=True,
        plot_uncertainty=True,
        filename=None,
        start=None,
        stop=None,
        highlight_active_periods=False,
        custom_lmp_vals=None,
        lmp_bounds=None,
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

    fig, ax1 = plt.subplots(figsize=(9, 8))
    ax1.grid(False)

    # get model blocks of interest
    start = model.current_time if start is None else start
    stop = model.current_time + len(active_blks) - 1 if stop is None else stop

    assert start >= 0
    assert stop < len(mp_model.pyomo_model.blocks)

    if lmp_bounds is not None:
        assert lmp_bounds[0] < lmp_bounds[1]

    periods = np.array(range(start, stop + 2))
    active_start = model.current_time
    blocks = list(
        blk.process
        for idx, blk in enumerate(mp_model.pyomo_model.blocks.values())
        if idx >= start and idx <= stop
    )

    # initialize containers
    battery_elecs = list()
    grid_elecs = list()
    wind_outputs = list()
    lmp_values = list()

    for t, blk in zip(periods, blocks):
        battery_elecs.append(pyo.value(blk.fs.battery.elec_out[0]))
        grid_elecs.append(pyo.value(blk.fs.splitter.grid_elec[0]))
        wind_outputs.append(pyo.value(blk.fs.windpower.electricity[0]))
        lmp_values.append(pyo.value(mp_model._pyomo_model.LMP[t]))

    battery_elecs.append(battery_elecs[-1])
    grid_elecs.append(grid_elecs[-1])
    wind_outputs.append(wind_outputs[-1])

    if periods[periods >= active_start].size > 0:
        active_qual = " (active)" if highlight_active_periods else ""
        ax1.step(
            periods[periods >= active_start],
            np.array(grid_elecs)[periods >= active_start] / 1e3,
            label="wind-to-grid" + active_qual,
            where="post",
            linewidth=1.8,
            color="red",
        )
        ax1.step(
            periods[periods >= active_start],
            np.array(battery_elecs)[periods >= active_start] / 1e3,
            label="battery-to-grid" + active_qual,
            linewidth=1.8,
            where="post",
            color="blue",
        )
        ax1.step(
            periods[periods >= active_start],
            np.array(wind_outputs)[periods >= active_start] / 1e3,
            where="post",
            label="wind production" + active_qual,
            linewidth=1.8,
            color="purple",
            linestyle="dashed",
        )

    if periods[periods < active_start].size > 0:
        alpha = 0.3 if highlight_active_periods else 1
        ax1.step(
            periods[periods <= active_start],
            np.array(grid_elecs)[periods <= active_start] / 1e3,
            label="wind-to-grid" if highlight_active_periods else None,
            where="post",
            linewidth=1.8,
            color="red",
            alpha=alpha,
        )
        ax1.step(
            periods[periods <= active_start],
            np.array(battery_elecs)[periods <= active_start] / 1e3,
            label="battery-to-grid" if highlight_active_periods else None,
            linewidth=1.8,
            where="post",
            color="blue",
            alpha=alpha,
        )
        ax1.step(
            periods[periods <= active_start],
            np.array(wind_outputs)[periods <= active_start] / 1e3,
            where="post",
            label="wind production" if highlight_active_periods else None,
            linewidth=1.8,
            color="purple",
            linestyle="dashed",
            alpha=alpha,
        )

    ax1.set_xlabel("Period (hr)")
    ax1.set_ylabel("Power Output (MW)")
    # ax1.set_xticks(model._pyomo_model.Horizon)
    ax1.legend(bbox_to_anchor=(0, -0.15), loc="upper left", ncol=1)

    if plot_lmp:
        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.set_ylabel("LMP Signal ($/MWh)", color="black")
        if plot_uncertainty and lmp_set is not None:
            lmp_set.lmp_sig_nom *= 1e3
            lmp_set.plot_bounds(ax2)
            lmp_set.lmp_sig_nom *= 1e-3
        else:
            ax2.plot(periods[:-1], np.array(lmp_values) * 1e3,
                     label="LMP", linewidth=1.8,
                     color="black")
        custom_lmp_vals = [] if custom_lmp_vals is None else custom_lmp_vals
        for val in custom_lmp_vals:
            lmp_arr = np.array(val) * 1e3
            ax2.plot(periods[:-1], lmp_arr, label="worst case", linewidth=1.8,
                     color="green")

        ax2.legend(bbox_to_anchor=(1, -0.15), loc="upper right", ncol=1)

        # set LMP axis bounds
        if lmp_bounds is None:
            if lmp_set is not None:
                lmp_bounds = lmp_set.bounds()
                y_min = min(bound[0] for bound in lmp_bounds) * 1e3
                y_max = max(bound[1] for bound in lmp_bounds) * 1e3
                ax2.set_ylim([y_min, y_max])
        else:
            y_min = lmp_bounds[0]
            y_max = lmp_bounds[1]
            ax2.set_ylim([y_min, y_max])

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
        model,
        lmp_set=lmp_set,
        plot_lmp=plot_lmp,
        plot_uncertainty=plot_uncertainty,
        custom_lmp_vals=custom_lmp_vals,
        filename=pwr_filename,
        start=start,
        stop=stop,
        highlight_active_periods=highlight_active_periods,
        lmp_bounds=lmp_bounds,
    )
    plot_soc_results(
        model,
        lmp_set=lmp_set,
        plot_lmp=plot_lmp,
        plot_uncertainty=plot_uncertainty,
        custom_lmp_vals=custom_lmp_vals,
        filename=soc_filename,
        start=start,
        stop=stop,
        highlight_active_periods=highlight_active_periods,
        lmp_bounds=lmp_bounds,
    )


def construct_profit_obj(model, lmp_signal):
    """
    Construct model LMP profit objective.

    Parameters
    ----------
    model : MultiPeriodModel
        Model of interest.
    lmp_signal : array-like
        LMP signal. Must contain as many entries are there are
        process blocks in the model.
    """
    horizon = len(lmp_signal)
    pyomo_model = model.pyomo_model

    assert horizon == len(pyomo_model.blocks)

    # reconstruct objective
    attrs_to_del = [
        "MaxProfitObj",
        "TotalPowerOutput",
        "OperationCost",
        "Horizon",
        "LMP",
    ]
    for attr in attrs_to_del:
        if hasattr(pyomo_model, attr):
            pyomo_model.del_component(attr)

    pyomo_model.Horizon = pyo.Set(initialize=range(len(lmp_signal)))

    # power output in MW
    pyomo_model.TotalPowerOutput = pyo.Expression(pyomo_model.Horizon)

    # Operation costs in $
    pyomo_model.OperationCost = pyo.Expression(pyomo_model.Horizon)

    for t, blk in pyomo_model.blocks.items():
        b = blk.process
        pyomo_model.TotalPowerOutput[t] = (
            b.fs.splitter.grid_elec[0] + b.fs.battery.elec_out[0]
        ) * 1e-3
        pyomo_model.OperationCost[t] = b.fs.windpower.op_total_cost

    # add LMPs
    pyomo_model.LMP = pyo.Param(
        pyomo_model.Horizon,
        initialize=lmp_signal,
        mutable=True,
        doc="Locational marginal prices, $/kWh",
        units=1/pyo.units.kWh,
    )

    @pyomo_model.Objective(doc="Total profit ($/hr)", sense=pyo.maximize)
    def MaxProfitObj(m):
        return sum(m.LMP[t] * m.TotalPowerOutput[t] - m.OperationCost[t]
                   for t in m.Horizon)


def get_dof_vars(model):
    """
    Obtain first-stage and second-stage degrees of freedom
    for all active model blocks.
    """
    active_blks = model.get_active_process_blocks()

    first_stage_vars = [
        list(active_blks)[0].fs.splitter.grid_elec[0],
        list(active_blks)[0].fs.battery.elec_out[0],
        list(active_blks)[0].fs.windpower.electricity[0.0],
    ]
    second_stage_vars = (
        list(blk.fs.splitter.grid_elec[0] for blk in list(active_blks)[1:])
        + list(blk.fs.battery.elec_out[0] for blk in list(active_blks)[1:])
        + list(
            blk.fs.windpower.electricity[0.0] for blk in list(active_blks)[1:]
        )
    )
    return first_stage_vars, second_stage_vars


def get_uncertain_params(model):
    """
    Obtain uncertain parameters for all active model blocks.
    """
    start_time = model.current_time
    return list(
        val for t, val in model.pyomo_model.LMP.items() if t >= start_time
    )


def create_two_stg_wind_battery_model(lmp_signal):
    """
    Create a two-stage surrogate of the multi-stage
    wind battery model.

    Parameters
    ----------
    lmp_signal : array-like
        LMP signal, of which the number of entries is equal
        to the model's prediction horizon length.

    Returns
    -------
    model : MultiPeriodModel
        Model.
    """
    horizon = len(lmp_signal)

    model = create_multiperiod_wind_battery_model(n_time_points=horizon)
    transform_design_model_to_operation_model(model)

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
        new_lmp_sig,
        lmp_set_class=None,
        lmp_set_class_params=None,
        wind_capacity=200e3,
        battery_power_capacity=25e3,
        battery_energy_capacity=100e3,
        ):
    """
    Advance model instance to next time period, and
    construct an updated uncertainty set.

    Parameters
    ----------
    model : MultiPeriodModel
        Model of interest.
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
        Wind production capacity.
    battery_power_capacity : float, optional
        Battery power capacity.
    battery_energy_capacity : float, optional
        Battery energy capacity.

    Returns
    -------
    : LMPBoxSet
        An updated LMP uncertainty set, if a constructor is provided
        through the `lmp_set_class` argument.
    """
    assert len(new_lmp_sig) == len(model.get_active_process_blocks())

    from dispatches.models.renewables_case.load_parameters import wind_speeds
    new_time = model.pyomo_model.TIME.last() + 1
    new_wind_resource = {
        "wind_resource_config": {
            "resource_probability_density": {
                0.0: ((wind_speeds[new_time], 180, 1),)
            }
        }
    }

    # get initial block
    block_init = model.get_active_process_blocks()[0]

    # fix variables, deactivate constraints on this block
    for var in block_init.component_data_objects(pyo.Var):
        var.fix()
    for con in block_init.component_data_objects(pyo.Constraint, active=True):
        con.deactivate()

    # now advance time (to obtain new process block)
    model.advance_time(**new_wind_resource)

    # retrieve new block
    b = model.get_active_process_blocks()[-1]

    # transform latest block to operation mode
    b.fs.windpower.system_capacity.fix(wind_capacity)
    b.fs.battery.nameplate_power.fix(battery_power_capacity)
    b.fs.battery.nameplate_energy.fix(battery_energy_capacity)
    b.periodic_constraints[0].deactivate()

    # fix initial state of charge and throughput to values
    # from previous period
    b_init = model.get_active_process_blocks()[0]
    b_init.fs.battery.initial_state_of_charge.fix()
    b_init.fs.battery.initial_energy_throughput.fix()

    # update LMP signal and model objective
    lmp_sig = {
        t: pyo.value(model.pyomo_model.LMP[t])
        for t in range(model.current_time + 1)
    }
    lmp_sig.update({
        t + model.current_time: new_lmp_sig[t]
        for t in range(len(new_lmp_sig))
    })
    construct_profit_obj(model, lmp_sig)

    # deactivate constraints in fixed variables
    for con in pyomo_model.component_data_objects(pyo.Constraint, active=True):
        if all(var.fixed for var in identify_variables(con.body)):
            con.deactivate()

    # construct an updated uncertainty set (if constructor provided)
    if lmp_set_class is not None:
        return lmp_set_class(new_lmp_sig, **lmp_set_class_params)


def solve_rolling_horizon(
        model,
        solver,
        lmp_signal_filename,
        control_length,
        num_steps,
        start,
        output_dir=None,
        **solver_kwargs,
        ):
    """
    Solve the deterministic wind-battery model on a rolling horizon.

    Parameters
    ----------
    model : MultiPeriodModel
        Model of interest.
    solver : pyomo SolverFactory object
        Optimizer used to solve the model.
    lmp_signal_filename : path-like
        Path to LMP signal data file.
    control_length : int
        Control horizon length, i.e. number of periods
        for which the optimal settings in each period
        are actually used.
    num_steps : int
        Number of prediction horizons for which to solve the model.
        (One plus the number of times to update the
        active time periods/blocks.)
    start : int
        Starting index for the LMP signal extracted from the
        data file at the path specified by `lmp_signal_filename`.
    solver_kwargs : dict, optional
        Keyword arguments to the method `solver.solve()`.
    """
    # cannot control beyond prediction horizon
    prediction_length = len(model.get_active_process_blocks())
    assert prediction_length >= control_length
    assert "load_solutions" not in solver_kwargs

    # obtain new LMP signal
    lmp_signal = get_lmp_data(
        lmp_signal_filename,
        (num_steps - 1) * control_length + prediction_length,
        start=start,
    ) / 1e3

    lmp_start = 0
    for idx in range(num_steps):
        if idx == 0:
            # set up extended LMP signal.
            # this changes the LMP values for the currently
            # active process blocks
            lmp_sig = {
                t: pyo.value(model.pyomo_model.LMP[t])
                for t in range(model.current_time + 1)
            }
            lmp_sig.update({
                t + model.current_time: lmp_signal[t]
                for t in range(prediction_length)
            })
            construct_profit_obj(model, lmp_sig)
        else:
            # advance the model in time, extend LMP signal
            for step in range(control_length):
                lmp_start += 1
                lmp_stop = lmp_start + prediction_length
                advance_time(model, lmp_signal[lmp_start:lmp_stop])

        # solve pyomo model
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

        revenue = pyo.value(
            pyo.dot_product(
                model.pyomo_model.LMP,
                model.pyomo_model.TotalPowerOutput,
            )
        )
        print(model.current_time, res.problem.lower_bound, revenue)

        if output_dir is not None:
            start = 0
            stop = prediction_length - 1
            lmp_bounds = (
                1e3 * min(lmp_signal[start:stop]) - 5,
                1e3 * max(lmp_signal[start:stop]) + 5,
            )
            plot_results(
                model,
                highlight_active_periods=True,
                output_dir=os.path.join(output_dir, f"step_{idx}"),
                lmp_bounds=lmp_bounds,
                start=0,
                stop=prediction_length - 1,
            )


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
        var for var in pyomo_model.component_data_objects(pyo.Var)
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
    nlp = PyomoNLP(pyomo_model)
    nlp.n_primals()
    jac = nlp.extract_submatrix_jacobian(list(state_vars), list(eqns))
    # splu(jac.tocsc())
    jacarr = jac.toarray()
    print("Jacobian determinant", det(jacarr))


if __name__ == "__main__":
    horizon = 14
    start = 4000
    solve_pyros = True

    logging.basicConfig(level=logging.INFO)

    # parameterization for uncertainty set
    n_recent = 4
    hyster_latency = 1
    box_growth_rate = 0.05
    moving_avg_multiplier = 0.1
    n_time_points = horizon

    # container for LMP set parameterization
    lmp_set_params = dict(
        n_recent=n_recent,
        growth_rate=box_growth_rate,
        avg_multiplier=moving_avg_multiplier,
        latency=hyster_latency,
        start_day_hour=0,
        include_peak_effects=True,
    )

    # make directory for storing results
    base_dir = f"../../../../results/results/hor_{horizon}_start_{start}"
    os.makedirs(base_dir, exist_ok=True)

    # get LMP data (and convert to $/kWh)
    lmp_signal_filename = (
        "../../../../results/lmp_data/rts_results_all_prices.npy"
    )
    lmp_signal = get_lmp_data(
        lmp_signal_filename,
        n_time_points,
        start=start,
    ) / 1e3

    # construct container for uncertainty set
    hyster_lmp_set = HysterLMPBoxSet(lmp_signal, **lmp_set_params)

    # create model, and obtain degree-of-freedom partitioning
    model = create_two_stg_wind_battery_model(hyster_lmp_set.lmp_sig_nom)
    first_stage_vars, second_stage_vars = get_dof_vars(model)
    uncertain_params = get_uncertain_params(model)
    pyomo_model = model.pyomo_model

    # set up solvers
    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    gams_baron = pyo.SolverFactory("gams")
    gams_baron.options["solver"] = "baron"
    couenne = pyo.SolverFactory("couenne")

    # solve deterministic model
    res = solver.solve(pyomo_model, tee=True)

    max_lmps = [bd[1] for bd in hyster_lmp_set.bounds()]
    min_lmps = [bd[0] for bd in hyster_lmp_set.bounds()]
    nom_lmps = hyster_lmp_set.lmp_sig_nom

    # evaluate objective value of deterministic solution in event
    # of nominal, max, and min LMP's
    for sig in [max_lmps, min_lmps, nom_lmps]:
        for t, val in enumerate(sig):
            pyomo_model.LMP[t].set_value(val)
        print(
            "Revenue",
            pyo.value(sum(
                pyomo_model.LMP[t] * pyomo_model.TotalPowerOutput[t]
                for t in pyomo_model.Horizon
            ))
        )

    plot_results(
        model,
        lmp_set=hyster_lmp_set,
        plot_uncertainty=False,
        highlight_active_periods=True,
        output_dir=os.path.join(base_dir, "deterministic"),
    )

    pdb.set_trace()

    pyros_solver = pyo.SolverFactory("pyros")
    ro_res = pyros_solver.solve(
        pyomo_model,
        first_stage_vars,
        second_stage_vars,
        uncertain_params,
        hyster_lmp_set.pyros_set(),
        solver,
        solver,
        backup_local_solvers=[gams_baron, couenne],
        backup_global_solvers=[gams_baron, couenne],
        decision_rule_order=1,
        bypass_local_separation=True,
        objective_focus=pyros.ObjectiveType.worst_case,
        solve_master_globally=True,
        keepfiles=True,
        load_solution=True,
        output_verbose_results=False,
        subproblem_file_directory=os.path.join(base_dir, "pyros_sublogs"),
    )

    for sig in [max_lmps, min_lmps, nom_lmps]:
        for t, val in enumerate(sig):
            pyomo_model.LMP[t].set_value(val)
        print(
            "Revenue",
            pyo.value(sum(
                pyomo_model.LMP[t] * pyomo_model.TotalPowerOutput[t]
                for t in pyomo_model.Horizon
            ))
        )

    for idx, val in pyomo_model.LMP.items():
        val.set_value(hyster_lmp_set.bounds()[idx][0])
    wc_rev = pyo.value(
        sum(
            pyomo_model.LMP[t] * pyomo_model.TotalPowerOutput[t]
            for t in pyomo_model.Horizon
        )
    )
    print("worst-case revenue", wc_rev)
    print(
        "final revenue (should be same)",
        ro_res.solver.final_objective_value + pyo.value(
            sum(
                pyomo_model.OperationCost[t]
                for t in pyomo_model.Horizon
            )
        )
    )

    for idx, val in pyomo_model.LMP.items():
        val.set_value(hyster_lmp_set.lmp_sig_nom[idx])

    # plot worst-case solution
    plot_results(
        model,
        hyster_lmp_set,
        custom_lmp_vals=[[bound[0] for bound in hyster_lmp_set.bounds()]],
        output_dir=os.path.join(base_dir, "ro"),
        plot_uncertainty=True,
    )
    ro_vars = [pyo.value(var) for var in first_stage_vars]

    for var in second_stage_vars:
        var.set_value(ro_res.solver.nom_ssv_vals[var.name])

    plot_results(
        model,
        hyster_lmp_set,
        output_dir=os.path.join(base_dir, "ro_nominal"),
        plot_uncertainty=True,
    )
    ro_nom_vars = [pyo.value(var) for var in first_stage_vars]

    nom_rev = pyo.value(
        sum(
            pyomo_model.LMP[t] * pyomo_model.TotalPowerOutput[t]
            for t in pyomo_model.Horizon
        )
    )
    print("nominal revenue", nom_rev)

    for var in second_stage_vars:
        var.set_value(ro_res.solver.best_case_ssv_vals[var.name])
    for idx, val in pyomo_model.LMP.items():
        val.set_value(hyster_lmp_set.bounds()[idx][1])

    plot_results(
        model,
        lmp_set=hyster_lmp_set,
        output_dir=os.path.join(base_dir, "ro_best"),
        plot_uncertainty=True,
    )
    rev = pyo.value(
        sum(
            pyomo_model.LMP[t] * pyomo_model.TotalPowerOutput[t]
            for t in pyomo_model.Horizon
        )
    )
    print("best-case revenue", rev)
    pdb.set_trace()

    # now finally, the rolling horizon simulation
    solve_rolling_horizon(
        model, solver, lmp_signal_filename, 1, horizon, start,
        output_dir=os.path.join(base_dir, "rolling_horizon"),
    )
    pdb.set_trace()
