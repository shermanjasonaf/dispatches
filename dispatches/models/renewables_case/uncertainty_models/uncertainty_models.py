#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-21 Thu 15:49:32

@author: jasherma

This module defines generic classes and methods for
modeling sources of uncertainty for the wind-battery model.
"""


import abc
import itertools


def lift_uncertain_params(uncertainty_set, reduced_params, lifting_values):
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
    fixed_dims = uncertainty_set.determine_fixed_dims()

    if fixed_dims != list(lifting_values.keys()):
        raise ValueError(
            "Fixed dimensions mismatch. Provided insertion into "
            f"dimensions {list(lifting_values.keys())}, but fixed "
            f"dimensions are actually {fixed_dims}"
        )

    lifted_params = list()
    reduced_dim_count = 0
    for idx in range(len(uncertainty_set.sig_nom)):
        if idx in fixed_dims:
            val = lifting_values[idx]
        else:
            val = reduced_params[reduced_dim_count]
            reduced_dim_count += 1
        lifted_params.append(val)

    return lifted_params


def get_uncertain_params(
        lmp_set,
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
    assert len(uncertain_params) == len(lmp_set.sig_nom)

    if not include_fixed_dims:
        fixed_dims = lmp_set.determine_fixed_dims()
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
        return list(itertools.chain.from_iterable(nested_param_list))


class UncertaintySet(abc.ABC):
    """
    Wrapper around a PyROS UncertaintySet object
    for modeling uncertainty in wind-battery model uncertain
    parameters.
    """
    @property
    @abc.abstractmethod
    def sig_nom(self):
        """Evaluate nominal value of the uncertain parameters."""
        ...

    @abc.abstractmethod
    def pyros_set(self, include_fixed_dims=True):
        """Construct PyROS Uncertainty set."""
        ...

    def determine_fixed_dims(self, tol=0):
        """Determine dimensions of the uncertainty set which are fixed."""
        unc_set = self.pyros_set(include_fixed_dims=False).parameter_bounds()

        return [
            idx for idx, bd in enumerate(unc_set)
            if abs(bd[0] - bd[1]) <= tol
        ]

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
        return get_uncertain_params(
            self,
            uncertain_params,
            include_fixed_dims=include_fixed_dims,
            nested=nested,
        )

    def lift_uncertain_params(self, reduced_params, lifting_values):
        """
        'Lift' the reduced dimensions of the set (i.e the dimensions
        for which the bounds are equal) back into the original set
        dimensions.

        See documentation for `_lift_uncertain_params`.
        """
        return lift_uncertain_params(self, reduced_params, lifting_values)

    @abc.abstractmethod
    def plot_set(self, **kwds):
        """Plot a visualization of the uncertainty set."""
        ...
