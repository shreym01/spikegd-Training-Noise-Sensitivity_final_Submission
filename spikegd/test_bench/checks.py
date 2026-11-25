"""Argument checks shared by the network, the tasks and the losses. A wrong type raises TypeError, a wrong value
ValueError; every message starts with the name of the argument.

Inside jit / grad the arguments are jax tracers: their shapes are known, their values are not. is_concrete() tells
the two apart, so value checks are skipped under tracing and only shape checks run.
"""
import numbers

import jax
import numpy as np


def is_concrete(value):
    """False for a jax tracer (inside jit / grad)."""
    return not isinstance(value, jax.core.Tracer)


def check_int(name, value, low=None, high=None):
    """An integer (not a bool) in [low, high]."""
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        upper = "inf" if high is None else high
        raise ValueError(f"{name} = {value} must be in [{low}, {upper}]")


def check_number(name, value, low=None, high=None, low_open=False):
    """A finite real number (not a bool), >= low (> low if low_open) and <= high."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not np.isfinite(value):
        raise TypeError(f"{name} must be a finite number, got {value!r}")
    if low is not None and (value < low or (low_open and value == low)):
        raise ValueError(f"{name} = {value} must be {'>' if low_open else '>='} {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} = {value} must be <= {high}")


def check_positive(name, value):
    """A finite number > 0 (skipped for a tracer)."""
    if is_concrete(value):
        check_number(name, value, low=0.0, low_open=True)


def check_choice(name, value, choices):
    """One of choices."""
    if value not in choices:
        raise ValueError(f"{name} = {value!r} must be one of {list(choices)}")


def check_bool(name, value):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, got {value!r}")


def check_type(name, value, cls):
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be a {cls.__name__}, got {type(value).__name__}")
