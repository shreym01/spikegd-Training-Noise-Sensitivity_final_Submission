"""Inputs, targets, and output spike times shared by the QIF RNN experiments."""

import jax.numpy as jnp
from jax import random
from jaxtyping import Array

from spikegd.models import AbstractPseudoIFNeuron


def init_input(key: Array, config: dict) -> tuple:
    """
    Creates input spikes.
    """
    ### Unpack arguments
    T: float = config["T"]
    N: int = config["N"]
    Nin = 2 * N
    r_in: float = config["r_in"]
    Tinit = 10.0  # To get same inputs as in paper

    ### Input spikes
    key, subkey = random.split(key)
    ISIs_exc = r_in * random.exponential(subkey, (N, int(5 * (T + Tinit) * r_in)))
    times_exc = jnp.cumsum(ISIs_exc, axis=1)
    key, subkey = random.split(key)
    ISIs_inh = 1 / r_in * random.exponential(subkey, (N, int(5 * (T + Tinit) * r_in)))
    times_inh = jnp.cumsum(ISIs_inh, axis=1)
    times = jnp.concatenate((times_exc, times_inh), axis=0)
    times_in = jnp.inf * jnp.ones_like(times)
    for i in range(Nin):
        times_after_init = times[i, times[i] > Tinit] - Tinit
        times_in = times_in.at[i, : len(times_after_init)].set(times_after_init)
    times_in = jnp.ravel(times_in)
    neurons_in = jnp.repeat(jnp.arange(Nin), times.shape[1])
    order = jnp.argsort(times_in)
    spikes_in = (times_in[order], neurons_in[order])

    return key, spikes_in


def init_weights(config: dict) -> Array:
    """
    Creates input weights.
    """
    ### Unpack arguments
    N: int = config["N"]
    Nin = 2 * N
    w_in_exc: float = config["w_in_exc"]
    w_in_inh: float = config["w_in_inh"]

    ### Weights
    weights_in = jnp.zeros((N, Nin))
    weights_in = weights_in.at[jnp.arange(N), jnp.arange(N)].set(w_in_exc)
    weights_in = weights_in.at[jnp.arange(N), jnp.arange(N, 2 * N)].set(w_in_inh)

    return weights_in


def init_target(key: Array, config: dict) -> tuple:
    """
    Creates target spike times.
    """
    ### Unpack arguments
    T: float = config["T"]
    Ntar: int = config["Ntar"]
    r_targets: list[float] = config["r_targets"]

    ### Create target spikes
    key, subkey = random.split(key)
    ISIs = random.exponential(subkey, (Ntar, int(2.5 * T)))
    for i, rtarget in enumerate(r_targets):
        ISIs = ISIs.at[i, :].divide(rtarget)
    ISIs = ISIs.at[:, 1:].add(1)
    times = jnp.cumsum(ISIs, axis=1)
    t_targets: list[Array] = []
    for i in range(Ntar):
        t_targets.append(jnp.append(times[i, times[i] < T], 1.1 * T))

    return key, t_targets


def w_to_weights_net(w: Array) -> Array:
    """
    Transforms learnable weights to network weight matrix.
    """
    weights_net = jnp.array([jnp.insert(w_i, i, 0) for i, w_i in enumerate(w)])
    return weights_net


def outfn(
    neuron: AbstractPseudoIFNeuron,
    out: tuple,
    p: list[Array],
    t_targets: list[Array],
    config: dict,
) -> list[Array]:
    """
    Computes output spike times given simulation results.
    """
    ### Unpack arguments
    Ntar: int = config["Ntar"]

    ### Unpack output
    times: Array = out[0]
    spikes_in: Array = out[1]
    neurons: Array = out[2]
    xs: Array = out[3]

    ### Set weights
    weights_net_p = w_to_weights_net(p[0])

    ### State at trial end
    Kord = jnp.sum(neurons >= 0)  # Number of ordinary spikes
    x_end = xs[Kord]
    pseudo_rates = neuron.construct_ratefn(x_end)(input=-x_end[1])

    ### Spike times for each learned neuron
    t_outs = []
    for i in range(Ntar):
        Ntargets_i = len(t_targets[i])

        ### Potential ordinary output spike times
        mask = (neurons == i) & (spikes_in == False)  # noqa: E712
        Kout = jnp.sum(mask)  # Number of ordinary output spikes
        t_out_ord = times[jnp.argsort(~mask)[:Ntargets_i]]

        ### Pseudospike times
        input = neuron.linear(pseudo_rates, weights_net_p[i])
        t_out_pseudo = neuron.t_pseudo(
            x_end[:, i], input, jnp.arange(Ntargets_i) + 1 - Kout, config
        )

        ### Output spike times
        t_out = jnp.where(jnp.arange(Ntargets_i) < Kout, t_out_ord, t_out_pseudo)
        t_outs.append(t_out)

    return t_outs
