"""Configuration and simulation for the event-driven QIF test bench.

The network keeps its structure and trainable parameters separate.

    from spikegd.test_bench.network import NetConfig, build_net, load_net

    config = NetConfig(architecture="fc_rnn", neurons_per_layer=[3], n_input_neurons=1, n_readout_neurons=2)
    net = build_net(config)
    params = net.init_params()                   # weights_net (N, N), weights_in (N, C), w_out (n_out, k), x0 (2, N)
    events = net.simulate(params, spikes_in, T=10.0, K=200)              # spikegd's neuron.event, unchanged
    y, events = net.forward(params, spikes_in, T=10.0, K=200, read_at=grid)  # simulate, then the output head
    net.export("net.npz", params)                # config and seeds are saved with the weights
    net, params = load_net("net.npz")

Conventions
  * Neurons are indexed by layer, starting with layer 1.
  * One layer (L = 1): the first n_input_neurons receive the task input, the last n_readout_neurons are read out,
    the ones in between are hidden.
    Several layers (L > 1): every neuron of layer 1 receives the task input, every neuron of layer L is read out,
    the layers in between are hidden.
  * Input channels: the task sets how many there are (n_input_channels, e.g. 784 for MNIST); the network sets how
    they reach the input neurons (InputChannels.mode):
      "shared"   every input neuron gets every channel, each with its own weight (MNIST, TIMIT)
      "private"  the channels are split into consecutive blocks, one block per input neuron, in neuron order.
                 InputChannels.fraction gives the split as decimals of C:
                   [f_1, ..., f_n]  input neuron k gets f_k C channels (one entry per input neuron)
                   f                every input neuron gets f C channels
                   None             every input neuron gets C / n_input_neurons channels
                 Every f_k C must be a whole number and the fractions must not add up to more than 1. If they add up
                 to less than 1 the last channels are left over: an error unless allow_leftover=True.
    A one-layer network can have no task input at all: n_input_neurons = 0 and n_input_channels = 0 (the cyclic task;
    the background drives it). Input channels are excitatory. Neurons of other layers get no task input channels.
    ("Input" neurons are hidden in the ML sense: they are not read out.)
  * Background drive: every neuron chosen by Background.applies_to gets one more private excitatory channel fed by a
    slow Poisson process (arbitrary spiking behaviour). Its weight is fixed (not a parameter). The background
    channels come after the task channels; simulate() adds the background spikes to the task spikes.
  * weights_net[i, j] is the synapse from j onto i (spikegd's layout). The parameters are kept in this (N, N) layout
    and multiplied by a fixed connectivity mask before every simulation.
  * Output head (Readout): y(t) = W_out sum over the read-out neurons' spikes of kernel(t - t_spike). forward()
    reads y at the times the task asks for: on a grid of times (n_times,), giving y (n_out, n_times), or in output
    windows (n_windows, 2) of (start, end), giving y (n_out, n_windows) as the mean over the window or the value at
    its end. forward() also returns the events, which the learning rules need.
  * Dale's law: every neuron has a sign, applied to all of its outgoing synapses. By default input neurons and
    hidden neurons are inhibitory and read-out neurons are free (their outgoing weights keep the sign the
    initialisation gives). hidden_excitatory_fraction turns a fraction of the hidden neurons excitatory.
    project_params() puts the signs and the mask back after an update.

Architectures
  "fcn"     feed-forward: layer l -> layer l + 1 only, nothing inside a layer, nothing backwards. Needs L > 1.
  "fc_rnn"  all-to-all between every pair of neurons (still indexed by layer).
  "custom"  weights_net (and optionally everything else) from an .npz file.

Errors: build_net checks every config field and the methods check their arguments; a TypeError or ValueError names
the field or argument that is wrong (see the validation helpers below).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from spikegd.qif import QIFNeuron

from .checks import (
    check_bool, check_choice, check_int, check_number, check_positive, check_type, is_concrete,
)


INHIBITORY, FREE, EXCITATORY = -1, 0, 1
ROLES = ("input", "hidden", "readout")
PARAM_NAMES = ("weights_net", "weights_in", "w_out", "x0")


# =====================================================================================================================
# Configuration
# =====================================================================================================================

@dataclass
class WeightInit:
    """How one group of weights is initialised.

    scheme          "zero", "constant" (every weight = scale) or "gaussian" (N(0, scale^2))
    scale           size of the weights
    normalisation   "none" or "fan_in": divide by sqrt(number of synapses onto the neuron)
    free_sign       sign of "constant" weights for neurons without a Dale sign (read-out neurons by default)
    seed            seed of the random draw (saved on export)
    """
    scheme: str = "constant"
    scale: float = 1.0
    normalisation: str = "none"
    free_sign: int = INHIBITORY
    seed: int = 0


@dataclass
class Connectivity:
    """Which of the synapses the architecture allows are actually present.

    kind            "full" (all of them), "uniform" or "gaussian"
    For "uniform" and "gaussian" every presynaptic neuron draws how many of its allowed targets it connects to:
      uniform:  out-degree ~ U{1, ..., max_targets}       (max_targets: default = number of allowed targets)
      gaussian: out-degree ~ round(N(mean, variance))     (mean: default = half the allowed targets), clipped
    and then picks that many targets at random.
    """
    kind: str = "full"
    max_targets: int | None = None
    mean: float | None = None
    variance: float = 1.0
    seed: int = 0


@dataclass
class SelfConnections:
    """Self-synapses (off by default).

    n_per_layer     how many neurons of each layer get a self-synapse
    selection       "random" (drawn with seed) or "indices" (taken from indices: global neuron indices)
    """
    enabled: bool = False
    n_per_layer: list[int] = field(default_factory=list)
    selection: str = "random"
    indices: list[int] = field(default_factory=list)
    seed: int = 0


@dataclass
class InputChannels:
    """How the task's input channels reach the input neurons.

    n_input_channels    number of task channels C (set by the task)
    mode                "shared" (every input neuron gets all C) or "private" (consecutive blocks, one per neuron)
    fraction            private only: list of decimals, one per input neuron (neuron k gets fraction[k] * C
                        channels), or one decimal for every neuron, or None (C / n_input_neurons each)
    allow_leftover      private only: allow the fractions to add up to less than 1 (the last channels go unused)
    """
    n_input_channels: int = 1
    mode: str = "private"
    fraction: float | list[float] | None = None
    allow_leftover: bool = False


@dataclass
class Background:
    """Slow excitatory Poisson drive, one private channel per chosen neuron.

    applies_to      "all", "non_input" (hidden and read-out) or "readout"
    rate            spikes per unit time on each channel (small: occasional spikes)
    weight          jump of the current b at each background spike (> pi^2 / 4 fires a neuron from rest)
    seed            seed of the background spike trains (one sample per seed)
    """
    enabled: bool = True
    applies_to: str = "all"
    rate: float = 0.1
    weight: float = 10.0
    seed: int = 0


@dataclass
class Readout:
    """The output head: y(t) = W_out sum_q kernel(t - t_q) over the read-out neurons' spikes t_q.

    kernel          "alpha": kappa(s) = (s / tau_r) exp(1 - s / tau_r) for s > 0, else 0 (peak 1 at s = tau_r)
    tau_r           time constant of the kernel
    window          how y is read in an output window (start, end): "mean" (average over window_points points
                    spread evenly in the window) or "end" (y at the end of the window)
    window_points   points per window for "mean"
    """
    kernel: str = "alpha"
    tau_r: float = 0.5
    window: str = "mean"
    window_points: int = 20


@dataclass
class NetConfig:
    # architecture
    architecture: str = "fc_rnn"                 # "fcn", "fc_rnn" or "custom"
    custom_path: str | None = None               # .npz for "custom"
    neurons_per_layer: list[int] = field(default_factory=lambda: [3])
    n_input_neurons: int = 1                     # one layer only: the first n_input_neurons get the task input
    n_readout_neurons: int = 2                   # one layer only: the last n_readout_neurons are read out
    input_channels: InputChannels = field(default_factory=InputChannels)
    n_output_channels: int = 1                   # rows of W_out
    readout: Readout = field(default_factory=Readout)
    background: Background = field(default_factory=Background)

    # neuron
    neuron_type: type = QIFNeuron
    neuron_kwargs: dict = field(default_factory=lambda: dict(tau=1.0, eps=1e-6, alpha=10.0))

    # connectivity and signs
    connectivity: Connectivity = field(default_factory=Connectivity)
    self_connections: SelfConnections = field(default_factory=SelfConnections)
    hidden_excitatory_fraction: float = 0.0      # fraction of hidden neurons that are excitatory (Dale's law)
    sign_seed: int = 0

    # initialisation
    recurrent_init: WeightInit = field(default_factory=WeightInit)
    input_init: WeightInit = field(default_factory=lambda: WeightInit(scheme="constant", scale=4.0))
    readout_init: WeightInit = field(default_factory=lambda: WeightInit(scheme="gaussian", scale=1.0))   # W_out
    # V(0) just above the separatrix V = 1 (the unstable fixed point at I = 0): with no input a neuron at
    # V(0) = 1 + d spikes at t = ln((1 + d) / d), about 4.6 for d = 0.01; below 1 it decays back to rest
    initial_voltage: dict = field(default_factory=lambda: dict(input=1.01, hidden=1.01, readout=1.01))
    initial_current: dict = field(default_factory=lambda: dict(input=0.0, hidden=0.0, readout=0.0))

    # which parameter groups are trained
    trainable: dict = field(default_factory=lambda: dict(weights_net=True, weights_in=True, w_out=True, x0=False))


# =====================================================================================================================
# The network
# =====================================================================================================================

class QIFNet:
    """A network built from a NetConfig. Holds the structure; the trainable values live in a params dict, so the same
    net can be simulated and differentiated with any params.

    Structure (set by the constructor):
      config, neuron                    the NetConfig and the spikegd neuron instance
      layer_sizes, n_neurons            neurons per layer, total
      layer_of_neuron                   (N,) layer index of every neuron
      roles                             (N,) "input" / "hidden" / "readout"
      input_neurons, hidden_neurons, readout_neurons      indices
      connection_mask                   (N, N) bool, synapse j -> i present
      neuron_signs                      (N,) Dale sign of every neuron (-1, 0 = free, +1)
      input_mask, n_input_channels      (N, C) bool, task channel c reaches neuron i
      background_neurons                indices of the neurons with a background channel
      background_weights                (N, B) fixed weights of the background channels
      kernel                            the read-out kernel kappa(s) of the output head (from config.readout)
      readout_column                    (N,) column of W_out each read-out neuron feeds, -1 for the others
    """

    def __init__(self, config, layer_sizes, roles, connection_mask, neuron_signs, input_mask):
        """Checks that the structure arrays fit together (ValueError otherwise); use build_net or load_net."""
        check_structure(layer_sizes, roles, connection_mask, neuron_signs, input_mask)
        self.config = config
        self.neuron = config.neuron_type(**config.neuron_kwargs)

        self.layer_sizes = list(layer_sizes)
        self.n_neurons = int(sum(layer_sizes))
        self.layer_of_neuron = np.repeat(np.arange(len(layer_sizes)), layer_sizes)

        self.roles = np.asarray(roles)
        self.input_neurons = np.where(self.roles == "input")[0]
        self.hidden_neurons = np.where(self.roles == "hidden")[0]
        self.readout_neurons = np.where(self.roles == "readout")[0]

        self.connection_mask = np.asarray(connection_mask, bool)
        self.neuron_signs = np.asarray(neuron_signs, int)
        self.input_mask = np.asarray(input_mask, bool)
        self.n_input_channels = self.input_mask.shape[1]

        # background channel b drives neuron background_neurons[b] with a fixed weight
        background = config.background
        if background.enabled:
            chosen = {"all": np.ones(self.n_neurons, bool),
                      "non_input": self.roles != "input",
                      "readout": self.roles == "readout"}[background.applies_to]
            self.background_neurons = np.where(chosen)[0]
        else:
            self.background_neurons = np.zeros(0, int)
        n_background = len(self.background_neurons)
        self.background_weights = np.zeros((self.n_neurons, n_background))
        self.background_weights[self.background_neurons, np.arange(n_background)] = background.weight

        # output head: the kernel, and which column of W_out each read-out neuron feeds
        self.kernel = KERNELS[config.readout.kernel](config.readout.tau_r)
        self.readout_column = np.full(self.n_neurons, -1)
        self.readout_column[self.readout_neurons] = np.arange(len(self.readout_neurons))

    # ---- parameters ----

    def init_params(self):
        """params = {weights_net (N, N), weights_in (N, C), w_out (n_out, k), x0 (2, N)} from the initialisations,
        with the mask and the Dale signs applied."""
        config = self.config
        weights_net = draw_weights(config.recurrent_init, self.connection_mask, self.neuron_signs)
        weights_in = draw_weights(config.input_init, self.input_mask, np.full(self.n_input_channels, EXCITATORY))
        k = len(self.readout_neurons)
        w_out = draw_weights(config.readout_init, np.ones((config.n_output_channels, k), bool), np.full(k, FREE))
        x0 = np.zeros((2, self.n_neurons))
        for role in ROLES:
            x0[0, self.roles == role] = config.initial_voltage.get(role, 0.0)
            x0[1, self.roles == role] = config.initial_current.get(role, 0.0)
        return dict(weights_net=jnp.asarray(weights_net), weights_in=jnp.asarray(weights_in),
                    w_out=jnp.asarray(w_out), x0=jnp.asarray(x0))

    def project_params(self, params):
        """After an update: zero the synapses that are not there, put Dale's signs back, keep input weights >= 0."""
        check_params(self, params)
        signs = jnp.asarray(self.neuron_signs)[None, :]            # sign of the presynaptic neuron, per column
        weights = params["weights_net"] * self.connection_mask
        weights = jnp.where(signs > 0, jnp.maximum(weights, 0.0), weights)
        weights = jnp.where(signs < 0, jnp.minimum(weights, 0.0), weights)
        w_in = jnp.maximum(params["weights_in"] * self.input_mask, 0.0)
        return dict(params, weights_net=weights, weights_in=w_in)

    def trainable_mask(self, params):
        """1 for the parameter groups that are trained, 0 otherwise (multiply the gradient by it)."""
        check_params(self, params)
        return {name: jnp.full_like(value, float(self.config.trainable.get(name, True))) for name, value in params.items()}

    # ---- inputs and simulation ----

    def background_spikes(self, T, seed=None):
        """(times, channel) of the background channels over [0, T]; channels numbered after the task channels."""
        check_positive("T", T)
        background = self.config.background
        rng = np.random.default_rng(background.seed if seed is None else seed)
        times, channels = [np.zeros(0)], [np.zeros(0, int)]
        for channel in range(len(self.background_neurons)):
            spike_times = rng.uniform(0.0, T, rng.poisson(background.rate * T))
            times.append(spike_times)
            channels.append(np.full(len(spike_times), self.n_input_channels + channel))
        return np.concatenate(times), np.concatenate(channels).astype(int)

    def merge_inputs(self, spikes_in, T, seed=None):
        """Task spikes plus background spikes, sorted by time (do this once outside jit if T is fixed)."""
        check_spikes_in(self, spikes_in, T)
        background_times, background_channels = self.background_spikes(T, seed)
        times = np.concatenate([np.asarray(spikes_in[0], float), background_times])
        channels = np.concatenate([np.asarray(spikes_in[1], int), background_channels])
        order = np.argsort(times)
        return jnp.asarray(times[order]), jnp.asarray(channels[order])

    def full_input_weights(self, params):
        """(N, C + B): trainable task-input weights (masked) next to the fixed background weights."""
        return jnp.concatenate([params["weights_in"] * self.input_mask, jnp.asarray(self.background_weights)], 1)

    def simulate(self, params, spikes_in, T, K, add_background=True, background_seed=None):
        """One trial with spikegd's neuron.event. spikes_in = (times, channel) of the task channels; the background
        spikes are added unless add_background is False. Returns (times, is_input, neuron, states).
        T and K come from the task / simulation, not from the network."""
        check_params(self, params)
        check_positive("T", T)
        check_int("K (maximum number of events)", K, low=1)
        check_spikes_in(self, spikes_in, T, task_only=add_background)
        if add_background:
            spikes_in = self.merge_inputs(spikes_in, T, background_seed)
        weights_net = params["weights_net"] * self.connection_mask
        weights_in = self.full_input_weights(params)
        if weights_in.shape[1] == 0:                                # no channels at all: spikegd needs one column
            weights_in = jnp.zeros((self.n_neurons, 1))
        config = dict(N=self.n_neurons, T=T, K=K)
        return self.neuron.event(params["x0"], weights_net, weights_in, spikes_in, config)

    # ---- output head ----

    def output(self, params, events, read_at):
        """y from the events of a trial: read_at (n_times,) gives y (n_out, n_times); read_at (n_windows, 2) of
        (start, end) gives y (n_out, n_windows), read as config.readout.window says."""
        check_params(self, params)
        points, average = _read_points(self, read_at)
        y = _kernel_trace(self, params, events, points)
        return y.mean(-1) if average else y

    def unit_response(self, spike_time, read_at):
        """(n_read,): what the head reads from one read-out spike at spike_time with weight 1 (its derivative in
        spike_time is the l(t_k) of the local rules)."""
        points, average = _read_points(self, read_at)
        response = self.kernel(points - spike_time)
        return response.mean(-1) if average else response

    def forward(self, params, spikes_in, T, K, read_at, add_background=True, background_seed=None):
        """simulate, then the output head: returns (y, events)."""
        events = self.simulate(params, spikes_in, T, K, add_background, background_seed)
        if is_concrete(events[0]) and bool(jnp.isnan(events[0]).any()):
            raise ValueError(f"forward: the simulation did not reach T = {T} within K = {K} events; raise K")
        return self.output(params, events, read_at), events

    # ---- export ----

    def export(self, path, params):
        """Weights, structure and the full config (with every seed) in one .npz."""
        check_params(self, params)
        folder = Path(path).resolve().parent
        if not folder.is_dir():
            raise FileNotFoundError(f"export: folder {folder} does not exist")
        config = asdict(self.config)
        config["neuron_type"] = self.config.neuron_type.__name__
        np.savez(path, **{name: np.asarray(value) for name, value in params.items()},
                 layer_sizes=np.array(self.layer_sizes), roles=self.roles, connection_mask=self.connection_mask,
                 neuron_signs=self.neuron_signs, input_mask=self.input_mask, config=json.dumps(config))

    def describe(self):
        """Print layers, roles, signs, synapses, input and background channels."""
        sign_names = {INHIBITORY: "inhibitory", EXCITATORY: "excitatory", FREE: "free"}
        print(f"{self.config.architecture}: {len(self.layer_sizes)} layer(s) {self.layer_sizes}, N = {self.n_neurons}, "
              f"neuron {type(self.neuron).__name__}")
        for role in ROLES:
            idx = np.where(self.roles == role)[0]
            signs = ", ".join(sorted({sign_names[sign] for sign in self.neuron_signs[idx]})) or "-"
            print(f"  {role:8s} neurons {idx.tolist()}  sign {signs}")
        n_possible = self.n_neurons * (self.n_neurons - 1)
        print(f"  synapses {int(self.connection_mask.sum())} of {n_possible} possible "
              f"(self-synapses {int(np.trace(self.connection_mask))}),  task input channels {self.n_input_channels}")
        background = self.config.background
        if background.enabled:
            print(f"  background: {len(self.background_neurons)} channels ({background.applies_to}), "
                  f"rate {background.rate}, weight {background.weight}")
        else:
            print("  background: off")


def alpha_kernel(tau_r):
    """kappa(s) = (s / tau_r) exp(1 - s / tau_r) for s > 0, else 0 (peak 1 at s = tau_r)."""
    check_positive("alpha_kernel: tau_r", tau_r)

    def kernel(s):
        s = jnp.where(s > 0, s, 0.0)
        return s / tau_r * jnp.exp(1.0 - s / tau_r)

    kernel.tau_r = tau_r
    return kernel


KERNELS = dict(alpha=alpha_kernel)


def _read_points(net, read_at):
    """The times the head evaluates the kernel at, and whether to average over the last axis: read_at (n_times,)
    -> (n_times,); windows (n_windows, 2) -> their ends ("end") or window_points points in each ("mean")."""
    read_at = jnp.asarray(read_at, float)
    if read_at.ndim == 1:
        return read_at, False
    if read_at.ndim != 2 or read_at.shape[1] != 2:
        raise ValueError(f"read_at must be times (n_times,) or windows (n_windows, 2), got shape {read_at.shape}")
    starts, ends = read_at[:, 0], read_at[:, 1]
    if is_concrete(read_at) and not bool(jnp.all(ends > starts)):
        raise ValueError("read_at: every window needs end > start")
    readout = net.config.readout
    if readout.window == "end":
        return ends, False
    fractions = (jnp.arange(readout.window_points) + 0.5) / readout.window_points
    return starts[:, None] + fractions[None, :] * (ends - starts)[:, None], True     # (n_windows, window_points)


def _kernel_trace(net, params, events, times):
    """y at times (any shape): W_out sum over the read-out spikes q of kernel(t - t_q); y has shape (n_out, *times).
    Only spikes of read-out neurons count; input events and padding (neuron -1) get weight 0."""
    spike_times, is_input, neuron, _ = events
    column = jnp.asarray(net.readout_column)[jnp.clip(neuron, 0)]
    counts = (neuron >= 0) & ~is_input & (column >= 0)
    event_weights = jnp.where(counts[None, :], params["w_out"][:, jnp.clip(column, 0)], 0.0)    # (n_out, K)
    lag = times[..., None] - jnp.where(counts, spike_times, 0.0)                                 # (*times, K)
    return jnp.tensordot(event_weights, net.kernel(lag), axes=([1], [-1]))


# ---- Building and loading ----

def build_net(config: NetConfig) -> QIFNet:
    """Build and validate a network, or load its structure from a custom archive."""
    check_type("build_net config", config, NetConfig)
    check_config(config)
    if config.architecture == "custom":
        return load_net(config.custom_path)[0]
    layer_sizes = list(config.neurons_per_layer)
    roles = assign_roles(config, layer_sizes)
    allowed = allowed_synapses(config.architecture, layer_sizes)
    allowed = add_self_connections(allowed, config.self_connections, layer_sizes)
    connection_mask = thin_connections(allowed, config.connectivity)
    neuron_signs = assign_signs(roles, config.hidden_excitatory_fraction, config.sign_seed)
    input_mask = input_channel_mask(roles, config.input_channels)
    return QIFNet(config, layer_sizes, roles, connection_mask, neuron_signs, input_mask)


def assign_roles(config, layer_sizes):
    """input / hidden / readout per neuron. One layer: the first n_input_neurons and the last n_readout_neurons.
    Several layers: all of layer 1 and all of layer L."""
    n_neurons = sum(layer_sizes)
    roles = np.array(["hidden"] * n_neurons, dtype=object)
    if len(layer_sizes) == 1:
        roles[:config.n_input_neurons] = "input"
        roles[n_neurons - config.n_readout_neurons:] = "readout"
    else:
        roles[:layer_sizes[0]] = "input"
        roles[n_neurons - layer_sizes[-1]:] = "readout"
    return roles.astype(str)


def allowed_synapses(architecture, layer_sizes):
    """(N, N) bool: the synapses j -> i the architecture allows (no self-synapses yet)."""
    n_neurons = sum(layer_sizes)
    if architecture == "fc_rnn":
        allowed = np.ones((n_neurons, n_neurons), bool)
    else:                                                       # "fcn": post layer = pre layer + 1
        layer_of_neuron = np.repeat(np.arange(len(layer_sizes)), layer_sizes)
        allowed = layer_of_neuron[:, None] == layer_of_neuron[None, :] + 1
    np.fill_diagonal(allowed, False)
    return allowed


def add_self_connections(allowed, self_connections, layer_sizes):
    if not self_connections.enabled:
        return allowed
    if self_connections.selection == "indices":
        chosen = list(self_connections.indices)
    else:                                                       # "random": n_per_layer[l] neurons of layer l
        rng = np.random.default_rng(self_connections.seed)
        starts = np.cumsum([0] + list(layer_sizes[:-1]))
        chosen = []
        for start, size, count in zip(starts, layer_sizes, self_connections.n_per_layer):
            chosen += (start + rng.choice(size, size=count, replace=False)).tolist()
    allowed = allowed.copy()
    allowed[chosen, chosen] = True
    return allowed


def thin_connections(allowed, connectivity):
    """"full": keep every allowed synapse. "uniform" / "gaussian": each presynaptic neuron draws an out-degree and
    keeps that many of its allowed targets, chosen at random."""
    if connectivity.kind == "full":
        return allowed
    rng = np.random.default_rng(connectivity.seed)
    mask = np.zeros_like(allowed)
    for pre in range(allowed.shape[1]):
        targets = np.where(allowed[:, pre])[0]
        n_allowed = len(targets)
        if n_allowed == 0:
            continue
        if connectivity.kind == "uniform":
            upper = min(connectivity.max_targets or n_allowed, n_allowed)
            out_degree = rng.integers(1, upper + 1)
        else:                                                   # "gaussian"
            mean = n_allowed / 2 if connectivity.mean is None else connectivity.mean
            out_degree = int(np.clip(round(rng.normal(mean, np.sqrt(connectivity.variance))), 0, n_allowed))
        mask[rng.choice(targets, size=out_degree, replace=False), pre] = True
    return mask


def assign_signs(roles, hidden_excitatory_fraction, seed):
    """Dale sign per neuron: input and hidden inhibitory (a fraction of the hidden ones excitatory), read-out free."""
    signs = np.full(len(roles), INHIBITORY)
    signs[roles == "readout"] = FREE
    hidden = np.where(roles == "hidden")[0]
    n_excitatory = int(round(hidden_excitatory_fraction * len(hidden)))
    if n_excitatory:
        rng = np.random.default_rng(seed)
        signs[rng.choice(hidden, size=n_excitatory, replace=False)] = EXCITATORY
    return signs


def input_channel_mask(roles, input_channels):
    """(N, C) bool. "shared": every input neuron gets all C channels. "private": consecutive blocks, input neuron k
    gets the next channel_counts()[k] channels."""
    input_neurons = np.where(roles == "input")[0]
    mask = np.zeros((len(roles), input_channels.n_input_channels), bool)
    if input_channels.n_input_channels == 0:
        return mask
    if input_channels.mode == "shared":
        mask[input_neurons] = True
        return mask
    counts = channel_counts(input_channels, len(input_neurons))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for neuron, start, count in zip(input_neurons, starts, counts):
        mask[neuron, start:start + count] = True
    return mask


def channel_counts(input_channels, n_input_neurons):
    """Channels per input neuron in "private" mode, from InputChannels.fraction (raises if the split does not work)."""
    n_channels, fraction = input_channels.n_input_channels, input_channels.fraction

    if fraction is None:                                        # an even split
        if n_channels < n_input_neurons:
            raise ValueError(f"input_channels: {n_channels} channels for {n_input_neurons} input neurons, some would "
                             f"get none")
        leftover = n_channels % n_input_neurons
        if leftover and not input_channels.allow_leftover:
            raise ValueError(f"input_channels: {n_channels} channels do not split evenly over {n_input_neurons} input "
                             f"neurons ({leftover} left over); give fractions, or set allow_leftover=True")
        return [n_channels // n_input_neurons] * n_input_neurons

    if isinstance(fraction, (list, tuple, np.ndarray)):
        fractions = [float(share) for share in fraction]
        if len(fractions) != n_input_neurons:
            raise ValueError(f"input_channels.fraction has {len(fractions)} entries but there are {n_input_neurons} "
                             f"input neurons (one entry per input neuron)")
    else:
        check_number("input_channels.fraction", fraction)
        fractions = [float(fraction)] * n_input_neurons

    counts = []
    for neuron, share in enumerate(fractions):
        if not 0.0 < share <= 1.0:
            raise ValueError(f"input_channels.fraction[{neuron}] = {share} is not in (0, 1]")
        count = share * n_channels
        if abs(count - round(count)) > 1e-9 * max(1.0, n_channels):
            raise ValueError(f"input_channels.fraction[{neuron}] = {share} gives {count:g} of the {n_channels} "
                             f"channels, not a whole number")
        counts.append(int(round(count)))

    total = sum(counts)
    if total > n_channels:
        raise ValueError(f"input_channels.fraction adds up to {total / n_channels:g} > 1 ({total} of {n_channels} "
                         f"channels)")
    if total < n_channels and not input_channels.allow_leftover:
        raise ValueError(f"input_channels.fraction adds up to {total / n_channels:g} < 1: {n_channels - total} of "
                         f"{n_channels} channels left over; set allow_leftover=True to allow this")
    return counts


def draw_weights(init, mask, presynaptic_signs):
    """Weights of the shape of mask. A neuron with a Dale sign gets |weight| times its sign; a free neuron keeps the
    sign of the draw. Entries outside mask are 0."""
    mask = np.asarray(mask, bool)
    if init.scheme == "zero":
        return np.zeros(mask.shape)
    if init.normalisation == "fan_in":
        size = init.scale / np.sqrt(np.maximum(mask.sum(1, keepdims=True), 1))
    else:
        size = np.full((mask.shape[0], 1), init.scale)
    if init.scheme == "constant":
        raw = np.broadcast_to(size * init.free_sign, mask.shape)
    else:                                                       # "gaussian"
        raw = size * np.random.default_rng(init.seed).standard_normal(mask.shape)
    signs = np.broadcast_to(np.asarray(presynaptic_signs)[None, :], mask.shape)
    weights = np.where(signs == FREE, raw, signs * np.abs(raw))
    return np.where(mask, weights, 0.0)


# =====================================================================================================================
# 3. Loading
# =====================================================================================================================

def load_net(path):
    """Needs weights_net (N, N); everything else is optional: layer_sizes (default one layer), roles or
    readout_indices (default: no input, last neuron read out), connection_mask (default weights_net != 0),
    neuron_signs (default: free), input_mask / weights_in, w_out, x0, config (json, as written by export)."""
    data = open_npz(path)
    weights_net = np.asarray(data["weights_net"], float)
    n_neurons = weights_net.shape[0]

    config = NetConfig(architecture="custom", custom_path=str(path))
    saved = json.loads(str(data["config"])) if "config" in data else {}
    for key in ("n_output_channels", "neuron_kwargs", "trainable"):
        if key in saved:
            setattr(config, key, saved[key])
    if "input_channels" in saved:
        config.input_channels = InputChannels(**saved["input_channels"])
    if "background" in saved:
        config.background = Background(**saved["background"])
    if "readout" in saved:
        config.readout = Readout(**saved["readout"])

    layer_sizes = data["layer_sizes"].tolist() if "layer_sizes" in data else [n_neurons]
    roles = data["roles"].astype(str) if "roles" in data else roles_from_readout_indices(data, n_neurons)
    connection_mask = data["connection_mask"] if "connection_mask" in data else weights_net != 0
    neuron_signs = data["neuron_signs"] if "neuron_signs" in data else np.full(n_neurons, FREE)
    if "input_mask" in data:
        input_mask = data["input_mask"]
    elif "weights_in" in data:
        input_mask = np.asarray(data["weights_in"]) != 0
    else:
        input_mask = np.zeros((n_neurons, 1), bool)
    net = QIFNet(config, layer_sizes, roles, connection_mask, neuron_signs, input_mask)

    params = net.init_params()
    params["weights_net"] = jnp.asarray(weights_net)
    for key in ("weights_in", "w_out", "x0"):
        if key in data:
            if data[key].shape != params[key].shape:
                raise ValueError(f"load_net: {key} has shape {data[key].shape}, the structure needs "
                                 f"{tuple(params[key].shape)}")
            params[key] = jnp.asarray(data[key], float)
    check_params(net, params)
    return net, params


def open_npz(path):
    """The .npz at path, with a square weights_net in it."""
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"load_net: no file {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"load_net: {path} is not a readable .npz ({error})") from error
    if "weights_net" not in data:
        raise KeyError(f"load_net: {path} has no weights_net (it has {sorted(data.files)})")
    shape = data["weights_net"].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"load_net: weights_net must be (N, N), got {shape}")
    return data


def roles_from_readout_indices(data, n_neurons):
    """Every neuron hidden except readout_indices (default: the last neuron)."""
    readout = np.asarray(data["readout_indices"] if "readout_indices" in data else [n_neurons - 1]).astype(int)
    if readout.size == 0 or readout.min() < 0 or readout.max() >= n_neurons:
        raise ValueError(f"load_net: readout_indices {readout.tolist()} must be non-empty and in [0, {n_neurons})")
    roles = np.array(["hidden"] * n_neurons, dtype=object)
    roles[readout] = "readout"
    return roles.astype(str)


# =====================================================================================================================
# 4. Checks
# =====================================================================================================================

def check_config(config):
    """Every field of a NetConfig (called by build_net). Returns nothing; raises on the first wrong field."""
    check_choice("architecture", config.architecture, ("fcn", "fc_rnn", "custom"))
    if config.architecture == "custom":
        if config.custom_path is None or not Path(config.custom_path).is_file():
            raise FileNotFoundError(f"architecture 'custom' needs custom_path to an .npz, got {config.custom_path!r}")
        return
    n_input_neurons = check_layers(config)
    check_readout(config.readout)
    check_input_channels(config.input_channels, n_input_neurons)
    check_background(config.background, n_input_neurons, sum(config.neurons_per_layer))
    check_neuron(config.neuron_type, config.neuron_kwargs)
    check_connectivity(config.connectivity)
    check_self_connections(config.self_connections, config.neurons_per_layer)
    check_number("hidden_excitatory_fraction", config.hidden_excitatory_fraction, low=0.0, high=1.0)
    check_int("sign_seed", config.sign_seed, low=0)
    for name in ("recurrent_init", "input_init", "readout_init"):
        check_weight_init(name, getattr(config, name))
    check_initial_state(config)
    check_trainable(config.trainable)


def check_layers(config):
    """Layer sizes, the input / read-out counts and n_output_channels. Returns the number of input neurons."""
    sizes = config.neurons_per_layer
    if not isinstance(sizes, (list, tuple)) or len(sizes) == 0:
        raise TypeError(f"neurons_per_layer must be a non-empty list, got {sizes!r}")
    for layer, size in enumerate(sizes):
        check_int(f"neurons_per_layer[{layer}]", size, low=1)
    if config.architecture == "fcn" and len(sizes) < 2:
        raise ValueError("architecture 'fcn' needs at least two layers")
    check_int("n_output_channels", config.n_output_channels, low=1)
    if len(sizes) > 1:
        return sizes[0]
    check_int("n_input_neurons", config.n_input_neurons, low=0)
    check_int("n_readout_neurons", config.n_readout_neurons, low=1)
    n_input_and_readout = config.n_input_neurons + config.n_readout_neurons
    if n_input_and_readout > sizes[0]:
        raise ValueError(f"n_input_neurons + n_readout_neurons = {n_input_and_readout} is more than the {sizes[0]} "
                         f"neurons of the layer")
    return config.n_input_neurons


def check_readout(readout):
    check_type("readout", readout, Readout)
    check_choice("readout.kernel", readout.kernel, tuple(KERNELS))
    check_number("readout.tau_r", readout.tau_r, low=0.0, low_open=True)
    check_choice("readout.window", readout.window, ("mean", "end"))
    check_int("readout.window_points", readout.window_points, low=1)


def check_input_channels(input_channels, n_input_neurons):
    check_type("input_channels", input_channels, InputChannels)
    check_int("input_channels.n_input_channels", input_channels.n_input_channels, low=0)
    check_choice("input_channels.mode", input_channels.mode, ("shared", "private"))
    check_bool("input_channels.allow_leftover", input_channels.allow_leftover)
    n_channels = input_channels.n_input_channels
    if n_input_neurons == 0 and n_channels > 0:
        raise ValueError(f"there are no input neurons to receive the {n_channels} task channels")
    if n_channels == 0:                                         # a network without task input (e.g. the cyclic task)
        if n_input_neurons > 0:
            raise ValueError(f"input_channels.n_input_channels = 0 needs no input neurons, got {n_input_neurons}")
        return
    if input_channels.mode == "shared" and input_channels.fraction is not None:
        raise ValueError("input_channels.fraction is only used with mode='private'")
    if input_channels.mode == "private":
        channel_counts(input_channels, n_input_neurons)


def check_background(background, n_input_neurons, n_neurons):
    check_type("background", background, Background)
    check_bool("background.enabled", background.enabled)
    if not background.enabled:
        return
    check_choice("background.applies_to", background.applies_to, ("all", "non_input", "readout"))
    check_number("background.rate", background.rate, low=0.0)
    check_number("background.weight", background.weight, low=0.0, low_open=True)
    check_int("background.seed", background.seed, low=0)
    if background.applies_to == "non_input" and n_input_neurons == n_neurons:
        raise ValueError("background.applies_to = 'non_input' but every neuron is an input neuron")


def check_neuron(neuron_type, neuron_kwargs):
    if not callable(neuron_type):
        raise TypeError(f"neuron_type must be a neuron class, got {neuron_type!r}")
    check_type("neuron_kwargs", neuron_kwargs, dict)
    try:
        neuron = neuron_type(**neuron_kwargs)
    except TypeError as error:
        raise ValueError(f"neuron_type {neuron_type.__name__} does not accept neuron_kwargs {neuron_kwargs}: "
                         f"{error}") from error
    if not hasattr(neuron, "event"):
        raise TypeError(f"neuron_type {neuron_type.__name__} has no event() method (needed by simulate)")


def check_connectivity(connectivity):
    check_type("connectivity", connectivity, Connectivity)
    check_choice("connectivity.kind", connectivity.kind, ("full", "uniform", "gaussian"))
    if connectivity.max_targets is not None:
        check_int("connectivity.max_targets", connectivity.max_targets, low=1)
    if connectivity.mean is not None:
        check_number("connectivity.mean", connectivity.mean, low=0.0)
    check_number("connectivity.variance", connectivity.variance, low=0.0)
    check_int("connectivity.seed", connectivity.seed, low=0)


def check_self_connections(self_connections, layer_sizes):
    check_type("self_connections", self_connections, SelfConnections)
    check_bool("self_connections.enabled", self_connections.enabled)
    if not self_connections.enabled:
        return
    check_choice("self_connections.selection", self_connections.selection, ("random", "indices"))
    if self_connections.selection == "random":
        if len(self_connections.n_per_layer) != len(layer_sizes):
            raise ValueError(f"self_connections.n_per_layer has {len(self_connections.n_per_layer)} entries for "
                             f"{len(layer_sizes)} layers")
        for layer, (count, size) in enumerate(zip(self_connections.n_per_layer, layer_sizes)):
            check_int(f"self_connections.n_per_layer[{layer}]", count, low=0, high=size)
        return
    if len(self_connections.indices) == 0:
        raise ValueError("self_connections.selection = 'indices' needs a non-empty indices list")
    if len(set(self_connections.indices)) != len(self_connections.indices):
        raise ValueError(f"self_connections.indices has repeats: {self_connections.indices}")
    for index in self_connections.indices:
        check_int("self_connections.indices entry", index, low=0, high=sum(layer_sizes) - 1)


def check_weight_init(name, init):
    check_type(name, init, WeightInit)
    check_choice(f"{name}.scheme", init.scheme, ("zero", "constant", "gaussian"))
    check_choice(f"{name}.normalisation", init.normalisation, ("none", "fan_in"))
    check_number(f"{name}.scale", init.scale, low=0.0)
    check_choice(f"{name}.free_sign", init.free_sign, (INHIBITORY, EXCITATORY))
    check_int(f"{name}.seed", init.seed, low=0)


def check_initial_state(config):
    for name in ("initial_voltage", "initial_current"):
        values = getattr(config, name)
        check_type(name, values, dict)
        for role, value in values.items():
            check_choice(f"{name} key", role, ROLES)
            check_number(f"{name}[{role!r}]", value)


def check_trainable(trainable):
    check_type("trainable", trainable, dict)
    for name, value in trainable.items():
        check_choice("trainable key", name, PARAM_NAMES)
        check_bool(f"trainable[{name!r}]", value)


def check_structure(layer_sizes, roles, connection_mask, neuron_signs, input_mask):
    """The arrays QIFNet is built from fit together."""
    n_neurons = int(sum(layer_sizes))
    roles, connection_mask = np.asarray(roles), np.asarray(connection_mask)
    neuron_signs, input_mask = np.asarray(neuron_signs), np.asarray(input_mask)
    if roles.shape != (n_neurons,):
        raise ValueError(f"roles has shape {roles.shape}, the layers have {n_neurons} neurons")
    unknown = set(roles.tolist()) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; allowed {list(ROLES)}")
    if not (roles == "readout").any():
        raise ValueError("the network has no read-out neuron")
    if connection_mask.shape != (n_neurons, n_neurons):
        raise ValueError(f"connection_mask has shape {connection_mask.shape}, needs ({n_neurons}, {n_neurons})")
    if neuron_signs.shape != (n_neurons,) or not np.isin(neuron_signs, (INHIBITORY, FREE, EXCITATORY)).all():
        raise ValueError(f"neuron_signs must be ({n_neurons},) with entries -1, 0, 1")
    if input_mask.ndim != 2 or input_mask.shape[0] != n_neurons:
        raise ValueError(f"input_mask has shape {input_mask.shape}, needs ({n_neurons}, C)")


def check_params(net, params):
    """Every parameter group, with the shape the structure needs (and, outside jit, finite values)."""
    check_type("params", params, dict)
    missing = [name for name in PARAM_NAMES if name not in params]
    if missing:
        raise KeyError(f"params is missing {missing}")
    shapes = dict(weights_net=(net.n_neurons, net.n_neurons),
                  weights_in=(net.n_neurons, net.n_input_channels),
                  w_out=(net.config.n_output_channels, len(net.readout_neurons)),
                  x0=(2, net.n_neurons))
    for name, shape in shapes.items():
        value = params[name]
        if tuple(np.shape(value)) != shape:
            raise ValueError(f"params[{name!r}] has shape {tuple(np.shape(value))}, needs {shape}")
        if is_concrete(value) and not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"params[{name!r}] has NaN or inf entries")


def check_spikes_in(net, spikes_in, T, task_only=True):
    """spikes_in = (times, channels): two 1-D arrays of the same length, times in [0, T], channels in [0, C) (task
    spikes only) or in [0, C + B) (task and background spikes already merged)."""
    if not isinstance(spikes_in, (tuple, list)) or len(spikes_in) != 2:
        raise TypeError("spikes_in must be a pair (times, channels)")
    times, channels = spikes_in
    if np.ndim(times) != 1 or np.ndim(channels) != 1 or np.shape(times) != np.shape(channels):
        raise ValueError(f"spikes_in: times {np.shape(times)} and channels {np.shape(channels)} must be 1-D arrays of "
                         f"the same length")
    if is_concrete(times) and np.size(times):
        time_values = np.asarray(times, float)
        after_T = is_concrete(T) and time_values.max() > T
        if not np.isfinite(time_values).all() or time_values.min() < 0.0 or after_T:
            raise ValueError(f"spikes_in: input spike times must be finite and in [0, T = {T}]")
    if is_concrete(channels) and np.size(channels):
        channel_values = np.asarray(channels)
        if not np.issubdtype(channel_values.dtype, np.integer):
            raise TypeError(f"spikes_in: channels must be integers, got {channel_values.dtype}")
        n_allowed = net.n_input_channels + (0 if task_only else len(net.background_neurons))
        if channel_values.min() < 0 or channel_values.max() >= n_allowed:
            raise ValueError(f"spikes_in: channel indices must be in [0, {n_allowed}), got "
                             f"[{channel_values.min()}, {channel_values.max()}]")
