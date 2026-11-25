"""QIF event Jacobians, learning-rule templates, and the online RFLO presets.

A rule receives the simulated events and dL/dy from the trainer. The template objects
configure sensitivity, trace updates, and how error is fed back to each neuron.

The event Jacobians use the tau = 1 closed forms. A neuron's state for the own maps is
(V, b) after an input and (t_spike, b) after an output. The reset maps use the analytic
limit V = -infinity, which differs from the finite reset in SpikeGD's simulator.
"""

import inspect
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


# ---- Closed-form QIF event maps ----

PI = jnp.pi


# =====================================================================================================================
# Closed forms
# =====================================================================================================================

def vector_field(V, b):
    """f(V, b) = (V^2 - V + b, -2 b)."""
    return jnp.stack([V ** 2 - V + b, -2.0 * b])


def voltage(V0, b, t):
    """V(t) from (V0, b), eq. (c-V), for b > 0, b = 0 and b < 0 (all branches)."""
    root_pos = jnp.sqrt(jnp.maximum(b, 1e-300))
    root_neg = jnp.sqrt(jnp.maximum(-b, 1e-300))
    decay = jnp.exp(-t)
    positive = root_pos * decay * jnp.tan(jnp.arctan(V0 / root_pos) + root_pos * (1.0 - decay))
    zero = V0 / (V0 - (V0 - 1.0) * jnp.exp(t))
    ratio = V0 / root_neg
    above = root_neg * decay / jnp.tanh(jnp.arctanh(1.0 / jnp.where(jnp.abs(ratio) > 1, ratio, 2.0))
                                         - root_neg * (1.0 - decay))
    below = root_neg * decay * jnp.tanh(jnp.arctanh(jnp.where(jnp.abs(ratio) < 1, ratio, 0.0))
                                        - root_neg * (1.0 - decay))
    on_line = jnp.sign(V0) * root_neg * decay
    negative = jnp.where(jnp.abs(ratio) > 1, above, jnp.where(jnp.abs(ratio) < 1, below, on_line))
    return jnp.where(b > 0, positive, jnp.where(b < 0, negative, zero))


def voltage_from_reset(b, t):
    """V(t) after a reset (V0 = -inf), b > 0: sqrt(b) e^{-t} tan(-pi/2 + sqrt(b) (1 - e^{-t}))."""
    root = jnp.sqrt(b)
    decay = jnp.exp(-t)
    return root * decay * jnp.tan(-PI / 2 + root * (1.0 - decay))


def spike_time(V0, b):
    """t_sp(V0, b), eq. (c-tsp); inf where no spike follows."""
    root_pos = jnp.sqrt(jnp.maximum(b, 1e-300))
    root_neg = jnp.sqrt(jnp.maximum(-b, 1e-300))
    arg_pos = 2.0 * root_pos / (2.0 * jnp.arctan(V0 / root_pos) - PI + 2.0 * root_pos)
    arg_zero = V0 / (V0 - 1.0)
    arcoth = jnp.arctanh(root_neg / jnp.where(V0 > root_neg, V0, 2.0 * root_neg))
    arg_neg = root_neg / (root_neg - arcoth)
    argument = jnp.where(b > 0, arg_pos, jnp.where(b < 0, jnp.where(V0 > root_neg, arg_neg, -1.0),
                                                   jnp.where(V0 > 1.0, arg_zero, -1.0)))
    return jnp.where(argument > 1.0, jnp.log(jnp.where(argument > 1.0, argument, 2.0)), jnp.inf)


def spike_time_from_reset(b):
    """t_sp(-inf, b) = -ln(1 - pi / sqrt(b)), eq. (c-reset); inf unless b > pi^2."""
    root = jnp.sqrt(jnp.maximum(b, 1e-300))
    fires = b > PI ** 2
    return jnp.where(fires, -jnp.log(1.0 - PI / jnp.where(fires, root, 2 * PI)), jnp.inf)


def grad_spike_time(V0, b):
    """grad t_sp = (dt_sp/dV0, dt_sp/db) at (V0, b)."""
    return jnp.stack(jax.grad(spike_time, argnums=(0, 1))(V0, b))


# =====================================================================================================================
# Own maps of one neuron
# =====================================================================================================================

def J_in_to_in(V0, b, gap):
    """Input -> input, a time gap apart: the flow of V and the decay of b."""
    dV = jax.grad(voltage, argnums=(0, 1))(V0, b, gap)
    return jnp.array([[dV[0], dV[1]], [0.0, jnp.exp(-2.0 * gap)]])


def J_in_to_out(V0, b):
    """Input (state (V0, b) after it) -> the next output: (t_out, b_out) = (t_in + t_sp, b e^{-2 t_sp})."""
    gradient = grad_spike_time(V0, b)
    t_sp = spike_time(V0, b)
    b_out = b * jnp.exp(-2.0 * t_sp)
    return jnp.stack([gradient, -2.0 * b_out * gradient + jnp.array([0.0, jnp.exp(-2.0 * t_sp)])])


def J_out_to_in(b_out, gap):
    """Output (reset current b_out) -> an input at a fixed time, a time gap after the reset."""
    V_before = voltage_from_reset(b_out, gap)
    b_before = b_out * jnp.exp(-2.0 * gap)
    dV_db = jax.grad(voltage_from_reset, argnums=0)(b_out, gap)
    return jnp.array([[-(V_before ** 2 - V_before + b_before), dV_db], [2.0 * b_before, jnp.exp(-2.0 * gap)]])


def J_out_to_out(b_out):
    """Output -> the next output with no input between (b_out = c^2 > pi^2)."""
    c = jnp.sqrt(b_out)
    return jnp.array([[1.0, -PI / (2.0 * c ** 2 * (c - PI))], [0.0, (c - PI) / c]])


# =====================================================================================================================
# Blocks from a spike that comes back as an input
# =====================================================================================================================

def J_rec(w):
    """The neuron's own spike re-entering through w: (-w, 2w)^T (1, 0); added to J_out_to_in at the re-entry."""
    return jnp.outer(jnp.array([-w, 2.0 * w]), jnp.array([1.0, 0.0]))


def J_cross(w, V_sender, b_sender):
    """Sender X's spike entering receiver Y through w: (-w, 2w)^T grad t_X, with grad t_X at X's state before its
    spike (row Y, column X of the network Jacobian), whatever Y's next event is."""
    return jnp.outer(jnp.array([-w, 2.0 * w]), grad_spike_time(V_sender, b_sender))


EVENT_MAPS = dict(in_to_in=J_in_to_in, in_to_out=J_in_to_out, out_to_in=J_out_to_in, out_to_out=J_out_to_out,
                  rec=J_rec, cross=J_cross)
OWN_MAPS = ("in_to_in", "in_to_out", "out_to_in", "out_to_out")


# ---- Rule interfaces and templates ----

# =====================================================================================================================
# Base class
# =====================================================================================================================

class LearningRule:
    """What the trainer calls on every trial.

    init_state(net, params, task)  -> the rule's state (traces, B, A, ...), kept between trials
    gradient(net, params, sample, y, events, dloss_dy, state)  -> (grads, state, info)
        dloss_dy  dL/dy for this trial's head output y (the trainer computes the loss and it)
        grads   a dict shaped like params (the weight update direction before the optimizer)
        info    anything the diagnostics may want from this trial (traces, learning signals, ...)
    algorithm      "online" (one gradient per trial) or "teacher_phases" (see Trainer._train_with_teacher)
    """

    name = "rule"
    algorithm = "online"

    def init_state(self, net, params, task):
        return {}

    def gradient(self, net, params, sample, y, events, dloss_dy, state):
        raise NotImplementedError


# =====================================================================================================================
# Temporally local: RTRL with masked or surrogate Jacobian blocks
# =====================================================================================================================

@dataclass
class JacobianMask:
    """Which blocks of J(k) are kept (True) or set to zero (False).

    own      per own map type ("in_to_in", "in_to_out", "out_to_in", "out_to_out")
    rec      the re-entry block J_rec of a neuron's own spike
    cross    the cross blocks J_{X -> Y}: True / False for all, or an (N, N) bool array indexed [receiver, sender]
    """
    own: dict = field(default_factory=lambda: {name: True for name in OWN_MAPS})
    rec: bool = True
    cross: object = True


@dataclass
class JacobianSurrogate:
    """Replacements for the true maps: map type -> function with the same arguments as the map in this module
    (e.g. {"out_to_in": lambda b_out, gap: ...}). Types not given use the true map."""
    maps: dict = field(default_factory=dict)


@dataclass
class TemporallyLocalRTRL(LearningRule):
    """RTRL over the events of a trial, with J(k) built from the event maps, then masked and/or surrogated.

    name         label used in the records
    mask         JacobianMask (default: keep every block, the exact gradient)
    surrogate    JacobianSurrogate (default: none)
    """
    name: str = "rtrl"
    mask: JacobianMask = field(default_factory=JacobianMask)
    surrogate: JacobianSurrogate = field(default_factory=JacobianSurrogate)

    algorithm = "online"

    def block(self, map_type, *arguments):
        """One block of J(k): the surrogate if there is one, else the true map; zero if the mask drops it.
        (For "cross" the mask can depend on the (receiver, sender) pair: see cross_kept.)"""
        function = self.surrogate.maps.get(map_type, EVENT_MAPS[map_type])
        value = function(*arguments)
        kept = self.mask.rec if map_type == "rec" else (True if map_type == "cross" else self.mask.own[map_type])
        return value if kept else 0.0 * value

    def cross_kept(self, receiver, sender):
        cross = self.mask.cross
        return bool(cross) if isinstance(cross, bool) else bool(cross[receiver, sender])

    def event_jacobian(self, net, params, events, k):
        """J(k), (2N, 2N): for every neuron its own map for its event at step k (identity if it has none), plus
        J_rec for a re-entering own spike and J_{X -> Y} in row Y, column X for every spike of X arriving at Y."""
        raise NotImplementedError

    def immediate_influence(self, net, events, k):
        """Mbar(k), (2N, P): (0, 1) in the b-row of every neuron receiving an input at step k, in the column of the
        synapse it came through."""
        raise NotImplementedError

    def learning_signal(self, net, params, sample, y, events, k):
        """cbar(k) = dL/da(k), (2N,): nonzero only in the time row of a read-out neuron that fires at step k, where it
        is W_out times the head's dL/dt for that spike."""
        raise NotImplementedError

    def gradient(self, net, params, sample, y, events, dloss_dy, state):
        """M(k) = J(k) M(k-1) + Mbar(k) over the events (lax.scan), dL/dw = sum_k cbar(k) M(k); dL/dW_out exact."""
        raise NotImplementedError


def exact_rtrl():
    """The exact gradient as RTRL: every block kept, no surrogate."""
    return TemporallyLocalRTRL(name="exact")


def no_cross_rtrl():
    """RTRL with every cross block and the re-entry block dropped (the e-prop / RFLO side of the paper)."""
    return TemporallyLocalRTRL(name="no_cross", mask=JacobianMask(rec=False, cross=False))


# =====================================================================================================================
# Spatially (neurally) local: the templates, and the rule that runs them
# =====================================================================================================================
#
# The trace of synapse j -> i (j a neuron or a task channel) is a two-vector E_ij = (E_V, E_b): how w_ij has moved
# neuron i's state. At every event k of the trial, for every neuron i, with D the time since the previous event and
# x_i its state after it:
#     E_i <- decay_i * own_map(prev_i -> cur_i)(x_i, D, alpha_i) E_i  [+ immediate(alpha_i) in the row of w_ij if
#                                                                       the event is an input to i through w_ij]
# prev_i = "out" if i fired at the previous event, else "in"; cur_i = "out" if i fires at this event, else "in" (the
# flow up to an event of another neuron is an "in" step with nothing arriving); decay_i applies only if the event
# reaches i (an input to i, a spike into i, or i's own spike). When i fires, before its trace is updated,
#     s_k = readout(E_i, x_i, prev_i == "out", alpha_i)            the time row: how w_i. has moved this spike
# and the update of w_ij collects sum_o c_io(k) l_o(t_k) s_k(ij), c from the LearningSignal and
# l_o(t_k) = sum_r dL/dy_or d(unit response of a spike at t_k)_r / dt_k.

def check_template(owner, name, function, arguments):
    """function must be callable with exactly these arguments (the template's declared signature)."""
    if not callable(function):
        raise TypeError(f"{owner}.{name} must be a function{arguments}, got {function!r}")
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return
    positional = [p for p in parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in positional if p.default is p.empty]
    takes_any = any(p.kind == p.VAR_POSITIONAL for p in parameters.values())
    if not takes_any and not len(required) <= len(arguments) <= len(positional):
        raise TypeError(f"{owner}.{name} must take {len(arguments)} arguments {arguments}, got {len(positional)}")


@dataclass
class LearningSignal:
    """c_update template.

    init_B(n_neurons, n_out, rng) -> B (n_neurons, n_out)         the feedback weights at the start
    factor(B, w_out, readout_column, is_readout) -> (n_neurons, n_out)
        c_io for every neuron i and output channel o (readout_column[i]: i's column of W_out, -1 if not a
        read-out neuron)
    """
    name: str
    init_B: object
    factor: object

    def __post_init__(self):
        check_template("LearningSignal", "init_B", self.init_B, ("n_neurons", "n_out", "rng"))
        check_template("LearningSignal", "factor", self.factor, ("B", "w_out", "readout_column", "is_readout"))


def bench_readout(E_i, x_i, previous_was_output, alpha_i):
    """The bench's read-out of the time row: grad t_sp(x_i) . E_i."""
    return E_i @ grad_spike_time(x_i[0], x_i[1])


def no_fit(gap_samples, alpha_old):
    return None


@dataclass
class Sensitivity:
    """A_update template.

    own_maps      {"in_to_in", "in_to_out", "out_to_in", "out_to_out"} -> f(x_i, gap, alpha_i) -> (2, 2)
                  x_i: the neuron's (V, b) after its previous event (after an output: V at the reset, b = b_out);
                  alpha_i: its column of alpha (2,). Must be jax-traceable (they run inside lax.scan, vmapped over
                  neurons).
    readout       f(E_i, x_i, previous_was_output, alpha_i) -> (n_synapses,): the time row at an output of i
    alpha_range   ((low, high) row 0, (low, high) row 1): alpha(0) ~ uniform; tie_alpha: the same a in both rows
    cross         how the cross maps are handled: "zero" (the only one the event loop runs so far)
    fit_alpha     f(gap_samples, alpha_old) -> new alpha (or None): the teacher's fit of alpha
    """
    name: str
    own_maps: dict
    readout: object = bench_readout
    alpha_range: tuple = ((0.0, 0.0), (0.0, 0.0))
    tie_alpha: bool = False
    cross: str = "zero"
    fit_alpha: object = no_fit

    def __post_init__(self):
        missing = [name for name in OWN_MAPS if name not in self.own_maps]
        if missing:
            raise ValueError(f"Sensitivity {self.name!r}: own_maps must give all four maps; missing {missing}")
        for name in OWN_MAPS:
            check_template("Sensitivity.own_maps", name, self.own_maps[name], ("x_i", "gap", "alpha_i"))
        check_template("Sensitivity", "readout", self.readout, ("E_i", "x_i", "previous_was_output", "alpha_i"))
        check_template("Sensitivity", "fit_alpha", self.fit_alpha, ("gap_samples", "alpha_old"))
        if self.cross != "zero":
            raise NotImplementedError(f"Sensitivity {self.name!r}: cross = {self.cross!r} is not written yet")

    def init_alpha(self, n_neurons, rng):
        (low_v, high_v), (low_b, high_b) = self.alpha_range
        row_v = rng.uniform(low_v, high_v, n_neurons)
        row_b = row_v if self.tie_alpha else rng.uniform(low_b, high_b, n_neurons)
        return jnp.asarray(np.stack([row_v, row_b]))

    def all_maps(self, x_previous, gap, alpha):
        """(4, N, 2, 2): the four own maps for every neuron, in the order of OWN_MAPS (index 2 prev_out + cur_out)."""
        return jnp.stack([jax.vmap(self.own_maps[name], in_axes=(1, None, 1))(x_previous, gap, alpha)
                          for name in OWN_MAPS])


@dataclass
class Trace:
    """B_update template.

    decay(alpha_i) -> (2,)        multiplies (E_V, E_b) at every event that reaches the neuron (after the own map)
    immediate(alpha_i) -> (2,)    added to E_ij when an input arrives through w_ij
    """
    name: str
    decay: object
    immediate: object

    def __post_init__(self):
        check_template("Trace", "decay", self.decay, ("alpha_i",))
        check_template("Trace", "immediate", self.immediate, ("alpha_i",))


@dataclass
class Teacher:
    """The teacher's settings (it runs inside the trainer's phased algorithm, Trainer._train_with_teacher).

    corrects        which parts it sets: "c" (B) and/or "A" (alpha, through Sensitivity.fit_alpha)
    n_corrections   P, the number of phases
    branch_trials   R, trials per phase at which its clean run branches (branch_fraction > 0: that fraction)
    copies          M, perturbed copies per branch point;  sigma: noise on b at every arrival
    pick            "draw": B_i ~ N(mean_i, se_i^2); "mean": B_i = mean_i
    average, trim   "mean", "median" or "trimmed" (with trim): the location of the samples and its standard error
    rescale         rescale the drawn B to |B(0)|
    hidden          hidden rows of B: "credit" (samples (dL/dt_Xk) / l(t_Xk), exact credit) or "random" (keep B)
    ell_min         spikes with |l| < ell_min max|l| are left out of the credit samples
    on_student_copy start every phase from a copy of the student ("+copy") instead of the teacher's own network
    """
    corrects: tuple = ("c",)
    n_corrections: int = 5
    branch_trials: int = 3
    branch_fraction: float = 0.0
    copies: int = 16
    sigma: float = 0.05
    pick: str = "draw"
    average: str = "trimmed"
    trim: float = 0.1
    rescale: bool = False
    hidden: str = "credit"
    ell_min: float = 0.05
    on_student_copy: bool = False

    def branch_count(self, phase_length):
        if self.branch_fraction > 0:
            return min(phase_length, max(1, int(round(self.branch_fraction * phase_length))))
        return min(phase_length, self.branch_trials)

    def location_spread(self, samples):
        """Mean (or median, or trimmed mean) of 1-d samples and its standard error."""
        samples = np.asarray(samples, float)
        if self.average == "mean":
            return samples.mean(), samples.std() / np.sqrt(len(samples))
        if self.average == "median":
            q1, median, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
            return median, 1.2533 * (q3 - q1) / 1.349 / np.sqrt(len(samples))
        low, high = np.quantile(samples, [self.trim, 1 - self.trim])
        kept = samples[(samples >= low) & (samples <= high)]
        kept = samples if len(kept) == 0 else kept
        return kept.mean(), kept.std() / np.sqrt(len(kept))

    def draw_B(self, samples, B_old, b_norm, rng, channel=0):
        """samples: neuron -> 1-d samples from the phase (read-outs: the branches' W_out; hidden: credit ratios).
        Returns the new B (only column `channel` changes) and (mean, se) per neuron."""
        B_new = np.asarray(B_old, float).copy()
        mean, spread = np.full(len(B_new), np.nan), np.full(len(B_new), np.nan)
        for neuron, values in samples.items():
            if len(values) >= 2:
                mean[neuron], spread[neuron] = self.location_spread(values)
                draw = rng.normal(mean[neuron], spread[neuron]) if self.pick == "draw" else mean[neuron]
                B_new[neuron, channel] = draw
        if self.rescale:
            B_new = B_new / max(np.linalg.norm(B_new), 1e-30) * b_norm
        return jnp.asarray(B_new), (mean, spread)

    def run_phase(self, trainer, phase, teacher_state):
        """The teacher's trials of one phase with the branches; returns the samples for draw_B (and the gap samples
        for fit_alpha). Still to write: the teacher part of qif_bench.train."""
        raise NotImplementedError


@dataclass
class SpatiallyLocalRule(LearningRule):
    """A neurally local rule, assembled from the four templates. The event loop (gradient) is written once.

    name, c_update (LearningSignal), A_update (Sensitivity), B_update (Trace), teacher (Teacher or None),
    seed (for B(0) and alpha(0))
    """
    name: str = "local"
    c_update: LearningSignal = None          # required (checked in __post_init__)
    A_update: Sensitivity = None
    B_update: Trace = None
    teacher: Teacher | None = None
    seed: int = 0

    def __post_init__(self):
        for part, template in (("c_update", LearningSignal), ("A_update", Sensitivity), ("B_update", Trace)):
            if not isinstance(getattr(self, part), template):
                raise TypeError(f"SpatiallyLocalRule.{part} must be a {template.__name__} object, got "
                                f"{type(getattr(self, part)).__name__}")
        if self.teacher is not None and not isinstance(self.teacher, Teacher):
            raise TypeError(f"SpatiallyLocalRule.teacher must be a Teacher object or None, got {type(self.teacher)}")

    @property
    def algorithm(self):
        return "online" if self.teacher is None else "teacher_phases"

    def init_state(self, net, params, task):
        rng = np.random.default_rng(self.seed)
        n_out = params["w_out"].shape[0]
        return dict(B=jnp.asarray(self.c_update.init_B(net.n_neurons, n_out, rng)),
                    alpha=self.A_update.init_alpha(net.n_neurons, rng))

    def traces(self, net, x0, events, alpha):
        """Run the traces over the events. Returns s (K, n_sources): the time row of the neuron firing at event k
        (zero where no neuron fires), sources = the N neurons then the C task channels."""
        times, is_input, neuron, states = events
        n_neurons, n_channels = net.n_neurons, net.n_input_channels
        n_sources = n_neurons + n_channels
        connection = jnp.asarray(net.connection_mask)                                 # [receiver, sender]
        reaches_channel = jnp.concatenate([jnp.asarray(net.input_mask),
                                           jnp.asarray(net.background_weights) != 0], 1)  # [neuron, channel]
        if reaches_channel.shape[1] == 0:
            reaches_channel = jnp.zeros((n_neurons, 1), dtype=bool)
        previous_times = jnp.concatenate([jnp.zeros(1), times[:-1]])
        previous_states = jnp.concatenate([x0[None], states[:-1]])
        sensitivity, trace = self.A_update, self.B_update
        decay = jax.vmap(trace.decay, in_axes=1)(alpha)                                # (N, 2)
        immediate = jax.vmap(trace.immediate, in_axes=1)(alpha)                        # (N, 2)
        readout = jax.vmap(sensitivity.readout, in_axes=(0, 1, 0, 1))

        def step(carry, k):
            E, prev_out = carry                                                        # (N, n_sources, 2), (N,)
            valid = neuron[k] >= 0
            source = jnp.maximum(neuron[k], 0)
            spike = valid & ~is_input[k]
            fired = spike & (jnp.arange(n_neurons) == source)
            gap = jnp.where(valid, times[k] - previous_times[k], 0.0)
            x_previous = previous_states[k]                                            # (2, N)
            s_all = readout(E, x_previous, prev_out, alpha)                            # (N, n_sources)
            s = jnp.where(spike, s_all[source], 0.0)
            maps = sensitivity.all_maps(x_previous, gap, alpha)                        # (4, N, 2, 2)
            choice = 2 * prev_out.astype(int) + fired.astype(int)
            own = maps[choice, jnp.arange(n_neurons)]                                  # (N, 2, 2)
            E_new = jnp.einsum("nab,nsb->nsa", own, E)
            channel = jnp.minimum(source, reaches_channel.shape[1] - 1)
            input_reach = jnp.where(is_input[k], reaches_channel[:, channel], False)
            spike_reach = spike & (connection[:, source] != 0)
            reached = valid & (input_reach | spike_reach | fired)
            E_new = E_new * jnp.where(reached[:, None, None], decay[:, None, :], 1.0)
            arriving = jnp.where(is_input[k], n_neurons + source, source)                # the source's column
            trained = jnp.where(is_input[k], source < n_channels, True)
            adds = (input_reach | spike_reach) & valid & trained
            E_new = E_new.at[:, jnp.minimum(arriving, n_sources - 1)].add(
                jnp.where(adds[:, None], immediate, 0.0))
            E_new = jnp.where(valid, E_new, E)
            prev_out = jnp.where(valid, fired, prev_out)
            return (E_new, prev_out), s

        start = (jnp.zeros((n_neurons, n_sources, 2)), jnp.zeros(n_neurons, bool))
        _, s = lax.scan(step, start, jnp.arange(len(times)))
        return s

    def learning_signals(self, net, events, read_at, dloss_dy):
        """l (K, n_out): l_o(t_k) = sum_r dL/dy_or d(unit response at t_k)_r / dt_k, for every event."""
        response_slope = jax.vmap(jax.jacfwd(lambda t: net.unit_response(t, read_at)))(events[0])  # (K, n_read)
        return response_slope @ jnp.asarray(dloss_dy).T

    def gradient(self, net, params, sample, y, events, dloss_dy, state):
        """grad w_ij = sum over outputs k of i of sum_o c_io l_o(t_k) s_k(ij); W_out by its exact gradient."""
        times, is_input, neuron, _ = events
        n_neurons, n_channels = net.n_neurons, net.n_input_channels
        s = self.traces(net, params["x0"], events, state["alpha"])                                 # (K, N + C)
        ell = self.learning_signals(net, events, sample.read_at, dloss_dy)           # (K, n_out)
        is_readout = jnp.asarray(net.readout_column) >= 0
        c = self.c_update.factor(state["B"], params["w_out"], jnp.asarray(net.readout_column), is_readout)
        spike = (neuron >= 0) & ~is_input
        firing = jnp.where(spike, neuron, 0)
        weight = jnp.where(spike, jnp.sum(c[firing] * ell, -1), 0.0)                # (K,)
        rows = jnp.zeros((n_neurons, n_neurons + n_channels)).at[firing].add(weight[:, None] * s)
        grad_w_out = jax.grad(lambda w: jnp.sum(jnp.asarray(dloss_dy) *
                                                net.output(dict(params, w_out=w), events, sample.read_at)))(
            params["w_out"])
        grads = dict(weights_net=rows[:, :n_neurons] * net.connection_mask,
                     weights_in=rows[:, n_neurons:] * net.input_mask,
                     w_out=grad_w_out, x0=jnp.zeros_like(params["x0"]))
        return grads, state, dict(s=s, ell=ell)


# ---- RFLO templates and presets ----

JCLIP = 1e3          # the running bench's clip on |J_VV|, |J_Vb| in exact_decay


# =====================================================================================================================
# c_update
# =====================================================================================================================

def random_B(n_neurons, n_out, rng):
    return jnp.asarray(rng.normal(size=(n_neurons, n_out)))


RANDOM_FEEDBACK = LearningSignal(
    name="B", init_B=random_B,
    factor=lambda B, w_out, readout_column, is_readout: B)

W_OUT_ON_READOUTS = LearningSignal(
    name="Wout", init_B=random_B,
    factor=lambda B, w_out, readout_column, is_readout: jnp.where(
        is_readout[:, None], w_out[:, jnp.maximum(readout_column, 0)].T, B))

READOUTS_ONLY = LearningSignal(
    name="targetsOnly", init_B=random_B,
    factor=lambda B, w_out, readout_column, is_readout: jnp.where(is_readout[:, None], B, 0.0))


# =====================================================================================================================
# A_update
# =====================================================================================================================

def decay_matrix(alpha, gap):
    """e-form flow over a gap D: [[e^{-alpha D}, (e^{-2D} - e^{-alpha D}) / (alpha - 2)], [0, e^{-2D}]]
    (D e^{-2D} for alpha = 2)."""
    decay_v, decay_b = jnp.exp(-alpha * gap), jnp.exp(-2.0 * gap)
    far = jnp.abs(alpha - 2.0) > 1e-6
    mix = jnp.where(far, (decay_b - decay_v) / jnp.where(far, alpha - 2.0, 1.0), gap * decay_b)
    return jnp.array([[decay_v, mix], [0.0, decay_b]])


def reset_voltage_row(matrix):
    """The map followed by e_V = 0 at the neuron's own spike."""
    return matrix.at[0].set(0.0)


def identity_map(x_i, gap, alpha_i):
    return jnp.eye(2)


def e_flow(x_i, gap, alpha_i):
    return decay_matrix(alpha_i[0], gap)


def e_flow_reset(x_i, gap, alpha_i):
    return reset_voltage_row(decay_matrix(alpha_i[0], gap))


def clipped_flow(x_i, gap, alpha_i):
    flow = J_in_to_in(x_i[0], x_i[1], gap)
    return flow.at[0].set(jnp.clip(flow[0], -JCLIP, JCLIP))


def clipped_flow_reset(x_i, gap, alpha_i):
    return reset_voltage_row(clipped_flow(x_i, gap, alpha_i))


def fit_murray(gap_samples, alpha_old, a_range=(0.0, 1.0)):
    """Per neuron, argmin_a sum over its gaps (J - (1 - a))^2 = 1 - mean J, for J_VV and J_bb separately.
    gap_samples: dict(J_VV (n_gaps, N), J_bb (n_gaps, N), ok (n_gaps, N)) from the teacher's trials."""
    ok = np.asarray(gap_samples["ok"])
    count = ok.sum(0)
    sum_v = np.where(ok, gap_samples["J_VV"], 0.0).sum(0)
    sum_b = np.where(ok, gap_samples["J_bb"], 0.0).sum(0)
    low, high = a_range
    new = np.stack([np.clip(1 - sum_v / np.maximum(count, 1), low, high),
                    np.clip(1 - sum_b / np.maximum(count, 1), low, high)])
    return jnp.asarray(np.where(count > 0, new, np.asarray(alpha_old)))


def fit_e_form(gap_samples, alpha_old, grid=(0.0, 3.0, 301)):
    """Per neuron, the alpha whose V row direction (e^{-alpha D}, mix) is closest to the exact one
    r = (dV/dV0, dV/db) over the teacher's gaps. gap_samples: dict(r (n_gaps, N, 2), D (n_gaps,), ok (n_gaps, N))."""
    alphas = np.linspace(grid[0], grid[1], int(grid[2]))
    rows = np.asarray(gap_samples["r"], float)
    rows = rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-30)
    gaps, ok = np.asarray(gap_samples["D"], float), np.asarray(gap_samples["ok"])
    cost = np.zeros((len(alphas), rows.shape[1]))
    for index, alpha in enumerate(alphas):
        model = np.asarray(decay_matrix(alpha, gaps)[0])                  # (2, n_gaps): the V row per gap
        model = (model / np.maximum(np.linalg.norm(model, axis=0, keepdims=True), 1e-30)).T[:, None, :]
        cost[index] = np.where(ok, ((rows - model) ** 2).sum(-1), 0.0).sum(0)
    new = np.asarray(alpha_old, float).copy()
    new[0] = np.where(ok.sum(0) > 0, alphas[np.argmin(cost, 0)], new[0])
    return jnp.asarray(new)


MURRAY = Sensitivity(
    name="murray",
    own_maps=dict(in_to_in=identity_map, in_to_out=identity_map, out_to_in=identity_map, out_to_out=identity_map),
    alpha_range=((0.1, 0.9), (0.1, 0.9)), fit_alpha=fit_murray)

E_FORM = Sensitivity(
    name="e",
    own_maps=dict(in_to_in=e_flow, in_to_out=e_flow_reset, out_to_in=e_flow, out_to_out=e_flow_reset),
    alpha_range=((0.5, 3.0), (0.0, 0.0)), fit_alpha=fit_e_form)

EXACT_DECAY = Sensitivity(
    name="exact_decay",
    own_maps=dict(in_to_in=clipped_flow, in_to_out=clipped_flow_reset, out_to_in=clipped_flow,
                  out_to_out=clipped_flow_reset),
    alpha_range=((0.0, 0.3), (0.0, 0.3)), tie_alpha=True)


def exact_readout(E_i, x_i, previous_was_output, alpha_i):
    """The time row of the exact map into the output: row 0 of J_out_to_out after an output, grad t_sp after an
    input."""
    after_output = E_i @ J_out_to_out(x_i[1])[0]
    after_input = E_i @ grad_spike_time(x_i[0], x_i[1])
    return jnp.where(previous_was_output, after_output, after_input)


EXACT_OWN = Sensitivity(
    name="exact_own",
    own_maps=dict(in_to_in=lambda x_i, gap, alpha_i: J_in_to_in(x_i[0], x_i[1], gap),
                  in_to_out=lambda x_i, gap, alpha_i: J_in_to_out(x_i[0], x_i[1]),
                  out_to_in=lambda x_i, gap, alpha_i: J_out_to_in(x_i[1], gap),
                  out_to_out=lambda x_i, gap, alpha_i: J_out_to_out(x_i[1])),
    readout=exact_readout, alpha_range=((0.0, 0.3), (0.0, 0.3)), tie_alpha=True)


# =====================================================================================================================
# B_update
# =====================================================================================================================

DECAYING_TRACE = Trace(name="decaying", decay=lambda alpha_i: 1.0 - alpha_i,
                       immediate=lambda alpha_i: jnp.array([0.0, 1.0]))

MURRAY_TRACE = Trace(name="murray", decay=lambda alpha_i: 1.0 - alpha_i,
                     immediate=lambda alpha_i: jnp.array([0.0, alpha_i[1]]))

PLAIN_TRACE = Trace(name="plain", decay=lambda alpha_i: jnp.ones(2),
                    immediate=lambda alpha_i: jnp.array([0.0, 1.0]))


# =====================================================================================================================
# The bench's rules
# =====================================================================================================================

MAPS = dict(murray=(MURRAY, MURRAY_TRACE), e=(E_FORM, PLAIN_TRACE), exact_decay=(EXACT_DECAY, DECAYING_TRACE),
            exact_own=(EXACT_OWN, DECAYING_TRACE))
SIGNALS = dict(B=RANDOM_FEEDBACK, W_out_targets=W_OUT_ON_READOUTS, targets_only=READOUTS_ONLY)


def rflo(jacobian="murray", B="random", P=0, signal="B", copy=False, seed=0, **teacher_settings):
    """A SpatiallyLocalRule under the bench's name RFLO_<source>_<P>@<jacobian>, e.g. rflo("e", "teacher", 5) =
    RFLO_teachB_5@e, rflo("murray", "teacher", 5, copy=True) = RFLO_teachB+copy_5@murray."""
    sensitivity, trace = MAPS[jacobian]
    teacher = None
    if B == "teacher":
        teacher = Teacher(n_corrections=P, on_student_copy=copy, **teacher_settings)
    source = "randB" if teacher is None else "teachB" + ("+copy" if copy else "")
    if signal != "B":
        source += "+" + SIGNALS[signal].name
    return SpatiallyLocalRule(name=f"RFLO_{source}_{P if teacher else 0}@{jacobian}", c_update=SIGNALS[signal],
                              A_update=sensitivity, B_update=trace, teacher=teacher, seed=seed)


def bench_rules():
    """The nine local rules of the running bench: random B, teacher B, teacher B with copy, for each jacobian."""
    jacobians = ("murray", "e", "exact_decay")
    return ([rflo(jacobian) for jacobian in jacobians]
            + [rflo(jacobian, "teacher", 5) for jacobian in jacobians]
            + [rflo(jacobian, "teacher", 5, copy=True) for jacobian in jacobians])
