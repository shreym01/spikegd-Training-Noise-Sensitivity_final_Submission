"""QIF RFLO benchmark, including random feedback and phased teacher corrections.

NET, TASKS and BENCH specify the baseline and the ablation grid. Build the system
with build(), draw per-seed values with make_init(), then pass both to train().
"""

import csv
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from . import config as C  # Set JAX defaults before importing it.

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random

from .gradients import make_neuron, w_to_weights_net


# =====================================================================================================================
# 1. NET and TASKS
# =====================================================================================================================

NET = dict(N=3,                 # neurons
           k=2,                 # the last k neurons feed the readout (read-out neurons); the rest are hidden
           init="inhibitory",   # "inhibitory": w = -w_scale; "gaussian": N(0, w_scale^2 / (N-1)); "zero"
           w_scale=1.0,
           seed_w=0,            # for "gaussian"
           n_in=None,           # input channels (None = one per neuron)
           w_in=4.0,            # input weight: channel c drives neuron c
           r_in=1.0,            # Poisson rate per input channel
           seed_in=0,
           T=10.0,              # trial length
           K="auto")            # max events per trial; "auto" = 3 x (events at init) + 100

# Every task: name of the target generator, its parameters, the readout (tau_r, dt_grid, n_max) and n_trials.
# n_max = None: every spike of a read-out neuron enters the readout (slots = K, the most a trial can hold).
# Low load (few target spikes): 2k-10k trials. High load: 20k-40k trials.
_READOUT = dict(tau_r=0.5, dt_grid=0.01, n_max=None)
TASKS = {
    "cycle1": dict(name="cycle", n_cycles=1, jitter=0.5, seed=0, **_READOUT, n_trials=4_000),
    "cycle2": dict(name="cycle", n_cycles=2, jitter=0.5, seed=0, **_READOUT, n_trials=10_000),
    "cycle4": dict(name="cycle", n_cycles=4, jitter=0.5, seed=0, **_READOUT, n_trials=20_000),
    "cycle8": dict(name="cycle", n_cycles=8, jitter=0.5, seed=0, **_READOUT, n_trials=40_000),
    "random": dict(name="random", rate=0.4, seed=0, **_READOUT, n_trials=20_000),
}
BASE_TASK = "cycle2"


def _cycle(p, T, k):
    # the k read-out neurons fire in turn, n_cycles times, at jittered times in [0, T]
    rng = np.random.default_rng(p["seed"])
    n_slots = max(1, int(round(k * p["n_cycles"])))          # n_cycles can be fractional: a partial cycle
    gaps = rng.uniform(1 - p["jitter"], 1 + p["jitter"], n_slots + 1)
    times = np.cumsum(gaps)[:n_slots] / gaps.sum() * T
    return [times[i::k] for i in range(k)]


def _random(p, T, k):
    # independent Poisson targets (at least one spike each)
    rng = np.random.default_rng(p["seed"])
    out = []
    for _ in range(k):
        t = np.sort(rng.uniform(0.05 * T, 0.95 * T, max(1, rng.poisson(p["rate"] * T))))
        out.append(t)
    return out


TASK_FNS = {"cycle": _cycle, "random": _random}     # add a task: f(task, T, k) -> list of k arrays of times


def trials_for(n_target_spikes):
    # 2.5k trials per target spike, kept in [2k, 40k]: low load 2k-10k, high load 20k-40k
    return int(min(40_000, max(2_000, 2_500 * n_target_spikes)))


def cycles_task(ratio, k=None):
    # the cycle task with n_cycles = ratio * k (k = NET["k"]); registered in TASKS, returns its name
    k = NET["k"] if k is None else k
    n_cycles = ratio * k
    if k == NET["k"] and n_cycles == TASKS[BASE_TASK]["n_cycles"]:
        return BASE_TASK
    name = f"cyc_per_k={ratio:g}" + ("" if k == NET["k"] else f",k={k}")
    slots = max(1, int(round(k * n_cycles)))
    TASKS[name] = dict(name="cycle", n_cycles=n_cycles, jitter=0.5, seed=0, **_READOUT, n_trials=trials_for(slots))
    return name


# =====================================================================================================================
# 2. The local rule object
# =====================================================================================================================

@dataclass(frozen=True)
class LocalRule:
    """One rule. kind = "local" (RFLO family), "exact" (autodiff) or "readout_only" (only W_out learns).

    jacobian  own block J^{ii} of the influence recursion, i.e. how the trace of neuron i evolves (JACOBIANS):
              "murray"       diag(1 - a_V, 1 - a_b) at every event of i, Mbar scaled by (a_V, a_b)
              "e"            e_V' = -alpha e_V + e_b, e_b' = -2 e_b between events, e_V = 0 at i's own spike
              "exact_decay"  (1 - a) x the exact subthreshold variational map of i over the gap
    cross     the cross blocks J^{iX} (i != X) in the trace evolution:
              "none"         J^{iX} = 0, every trace evolves with its own block only (RFLO)
    signal    the learning signal of a neuron with an output at t_k is c_i * l(t_k), with c_i:
              "B"            c = B for every neuron (Murray)
              "W_out_targets" c = W_out for read-out neurons, B for hidden ones
              "targets_only" c = B for read-out neurons, 0 for hidden ones
    B_from, alpha_from   "random" (drawn once per seed, fixed) or "teacher" (corrected P times)
    P                    number of teacher corrections (phases)
    pick, R, M, sigma, average, trim, rescale     teacher settings: at each of R trials of a phase the teacher's clean
                         run branches into M perturbed copies (noise sigma on b at every arrival); every branch keeps
                         training with the exact gradient for the rest of the phase and is perturbed again at every
                         later R trial. B is drawn from the branches' W_out (read-outs) and their exact credit
                         ratios (hidden) at the end of the phase (see train, pick_B)
    """
    kind: str = "local"
    jacobian: str = "murray"
    cross: str = "none"
    signal: str = "B"
    B_from: str = "random"
    alpha_from: str = "random"
    P: int = 0
    pick: str = "draw"          # "draw": B ~ N(mu, se^2), se = standard error of mu over the phase's samples; "mean": B = mu
    R: int = 3                  # perturbed teacher trials per phase
    R_frac: float = 0.0         # > 0: perturbed trials per phase = max(1, round(R_frac * trials in the phase)), else R
    M: int = 16                 # perturbations per perturbed trial
    sigma: float = 0.05         # noise on b at every arrival of a network spike
    average: str = "trimmed"    # "mean" | "median" | "trimmed"
    trim: float = 0.1
    rescale: bool = False       # rescale the drawn B to |B(0)| (off: B keeps the scale of W_out)
    train_w: bool = True
    hidden_B: str = "credit"    # teacher B for hidden neurons: "credit" = drawn from (dL/dt_Xk) / l(t_Xk) over X's spikes
                                # in the teacher's trials (dL/dt_Xk exact, all paths); "random" = keep B(0)
    teacher_on: str = "own"     # "own": perturb the teacher's own network (trained one phase ahead with the exact
                                # gradient); "student": perturb a copy of the student's current weights (and use its
                                # optimizer state) at the start of each phase, no separate teacher network
    ell_min: float = 0.05       # spikes with |l(t_Xk)| < ell_min * max|l| of that trial are left out of the ratio
    label: str = ""             # optional extra tag for the name

    @property
    def teacher(self):
        return self.kind == "local" and self.P > 0 and "teacher" in (self.B_from, self.alpha_from)

    @property
    def name(self):
        if self.kind != "local":
            return self.kind
        src = "randB" if self.B_from == "random" else "teachB" + ("" if self.pick == "draw" else "-" + self.pick)
        if self.alpha_from == "teacher":
            src += "+teachA"
        if self.cross != "none":
            src += "+" + self.cross
        if self.teacher and self.teacher_on == "student":
            src += "+copy"
        if self.B_from == "teacher" and self.hidden_B != "credit":
            src += "+hid" + self.hidden_B
        if self.signal != "B":
            src += "+" + {"W_out_targets": "Wout", "targets_only": "targetsOnly"}[self.signal]
        if self.label:
            src += "+" + self.label
        return f"RFLO_{src}_{self.P if self.teacher else 0}@{self.jacobian}"


EXACT = LocalRule(kind="exact")
READOUT_ONLY = LocalRule(kind="readout_only")


def rflo(jacobian="murray", B="random", P=0, **kw):
    # shorthand: rflo("e", "teacher", 5) -> RFLO_teachB_5@e
    return LocalRule(jacobian=jacobian, B_from=B, P=P, **kw)


# =====================================================================================================================
# 3. BENCH: optimizer, ablations (net, tasks, rules), seeds
# =====================================================================================================================

OPTIMIZER = dict(name="adabelief",        # "adabelief" | "adam" | "sgd"
                 lr=C.LR,
                 decay_steps=C.DECAY_STEPS,  # lr * exp(-n / decay_steps)
                 clip_inhibitory=False,      # keep w <= 0 after every update
                 n_probes=50)                # probes (cosines, alignment) per run

BENCH = dict(
    optimizer=OPTIMIZER,
    # one-at-a-time around NET: every value != NET[p] is one config (with the base task)
    # "k/N": read-out ratio, crossed with every N in "N": k = max(1, round(ratio * N)); repeated (N, k) are run once
    net={"N": [3, 5, 10, 15, 20],
         "k/N": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
         "w_scale": [0.5, 1.0, 2.0],
         "w_in": [2.0, 4.0, 8.0],
         "r_in": [0.5, 1.0, 2.0]},
    # every task != BASE_TASK is one config (with NET): the cycle task at n_cycles / k = 0.25 ... 8 (k = NET["k"])
    tasks=[cycles_task(r) for r in (0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0)],
    # teacher ablation, one at a time at the base net and task, only for the rules with a teacher B:
    #   "P/trials": corrections P = max(1, round(ratio * n_trials));  "R/phase": perturbed trials per phase as a
    #   fraction of the trials in the phase;  "M": perturbed copies per perturbed trial
    teacher={"P/trials": [0.0002, 0.002, 0.02, 0.2, 1.0],
             "R/phase": [0.001, 0.01, 0.1],
             "M": [4, 16, 64]},
    # every rule runs at every config
    rules=[EXACT, READOUT_ONLY,
           *[rflo(j, "random", 0) for j in ("murray", "e", "exact_decay")],
           *[rflo(j, "teacher", 5) for j in ("murray", "e", "exact_decay")],
           *[rflo(j, "teacher", 5, teacher_on="student") for j in ("murray", "e", "exact_decay")]],
    seeds=5,
    # which configs the bench runs:
    #   "minmidmax": for each task in mmm_tasks (cycles / k), every ablation axis above (net and teacher) one at a time
    #                at its smallest, middle and largest value, everything else at base
    #   "full":      every value of every axis, one at a time (around the base task), plus every task
    grid="minmidmax",
    mmm_tasks=(0.25, 1.0, 8.0),
)

# what Init draws per seed
INIT = dict(w_star_max="2k",          # W_star ~ integers in [1, 2k)
            w_out0=(0.5, 1.5),        # W_out(0) ~ U(lo, hi)
            b_scale=1.0)              # B(0) ~ N(0, b_scale^2), one entry per neuron


# =====================================================================================================================
# 4. System: simulation, readout, loss, exact gradient
# =====================================================================================================================

def neuron_names(n):
    return [chr(65 + i) if i < 26 else f"n{i}" for i in range(n)]


def init_weights(p):
    n = p["N"]
    if p["init"] == "inhibitory":
        return -p["w_scale"] * jnp.ones((n, n - 1))
    if p["init"] == "gaussian":
        return p["w_scale"] / np.sqrt(n - 1) * random.normal(random.PRNGKey(p["seed_w"]), (n, n - 1))
    if p["init"] == "zero":
        return jnp.zeros((n, n - 1))
    raise ValueError(p["init"])


def poisson_inputs(n_in, rate, T, seed):
    rng = np.random.default_rng(seed)
    t = [rng.uniform(0, T, rng.poisson(rate * T)) for _ in range(n_in)]
    ch = np.concatenate([np.full(len(x), c) for c, x in enumerate(t)]).astype(int)
    t = np.concatenate(t)
    order = np.argsort(t)
    return jnp.asarray(t[order]), jnp.asarray(ch[order])


def build(net=None, task=None):
    """NET + task -> System S. Everything here is fixed for the config; nothing is drawn per seed."""
    if not jax.config.x64_enabled:
        raise RuntimeError("Start with a fresh kernel and set JAX_ENABLE_X64=True before importing JAX.")
    net = dict(NET if net is None else net)
    task = dict(TASKS[BASE_TASK] if task is None else task)
    S = SimpleNamespace(net=net, task=task, _rules={}, _steps={}, _misc={})
    N, k, T = net["N"], net["k"], net["T"]
    assert 1 <= k <= N, "need 1 <= k <= N"
    neuron = make_neuron()               # QIF, tau = 1, reset to -1/eps (spikegd's -inf)
    n_in = N if net["n_in"] is None else net["n_in"]
    WEIGHTS_IN = net["w_in"] * jnp.eye(N, n_in)
    w0, x0 = init_weights(net), jnp.zeros((2, N))
    spikes = poisson_inputs(n_in, net["r_in"], T, net["seed_in"])     # one input sample, used every trial

    if net["K"] == "auto":
        _, _, who, _ = neuron.event(x0, w_to_weights_net(w0), WEIGHTS_IN, spikes, dict(N=N, T=T, K=5000))
        K = 3 * int(jnp.sum(who >= 0)) + 100
    else:
        K = int(net["K"])
    cfg = dict(N=N, T=T, K=K)
    TARGETS, HIDDEN = np.arange(N - k, N), np.arange(N - k)
    S.__dict__.update(N=N, k=k, T=T, K=K, cfg=cfg, neuron=neuron, n_in=n_in, WEIGHTS_IN=WEIGHTS_IN, w0=w0, x0=x0,
                      spikes=spikes, TARGETS=TARGETS, HIDDEN=HIDDEN, NTAR=k, NHID=N - k, NAMES=neuron_names(N),
                      COL=jnp.array([[j if j < i else j - 1 for j in range(N)] for i in range(N)]),
                      J_OF=jnp.array([[j for j in range(N) if j != i] for i in range(N)]),
                      TOUCHED_IN=WEIGHTS_IN != 0)

    def simulate(w):
        return neuron.event(x0, w_to_weights_net(w), WEIGHTS_IN, spikes, cfg)

    def event_list(out):
        ts, is_in, who, xs = out
        return dict(t=ts, t_prev=jnp.concatenate([jnp.zeros(1), ts[:-1]]), is_input=is_in, node=who,
                    valid=who >= 0, x=xs, x_prev=jnp.concatenate([x0[None], xs[:-1]]))

    # ---- readout and loss: y(t) = sum_o W_out,o sum_q kappa(t - t_oq),  L = 1/(2T) int (y - y*)^2 dt ----
    DT, PAD, tau_r = task["dt_grid"], 2 * T, task["tau_r"]
    NMAX = K if task.get("n_max") is None else int(task["n_max"])     # None: no cap below K
    GRID = jnp.arange(0.0, T, DT) + DT / 2

    def kernel(s):
        sc = jnp.where(s > 0, s, 0.0)
        return sc / tau_r * jnp.exp(1 - sc / tau_r)

    def target_times(out):
        ts, is_in, who, _ = out
        rows, counts = [], []
        for o in TARGETS:
            mask = (who == int(o)) & ~is_in
            t = ts[jnp.argsort(~mask)[:NMAX]]
            n = jnp.sum(mask)
            rows.append(jnp.where(jnp.arange(NMAX) < n, t, PAD))
            counts.append(n)
        return jnp.stack(rows), jnp.stack(counts)

    def readout(W_out, t_spk):
        return W_out @ kernel(GRID[None, None, :] - t_spk[:, :, None]).sum(1)

    def loss_from_spikes(W_out, t_spk, Y_star):
        return 0.5 * jnp.sum((readout(W_out, t_spk) - Y_star) ** 2) * DT / T

    def dkernel(s):
        # d kappa / ds
        sc = jnp.where(s > 0, s, 0.0)
        return jnp.where(s > 0, (1 - sc / tau_r) / tau_r * jnp.exp(1 - sc / tau_r), 0.0)

    def ell_at(tk, W_out, t_spk, Y_star):
        # l(t_k) = dL/dt_k for a spike at tk read out with weight 1, so dL/dt_ok = W_out,o l(t_ok).
        # tk can have any shape; eps is computed once:  l(t_k) = -(1/T) int eps(t) kappa'(t - t_k) dt
        eps = readout(W_out, t_spk) - Y_star
        return -jnp.tensordot(dkernel(GRID - jnp.asarray(tk)[..., None]), eps, axes=([-1], [0])) * DT / T

    def loss_fn(params, Y_star):
        w, W_out = params
        t_spk, _ = target_times(simulate(w))
        return loss_from_spikes(W_out, t_spk, Y_star)

    t_targets = TASK_FNS[task["name"]](task, T, k)
    assert max(len(t) for t in t_targets) <= NMAX, "raise n_max"
    t_star = jnp.stack([jnp.pad(jnp.asarray(t), (0, NMAX - len(t)), constant_values=PAD) for t in t_targets])

    _grad_dt = jax.grad(lambda V, b: neuron.dt_spike(jnp.stack([V, b])), argnums=(0, 1))

    def grad_tsp(x_j):
        gV, gb = _grad_dt(x_j[0], x_j[1])
        return jnp.stack([gV, gb])

    _jac_flow = jax.vmap(jax.jacfwd(lambda V, b, D: neuron.flow_V(V, b, D), argnums=(0, 1)), in_axes=(0, 0, None))

    def flow_jac(V, b, D):
        # exact V row of each neuron's own subthreshold map over a gap D: (dV(D)/dV0, dV(D)/db0); b row = (0, e^{-2D})
        JVV, JVb = _jac_flow(V, b, D)
        ok = jnp.isfinite(JVV) & jnp.isfinite(JVb)
        return jnp.where(ok, JVV, 0.0), jnp.where(ok, JVb, 0.0)

    _jac_pw = jax.vmap(jax.jacfwd(lambda V, b, D: neuron.flow_V(V, b, D), argnums=(0, 1)))

    def flow_jac_pw(V, b, D):
        # same, with one gap per neuron
        JVV, JVb = _jac_pw(V, b, D)
        ok = jnp.isfinite(JVV) & jnp.isfinite(JVb)
        return jnp.where(ok, JVV, 0.0), jnp.where(ok, JVb, 0.0)

    @jax.jit
    def activity(w):
        ts, is_in, who, _ = simulate(w)
        net_ = (who >= 0) & ~is_in
        return jnp.sum((who[:, None] == jnp.arange(N)[None]) & net_[:, None], 0), jnp.sum(who >= 0)

    S.__dict__.update(simulate=simulate, event_list=event_list, kernel=kernel, target_times=target_times,
                      readout=readout, loss_from_spikes=loss_from_spikes, ell_at=ell_at, loss_fn=loss_fn,
                      exact=jax.jit(jax.value_and_grad(loss_fn)), t_targets=t_targets, t_star=t_star, GRID=GRID,
                      DT=DT, NMAX=NMAX, PAD=PAD, grad_tsp=grad_tsp, flow_jac=flow_jac, flow_jac_pw=flow_jac_pw, activity=activity,
                      event_list_jit=jax.jit(lambda w: event_list(simulate(w))))
    _build_noisy(S)
    return S


def describe(S):
    n_spk, n_ev = (np.asarray(x) for x in S.activity(S.w0))
    print(f"network: N = {S.N}, read-out neurons {[S.NAMES[o] for o in S.TARGETS]} (their spikes make y), "
          f"hidden {[S.NAMES[h] for h in S.HIDDEN]}, init '{S.net['init']}' w = {S.net['w_scale']}")
    print(f"inputs: {S.n_in} Poisson channels at rate {S.net['r_in']}, weight {S.net['w_in']}, "
          f"{len(S.spikes[0])} input spikes, the same every trial")
    print(f"events at init: {int(n_ev)} of K = {S.K};  spikes per neuron:",
          dict(zip(S.NAMES, n_spk.tolist())))
    seq = " ".join(S.NAMES[S.TARGETS[i]] for t, i in sorted((t, i) for i, ts in enumerate(S.t_targets) for t in ts))
    print(f"task '{S.task['name']}': target sequence {seq}  at",
          {S.NAMES[o]: np.round(t, 2).tolist() for o, t in zip(S.TARGETS, S.t_targets)},
          f" ({S.task['n_trials']} trials)")


# =====================================================================================================================
# 5. Init: everything drawn per seed
# =====================================================================================================================

@dataclass
class Init:
    seed: int
    W_star: jnp.ndarray          # readout weights that make y*
    Y_star: jnp.ndarray          # y*(t) on the grid
    W_out0: jnp.ndarray          # W_out(0), the same for every rule
    B0: jnp.ndarray              # random feedback B(0), one entry per neuron
    alpha0: dict                 # Jacobian form -> (2, N)

    def describe(self, S):
        print(f"seed {self.seed}:  W* = {np.round(self.W_star, 3)}   W_out(0) = {np.round(self.W_out0, 3)}")
        print(f"  B(0) = {np.round(self.B0, 3)}   (read-out entries {np.round(self.B0[S.TARGETS], 3)})")
        c0 = _cos(self.B0[S.TARGETS], self.W_out0)
        print(f"  cos(B_targets, W_out(0)) = {c0:+.2f}: " + _read_align(c0))
        for j, a in self.alpha0.items():
            print(f"  alpha(0) for '{j}':", JACOBIANS[j]["alpha_doc"], np.round(np.asarray(a), 3).tolist())


def make_init(S, seed):
    rng = np.random.default_rng(seed)
    hi = 2 * S.k if INIT["w_star_max"] == "2k" else int(INIT["w_star_max"])
    W_star = jnp.asarray(rng.integers(1, max(hi, 2), size=S.k).astype(float))
    W_out0 = jnp.asarray(rng.uniform(*INIT["w_out0"], size=S.k))
    B0 = jnp.asarray(INIT["b_scale"] * rng.normal(size=S.N))
    alpha0 = {}
    for name, J in JACOBIANS.items():
        (lv, hv), (lb, hb) = J["a_range"]
        a = rng.uniform(lv, hv, S.N)
        b = a if J.get("tie") else rng.uniform(lb, hb, S.N)
        alpha0[name] = jnp.asarray(np.stack([a, b]))
    return Init(seed, W_star, S.readout(W_star, S.t_star), W_out0, B0, alpha0)


# =====================================================================================================================
# 6. Jacobian approximations (trace evolution). Each: f(S, ev, alpha) -> (s (K, N-1), traces before every event)
#    s[k] = the student's d t_k / d w_{j,.} for an output of neuron j at event k (0 for other events).
#    The trace of neuron i is E_i (N-1, 2): one column per synapse j -> i, rows (V_i, b_i).
#    Add a form: write f, put it in JACOBIANS with its alpha range (and a teacher fit if there is one).
# =====================================================================================================================

def _common(S, ev, k, E):
    valid, j, is_in = ev["valid"][k], ev["node"][k], ev["is_input"][k]
    spike = valid & ~is_in
    jj = jnp.maximum(j, 0)
    touched = valid & jnp.where(is_in, S.TOUCHED_IN[:, jj], True)
    # output of j: the time row read off the trace before this event (the state the spike time came from)
    s = jnp.where(spike, E[jj] @ S.grad_tsp(ev["x_prev"][k][:, jj]), 0.0)
    recv = (spike & (jnp.arange(S.N) != j)).astype(float)
    fired = spike & (jnp.arange(S.N) == j)
    return valid, jj, touched, s, recv, fired


def trace_murray(S, ev, alpha):
    # M_i(k) = diag(1 - a_V, 1 - a_b) M_i(k-1) + diag(a_V, a_b) Mbar_i(k), at every event of i (identity otherwise)
    a_v, a_b = alpha[0], alpha[1]

    def step(E, k):
        valid, jj, touched, s, recv, _ = _common(S, ev, k, E)
        EV = jnp.where(touched, 1.0 - a_v, 1.0)[:, None] * E[..., 0]
        Eb = jnp.where(touched, 1.0 - a_b, 1.0)[:, None] * E[..., 1]
        Eb = Eb.at[jnp.arange(S.N), S.COL[:, jj]].add(recv * a_b)      # Mbar = (0, 1): the V entry is 0
        return jnp.stack([EV, Eb], -1), (s, E)

    _, (s, E) = lax.scan(step, jnp.zeros((S.N, S.N - 1, 2)), jnp.arange(S.K))
    return s, E


def trace_e(S, ev, alpha):
    # between events: e_V' = -alpha e_V + e_b, e_b' = -2 e_b (closed form over the gap D, for every neuron);
    # at an arrival j -> i: e_b += 1 in column j; at i's own spike: e_V = 0
    al = alpha[0]

    def step(E, k):
        valid, jj, _, s, recv, fired = _common(S, ev, k, E)
        D = jnp.where(valid, ev["t"][k] - ev["t_prev"][k], 0.0)
        d2, da = jnp.exp(-2.0 * D), jnp.exp(-al * D)
        far = jnp.abs(al - 2.0) > 1e-6
        mix = jnp.where(far, (d2 - da) / jnp.where(far, al - 2.0, 1.0), D * d2)
        EV = da[:, None] * E[..., 0] + mix[:, None] * E[..., 1]
        Eb = d2 * E[..., 1]
        EV = jnp.where(fired[:, None], 0.0, EV)
        Eb = Eb.at[jnp.arange(S.N), S.COL[:, jj]].add(recv)
        return jnp.stack([EV, Eb], -1), (s, E)

    _, (s, E) = lax.scan(step, jnp.zeros((S.N, S.N - 1, 2)), jnp.arange(S.K))
    return s, E


JCLIP = 1e3          # |J_VV|, |J_Vb| clip for exact_decay (near threshold the exact map blows up)


def trace_exact_decay(S, ev, alpha):
    # every gap: E_i <- J_i(D) E_i with the exact own map J_i = [[dV/dV0, dV/db0], [0, e^{-2D}]] of the QIF flow;
    # then (1 - a_V, 1 - a_b) at every event of i (as in Murray); arrival j -> i: e_b += 1; own spike: e_V = 0
    a_v, a_b = alpha[0], alpha[1]

    def step(E, k):
        valid, jj, touched, s, recv, fired = _common(S, ev, k, E)
        D = jnp.where(valid, ev["t"][k] - ev["t_prev"][k], 0.0)
        xp = ev["x_prev"][k]
        JVV, JVb = S.flow_jac(xp[0], xp[1], D)
        JVV, JVb = jnp.clip(JVV, -JCLIP, JCLIP), jnp.clip(JVb, -JCLIP, JCLIP)
        EV = JVV[:, None] * E[..., 0] + JVb[:, None] * E[..., 1]
        Eb = jnp.exp(-2.0 * D) * E[..., 1]
        EV = jnp.where(valid, EV, E[..., 0])
        EV = jnp.where(touched, 1.0 - a_v, 1.0)[:, None] * EV
        Eb = jnp.where(touched, 1.0 - a_b, 1.0)[:, None] * Eb
        EV = jnp.where(fired[:, None], 0.0, EV)
        Eb = Eb.at[jnp.arange(S.N), S.COL[:, jj]].add(recv)
        return jnp.stack([EV, Eb], -1), (s, E)

    _, (s, E) = lax.scan(step, jnp.zeros((S.N, S.N - 1, 2)), jnp.arange(S.K))
    return s, E


# ---- teacher fits of alpha (only where the form has one) ----

def fit_alpha_murray(S, evs, alpha_old, a_range=(0.0, 1.0)):
    # per neuron: argmin_a sum over its gaps (J - (1 - a))^2 = 1 - mean J, for J_VV and J_bb separately
    if "local_diag" not in S._misc:
        @jax.jit
        def local_diag(ev):
            def step(carry, k):
                t_last, x_last = carry
                valid, j, isin = ev["valid"][k], ev["node"][k], ev["is_input"][k]
                jj = jnp.maximum(j, 0)
                touched = valid & jnp.where(isin, S.TOUCHED_IN[:, jj], True)
                D = ev["t"][k] - t_last
                JVV, _ = S.flow_jac_pw(x_last[0], x_last[1], D)
                base = jax.vmap(lambda V, b, d: S.neuron.flow_V(V, b, d))(x_last[0], x_last[1], D)
                fires = valid & ~isin & (jnp.arange(S.N) == j)
                ok = touched & (D > 1e-6) & (jnp.abs(x_last[0]) < 50) & (jnp.abs(base) < 50) & ~fires
                t_last = jnp.where(touched, ev["t"][k], t_last)
                x_last = jnp.where(touched[None], ev["x"][k], x_last)
                return (t_last, x_last), (JVV, jnp.exp(-2.0 * D), ok)
            _, out = lax.scan(step, (jnp.zeros(S.N), S.x0), jnp.arange(S.K))
            return out
        S._misc["local_diag"] = local_diag
    sV, sb, cnt = np.zeros(S.N), np.zeros(S.N), np.zeros(S.N)
    for ev_ in evs:
        JVV, Jbb, ok = (np.asarray(x) for x in S._misc["local_diag"](ev_))
        sV += np.where(ok, JVV, 0).sum(0); sb += np.where(ok, Jbb, 0).sum(0); cnt += ok.sum(0)
    lo, hi = a_range
    new = np.stack([np.clip(1 - sV / np.maximum(cnt, 1), lo, hi), np.clip(1 - sb / np.maximum(cnt, 1), lo, hi)])
    return jnp.asarray(np.where(cnt > 0, new, np.asarray(alpha_old))), cnt


def fit_alpha_e(S, evs, alpha_old, grid=(0.0, 3.0, 301)):
    # per neuron: the direction of the exact V row over each gap, r = (dV/dV0, dV/db0), against the student's
    # rho_alpha(D) = (e^{-alpha D}, (e^{-2D} - e^{-alpha D}) / (alpha - 2)); alpha = argmin sum |r/|r| - rho/|rho||^2
    A = jnp.linspace(*grid[:2], int(grid[2]))
    if "vrows" not in S._misc:
        @jax.jit
        def vrows(ev):
            D = jnp.where(ev["valid"], ev["t"] - ev["t_prev"], 0.0)                        # (K,)
            JVV, JVb = jax.vmap(lambda x, d: S.flow_jac(x[0], x[1], d))(ev["x_prev"], D)    # (K, N)
            base = jax.vmap(lambda x, d: S.neuron.flow_V(x[0], x[1], d))(ev["x_prev"], D)
            fires = ev["valid"][:, None] & ~ev["is_input"][:, None] & (ev["node"][:, None] == jnp.arange(S.N)[None])
            ok = (ev["valid"][:, None] & (D[:, None] > 1e-6) & (jnp.abs(ev["x_prev"][:, 0]) < 50)
                  & (jnp.abs(base) < 50) & ~fires)
            return JVV, JVb, D, ok
        S._misc["vrows"] = vrows
    cost, cnt = np.zeros((len(A), S.N)), np.zeros(S.N)
    for ev_ in evs:
        JVV, JVb, D, ok = S._misc["vrows"](ev_)
        r = jnp.stack([JVV, JVb], -1)
        r = r / jnp.maximum(jnp.linalg.norm(r, axis=-1, keepdims=True), 1e-30)
        a = A[:, None, None]
        da, d2 = jnp.exp(-a * D[None, :, None]), jnp.exp(-2.0 * D)[None, :, None]
        far = jnp.abs(a - 2.0) > 1e-6
        mix = jnp.where(far, (d2 - da) / jnp.where(far, a - 2.0, 1.0), D[None, :, None] * d2)
        rho = jnp.stack([da * jnp.ones_like(mix), mix], -1)                                # (G, K, N, 2)
        rho = rho / jnp.maximum(jnp.linalg.norm(rho, axis=-1, keepdims=True), 1e-30)
        c = jnp.sum((r[None] - rho) ** 2, -1)
        cost += np.asarray(jnp.sum(jnp.where(ok[None], c, 0.0), 1)); cnt += np.asarray(ok.sum(0))
    best = np.asarray(A)[np.argmin(cost, 0)]
    new = np.asarray(alpha_old).copy()
    new[0] = np.where(cnt > 0, best, new[0])
    return jnp.asarray(new), cnt


JACOBIANS = {
    "murray": dict(fn=trace_murray, a_range=((0.1, 0.9), (0.1, 0.9)), fit=fit_alpha_murray,
                   alpha_doc="rows (a_V, a_b); a_V does nothing here (the V row of the trace stays 0)"),
    "e": dict(fn=trace_e, a_range=((0.5, 3.0), (0.0, 0.0)), fit=fit_alpha_e,
              alpha_doc="row 0 = alpha (decay rate of e_V), row 1 unused"),
    "exact_decay": dict(fn=trace_exact_decay, a_range=((0.0, 0.3), (0.0, 0.3)), fit=None, tie=True,
                        alpha_doc="rows (a_V, a_b) = the same a per neuron: factor (1 - a) at every event"),
}


# =====================================================================================================================
# 7. Rule assembly: dL/dw ~ sum over outputs k of neuron j:  c_j(k) s_k
# =====================================================================================================================

def _student_terms(S, params, Y_star, alpha, jacobian):
    w, W_out = params
    out = S.simulate(w)
    ev = S.event_list(out)
    t_spk, _ = S.target_times(out)
    loss, g_Wout = jax.value_and_grad(S.loss_from_spikes)(W_out, t_spk, Y_star)
    ell = S.ell_at(ev["t"], W_out, t_spk, Y_star)
    s, _ = JACOBIANS[jacobian]["fn"](S, ev, alpha)
    is_out = ev["valid"] & ~ev["is_input"]
    j = jnp.where(is_out, ev["node"], 0)
    return dict(loss=loss, g_Wout=g_Wout, ev=ev, t_spk=t_spk, ell=ell, s=s, is_out=is_out, j=j, w=w, W_out=W_out,
                Y_star=Y_star)


def _signal(S, rule, B, T_):
    j, ell = T_["j"], T_["ell"]
    cred = B.at[S.TARGETS].set(T_["W_out"]) if rule.signal == "W_out_targets" else B
    own = cred[j] * ell
    if rule.cross != "none":
        raise ValueError(f"cross = {rule.cross!r}: only 'none' is implemented")
    hid = jnp.zeros_like(own) if rule.signal == "targets_only" else own
    is_ro = jnp.zeros(S.N, bool).at[S.TARGETS].set(True)
    return jnp.where(T_["is_out"], jnp.where(is_ro[j], own, hid), 0.0)


def rule_fn(S, rule):
    """jitted (params, Y_star, alpha, B) -> (loss, (g_w, g_W_out)). The W_out gradient is exact for every rule."""
    if rule in S._rules:
        return S._rules[rule]
    if rule.kind == "exact":
        def f(p, Y, a, B):
            return S.exact(p, Y)
    elif rule.kind == "readout_only":
        def f(p, Y, a, B):
            L, gW = jax.value_and_grad(lambda Wo: S.loss_fn((p[0], Wo), Y))(p[1])
            return L, (jnp.zeros_like(p[0]), gW)
    else:
        def f(p, Y, a, B):
            T_ = _student_terms(S, p, Y, a, rule.jacobian)
            Lk = _signal(S, rule, B, T_)
            g_w = jnp.zeros((S.N, S.N - 1)).at[T_["j"]].add(Lk[:, None] * T_["s"])
            return T_["loss"], (g_w if rule.train_w else jnp.zeros_like(g_w), T_["g_Wout"])
    S._rules[rule] = jax.jit(f)
    return S._rules[rule]


def U_fn(S, jacobian):
    # U_i = sum over outputs k of i of l(t_k) s_k  (the student's row before the factor c_i), (N, N-1)
    key = ("U", jacobian)
    if key not in S._misc:
        def f(p, Y, a):
            T_ = _student_terms(S, p, Y, a, jacobian)
            return jnp.zeros((S.N, S.N - 1)).at[T_["j"]].add(jnp.where(T_["is_out"], T_["ell"], 0.0)[:, None] * T_["s"])
        S._misc[key] = jax.jit(f)
    return S._misc[key]


# =====================================================================================================================
# 8. Teacher: B from node perturbation on the teacher's own trials
# =====================================================================================================================

def _build_noisy(S):
    N, K = S.N, S.K

    def simulate_noisy(w, xi_b):
        # neuron.event with xi_b[k, i] added to b_i when the network spike of event k reaches i
        wn = w_to_weights_net(w)
        order = jnp.argsort(S.spikes[0])
        queue = (jnp.append(S.spikes[0][order], jnp.inf), jnp.append(S.spikes[1][order], 0))

        def f(state, k):
            state, out = S.neuron.spike(state, wn, S.WEIGHTS_IN, queue, S.cfg)
            t, x, dt_net, c = state
            _, spike_in, j, _ = out
            recv = (~spike_in) & (j >= 0) & (jnp.arange(N) != j)
            db = jnp.where(recv, xi_b[k], 0.0)
            x = x.at[1].add(db)
            dt_net = jnp.where(db != 0, S.neuron.dt_spike(x), dt_net)
            return (t, x, dt_net, c), (t, spike_in, j, x)

        state0 = (jnp.array(0.0), S.x0, S.neuron.dt_spike(S.x0), jnp.array(0))
        _, out = lax.scan(f, state0, jnp.arange(K))
        return out

    def simulate_kick(w, xi):
        # neuron.event with xi[k] (2, N) added to the state (V, b) of every neuron after event k (not to the neuron
        # that just spiked); next spike times are recomputed, so autodiff at xi = 0 gives dL/dx_i after event k
        wn = w_to_weights_net(w)
        order = jnp.argsort(S.spikes[0])
        queue = (jnp.append(S.spikes[0][order], jnp.inf), jnp.append(S.spikes[1][order], 0))

        def f(state, k):
            state, out = S.neuron.spike(state, wn, S.WEIGHTS_IN, queue, S.cfg)
            t, x, dt_net, c = state
            _, spike_in, j, _ = out
            keep = (j >= 0) & ~((~spike_in) & (jnp.arange(N) == j))
            x = x + jnp.where(keep[None], xi[k], 0.0)
            dt_net = jnp.where(keep, S.neuron.dt_spike(x), dt_net)
            return (t, x, dt_net, c), (t, spike_in, j, x)

        state0 = (jnp.array(0.0), S.x0, S.neuron.dt_spike(S.x0), jnp.array(0))
        _, out = lax.scan(f, state0, jnp.arange(K))
        return out

    @jax.jit
    def spike_credit(params, Y):
        # total credit of every network spike, C_k = dL/dt_k (all paths), and l(t_k), from the exact network.
        # The state of the spiking neuron j after the event before k affects the loss only through t_k, so
        # dL/dV_j(after k-1) = C_k * d t_sp / dV (x_j after k-1)  ->  C_k = dL/dV_j(after k-1) / (d t_sp / dV).
        # Not defined when event k-1 is j's own spike (j is at its reset); those spikes get ok = False.
        def Lxi(xi):
            t_spk, _ = S.target_times(simulate_kick(params[0], xi))
            return S.loss_from_spikes(params[1], t_spk, Y)
        G = jax.grad(Lxi)(jnp.zeros((K, 2, N)))                             # dL/dx after every event
        out = S.simulate(params[0])
        ev = S.event_list(out)
        t_spk, _ = S.target_times(out)
        j = jnp.maximum(ev["node"], 0)
        xj = ev["x_prev"][jnp.arange(K), :, j]                              # (K, 2): state of j after event k-1
        gV = jax.vmap(lambda x: S.grad_tsp(x)[0])(xj)
        G_prev = jnp.concatenate([jnp.zeros((1, 2, N)), G[:-1]])
        dLdV = G_prev[jnp.arange(K), 0, j]
        own_prev = jnp.concatenate([jnp.zeros(1, bool), (~ev["is_input"][:-1]) & (ev["node"][:-1] == ev["node"][1:])])
        is_spk = ev["valid"] & ~ev["is_input"]
        ok = is_spk & (jnp.arange(K) > 0) & ~own_prev & jnp.isfinite(gV) & (jnp.abs(gV) > 1e-12)
        C = jnp.where(ok, dLdV / jnp.where(ok, gV, 1.0), 0.0)
        ell = S.ell_at(ev["t"], params[1], t_spk, Y)
        return C, ell, ev["node"], ok, G

    def loss_noisy(params, Y, xi_b):
        out = simulate_noisy(params[0], xi_b)
        t_spk, _ = S.target_times(out)
        return S.loss_from_spikes(params[1], t_spk, Y), out

    @partial(jax.jit, static_argnums=3)
    def np_gradients(params, Y, key, M, sig):
        # M single-perturbation estimates of dL/dw, (M, N, N-1)
        L0, _ = loss_noisy(params, Y, jnp.zeros((K, N)))
        xi = sig * random.normal(key, (M, K, N))

        def one(z):
            L, (_, s_in, who, _) = loss_noisy(params, Y, z)
            net_ = (~s_in) & (who >= 0)
            onehot = (who[:, None] == jnp.arange(N)[None]) & net_[:, None]
            E = jnp.einsum("ki,kj->ij", z, onehot.astype(float))            # (receiver, sender)
            return (L - L0) / sig ** 2 * E[jnp.arange(N)[:, None], S.J_OF]
        return jax.vmap(one)(xi)

    def noisy_value_grad(params, Y, xi):
        # loss and exact gradient of one trial with noise xi (K, N) on b at every arrival (xi = 0: the clean trial)
        return jax.value_and_grad(lambda p_: loss_noisy(p_, Y, xi)[0])(params)

    S.simulate_noisy, S.np_gradients, S.noisy_value_grad = simulate_noisy, np_gradients, noisy_value_grad
    S.simulate_kick, S.spike_credit = simulate_kick, spike_credit


def hidden_samples(S, rule, pT, Y):
    # hidden entries: for every spike k of a hidden neuron X in the teacher's trial, (dL/dt_Xk) / l(t_Xk), with
    # dL/dt_Xk the exact total credit (all paths). Spikes with |l| < ell_min * max|l| are left out.
    C, ell, node, ok, _ = (np.asarray(x) for x in S.spike_credit(pT, Y))
    out = {}
    lmax = np.max(np.abs(ell[ok])) if ok.any() else 0.0
    for X in S.HIDDEN:
        m = ok & (node == X) & (np.abs(ell) > rule.ell_min * lmax)
        out[int(X)] = C[m] / ell[m]
    return out


def _location_spread(x, rule):
    # mean / median / trimmed mean of 1-d samples and its standard error
    x = np.asarray(x, float)
    n = len(x)
    if rule.average == "mean":
        return x.mean(), x.std() / np.sqrt(n)
    if rule.average == "median":
        q1, mu, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        return mu, 1.2533 * (q3 - q1) / 1.349 / np.sqrt(n)
    lo, hi = np.quantile(x, [rule.trim, 1 - rule.trim])
    k = x[(x >= lo) & (x <= hi)]
    if len(k) == 0:
        k = x
    return k.mean(), k.std() / np.sqrt(len(k))


def pick_B(rule, S, w_samples, h_samples, B_old, b_norm, rng):
    # per neuron: (mu_i, se_i) from its samples, then B_i ~ N(mu_i, se_i^2) (pick="draw") or mu_i (pick="mean").
    # read-out o: samples = teacher's candidate W_out,o.  hidden X: samples = (dL/dt_Xk) / l(t_Xk) ("credit"),
    # or none ("random": X keeps its B). A neuron with fewer than 2 samples keeps its B.
    B_new = np.asarray(B_old, float).copy()
    mu, se = np.full(S.N, np.nan), np.full(S.N, np.nan)
    per = {int(o): np.concatenate([w[:, oi] for w in w_samples]) for oi, o in enumerate(S.TARGETS)}
    if rule.hidden_B == "credit":
        for X in S.HIDDEN:
            per[int(X)] = np.concatenate([h[int(X)] for h in h_samples]) if h_samples else np.zeros(0)
    for i, x in per.items():
        if len(x) >= 2:
            mu[i], se[i] = _location_spread(x, rule)
            B_new[i] = rng.normal(mu[i], se[i]) if rule.pick == "draw" else mu[i]
    if rule.rescale:                                        # after the draw: same norm as B(0)
        B_new = B_new / max(np.linalg.norm(B_new), 1e-30) * b_norm
    return jnp.asarray(B_new), (mu, se)


# =====================================================================================================================
# 9. Training
# =====================================================================================================================

def _cos(a, b):
    a, b = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else np.nan


def _row_cos(g, G):
    g, G = np.asarray(g), np.asarray(G)
    den = np.linalg.norm(g, axis=1) * np.linalg.norm(G, axis=1)
    return np.where(den > 0, np.sum(g * G, 1) / np.where(den > 0, den, 1.0), np.nan)


def make_optimizer(opt):
    sched = optax.exponential_decay(opt["lr"], opt["decay_steps"], 1 / np.e)
    return {"adabelief": optax.adabelief, "adam": optax.adam, "sgd": optax.sgd}[opt["name"]](sched)


def _step_fn(S, rule, opt):
    key = (rule, json.dumps(opt, sort_keys=True))
    if key not in S._steps:
        g_fn, optim = rule_fn(S, rule), make_optimizer(opt)

        @jax.jit
        def step(p, st, Y, a, B):
            loss, g = g_fn(p, Y, a, B)
            upd, st = optim.update(g, st, p)
            p = optax.apply_updates(p, upd)
            if opt["clip_inhibitory"]:
                p = (jnp.minimum(p[0], 0.0), p[1])
            return p, st, loss
        S._steps[key] = (step, optim)
    return S._steps[key]


def _R_of(rule, phase_len):
    # perturbed trials in a phase of phase_len trials
    return max(1, int(round(rule.R_frac * phase_len))) if rule.R_frac > 0 else rule.R


def train(S, ini, rule, n_trials=None, opt=None, verbose=False):
    """One run. Without a teacher: n_trials student trials with fixed (alpha, B).
    With a teacher (rule.P > 0): P phases. In each phase the teacher (exact gradient, same start) runs its trials
    first, on its own weights. At each of R chosen trials the clean run branches into M copies; the branches take a
    perturbed trial (noise on b at every arrival) instead of the clean one, keep training with the exact gradient on
    the clean trials after it, and are perturbed again (each with its own noise) at every later chosen trial; the
    clean run itself is never perturbed. At the end of the phase B is drawn from the branches (pick_B) and alpha is
    fitted from the clean run's event lists at the chosen trials; then the student runs the same phase with them.
    So the teacher corrects B and alpha P times.
    teacher_on = "student": the same, but the teacher starts every phase from a copy of the student's current
    weights and optimizer state."""
    opt = OPTIMIZER if opt is None else opt
    n_trials = S.task["n_trials"] if n_trials is None else n_trials
    g_fn = rule_fn(S, rule)
    s_step, optim = _step_fn(S, rule, opt)
    Y = ini.Y_star
    alpha = ini.alpha0[rule.jacobian]
    B = ini.B0
    p = (S.w0, ini.W_out0)
    st = optim.init(p)
    teach = rule.teacher
    pT = None
    if teach:
        t_step, t_optim = _step_fn(S, EXACT, opt)
        pT, stT = p, t_optim.init(p)
        clip = opt["clip_inhibitory"]

        @jax.jit
        def branch_step(pB, stB, Y_, xiB):
            # one trial for every branch: exact gradient on its own (perturbed or clean) trial, teacher's optimizer
            def one(p_, st_, xi):
                L, g = S.noisy_value_grad(p_, Y_, xi)
                upd, st_ = t_optim.update(g, st_, p_)
                p_ = optax.apply_updates(p_, upd)
                if clip:
                    p_ = (jnp.minimum(p_[0], 0.0), p_[1])
                return p_, st_, L
            return jax.vmap(one)(pB, stB, xiB)

        def spawn(pB, stB, p_, st_, m):
            # m new branches, copies of the clean run (weights and optimizer state)
            rep_ = lambda a: jnp.broadcast_to(a, (m,) + jnp.shape(a))
            new_p, new_s = jax.tree_util.tree_map(rep_, p_), jax.tree_util.tree_map(rep_, st_)
            if pB is None:
                return new_p, new_s
            cat = lambda a, b: jnp.concatenate([a, b], 0)
            return jax.tree_util.tree_map(cat, pB, new_p), jax.tree_util.tree_map(cat, stB, new_s)
        if rule.alpha_from == "teacher" and JACOBIANS[rule.jacobian]["fit"] is None:
            raise ValueError(f"no teacher fit of alpha for '{rule.jacobian}'")
    rng, key = np.random.default_rng(10_000 + ini.seed), random.PRNGKey(10_000 + ini.seed)
    b_norm = float(jnp.linalg.norm(ini.B0))
    probe_every = max(1, n_trials // opt["n_probes"])
    copy = teach and rule.teacher_on == "student"
    rec = dict(loss=[], probe_trial=[], cos_row=[], ceil_row=[], cos_all=[], cos_tar=[], cos_hid=[], cos_Wout=[], align_B=[], align_dWout=[],
               align_teacher=[], gnorm_row=[], n_spk=[], n_events=[], teacher_loss=[], B_hist=[np.asarray(B)],
               B_star=[], B_sd=[], n_branches=[], alpha_hist=[np.asarray(alpha)], n_perturbed=0)

    def probe(n):
        _, (gw, gWo) = g_fn(p, Y, alpha, B)
        _, (Gw, GWo) = S.exact(p, Y)
        cr = _row_cos(gw, Gw)
        rec["probe_trial"].append(n)
        rec["cos_row"].append(cr)
        # best cosine any B could give a row: |cos(U_i, G_i)|, U_i = sum_k l(t_ik) s_k (B_i only sets the sign)
        if rule.kind == "local":
            rec["ceil_row"].append(np.abs(_row_cos(U_fn(S, rule.jacobian)(p, Y, alpha), Gw)))
        else:
            rec["ceil_row"].append(np.full(S.N, np.nan))
        rec["cos_all"].append(_cos(gw, Gw))
        rec["cos_tar"].append(_cos(np.asarray(gw)[S.TARGETS], np.asarray(Gw)[S.TARGETS]))
        rec["cos_hid"].append(_cos(np.asarray(gw)[S.HIDDEN], np.asarray(Gw)[S.HIDDEN]) if S.NHID else np.nan)
        rec["cos_Wout"].append(_cos(gWo, GWo))
        # feedback alignment of the read-out: W_out against B's read-out entries, and the W_out update (-dL/dW_out)
        # against them. For exact / readout_only B = B(0) is never used: their curves are the control.
        rec["align_B"].append(_cos(np.asarray(B)[S.TARGETS], p[1]))
        rec["align_dWout"].append(_cos(np.asarray(B)[S.TARGETS], -np.asarray(gWo)))
        rec["align_teacher"].append(_cos(np.asarray(B)[S.TARGETS], pT[1]) if pT is not None else np.nan)
        rec["gnorm_row"].append(np.linalg.norm(np.asarray(Gw), axis=1))
        ns, ne = S.activity(p[0])
        rec["n_spk"].append(np.asarray(ns)); rec["n_events"].append(int(ne))

    P = rule.P if teach else 1
    base_len = n_trials // P
    n, t0 = 0, time.time()
    for ph in range(P):
        L_ph = base_len + (n_trials - P * base_len if ph == P - 1 else 0)
        if teach:
            if copy:
                pT, stT = p, st                       # the teacher starts the phase from a copy of the student
            R_ph = min(_R_of(rule, L_ph), L_ph)
            chosen = set(rng.choice(L_ph, size=R_ph, replace=False).tolist())
            pB = stB = None
            nB, evs = 0, []
            for i in range(L_ph):
                if i in chosen:
                    if rule.B_from == "teacher":
                        # branch: M new copies of the clean run; this trial is perturbed for every branch (new and old)
                        pB, stB = spawn(pB, stB, pT, stT, rule.M)
                        nB += rule.M
                        key, sub = random.split(key)
                        pB, stB, _ = branch_step(pB, stB, Y, rule.sigma * random.normal(sub, (nB, S.K, S.N)))
                        rec["n_perturbed"] += nB
                    if rule.alpha_from == "teacher":
                        evs.append(S.event_list_jit(pT[0]))
                elif pB is not None:
                    pB, stB, _ = branch_step(pB, stB, Y, jnp.zeros((nB, S.K, S.N)))   # clean trial for every branch
                pT, stT, lT = t_step(pT, stT, Y, alpha, B)
                rec["teacher_loss"].append(float(lT))
            if rule.B_from == "teacher":
                # samples at the end of the phase: each branch's W_out (read-outs) and its exact credit ratios (hidden)
                W_samples = [np.asarray(pB[1])]
                hsamples = ([hidden_samples(S, rule, (pB[0][b_], pB[1][b_]), Y) for b_ in range(nB)]
                            if rule.hidden_B == "credit" and S.NHID else [])
                B, (mu, sd) = pick_B(rule, S, W_samples, hsamples, B, b_norm, rng)
                rec["B_star"].append(mu); rec["B_sd"].append(sd); rec["n_branches"].append(nB)
            if rule.alpha_from == "teacher":
                alpha, _ = JACOBIANS[rule.jacobian]["fit"](S, evs, alpha)
            rec["B_hist"].append(np.asarray(B)); rec["alpha_hist"].append(np.asarray(alpha))
        for i in range(L_ph):
            if n % probe_every == 0:
                probe(n)
            p, st, loss = s_step(p, st, Y, alpha, B)
            rec["loss"].append(float(loss))
            n += 1
        if verbose:
            print(f"  phase {ph + 1}/{P}: loss {np.mean(rec['loss'][-50:]):.3e}  ({time.time() - t0:.0f}s)")
    probe(n)
    out = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in rec.items()}
    out.update(w=np.asarray(p[0]), W_out=np.asarray(p[1]), B=np.asarray(B), alpha=np.asarray(alpha),
               seconds=time.time() - t0, K=S.K, align_B0=_cos(np.asarray(ini.B0)[S.TARGETS], ini.W_out0))
    if np.nanmax(out["n_events"]) >= S.K - 1 or not np.all(np.isfinite(out["loss"])):
        print(f"WARNING {rule.name} seed {ini.seed}: events reached K = {S.K} or the loss is not finite")
    return out


# =====================================================================================================================
# 10. Reading a run (the text the notebook prints)
# =====================================================================================================================

def _read_cos(c):
    if not np.isfinite(c):
        return "no update (undefined)"
    if c < -0.2:
        return "points against the gradient: on average this update raises the loss"
    if c < 0.2:
        return "about orthogonal to the gradient: carries almost no information"
    if c < 0.6:
        return "partly aligned: a descent direction, but a poor one"
    return "aligned with the gradient"


def _read_align(c):
    if not np.isfinite(c):
        return "-"
    if c > 0.6:
        return "B's read-out entries already point along W_out: the read-out neurons get roughly the right credit"
    if c > 0.2:
        return "B and W_out partly agree"
    if c > -0.2:
        return "B and W_out about orthogonal: the read-out neurons' credit has the wrong mix"
    return "B and W_out disagree in sign: read-out neurons are pushed the wrong way"


def explain(S, r, rule, refs=None):
    """Print what a run tells us. refs: {'exact': run, 'readout_only': run} for comparison."""
    tail = max(1, len(r["loss"]) // 50)
    fl = float(np.mean(r["loss"][-tail:]))
    print(f"{rule.name}: final loss {fl:.3e} (mean of the last {tail} trials, start {r['loss'][0]:.3e})")
    if refs:
        ex = float(np.mean(refs["exact"]["loss"][-tail:])) if "exact" in refs else np.nan
        ro = float(np.mean(refs["readout_only"]["loss"][-tail:])) if "readout_only" in refs else np.nan
        if np.isfinite(ro):
            verdict = ("the recurrent update helps over training W_out alone" if fl < 0.95 * ro else
                       "the recurrent update hurts compared to training W_out alone" if fl > 1.05 * ro else
                       "about the same as training W_out alone: the recurrent update does little")
            print(f"  vs readout-only {ro:.3e}: {verdict}")
        if np.isfinite(ex):
            print(f"  vs exact {ex:.3e}: {fl / ex:.1f}x the exact gradient's loss")
    if rule.kind != "local":
        a1 = r["align_B"][-1]
        print(f"  control, B(0) is not used: cos(B(0)_targets, W_out) start {r['align_B0']:+.2f}, end {a1:+.2f}"
              " (any change here is W_out moving on its own gradient, not toward B)")
        return
    half = len(r["cos_tar"]) // 2
    ct, ch = np.nanmedian(r["cos_tar"][half:]), np.nanmedian(r["cos_hid"][half:]) if S.NHID else np.nan
    print(f"  read-out rows: median cos(update, exact) over the 2nd half {ct:+.2f}: {_read_cos(ct)}")
    if S.NHID:
        print(f"  hidden rows:   median cos(update, exact) over the 2nd half {ch:+.2f}: {_read_cos(ch)}")
    a0, a1 = r["align_B0"], r["align_B"][-1]
    print(f"  cos(B_targets, W_out): start {a0:+.2f} (B(0), W_out(0)), end {a1:+.2f}. " +
          ("W_out turned toward B (feedback alignment)" if a1 > a0 + 0.1 else
           "W_out turned away from B" if a1 < a0 - 0.1 else "no alignment during training") +
          f"; at the end: {_read_align(a1)}")
    du = np.nanmedian(r["align_dWout"][half:])
    print(f"  cos(B_targets, W_out update) median over the 2nd half {du:+.2f}: " +
          ("the read-out update keeps moving W_out toward B" if du > 0.2 else
           "the read-out update moves W_out away from B" if du < -0.2 else
           "the read-out update is about orthogonal to B (alignment, if any, has settled)"))
    if rule.teacher and rule.B_from == "teacher":
        bh = r["B_hist"]
        print(f"  teacher corrected B {len(bh) - 1} times; cos(B, B(0)) after each correction:",
              np.round([_cos(b, bh[0]) for b in bh[1:]], 2).tolist(),
              f"; {r['n_perturbed']} perturbed simulations used")
        at = r["align_teacher"][~np.isnan(r["align_teacher"])]
        if len(at):
            print(f"  cos(B_targets, teacher's W_out) at the end {at[-1]:+.2f}: whether B carries the teacher's read-out")
    if rule.alpha_from == "teacher":
        print("  alpha after each correction (row 0):", np.round(np.asarray(r["alpha_hist"])[:, 0], 2).tolist())


# =====================================================================================================================
# 11. Jobs, saving, collecting
# =====================================================================================================================

def _min_mid_max(vals):
    v = sorted(vals)
    return [v[0], v[(len(v) - 1) // 2], v[-1]]


def _teacher_override(axis, v, n_trials):
    if axis == "P/trials":
        return {"P": max(1, int(round(v * n_trials)))}
    if axis == "R/phase":
        return {"R_frac": float(v)}
    return {"M": int(v)}


def configs_minmidmax(bench=BENCH):
    """For each task (cycles / k in mmm_tasks) and each ablation axis: the axis at its min / middle / max value, the
    rest at base. k/N is applied at the base N. Duplicates (an axis value equal to base) are run once, as the task's
    base config. Teacher axes apply to the teacher-B rules only."""
    safe = {"k/N": "k_over_N", "P/trials": "P_per_trial", "R/phase": "R_per_phase"}
    base_rule = next((r for r in bench["rules"] if r.B_from == "teacher"), None)
    out, seen = [], set()

    def add(cid, net, ratio, ov):
        tname = cycles_task(ratio, net["k"])
        key = (json.dumps(net, sort_keys=True), tname, json.dumps(ov, sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.append((cid, net, tname, ov))

    for ratio in bench["mmm_tasks"]:
        tag = f"mmm.cyc_per_k={ratio:g}"
        add(f"{tag}.base", dict(NET), ratio, None)
        for p_, vals in bench["net"].items():
            for v in _min_mid_max(vals):
                if p_ == "k/N":
                    k_ = min(NET["N"], max(1, int(np.floor(v * NET["N"] + 0.5))))
                    net = {**NET, "k": k_}
                else:
                    net = {**NET, p_: v}
                if net == NET:
                    continue
                add(f"{tag}.{safe.get(p_, p_)}={v:g}", net, ratio, None)
        n_tr = TASKS[cycles_task(ratio)]["n_trials"]
        for p_, vals in bench.get("teacher", {}).items():
            for v in _min_mid_max(vals):
                ov = _teacher_override(p_, v, n_tr)
                if base_rule is not None and all(getattr(base_rule, f) == x for f, x in ov.items()):
                    continue
                add(f"{tag}.{safe.get(p_, p_)}={v:g}", dict(NET), ratio, ov)
    return out


def configs(bench=BENCH):
    """[(config id, net, task name, rule overrides or None)]. bench["grid"] = "minmidmax": configs_minmidmax.
    "full": base, one per changed NET value, the k/N ratios crossed with N, one per extra task, and the teacher
    ablation (overrides applied to the teacher-B rules only)."""
    if bench.get("grid", "full") == "minmidmax":
        return configs_minmidmax(bench)
    out = [("base", dict(NET), BASE_TASK, None)]
    seen = {(NET["N"], NET["k"])}
    for p_, vals in bench["net"].items():
        if p_ == "k/N":
            continue
        for v in vals:
            if v != NET[p_]:
                out.append((f"net.{p_}={v}", {**NET, p_: v}, BASE_TASK, None))
                if p_ == "N":
                    seen.add((v, NET["k"]))
    for n_ in bench["net"].get("N", [NET["N"]]) if "k/N" in bench["net"] else []:
        for r_ in bench["net"]["k/N"]:
            k_ = min(n_, max(1, int(np.floor(r_ * n_ + 0.5))))
            if (n_, k_) not in seen:
                seen.add((n_, k_))
                out.append((f"net.N={n_},k={k_}", {**NET, "N": n_, "k": k_}, BASE_TASK, None))
    for t in bench["tasks"]:
        if t != BASE_TASK:
            out.append((f"task.{t}", dict(NET), t, None))
    n_base = TASKS[BASE_TASK]["n_trials"]
    names = {"P/trials": "P_per_trial", "R/phase": "R_per_phase", "M": "M"}
    for p_, vals in bench.get("teacher", {}).items():
        for v in vals:
            if p_ == "P/trials":
                ov = {"P": max(1, int(round(v * n_base)))}
            elif p_ == "R/phase":
                ov = {"R_frac": float(v)}
            else:
                ov = {"M": int(v)}
            base_rule = next((r for r in bench["rules"] if r.B_from == "teacher"), None)
            if base_rule is not None and all(getattr(base_rule, f) == x for f, x in ov.items()):
                continue                                   # same as the base teacher rules: already in "base"
            out.append((f"teach.{names[p_]}={v:g}", dict(NET), BASE_TASK, ov))
    return out


def jobs(bench=BENCH):
    out = []
    for cid, net, t, ov in configs(bench):
        rules = bench["rules"] if ov is None else [replace(r, **ov) for r in bench["rules"] if r.B_from == "teacher"]
        out += [(cid, net, t, rule, s) for rule in rules for s in range(bench["seeds"])]
    return out


def job_path(out_dir, cid, rule, seed):
    return Path(out_dir) / cid / rule.name / f"seed{seed}.npz"


def run_job(i, out_dir=C.HERE / "bench_results", bench=BENCH):
    cid, net, tname, rule, seed = jobs(bench)[i]
    path = job_path(out_dir, cid, rule, seed)
    if path.exists():
        print("exists:", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    S = build(net, TASKS[tname])
    ini = make_init(S, seed)
    t0 = time.time()
    r = train(S, ini, rule, opt=bench["optimizer"])
    meta = dict(config=cid, task=tname, net=net, rule=rule.name, rule_fields=asdict(rule), seed=seed,
                n_trials=TASKS[tname]["n_trials"], optimizer=bench["optimizer"], K=S.K,
                targets=S.TARGETS.tolist(), hidden=S.HIDDEN.tolist(), names=S.NAMES)
    arrays = {key: np.asarray(value) for key, value in r.items() if isinstance(value, (np.ndarray, float, int))}
    with TemporaryDirectory(dir=path.parent, prefix=".run-") as folder:
        staged = Path(folder) / path.name
        np.savez_compressed(staged, **arrays, W_star=np.asarray(ini.W_star),
                            W_out0=np.asarray(ini.W_out0), B0=np.asarray(ini.B0), meta=json.dumps(meta))
        staged.replace(path)
    print(f"[{i}] {cid} {rule.name} seed {seed}: final loss {np.mean(r['loss'][-50:]):.3e} ({time.time() - t0:.0f}s)")


def load_results(out_dir=C.HERE / "bench_results"):
    """{config: {rule: [run per seed]}} with the npz contents as dicts (meta parsed)."""
    res = {}
    for f in sorted(Path(out_dir).glob("*/*/seed*.npz")):
        with np.load(f, allow_pickle=False) as archive:
            d = {key: archive[key] for key in archive.files}
        d["meta"] = json.loads(str(d["meta"]))
        res.setdefault(d["meta"]["config"], {}).setdefault(d["meta"]["rule"], []).append(d)
    return res


def summarize(out_dir=C.HERE / "bench_results"):
    res = load_results(out_dir)
    rows = []
    if not res:
        print(f"No RFLO runs found in {out_dir}.")
        return rows
    stats = ("cos_tar", "cos_hid", "cos_Wout", "align_B", "align_dWout")
    with warnings.catch_warnings():
        # Some rules have no gradient for a group of neurons, so its cosine is undefined.
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        for config, rules in res.items():
            for name, runs in rules.items():
                final = [np.mean(r["loss"][-max(1, len(r["loss"]) // 50):]) for r in runs]
                cosines = {key: np.nanmean([run[key][-1] for run in runs]) for key in stats}
                rows.append(dict(config=config, rule=name, seeds=len(runs),
                                 final_loss=np.mean(final), final_loss_sd=np.std(final),
                                 **cosines, seconds=np.mean([run["seconds"] for run in runs])))
    path = Path(out_dir) / "summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")
    return rows
