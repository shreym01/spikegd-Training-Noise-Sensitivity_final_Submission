"""Run the published input-jitter and initial-weight ablations.

The workflow records an exact-gradient run, fits corrections to local gradients,
trains exact, local, and fitted rules, and computes rate and reproduction statistics.
"""

import json
import time

from . import config as C  # Set JAX defaults before importing it.

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random

from spikegd.utils.rnn import init_input, init_target, init_weights, outfn

from .gradients import make_fast_grad, make_lyapunov, make_neuron, make_spike_counts, w_to_weights_net


# ---- Task and perturbations ----


def make_task(cfg=C.TASK):
    """Input spikes (one Poisson sample), input weights, targets and a zero start."""
    key = random.PRNGKey(cfg["seed"])
    key, spikes_in = init_input(key, cfg)
    key, t_targets = init_target(key, cfg)
    N = cfg["N"]
    return {"cfg": dict(cfg), "spikes_in": spikes_in, "weights_in": init_weights(cfg), "t_targets": t_targets,
            "w0": jnp.zeros((N, N - 1)), "x0": jnp.zeros((2, N)),
            "denom": float(sum(len(t) for t in t_targets))}


def fresh_input(task, k):
    """Input k of the fit: 0 is the task input, k > 0 a new Poisson sample from the same process (same targets)."""
    if k == 0:
        return task["spikes_in"]
    _, spikes = init_input(random.PRNGKey(C.SEED_FIT_INPUTS + k), task["cfg"])
    return spikes


def loss_from_outputs(t_outs, t_targets, denom):
    """Relative squared error; the last entry (1.1 T) counts only while the output comes before it."""
    total = 0.0
    for t_out, t_target in zip(t_outs, t_targets):
        total = total + jnp.sum(((t_out[:-1] - t_target[:-1]) / (t_target[:-1] + 2)) ** 2)
        total = total + ((t_out[-1] - t_target[-1]) / (t_target[-1] + 2)) ** 2 * (t_out[-1] < t_target[-1])
    return total / denom


def make_loss(task, neuron):
    """L(p, spikes) with p = (w, x0)."""
    cfg = task["cfg"]

    def loss(p, spikes):
        out = neuron.event(p[1], w_to_weights_net(p[0]), task["weights_in"], spikes, cfg)
        t_outs = outfn(neuron, out, [p[0], p[1]], task["t_targets"], cfg)
        return loss_from_outputs(t_outs, task["t_targets"], task["denom"])

    return loss


def jitter(spikes, key, sigma):
    """Gaussian jitter on every finite input time, clipped at 0 and re-sorted."""
    times, neurons = spikes
    t = jnp.maximum(times + sigma * random.normal(key, times.shape), 0.0)
    t = jnp.where(jnp.isfinite(times), t, jnp.inf)
    order = jnp.argsort(t)
    return t[order], neurons[order]


def perturbed_input(task, level_index, sigma, seed):
    """One jittered copy of the task input, kept for the whole run."""
    key = random.fold_in(random.PRNGKey(C.SEED_INPUT + seed), level_index)
    return jitter(task["spikes_in"], key, sigma)


def perturbed_weights(task, w_sigma, seed):
    """w0 = (w_sigma / sqrt(N)) xi, xi ~ N(0, 1) fixed per seed."""
    xi = random.normal(random.PRNGKey(C.SEED_WEIGHTS + seed), task["w0"].shape)
    return w_sigma / jnp.sqrt(task["cfg"]["N"]) * xi


def make_runs(task, case, levels=C.LEVELS, n_seeds=C.N_SEEDS):
    """Stacked (spikes, w0, x0) of every run; level 0 (the fit's own input and start) once."""
    spikes, w0s, level, seed = [], [], [], []
    for li, s in enumerate((0.0,) + tuple(levels)):
        for r in range(1 if s == 0 else n_seeds):
            if case == "perturb":
                spikes.append(perturbed_input(task, li, s, r))
                w0s.append(task["w0"])
            else:
                spikes.append(task["spikes_in"])
                w0s.append(perturbed_weights(task, s, r))
            level.append(s)
            seed.append(r)
    stacked = (jnp.stack([a for a, _ in spikes]), jnp.stack([b for _, b in spikes]))
    x0 = jnp.zeros((len(level),) + task["x0"].shape)
    return stacked, jnp.stack(w0s), x0, np.array(level), np.array(seed)


# ---- Training rules ----


def exact(task, neuron):
    """Autodiff through spikegd's event-based simulation."""
    return jax.value_and_grad(make_loss(task, neuron))


def local(task, neuron):
    """Eligibility trace of each target neuron times its own errors; off-target rows get nothing."""
    g = make_fast_grad(neuron, task, task["cfg"]["Ntar"], "trace")
    return lambda p, spikes: g(p[0], p[1], spikes)


def load_fit(path, task):
    """Fit coefficients, checked against this task (the task depends on the JAX version)."""
    with open(path) as stream:
        fit = json.load(stream)
    n_out = [len(t) for t in task["t_targets"]]
    if list(fit["n_outputs"]) != n_out:
        raise SystemExit(f"{path} was fitted for target counts {fit['n_outputs']}, this task has {n_out}: refit first.")
    return fit


def fitted(task, neuron, fit):
    """Local trace mixed by A, plus the cross-target C and the off-target C_off (or a fixed signal)."""
    fixed = np.array(fit["offtarget_fixed"]) if "offtarget_fixed" in fit else None
    g = make_fast_grad(neuron, task, task["cfg"]["Ntar"], "trace", spike_mix=np.array(fit["spike_mix"]),
                       cross=np.array(fit["cross"]), offtarget=np.array(fit["offtarget"]), offtarget_fixed=fixed)
    return lambda p, spikes: g(p[0], p[1], spikes)


def make_rule(name, task, neuron, fit_path=None):
    if name == "exact":
        return exact(task, neuron)
    if name == "local":
        return local(task, neuron)
    return fitted(task, neuron, load_fit(fit_path, task))


# ---- Training and saved runs ----


def make_optimizer():
    return optax.adabelief(optax.exponential_decay(C.LR, C.DECAY_STEPS, 1 / jnp.e))


def row_norms(g):
    return jnp.concatenate([jnp.linalg.norm(g[0], axis=1), jnp.linalg.norm(g[1])[None]])


def row_cosine(g, G):
    num = jnp.sum(g[0] * G[0], axis=1)
    den = jnp.linalg.norm(g[0], axis=1) * jnp.linalg.norm(G[0], axis=1)
    return jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), jnp.nan)


def make_trainer(task, neuron, rule, with_lyapunov=True, n_trials=C.N_TRIALS, probe_every=C.PROBE_EVERY):
    """train(spikes, w0, x0) -> record of one run; vmap over runs."""
    exact_rule = exact(task, neuron)
    spike_counts = make_spike_counts(task, neuron)
    lyapunov = make_lyapunov(task, neuron) if with_lyapunov else None
    optim = make_optimizer()
    D = 2 * task["cfg"]["N"] + 1

    def probe(p, spikes):
        _, g = rule(p, spikes)
        _, G = exact_rule(p, spikes)
        if lyapunov is None:
            full = block = jnp.full(D, jnp.nan)
            n = jnp.array(0)
        else:
            full, block, n = lyapunov(p[0], p[1], spikes)
        theta = jnp.concatenate([jnp.ravel(p[0]), jnp.ravel(p[1])])
        return theta, row_cosine(g, G), row_norms(G), full, block, n

    def train(spikes, w0, x0):
        p0 = (w0, x0)

        def one_trial(carry, _):
            p, opt_state = carry
            loss, g = rule(p, spikes)
            counts = spike_counts(p[0], p[1], spikes)
            updates, opt_state = optim.update(g, opt_state)
            return (optax.apply_updates(p, updates), opt_state), (loss, counts, row_norms(g))

        def block(carry, _):
            probed = probe(carry[0], spikes)
            carry, per_trial = lax.scan(one_trial, carry, None, length=probe_every)
            return carry, (per_trial, probed)

        (p_end, _), (per_trial, probed) = lax.scan(block, (p0, optim.init(p0)), None, length=n_trials // probe_every)
        last = probe(p_end, spikes)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])  # noqa: E731
        append = lambda a, b: jnp.concatenate([a, b[None]])  # noqa: E731
        keys = ("theta", "cos", "gnorm_exact", "lyap", "lyap_block", "n_events")
        out = {"loss": flat(per_trial[0]), "counts": flat(per_trial[1]), "gnorm": flat(per_trial[2])}
        out.update({k: append(a, b) for k, a, b in zip(keys, probed, last)})
        return out

    return train


def out_path(case, rule_name, fit_name=None):
    name = rule_name if fit_name is None else f"fitted_{fit_name}"
    return C.RUNS / f"train_v2_{case}_{name}.npz"


def train_case(case, rule_name, fit_name=None, with_lyapunov=True):
    """Train all runs of a case (vmapped) and save them in the train_v2 format."""
    task, neuron = make_task(), make_neuron()
    fit_path = None if fit_name is None else C.FITS / f"fit_{fit_name}.json"
    rule = make_rule(rule_name, task, neuron, fit_path)
    spikes, w0, x0, level, seed = make_runs(task, case)
    t0 = time.time()
    record = jax.jit(jax.vmap(make_trainer(task, neuron, rule, with_lyapunov)))(spikes, w0, x0)
    r = "exact" if rule_name == "exact" else "local" if rule_name == "local" else "fitted"
    out = {f"{r}_{k}": np.asarray(v).astype(np.int16) if k == "counts" else np.asarray(v) for k, v in record.items()}
    out.update(level=level, seed=seed, probe_trials=np.arange(0, C.N_TRIALS + 1, C.PROBE_EVERY),
               args=json.dumps({"case": case, "rule": rule_name, "fit": fit_name, "trials": C.N_TRIALS}))
    path = out_path(case, rule_name, fit_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    final = out[f"{r}_loss"][:, -C.LAST:].mean(1)
    for v in sorted(set(level)):
        print(f"  level {v:<5g} final loss {final[level == v].mean():.3e}", flush=True)
    print(f"  -> {path} ({time.time() - t0:.0f}s)", flush=True)


# ---- Recordings and ridge fits ----


def recording_path(k):
    return C.RECORDING if k == 0 else C.FIT_INPUTS / f"input_{k:02d}.npz"


def record_exact_run(spikes_in, path, n_trials=C.N_TRIALS, stride=C.RECORD_STRIDE):
    """Train with the exact gradient on one input and record the pieces the fit needs."""
    task, neuron = make_task(), make_neuron()
    loss = make_loss(task, neuron)
    pieces = jax.jit(make_fast_grad(neuron, task, task["cfg"]["Ntar"], "trace",
                                    n_hidden_spikes=C.N_HIDDEN_SPIKES, return_pieces=True))
    optim = make_optimizer()

    @jax.jit
    def step(p, opt_state):
        value, grad = jax.value_and_grad(loss)(p, spikes_in)
        updates, opt_state = optim.update(grad, opt_state)
        return optax.apply_updates(p, updates), opt_state, value, grad[0]

    p = (task["w0"], task["x0"])
    opt_state = optim.init(p)
    rec = {k: [] for k in ("DL", "EL", "EH", "G", "trial")}
    t0 = time.time()
    for n in range(n_trials):
        p_before = p
        p, opt_state, value, g = step(p, opt_state)
        if n % stride == 0:
            _, _, pc = pieces(p_before[0], p_before[1], spikes_in)
            for k in ("DL", "EL", "EH"):
                rec[k].append(np.asarray(pc[k]))
            rec["G"].append(np.asarray(g))
            rec["trial"].append(n)
        if n % 1000 == 0:
            print(f"  trial {n:6d}  loss {float(value):.4e}  ({time.time() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.array(v) for k, v in rec.items()}, stride=stride)


def record(inputs):
    """Exact runs on the given inputs; inputs already recorded are kept."""
    task = make_task()
    for k in inputs:
        path = recording_path(k)
        if path.exists():
            continue
        print(f"input {k} -> {path}", flush=True)
        record_exact_run(fresh_input(task, k), path)


def load_recordings(n_inputs):
    """The recordings of inputs 0..n_inputs-1 stacked along the trial axis."""
    parts = []
    for k in range(n_inputs):
        if not recording_path(k).exists():
            raise FileNotFoundError(f"{recording_path(k)} is missing. Run the paper record command first.")
        parts.append(dict(np.load(recording_path(k))))
    return {key: np.concatenate([d[key] for d in parts]) for key in ("DL", "EL", "EH", "G", "trial")}


# ---------------- least squares ----------------

def ridge(X, Y, q, penalty):
    """argmin sum_n q_n |Y_n - X_n theta|^2 + sum penalty * theta^2, X (S, P, D), Y (S, P)."""
    P = X.shape[1]
    Xt, Yt, Wt = X.reshape(-1, X.shape[2]), Y.reshape(-1), np.repeat(q, P)
    M = Xt.T @ (Wt[:, None] * Xt)
    return np.linalg.solve(M + np.diag(penalty(np.diag(M))), Xt.T @ (Wt * Yt))


def scaled_ridge(r):
    """Ridge r times the mean diagonal of the normal matrix."""
    return lambda diag: np.full(len(diag), r * max(diag.mean(), 1e-300))


def metrics(E, P):
    """Median relative error and median cosine of rows E (S, D) against predictions P."""
    ne = np.linalg.norm(E, axis=1)
    ok = ne > 0
    rel = np.linalg.norm(E - P, axis=1)[ok] / ne[ok]
    cos = np.sum(E * P, 1)[ok] / (ne[ok] * np.maximum(np.linalg.norm(P, axis=1)[ok], 1e-30))
    return float(np.median(rel)), float(np.median(cos))


def trial_weights(cfg, G, ntar, i):
    """q^(n) for the regression of row i: target-row norm (base), one (no q) or the row's own norm (q per row)."""
    if cfg["q"] is True:
        return 1.0 / np.maximum(np.linalg.norm(G[:, :ntar], axis=(1, 2)), 1e-30) ** 2
    if cfg["q"] is False:
        return np.ones(len(G))
    nr = np.linalg.norm(G[:, i], axis=1)
    ok = nr > 1e-3 * np.median(nr[nr > 0]) if np.any(nr > 0) else np.zeros_like(nr, bool)
    return np.where(ok, 1.0 / np.where(ok, nr, 1.0) ** 2, 0.0)


# ---------------- the fit ----------------

def fit_target_row(cfg, d, i, ns, ntar, train, test, q):
    """A (mixing of the traces into i) and C (errors of the other targets) for target row i."""
    DL, EL, G = d["DL"], d["EL"], d["G"]
    S, N1 = len(G), G.shape[2]
    off = [j for j in range(N1 + 1) if j != i]
    n = ns[i]
    e = EL[:, i, :n][:, :, off]
    F = DL[:, i, :n, None] * e
    local = F.sum(1)
    others = [m for m in range(ntar) if m != i]
    blocks, kinds = [], []
    if cfg["A"]:
        blocks.append(np.einsum("skp,jq->sjkpq", F, np.eye(N1)).reshape(S, N1, -1))
        kinds.append("A")
    if cfg["C"] and cfg["Cx"] and others:
        blocks.append(np.concatenate([np.einsum("sa,scj->sjac", DL[:, m, :ns[m]], e).reshape(S, N1, -1)
                                      for m in others], axis=2))
        kinds.append("C")
    mix = np.tile(np.eye(N1), (n, 1, 1))
    cross = {}
    if not blocks:
        return mix, cross, local, metrics(G[test, i], local[test]), metrics(G[test, i], local[test])
    X = np.concatenate(blocks, axis=2)
    sizes = [b.shape[2] for b in blocks]

    def penalty(diag):
        out, o = [], 0
        for size, kind in zip(sizes, kinds):
            r = cfg["r_A"] if kind == "A" else cfg["r_C"]
            out.append(np.full(size, r * max(diag[o:o + size].mean(), 1e-300)))
            o += size
        return np.concatenate(out)

    theta = ridge(X[train], (G[:, i] - local)[train], q[train], penalty)
    o = 0
    for size, kind in zip(sizes, kinds):
        th = theta[o:o + size]
        if kind == "A":
            mix = np.eye(N1)[None] + th.reshape(n, N1, N1)
        else:
            oo = 0
            for m in others:
                cross[m] = th[oo:oo + ns[m] * n].reshape(ns[m], n)
                oo += ns[m] * n
        o += size
    pred = local + X @ theta
    return mix, cross, pred, metrics(G[test, i], local[test]), metrics(G[test, i], pred[test])


def fit_offtarget_row(cfg, d, i, ntar, train, q):
    """C_off (errors of all targets) or a fixed signal per spike for off-target row i."""
    DL, EH, G = d["DL"], d["EH"], d["G"]
    S, N1 = len(G), G.shape[2]
    off = [j for j in range(N1 + 1) if j != i]
    e = EH[:, i - ntar][:, :, off]                       # (S, Kh, N-1)
    if cfg["off"] == "fixed":
        X = np.transpose(e, (0, 2, 1))                   # regressor of S[i, c] is e_i.(t_c)
    else:
        X = np.einsum("smc,sdj->sjmcd", DL, e).reshape(S, N1, -1)
    theta = ridge(X[train], G[train, i], q[train], scaled_ridge(cfg["r_C_off"]))
    return theta, X @ theta


def fit_config(cfg, d, ns, N, ntar):
    """All coefficients of one configuration, in the json layout the rules read."""
    S, nmax, Kh, N1 = len(d["G"]), d["DL"].shape[2], d["EH"].shape[2], N - 1
    blocks = (np.arange(S) // C.BLOCK) % 2
    train, test = blocks == 0, blocks == 1
    MIX = np.tile(np.eye(N1), (ntar, nmax, 1, 1))
    CROSS = np.zeros((ntar, ntar, nmax, nmax))
    OFFT = np.zeros((N - ntar, ntar, nmax, Kh))
    OFFS = np.zeros((N - ntar, Kh))
    met = {}
    for i in range(ntar):
        q = trial_weights(cfg, d["G"], ntar, i)
        mix, cross, _, m_local, m_fit = fit_target_row(cfg, d, i, ns, ntar, train, test, q)
        MIX[i, :ns[i]] = mix
        for m, c in cross.items():
            CROSS[m, i, :ns[m], :ns[i]] = c
        met[f"target {i}|local"], met[f"target {i}|fit"] = m_local, m_fit
    pred_off = np.zeros((S, N - ntar, N1))
    for i in range(ntar, N):
        if cfg["C"] and cfg["Coff"]:
            q = trial_weights(cfg, d["G"], ntar, i)
            theta, pred_off[:, i - ntar] = fit_offtarget_row(cfg, d, i, ntar, train, q)
            if cfg["off"] == "fixed":
                OFFS[i - ntar] = theta
            else:
                OFFT[i - ntar] = theta.reshape(ntar, nmax, Kh)
        met[f"off-target {i}"] = metrics(d["G"][test, i], pred_off[test, i - ntar])
    met["off-target, all rows"] = metrics(d["G"][test, ntar:].reshape(test.sum(), -1), pred_off[test].reshape(test.sum(), -1))
    out = {"config": cfg, "metrics": met, "spike_mix": MIX.tolist(), "cross": CROSS.tolist(), "offtarget": OFFT.tolist(),
           "n_outputs": ns, "n_hidden_spikes": int(Kh), "r_A": cfg["r_A"], "r_C": cfg["r_C"], "r_C_off": cfg["r_C_off"]}
    if cfg["off"] == "fixed":
        out["offtarget_fixed"] = OFFS.tolist()
    return out


def configurations():
    """Base, the switches and one sweep per ridge (others at base), as in App. J of the paper."""
    out = {"base": dict(C.BASE_FIT),
           "noA": dict(C.BASE_FIT, A=False), "noC": dict(C.BASE_FIT, C=False),
           "noCx": dict(C.BASE_FIT, Cx=False), "noCoff": dict(C.BASE_FIT, Coff=False)}
    for key, tag in (("r_A", "rA"), ("r_C", "rC"), ("r_C_off", "rCoff")):
        for r in C.RIDGES:
            if r != C.BASE_FIT[key]:
                out[f"{tag}_{r:g}"] = dict(C.BASE_FIT, **{key: r})
    return out


def fit_all(n_inputs=C.N_FIT_INPUTS, force=False):
    """One input: every configuration. Several inputs: the base configuration, saved as fit_base_in<n>."""
    task = make_task()
    N, ntar = task["cfg"]["N"], task["cfg"]["Ntar"]
    ns = [len(t) for t in task["t_targets"]]
    d = load_recordings(n_inputs)
    if n_inputs == 1:
        todo = configurations()
    else:
        todo = {f"base_in{n_inputs}": dict(C.BASE_FIT)}
    C.FITS.mkdir(parents=True, exist_ok=True)
    for name, cfg in todo.items():
        path = C.FITS / f"fit_{name}.json"
        if path.exists() and not force:
            continue
        out = fit_config(cfg, d, ns, N, ntar)
        out["n_inputs"] = n_inputs
        with path.open("w") as stream:
            json.dump(out, stream)
        m = out["metrics"]
        print(f"  {name:<20} target cos {m['target 0|fit'][1]:.3f} / {m['target 1|fit'][1]:.3f}"
              f"   off-target cos {m['off-target, all rows'][1]:.3f}", flush=True)


# ---- Firing rates ----


BIN = 0.5
RATES = C.DATA / "rates"


def make_spike_times(task, neuron):
    """(w, x0, spikes) -> spike times (K,) and neuron of each spike (K,), -1 where the event is not a network spike."""
    cfg = task["cfg"]

    def spike_times(w, x0, spikes):
        times, is_in, neurons, _ = neuron.event(x0, w_to_weights_net(w), task["weights_in"], spikes, cfg)
        net = (~is_in) & (neurons >= 0) & (times <= cfg["T"])
        return times, jnp.where(net, neurons, -1)

    return spike_times


def binned(times, neurons, edges, N):
    """Spike counts (bins, N) of one simulation."""
    b = jnp.clip(jnp.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)
    one_hot = (neurons[:, None] == jnp.arange(N)) & (neurons[:, None] >= 0)
    return jnp.zeros((len(edges) - 1, N)).at[b].add(one_hot)


def unpack(theta, N):
    """theta = (ravel(w), ravel(x0)) as saved by train_case."""
    return theta[: N * (N - 1)].reshape(N, N - 1), theta[N * (N - 1):].reshape(2, N)


def rates(case, rule_name, fit_name=None):
    task, neuron = make_task(), make_neuron()
    N, T = task["cfg"]["N"], task["cfg"]["T"]
    edges = jnp.arange(0.0, T + BIN / 2, BIN)
    d = np.load(out_path(case, rule_name, fit_name))
    r = "exact" if rule_name == "exact" else "local" if rule_name == "local" else "fitted"
    thetas = d[f"{r}_theta"]                                  # (runs, checkpoints, D)
    spikes, _, _, level, seed = make_runs(task, case)
    spike_times = make_spike_times(task, neuron)

    @jax.jit
    def one_run(thetas_run, spikes_run):
        def at(theta):
            w, x0 = unpack(theta, N)
            t, n = spike_times(w, x0, spikes_run)
            return binned(t, n, edges, N)
        return jax.vmap(at)(thetas_run)                       # (checkpoints, bins, N)

    counts = np.stack([np.asarray(one_run(thetas[k], (spikes[0][k], spikes[1][k]))) for k in range(len(level))])
    RATES.mkdir(parents=True, exist_ok=True)
    name = rule_name if fit_name is None else f"fitted_{fit_name}"
    path = RATES / f"rates_{case}_{name}.npz"
    np.savez(path, rate=counts.mean(1) / BIN, count=counts.sum(2), edges=np.asarray(edges), level=level, seed=seed)
    print(f"  -> {path}", flush=True)


# ---- Reproduction check ----


N_CHECK, RUNS_CHECKED = 400, 3


def check(case, rule_name, fit_name=None):
    task, neuron = make_task(), make_neuron()
    fit_path = None if fit_name is None else C.FITS / f"fit_{fit_name}.json"
    rule = make_rule(rule_name, task, neuron, fit_path)
    spikes, w0, x0, _, _ = make_runs(task, case)
    sl = slice(0, RUNS_CHECKED)
    train = jax.jit(jax.vmap(make_trainer(task, neuron, rule, with_lyapunov=False, n_trials=N_CHECK)))
    new = np.asarray(train((spikes[0][sl], spikes[1][sl]), w0[sl], x0[sl])["loss"])
    r = "exact" if rule_name == "exact" else "local" if rule_name == "local" else "fitted"
    old = np.load(out_path(case, rule_name, fit_name))[f"{r}_loss"][sl, :N_CHECK]
    print(f"  {case:8s} {fit_name or rule_name:8s} max |new - saved| over {N_CHECK} trials: {np.max(np.abs(new - old)):.2e}")
