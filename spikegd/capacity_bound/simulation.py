"""Exact event-driven QIF simulation and complex-step event derivatives.

The simulator uses tau_m=1 and tau_s=1/2. Its state is (V0, b) after an input and
(t_out, b_out) after an output. Jacobians use f'(x) = Im f(x + i h) / h.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Scalar = float | complex | np.number | np.ndarray


# %%
############################
### Configuration
############################


H = 1e-30  # complex-step size
PI = np.pi


# %%
############################
### Closed forms
############################


def flow(V0: Scalar, b: Scalar, t: Scalar) -> Scalar:
    """V(t) from a finite V0 (eq. c-V). Branch chosen on the real parts."""
    br, Vr = np.real(b), np.real(V0)
    if br > 0:
        s = np.sqrt(b)
        return s * np.exp(-t) * np.tan(np.arctan(V0 / s) + s * (1 - np.exp(-t)))
    if br == 0:
        return V0 / (V0 - (V0 - 1) * np.exp(t))
    s = np.sqrt(-b)
    if abs(Vr) > np.real(s):  # coth branch, arcoth(x) = artanh(1/x)
        return s * np.exp(-t) / np.tanh(np.arctanh(s / V0) - s * (1 - np.exp(-t)))
    return s * np.exp(-t) * np.tanh(np.arctanh(V0 / s) - s * (1 - np.exp(-t)))


def flow_reset(b: Scalar, t: Scalar) -> Scalar:
    """V(t) a time t after a reset (V0 = -inf)."""
    br = np.real(b)
    if br > 0:
        s = np.sqrt(b)
        return s * np.exp(-t) * np.tan(-PI / 2 + s * (1 - np.exp(-t)))
    if br == 0:
        return -1 / (np.exp(t) - 1)
    s = np.sqrt(-b)
    return s * np.exp(-t) / np.tanh(-s * (1 - np.exp(-t)))


def tsp(V0: Scalar, b: Scalar) -> Scalar:
    """Time to the next spike from (V0,b) (eq. c-tsp); np.inf if there is none."""
    br, Vr = np.real(b), np.real(V0)
    if br > 0:
        s = np.sqrt(b)
        D = 1 - (PI / 2 - np.arctan(V0 / s)) / s
    elif br == 0:
        if Vr <= 1:
            return np.inf
        D = 1 - 1 / V0
    else:
        s = np.sqrt(-b)
        if Vr <= np.real(s):
            return np.inf
        D = 1 - np.arctanh(s / V0) / s
    return np.inf if np.real(D) <= 0 else -np.log(D)


def tsp_reset(b: Scalar) -> Scalar:
    """Time from a reset to the next spike (eq. c-reset); np.inf if b <= pi^2."""
    if np.real(b) <= PI**2:
        return np.inf
    return -np.log(1 - PI / np.sqrt(b))


# %%
############################
### Event maps
############################


def map_in_in(s: np.ndarray, t_last: float, t_in: float, w: float) -> np.ndarray:
    D = t_in - t_last
    return np.array([flow(s[0], s[1], D), s[1] * np.exp(-2 * D) + w])


def map_in_out(s: np.ndarray, t_last: float) -> np.ndarray:
    T = tsp(s[0], s[1])
    return np.array([t_last + T, s[1] * np.exp(-2 * T)])


def map_out_in(s: np.ndarray, t_in: float, w: float) -> np.ndarray:
    D = t_in - s[0]
    return np.array([flow_reset(s[1], D), s[1] * np.exp(-2 * D) + w])


def map_out_out(s: np.ndarray) -> np.ndarray:
    T = tsp_reset(s[1])
    return np.array([s[0] + T, s[1] * np.exp(-2 * T)])


def jac(f: Callable, s: np.ndarray) -> np.ndarray:
    """2x2 Jacobian of f at the real state s by the complex step."""
    J = np.empty((2, 2))
    for j in range(2):
        z = s.astype(complex)
        z[j] += 1j * H
        J[:, j] = np.imag(f(z)) / H
    return J


# %%
############################
### Simulation
############################


def simulate(
    t_inputs: np.ndarray,
    syn: np.ndarray,
    weights: np.ndarray,
    V0: float = 0.0,
    b0: float = 0.0,
    T_end: float = 10.0,
    max_events: int = 100000,
) -> dict:
    """Event-driven run with RTRL traces.

    t_inputs : sorted input times (external, fixed);  syn : synapse index of each
    input;  weights : one weight per synapse.
    Parameters are theta = (V0, w_1, ..., w_l); the trace M is 2 x (1+l).
    Returns a dict with the event list (kind, time, state, Jacobian) and the traces
    at every event.

    """
    ### Unpack arguments
    t_inputs = np.asarray(t_inputs, float)
    syn = np.asarray(syn, int)
    weights = np.asarray(weights, float)

    ### Initialize the simulation
    n_weights = len(weights)
    kind, s, t_last = (
        "in",
        np.array([V0, b0], float),
        0.0,
    )  # t=0 acts like an input with no jump
    M = np.zeros((2, 1 + n_weights))
    M[0, 0] = 1.0
    ev = dict(kind=[], time=[], state=[], J=[], M=[], b_after_last_input=[])
    j = 0

    ### Event and gradient updates
    while len(ev["kind"]) < max_events:
        t_next = t_inputs[j] if j < len(t_inputs) else np.inf
        t_spk = (t_last + tsp(*s)) if kind == "in" else (s[0] + tsp_reset(s[1]))
        if t_spk < min(t_next, T_end):  # an output comes first
            if kind == "in":

                def f(z: np.ndarray, tl: float = t_last) -> np.ndarray:
                    return map_in_out(z, tl)

                new = "in->out"
            else:
                f = map_out_out
                new = "out->out"
            Jk = jac(f, s)
            s = np.real(f(s))
            kind = "out"
            M = Jk @ M
        elif t_next < T_end:  # an input comes first
            w = weights[syn[j]]
            if kind == "in":

                def f(
                    z: np.ndarray,
                    tl: float = t_last,
                    tn: float = t_next,
                    ww: float = w,
                ) -> np.ndarray:
                    return map_in_in(z, tl, tn, ww)

                new = "in->in"
            else:

                def f(
                    z: np.ndarray,
                    tn: float = t_next,
                    ww: float = w,
                ) -> np.ndarray:
                    return map_out_in(z, tn, ww)

                new = "out->in"
            Jk = jac(f, s)
            s = np.real(f(s))
            kind = "in"
            t_last = t_next
            M = Jk @ M
            M[1, 1 + syn[j]] += 1.0  # injection (0,1) in the column of the synapse
            j += 1
        else:
            break
        ev["kind"].append(new)
        ev["time"].append(s[0] if kind == "out" else t_last)
        ev["state"].append(s.copy())
        ev["J"].append(Jk)
        ev["M"].append(M.copy())

    ### Collect results
    for k in ev:
        ev[k] = list(ev[k])
    ev["outputs"] = [t for t, k in zip(ev["time"], ev["kind"]) if k.endswith("out")]
    return ev


# %%
############################
### QR accumulation
############################


def initial_basis(seed_ons: int | None = None) -> np.ndarray:
    """Random orthonormal 2x2 basis (Gaussian QR, as in Toolkit 2); identity if
    seed_ons is None.
    """
    if seed_ons is None:
        return np.eye(2)
    Q, _ = np.linalg.qr(np.random.default_rng(seed_ons).standard_normal((2, 2)))
    return Q


def qr_accumulate(Js: np.ndarray | list, Q0: np.ndarray | None = None) -> np.ndarray:
    """Benettin/QR along a product of event Jacobians, QR after every event, no
    alignment events discarded.
    Returns the running sums of log|R_ii| after every event (shape (K,2)), unsorted
    (Q is never reordered).
    """
    Q = np.eye(2) if Q0 is None else np.array(Q0, float)
    logs = np.zeros(2)
    hist = []
    for Jk in Js:
        Q, R = np.linalg.qr(Jk @ Q)
        with np.errstate(divide="ignore"):
            logs = logs + np.log(np.abs(np.diag(R)))
        hist.append(logs.copy())
    return np.array(hist)


# %%
############################
### Groups and the gap count
############################


def groups(ev: dict, t_inputs: np.ndarray, T_end: float = np.inf) -> list[dict]:
    """Split the run at its outputs: for every maximal run of inputs with no output in
    between, return
    (index of last input, current B just after it, sqrt(B), c2, gap G to the next
    input or trial end, outputs in the gap).
    """
    out = []
    t_inputs = np.asarray(t_inputs)
    kinds, times, states = ev["kind"], ev["time"], ev["state"]
    for e, k in enumerate(kinds):
        if not k.endswith("in"):
            continue
        nxt = e + 1
        if nxt < len(kinds) and kinds[nxt].endswith("in"):
            continue  # not the last input of its group
        V0, B = states[e]
        tl = times[e]
        later = t_inputs[t_inputs > tl + 1e-12]
        G = min(later[0] if len(later) else np.inf, T_end) - tl
        n_out = 0
        while nxt < len(kinds) and kinds[nxt].endswith("out"):
            n_out += 1
            nxt += 1
        c2 = np.sqrt(B) - PI / 2 + np.arctan(V0 / np.sqrt(B)) if B > 0 else -np.inf
        out.append(dict(event=e, t=tl, V0=V0, B=B, c2=c2, G=G, n_out=n_out))
    return out


# ---- Delayed re-entry (values only) ----


def r1_outputs(
    w_ext: float,
    t_ext: float,
    wj: float,
    d: float,
    T: float,
    V0: float = 0.0,
    b0: float = 0.0,
) -> np.ndarray:
    """Return output times with delayed recurrence, for finite differences."""
    ### Initialize the neuron
    V, b = flow(V0, b0, t_ext), b0 * np.exp(-2 * t_ext) + w_ext
    t = t_ext
    outs = []

    ### Advance outputs and delayed inputs
    while True:
        Ts = tsp(V, b)
        if not np.isfinite(Ts) or t + Ts >= T:
            return np.array(outs)
        t = t + Ts
        b = b * np.exp(-2 * Ts)
        outs.append(t)
        if wj == 0:  # no re-entry: the train continues from the reset
            while b > PI**2 and t + tsp_reset(b) < T:
                Ts = tsp_reset(b)
                t += Ts
                b *= np.exp(-2 * Ts)
                outs.append(t)
            return np.array(outs)
        if t + d >= T:
            return np.array(outs)
        if b > PI**2 and tsp_reset(b) < d:
            raise ValueError("output before its re-entry: outside (R3)")
        V, b = flow_reset(b, d), b * np.exp(-2 * d) + wj
        t = t + d
