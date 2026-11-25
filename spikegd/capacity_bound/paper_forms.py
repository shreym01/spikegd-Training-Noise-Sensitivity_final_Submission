"""Every closed form of the paper, transcribed literally from the equations
(section/eq. given in each docstring).

Value functions take a backend `m` (numpy, or mpmath for 50-digit finite
differences). Derivative functions are the paper's
closed-form derivatives in float64 and are what the checks compare against.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

Scalar = float | complex | np.number | mp.mpf | mp.mpc


# %%
############################
### Numerical backends
############################


class NP:  # numpy backend with the functions numpy lacks
    pi = np.pi
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    tan = staticmethod(np.tan)
    atan = staticmethod(np.arctan)
    tanh = staticmethod(np.tanh)
    atanh = staticmethod(np.arctanh)

    @staticmethod
    def coth(x):
        return 1 / np.tanh(x)

    @staticmethod
    def acoth(x):
        return np.arctanh(1 / x)


class MP:
    pi = mp.pi
    exp = staticmethod(mp.exp)
    log = staticmethod(mp.log)
    sqrt = staticmethod(mp.sqrt)
    tan = staticmethod(mp.tan)
    atan = staticmethod(mp.atan)
    tanh = staticmethod(mp.tanh)
    atanh = staticmethod(mp.atanh)
    coth = staticmethod(mp.coth)
    acoth = staticmethod(mp.acoth)


# %%
############################
### Sec. 2.1, eq. (3): voltage, five branches
############################


def branch(V0: Scalar, b: Scalar) -> str:
    if b > 0:
        return "tan"
    if b == 0:
        return "b0"
    s = (-b) ** 0.5 if not isinstance(b, mp.mpf) else mp.sqrt(-b)
    if abs(V0) == s:
        return "line"
    return "coth" if abs(V0) > s else "tanh"


def V(
    V0: Scalar, b: Scalar, t: Scalar, m: type[NP] | type[MP] = NP, br: str | None = None
) -> Scalar:
    br = br or branch(V0, b)
    if br == "b0":
        return V0 / (V0 - (V0 - 1) * m.exp(t))
    if br == "tan":
        s = m.sqrt(b)
        return s * m.exp(-t) * m.tan(m.atan(V0 / s) + s * (1 - m.exp(-t)))
    s = m.sqrt(-b)
    if br == "line":
        return (1 if V0 > 0 else -1) * s * m.exp(-t)
    if br == "coth":
        return s * m.exp(-t) * m.coth(m.acoth(V0 / s) - s * (1 - m.exp(-t)))
    return s * m.exp(-t) * m.tanh(m.atanh(V0 / s) - s * (1 - m.exp(-t)))


# %%
############################
### Eq. (4): spike time;  App. A.5: D
############################


def D(V0: Scalar, b: Scalar, m: type[NP] | type[MP] = NP) -> Scalar:
    if b == 0:
        return 1 - 1 / V0
    if b > 0:
        s = m.sqrt(b)
        return 1 - m.pi / (2 * s) + m.atan(V0 / s) / s
    s = m.sqrt(-b)
    return 1 - m.acoth(V0 / s) / s


def tsp(V0: Scalar, b: Scalar, m: type[NP] | type[MP] = NP) -> Scalar:
    if b == 0:
        return m.log(V0 / (V0 - 1))
    if b > 0:
        s = m.sqrt(b)
        return m.log(2 * s / (2 * m.atan(V0 / s) - m.pi + 2 * s))
    s = m.sqrt(-b)
    return m.log(s / (s - m.acoth(V0 / s)))


# eq. (5): from a reset
def tsp_reset(b: Scalar, m: type[NP] | type[MP] = NP) -> Scalar:
    return -m.log(1 - m.pi / m.sqrt(b))


# %%
############################
### App. A.5: derivatives of the spike time
############################


def dtsp_dV0(V0: float, b: float) -> float:
    if b == 0:
        return -1 / (V0 * (V0 - 1))
    return -1 / (D(V0, b) * (b + V0**2))


def dtsp_db(V0: float, b: float) -> float:
    if b == 0:
        return -1 / (3 * V0**2 * (V0 - 1))
    Dv = D(V0, b)
    if b > 0:
        s = np.sqrt(b)
        return (
            -(
                np.pi / (4 * b * s)
                - np.arctan(V0 / s) / (2 * b * s)
                - V0 / (2 * b * (b + V0**2))
            )
            / Dv
        )
    s = np.sqrt(-b)
    return -(NP.acoth(V0 / s) / (2 * b * s) + V0 / (2 * (-b) * (V0**2 + b))) / Dv


# %%
############################
### App. A.6: derivatives of the voltage
############################


def K(V0: float, b: float, t: float, br: str | None) -> float:
    if br == "tan":
        s = np.sqrt(b)
        return np.arctan(V0 / s) + s * (1 - np.exp(-t))
    s = np.sqrt(-b)
    if br == "coth":
        return NP.acoth(V0 / s) - s * (1 - np.exp(-t))
    return np.arctanh(V0 / s) - s * (1 - np.exp(-t))


def dK_db(V0: float, b: float, t: float, br: str | None) -> float:
    if br == "tan":
        s = np.sqrt(b)
        return -V0 / (2 * s * (b + V0**2)) + (1 - np.exp(-t)) / (2 * s)
    s = np.sqrt(-b)
    return -V0 / (2 * s * (V0**2 + b)) + (1 - np.exp(-t)) / (2 * s)


def dV_dV0(V0: float, b: float, t: float, br: str | None = None) -> float:
    br = br or branch(V0, b)
    if br == "b0":
        return np.exp(t) / (V0 - (V0 - 1) * np.exp(t)) ** 2
    k = K(V0, b, t, br)
    if br == "tan":
        return b * np.exp(-t) / np.cos(k) ** 2 / (b + V0**2)
    if br == "coth":
        return -b * np.exp(-t) / np.sinh(k) ** 2 / (b + V0**2)
    if br == "tanh":
        return -b * np.exp(-t) / np.cosh(k) ** 2 / (-b - V0**2)


def dV_db(V0: float, b: float, t: float, br: str | None = None) -> float:
    br = br or branch(V0, b)
    if br == "b0":
        if V0 == 0:
            return np.exp(-t) * (
                1 - np.exp(-t)
            )  # removable 0/0 of the paper's b=0 entry at V0=0 (limit V0->0)
        Vt = V(V0, 0.0, t)
        return (
            Vt**2
            * np.exp(t)
            / V0**2
            * (
                V0**2 * (1 - np.exp(-3 * t)) / 3
                - V0 * (V0 - 1) * (1 - np.exp(-2 * t))
                + (V0 - 1) ** 2 * (1 - np.exp(-t))
            )
        )
    if br == "line":
        return (
            -np.sign(V0) * np.exp(-t) / (2 * np.sqrt(-b))
        )  # as printed in the paper (derivative along the line)
    k = K(V0, b, t, br)
    dk = dK_db(V0, b, t, br)
    if br == "tan":
        s = np.sqrt(b)
        return np.exp(-t) * np.tan(k) / (2 * s) + s * np.exp(-t) / np.cos(k) ** 2 * dk
    s = np.sqrt(-b)
    if br == "coth":
        return (
            -np.exp(-t) / np.tanh(k) / (2 * s) - s * np.exp(-t) / np.sinh(k) ** 2 * dk
        )
    return -np.exp(-t) * np.tanh(k) / (2 * s) + s * np.exp(-t) / np.cosh(k) ** 2 * dk


def dV_dt(V0: float, b: float, t: float, br: str | None = None) -> float:
    Vt = V(V0, b, t, br=br)
    return Vt**2 - Vt + b * np.exp(-2 * t)


def dV_db_line_corrected(V0: float, b: float, t: float) -> float:
    """Partial derivative at fixed V0 on the line |V0| = sqrt(-b), from the variational
    equation (correction found by this check).
    """
    return np.exp(-t) * (np.exp(2 * V0 * (1 - np.exp(-t))) - 1) / (2 * V0)


# %%
############################
### App. A.7: from a reset (V0 -> -infinity)
############################


def V_reset(b: Scalar, tau: Scalar, m: type[NP] | type[MP] = NP) -> Scalar:
    if b > 0:
        s = m.sqrt(b)
        return s * m.exp(-tau) * m.tan(-m.pi / 2 + s * (1 - m.exp(-tau)))
    if b == 0:
        return -1 / (m.exp(tau) - 1)
    s = m.sqrt(-b)
    return s * m.exp(-tau) * m.coth(-s * (1 - m.exp(-tau)))


def dVreset_db(b: float, tau: float) -> float:
    if b > 0:
        s = np.sqrt(b)
        k = -np.pi / 2 + s * (1 - np.exp(-tau))
        return (
            np.exp(-tau) * np.tan(k) / (2 * s)
            + np.exp(-tau) / np.cos(k) ** 2 * (1 - np.exp(-tau)) / 2
        )
    if b == 0:
        Vt = V_reset(0.0, tau)
        return (
            Vt**2
            * np.exp(tau)
            * ((1 - np.exp(-3 * tau)) / 3 - (1 - np.exp(-2 * tau)) + (1 - np.exp(-tau)))
        )
    s = np.sqrt(-b)
    k = -s * (1 - np.exp(-tau))
    return (
        -np.exp(-tau) / np.tanh(k) / (2 * s)
        - np.exp(-tau) / np.sinh(k) ** 2 * (1 - np.exp(-tau)) / 2
    )


# %%
############################
### Sec. 2.1: vector field, Props. 1-2
############################


def f(V0: float, b: float) -> np.ndarray:
    return np.array([V0**2 - V0 + b, -2 * b])


def grad_tsp(V0: float, b: float) -> np.ndarray:
    return np.array([dtsp_dV0(V0, b), dtsp_db(V0, b)])


def input_time_sensitivity(
    V0: float, b: float, w: float
) -> float:  # Prop. 2, right-hand side
    return grad_tsp(V0, b) @ np.array([-w, 2 * w])


# %%
############################
### App. B: the four event maps (state after in: (V0,b); after out: (t_out,b_out))
############################


def J_in_in(V0: float, b: float, Delta: float) -> np.ndarray:
    return np.array(
        [[dV_dV0(V0, b, Delta), dV_db(V0, b, Delta)], [0.0, np.exp(-2 * Delta)]]
    )


def J_in_out(V0: float, b: float) -> np.ndarray:
    T = tsp(V0, b)
    bo = b * np.exp(-2 * T)
    g = grad_tsp(V0, b)
    return np.array([g, -2 * bo * g + np.array([0.0, np.exp(-2 * T)])])


def J_out_out(b_out: float) -> np.ndarray:
    c = np.sqrt(b_out)
    return np.array([[1.0, -np.pi / (2 * c**2 * (c - np.pi))], [0.0, (c - np.pi) / c]])


def J_out_in(b_out: float, Delta: float) -> np.ndarray:
    Vd = V_reset(b_out, Delta)
    bm = b_out * np.exp(-2 * Delta)
    return np.array(
        [[-(Vd**2 - Vd + bm), dVreset_db(b_out, Delta)], [2 * bm, np.exp(-2 * Delta)]]
    )


# Sec. 4.1, eq. (9): recurrent block; Prop. 7: cross block
def J_rec(w: float) -> np.ndarray:
    return np.array([[-w, 0.0], [2 * w, 0.0]])


def J_cross(w: float, grad_tX: np.ndarray) -> np.ndarray:
    return np.outer([-w, 2 * w], grad_tX)


# %%
############################
### Prop. 3 (horizon) and Prop. 4 (gap)
############################


def c2(V0: float, b: float) -> float:
    return np.sqrt(b) - np.pi / 2 + np.arctan(V0 / np.sqrt(b))


def horizon_count(V0: float, b: float) -> int:
    return int(np.ceil(c2(V0, b) / np.pi))


def output_times_after_input(
    V0: float, b: float, k: int
) -> np.ndarray:  # t_i = 1/2 ln b - ln(c2-(i-1)pi), measured from the last input
    return np.array(
        [0.5 * np.log(b) - np.log(c2(V0, b) - (i - 1) * np.pi) for i in range(1, k + 1)]
    )


def trace_after_input(
    V0: float,
    b: float,
    beta: float | np.ndarray,
    dV0_dtheta: float | np.ndarray,
    i: int,
) -> tuple:  # Prop. 3(iii)
    s = np.sqrt(b)
    gamma = (
        s / (b + V0**2) * dV0_dtheta + (1 / (2 * s) - V0 / (2 * s * (b + V0**2))) * beta
    )
    ci = c2(V0, b) - (i - 1) * np.pi
    return beta / (2 * b) - gamma / ci, 2 * ci * gamma


def gap_count(V0: float, B: float, G: float) -> int:  # Prop. 4
    return max(0, int(np.ceil((c2(V0, B) - np.sqrt(B) * np.exp(-G)) / np.pi)))


# %%
############################
### R1: App. D, four numbers of a cycle and the cycle maps (Prop. 5)
############################


def r1_cycle(B: float, wj: float, d: float) -> tuple:
    """From output i with reset current B: returns (sigma, rho, K, lambda, T_i,
    b_next_out).
    """
    V1 = V_reset(B, d)
    b1 = B * np.exp(-2 * d) + wj
    T = tsp(V1, b1)
    bn = b1 * np.exp(-2 * T)
    g = grad_tsp(V1, b1)
    sigma = g @ np.array([-wj, 2 * wj])
    rho = 2 * wj * np.exp(-2 * T) - 2 * bn * sigma
    Kc = g[0] * dVreset_db(B, d) + g[1] * np.exp(-2 * d)
    lam = np.exp(-2 * d) * np.exp(-2 * T) - 2 * bn * Kc
    return sigma, rho, Kc, lam, T, bn


def F_map(B: float, wj: float, d: float) -> np.ndarray:
    s, r, Kc, lam, _, _ = r1_cycle(B, wj, d)
    return np.array([[1.0, Kc], [0.0, lam]])


def G_map(B: float, wj: float, d: float) -> np.ndarray:
    s, r, Kc, lam, _, _ = r1_cycle(B, wj, d)
    return np.array([[1 - s, Kc], [-r, lam]])


def continuation(Bk: float, wj: float, d: float) -> bool:  # Prop. 6: (A) or (B)
    return (Bk > np.pi**2) or (Bk * np.exp(-2 * d) + wj > np.pi**2)


# ---- Analytic RTRL ----


def single_neuron(
    t_in: np.ndarray,
    syn: np.ndarray,
    w: np.ndarray,
    V0: float,
    b0: float,
    T: float,
    with_b0: bool = False,
) -> tuple:
    """RTRL of Sec. 2.2 for one neuron with external inputs. Parameters theta = (V0,
    w_1..w_l[, b0]).
    Returns output times and the time row of the trace at every output (list of
    arrays of length 1+l[+1]).
    """
    ### Initialize the simulation
    n_weights = len(w)
    M = np.zeros((2, 1 + n_weights + int(with_b0)))
    M[0, 0] = 1.0
    if with_b0:
        M[1, -1] = 1.0  # the initial current is a parameter: d b0 / d b0 = 1
    kind, s, t_last = "in", np.array([V0, b0], float), 0.0
    outs, rows, j = [], [], 0

    ### Event and gradient updates
    while True:
        t_next = t_in[j] if j < len(t_in) else np.inf
        if kind == "in":
            ts = t_last + tsp(*s) if _fires(s) else np.inf
        else:
            ts = s[0] + tsp_reset(s[1]) if s[1] > np.pi**2 else np.inf
        if ts < min(t_next, T):
            if kind == "in":
                J = J_in_out(*s)
                Tsp = tsp(*s)
                s = np.array([t_last + Tsp, s[1] * np.exp(-2 * Tsp)])
            else:
                J = J_out_out(s[1])
                Tsp = tsp_reset(s[1])
                s = np.array([s[0] + Tsp, s[1] * np.exp(-2 * Tsp)])
            M = J @ M
            kind = "out"
            outs.append(s[0])
            rows.append(M[0].copy())
        elif t_next < T:
            ww = w[syn[j]]
            if kind == "in":
                D = t_next - t_last
                J = J_in_in(s[0], s[1], D)
                s = np.array([V(s[0], s[1], D), s[1] * np.exp(-2 * D) + ww])
            else:
                D = t_next - s[0]
                J = J_out_in(s[1], D)
                s = np.array([V_reset(s[1], D), s[1] * np.exp(-2 * D) + ww])
            M = J @ M
            M[1, 1 + syn[j]] += 1.0
            kind = "in"
            t_last = t_next
            j += 1
        else:
            return np.array(outs), np.array(rows), (kind, s, t_last, M)


def _fires(s: np.ndarray) -> bool:
    V0, b = s
    if b > 0:
        return D(V0, b) > 0
    if b == 0:
        return V0 > 1
    return V0 > np.sqrt(-b) and D(V0, b) > 0


# %%
############################
### Recurrent gradients
############################


def r1(
    w_ext: float,
    t_ext: float,
    wj: float,
    d: float,
    T: float,
    V0: float = 0.0,
    b0: float = 0.0,
) -> dict:
    """R1 (Sec. 4.1): one external input, then every output re-enters after d through
    w_j. theta = (V0, w_ext, w_j).
    Exact trace with J_rec (eq. 9) and the e-prop trace without it.
    """
    out = {}
    for exact in (True, False):
        M = np.zeros((2, 3))
        M[0, 0] = 1.0
        D = t_ext
        s = np.array([V(V0, b0, D), b0 * np.exp(-2 * D) + w_ext])
        J = J_in_in(V0, b0, D)
        M = J @ M
        M[1, 1] += 1.0
        t_last = t_ext
        outs, rows = [], []
        while True:
            if not _fires(s):
                break
            Tsp = tsp(*s)
            if t_last + Tsp >= T:
                break
            J = J_in_out(*s)
            s = np.array([t_last + Tsp, s[1] * np.exp(-2 * Tsp)])
            M = J @ M
            outs.append(s[0])
            rows.append(M[0].copy())
            if s[0] + d >= T:
                break
            J = J_out_in(s[1], d) + (J_rec(wj) if exact else 0)
            s = np.array([V_reset(s[1], d), s[1] * np.exp(-2 * d) + wj])
            M = J @ M
            M[1, 2] += 1.0
            t_last = outs[-1] + d
        out["exact" if exact else "eprop"] = (np.array(outs), np.array(rows))
    return out


# %%
############################
### Network gradients
############################


def network(
    t_ext: np.ndarray,
    idx_ext: np.ndarray,
    W_in: np.ndarray,
    W_net: np.ndarray,
    x0: np.ndarray,
    T: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """N-neuron network with zero delays (spikegd's convention): a spike reaches every
    target at its own time.
    State a(k) stacks the two-vectors; J(k) has the single-neuron maps on the
    diagonal and the cross blocks of Prop. 7.
    theta = (W_in.ravel(), W_net.ravel()). Returns, per neuron, the output times and
    the time row of the trace at each output.
    """
    ### Initialize the network
    N, Nin = W_in.shape
    P_ = N * Nin + N * N
    kinds = ["in"] * N
    S = [np.array(x0[:, n], float) for n in range(N)]
    tl = [0.0] * N
    M = np.zeros((2 * N, P_))
    outs = [[] for _ in range(N)]
    rows = [[] for _ in range(N)]
    j = 0

    # Next output time
    def next_spike(n: int) -> float:
        s = S[n]
        if kinds[n] == "in":
            return tl[n] + tsp(*s) if _fires(s) else np.inf
        return s[0] + tsp_reset(s[1]) if s[1] > np.pi**2 else np.inf

    # State and Jacobian for a receiving neuron
    def recv(n: int, t: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        """diagonal block and new state of neuron n receiving an input of weight w at
        time t (input time held fixed).
        """
        s = S[n]
        if kinds[n] == "in":
            D = t - tl[n]
            J = J_in_in(s[0], s[1], D)
            new = np.array([V(s[0], s[1], D), s[1] * np.exp(-2 * D) + w])
        else:
            D = t - s[0]
            J = J_out_in(s[1], D)
            new = np.array([V_reset(s[1], D), s[1] * np.exp(-2 * D) + w])
        return J, new

    while True:
        t_next = t_ext[j] if j < len(t_ext) else np.inf
        tsps = [next_spike(n) for n in range(N)]
        X = int(np.argmin(tsps))
        tX = tsps[X]
        Jk = np.eye(2 * N)
        if tX < min(t_next, T):  # neuron X fires; its spike reaches every Y != X now
            s = S[X]
            if kinds[X] == "in":
                JX = J_in_out(*s)
                Tsp = tsp(*s)
                newX = np.array([tl[X] + Tsp, s[1] * np.exp(-2 * Tsp)])
            else:
                JX = J_out_out(s[1])
                Tsp = tsp_reset(s[1])
                newX = np.array([s[0] + Tsp, s[1] * np.exp(-2 * Tsp)])
            Jk[2 * X : 2 * X + 2, 2 * X : 2 * X + 2] = JX
            grad_tX = JX[0]
            inj = []
            for Y in range(N):
                if Y == X:
                    continue
                w = W_net[Y, X]
                JY, newY = recv(Y, tX, w)
                Jk[2 * Y : 2 * Y + 2, 2 * Y : 2 * Y + 2] = JY
                Jk[2 * Y : 2 * Y + 2, 2 * X : 2 * X + 2] = J_cross(
                    w, grad_tX
                )  # Prop. 7
                S[Y] = newY
                kinds[Y] = "in"
                tl[Y] = tX
                inj.append((Y, N * Nin + Y * N + X))
            S[X] = newX
            kinds[X] = "out"
            M = Jk @ M
            for Y, col in inj:
                M[2 * Y + 1, col] += 1.0
            outs[X].append(tX)
            rows[X].append(M[2 * X].copy())
        elif t_next < T:
            # external input j reaches every neuron Y with weight W_in[Y, idx]; a zero
            # weight is an event with no jump
            Jk = np.eye(2 * N)
            cols = []
            for Y in range(N):
                JY, newY = recv(Y, t_next, W_in[Y, idx_ext[j]])
                Jk[2 * Y : 2 * Y + 2, 2 * Y : 2 * Y + 2] = JY
                S[Y] = newY
                kinds[Y] = "in"
                tl[Y] = t_next
                cols.append((Y, Y * Nin + idx_ext[j]))
            M = Jk @ M
            for Y, col in cols:
                M[2 * Y + 1, col] += 1.0
            j += 1
        else:
            break
    return [np.array(o) for o in outs], [np.array(r) for r in rows]
