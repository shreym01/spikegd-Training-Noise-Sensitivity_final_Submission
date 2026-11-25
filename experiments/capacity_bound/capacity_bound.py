"""Generate the capacity-bound paper figures and run the numerical checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from spikegd.capacity_bound.simulation import PI, groups, initial_basis, qr_accumulate, simulate
from spikegd.capacity_bound.validation import (
    capture_results,
    check_simulator,
    check_weight_bounds,
    run_gradient_checks,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ---- Paper figures ----


WIDTH = 5.5  # inches, NeurIPS text width
# colour-blind safe (Okabe-Ito)
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERM, PURPLE = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)
GREY = "0.6"


# %%
############################
### Plotting
############################


def configure_style() -> ModuleType:
    """Apply the paper's plotting style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def save(fig: Figure, name: str, output_dir: str | Path = OUTPUT_DIR) -> None:
    """Save a figure as SVG, then release it."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_dir / f"{name}.svg")
    finally:
        plt.close(fig)


def generate_r1_train(output_dir: str | Path = OUTPUT_DIR) -> None:
    """Fig. 2(a): R1, current with and without the recurrent synapse.
    One external input (w=10 at t=0.3) starts the neuron from V0=b0=0; each output
    re-enters after d=0.15 through w_j.
    Uses the exact event-driven simulator.
    """
    plt = configure_style()
    wj, d, T, t_ext, w_ext = 12.0, 0.15, 6.5, 0.3, 10.0

    def run(wj):
        """Event list, plus the current I(t) sampled between events (I
        decays as e^{-2t} and jumps by w at inputs).
        """
        t_in, w_in = [t_ext], [w_ext]
        ev = None
        # re-entries depend on the outputs, so iterate: add the re-entry of every
        # output found so far, rerun until stable
        for _ in range(50):
            ev = simulate(t_in, list(range(len(t_in))), w_in, V0=0.0, b0=0.0, T_end=T)
            re = [o + d for o in ev["outputs"] if o + d < T] if wj else []
            new_t = sorted([t_ext] + re)
            if new_t == t_in:
                break
            t_in = new_t
            w_in = [w_ext if abs(t - t_ext) < 1e-12 else wj for t in t_in]
        ts = np.linspace(0, T, 4000)
        current = np.zeros_like(ts)
        for t, w in zip(t_in, w_in):
            current += w * np.exp(-2 * (ts - t)) * (ts >= t)
        return ev["outputs"], t_in, ts, current

    outs1, tin1, ts, I1 = run(wj)
    outs0, tin0, _, I0 = run(0.0)
    print(
        "with w_j=12: outputs",
        np.round(outs1, 3),
        " reset currents",
        np.round([np.interp(o, ts, I1) for o in outs1], 3),
    )
    print("with w_j=0 : outputs", np.round(outs0, 3))

    fig, ax = plt.subplots(figsize=(WIDTH * 0.92, 1.9))
    ax.axhline(PI**2, color=GREY, lw=0.8, ls="--")
    ax.text(T + 0.05, PI**2, r"$\pi^2$", va="center", fontsize=9, color="0.35")
    ax.plot(ts, I0, color=GREY, lw=1.4)
    ax.plot(ts, I1, color=BLACK, lw=1.2)
    for o in outs1:
        ax.plot([o, o], [17.2, 18.6], color=BLACK, lw=1.4)
    for o in outs0:
        ax.plot([o, o], [15.2, 16.6], color=GREY, lw=1.4)
    for t in tin1[1:]:
        ax.plot(t, 13.9, "v", color=VERM, ms=4)
    ax.plot(t_ext, 13.9, "v", color=BLUE, ms=4)
    ax.text(T + 0.05, 17.9, "outputs, $w_j=12$", va="center", fontsize=8)
    ax.text(T + 0.05, 15.9, "outputs, $w_j=0$", va="center", fontsize=8, color="0.4")
    ax.text(T + 0.05, 13.9, "inputs", va="center", fontsize=8)
    ax.text(6.0, 3.9, "$w_j=12$", fontsize=9, ha="center")
    ax.text(2.2, 0.9, "$w_j=0$", fontsize=9, color="0.4")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 19)
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("current $I(t)$")
    save(fig, "r1_train", output_dir)


def generate_qr_train(output_dir: str | Path = OUTPUT_DIR) -> None:
    """Fig. (App. G): along the outputs after one input, QR exponents go to 0 while the
    gradient magnitude grows.
    Case 1b: one input of weight w at t=0.1 from V0=b0=0, chosen so that c2/pi =
    5.5, 10.5, 20.5 (k = 6, 11, 21 outputs).
    QR (Benettin) on the product of the out->out event Jacobians, QR after every
    map, Q0 = identity.
    Magnitude: time row of the RTRL trace for V0 and for w, relative to the first
    output.
    """
    plt = configure_style()
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    ax[0].axhline(0, color=GREY, lw=0.8, zorder=0)
    for col, target in zip([BLUE, ORANGE, GREEN], [5.5, 10.5, 20.5]):
        w = (target * PI + PI / 2) ** 2
        ev = simulate([0.1], [0], [w], T_end=200.0)
        J_out = [J for J, k in zip(ev["J"], ev["kind"]) if k.endswith("out")][1:]
        logs = qr_accumulate(J_out)
        i = np.arange(1, len(logs) + 1)
        g = groups(ev, [0.1], 200.0)[0]
        c2, k = g["c2"], g["n_out"]
        idx = np.arange(1, k + 1)
        c = c2 - (idx - 1) * PI
        ax[0].plot(i + 1, logs[:, 0] / i, "-", color=col)
        ax[0].plot(i + 1, logs[:, 1] / i, "--", color=col)
        ax[0].text(
            k + 0.4, logs[-1, 1] / i[-1], f"$k={k}$", color=col, va="center", fontsize=8
        )
        MV = np.array(
            [M[0, 0] for M, kk in zip(ev["M"], ev["kind"]) if kk.endswith("out")]
        )
        Mw = np.array(
            [M[0, 1] for M, kk in zip(ev["M"], ev["kind"]) if kk.endswith("out")]
        )
        ax[1].semilogy(idx, np.abs(Mw / Mw[0]), "-", color=col)
        ax[1].semilogy(idx, np.abs(MV / MV[0]), "--", color=col)
        print(
            (
                "c2/pi=%.2f k=%d | QR exponents per output %s, predicted "
                "(0, %.4f) | last-output amplification V0 %.1f "
                "(c2/c_{k+1}=%.1f >= k), w %.1f (=(2k-1)c2/c_{k+1}=%.1f)"
            )
            % (
                c2 / PI,
                k,
                np.round(logs[-1] / i[-1], 4),
                np.log(c[-1] / c2) / (k - 1),
                MV[-1] / MV[0],
                c2 / c[-1],
                Mw[-1] / Mw[0],
                (2 * k - 1) * c2 / c[-1],
            )
        )
    ax[0].set_xlabel("output $i$")
    ax[0].set_ylabel("QR exponent per output")
    ax[0].text(14, 0.025, r"$\lambda_1=0$ (solid)", fontsize=8)
    ax[0].text(12.5, -0.42, r"$\lambda_2$ (dashed)", fontsize=8)
    ax[0].set_title("(a) exponents tend to 0")
    ax[0].set_ylim(-0.52, 0.08)
    ax[0].set_xlim(1.5, 24)
    ax[1].set_xlabel("output $i$")
    ax[1].set_ylabel(r"$|\partial t_i/\partial\theta\,/\,\partial t_1/\partial\theta|$")
    ax[1].text(16, 2.2e2, r"$\theta=w$ (solid)", fontsize=8)
    ax[1].text(14.5, 2.0, r"$\theta=V_0$ (dashed)", fontsize=8)
    ax[1].set_title("(b) magnitude grows")
    fig.tight_layout(w_pad=1.5)
    save(fig, "qr_train", output_dir)


def generate_capacity(output_dir: str | Path = OUTPUT_DIR) -> None:
    """Fig. (App. G): the number of learnable outputs is set by the current at the last
    input of each group and by the gap.
    Poisson inputs: n input times uniform on [0,T] (Poisson conditioned on n), one
    synapse of weight W=250, V0=b0=0, 300 draws per (n,T).
    (a) outputs in each gap against x = sqrt(B)(1-e^{-G})/pi, with the exact count
    of Prop. 4.
    (b) outputs in the trial against T, median and 10-90% band.
    (c) c2/pi at the last input of each group, T=4.
    """
    plt = configure_style()
    W, N_LIST, T_LIST, DRAWS, SEED_IN = 250.0, [1, 2, 4, 8], [1, 2, 4, 8, 16], 300, 11
    colors = [BLUE, ORANGE, GREEN, VERM]
    rng = np.random.default_rng(SEED_IN)
    c2_by_n = {n: [] for n in N_LIST}
    cap = {n: [] for n in N_LIST}
    gap = []
    for n in N_LIST:
        for T in T_LIST:
            tot = []
            for _ in range(DRAWS):
                t = np.sort(rng.uniform(0, T, n))
                ev = simulate(t, np.zeros(n, int), [W], T_end=T)
                tot.append(len(ev["outputs"]))
                for g in groups(ev, t, T):
                    gap.append((g["n_out"], g["c2"], g["B"], g["G"]))
                    if T == 4:
                        c2_by_n[n].append(g["c2"] / PI)
            cap[n].append(np.percentile(tot, [10, 50, 90]))
    fig, ax = plt.subplots(
        1, 3, figsize=(WIDTH, 2.1), gridspec_kw={"width_ratios": [1, 1.1, 1]}
    )
    gd = np.array(gap)
    n_out, c2, B, G = gd.T
    x = np.sqrt(B) * (1 - np.exp(-G)) / PI
    pred = np.maximum(0, np.ceil((c2 - np.sqrt(B) * np.exp(-G)) / PI))
    ax[0].plot(x, n_out, ".", ms=2, color=BLACK, alpha=0.25)
    xx = np.linspace(0, x.max(), 400)
    ax[0].plot(xx, np.ceil(xx), color=VERM, lw=0.9)
    ax[0].plot(xx, np.ceil(xx) - 1, color=VERM, lw=0.9, ls="--")
    ax[0].set_xlabel(r"$x=\sqrt{B}\,(1-e^{-G})/\pi$")
    ax[0].set_ylabel("outputs in the gap")
    ax[0].set_title("(a) exact gap count")
    ax[0].set_yticks([0, 2, 4, 6, 8, 10])
    print(
        (
            "(a) groups %d, mismatches with the exact count %d, "
            "outside {ceil(x)-1, ceil(x)} %d"
        )
        % (
            len(gd),
            int(np.sum(pred != n_out)),
            int(np.sum((n_out > np.ceil(x)) | (n_out < np.ceil(x) - 1))),
        )
    )
    for col, n in zip(colors, N_LIST):
        q = np.array(cap[n])
        ax[1].plot(T_LIST, q[:, 1], "o-", ms=3, color=col)
        ax[1].fill_between(T_LIST, q[:, 0], q[:, 2], color=col, alpha=0.15)
        ax[1].text(17.5, q[-1, 1], f"$n={n}$", color=col, va="center", fontsize=8)
        print("(b) n=%d median outputs over T=%s: %s" % (n, T_LIST, q[:, 1]))
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(T_LIST, [str(t) for t in T_LIST])
    ax[1].set_xlim(0.9, 30)
    ax[1].set_xlabel("trial length $T$")
    ax[1].set_ylabel("outputs in the trial")
    ax[1].set_title("(b) outputs vs. $T$")
    bins = np.linspace(4, 9, 41)
    for col, n in zip(colors, N_LIST):
        ax[2].hist(
            c2_by_n[n], bins=bins, histtype="step", color=col, density=True, lw=1.1
        )
        v = np.array(c2_by_n[n])
        print(
            "(c) n=%d c2/pi median %.2f 10-90%% [%.2f, %.2f]"
            % (n, np.median(v), *np.percentile(v, [10, 90]))
        )
    ax[2].set_xlabel(r"$c_2/\pi$ at the last input")
    ax[2].set_ylabel("density")
    ax[2].set_yticks([])
    ax[2].set_title("(c) spread of $c_2$")
    ax[2].text(4.75, 7.0, "$n=1$", color=BLUE, fontsize=8)
    ax[2].text(6.6, 0.9, "$n=8$", color=VERM, fontsize=8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "capacity", output_dir)


def generate_qr_seeds(output_dir: str | Path = OUTPUT_DIR) -> None:
    """Fig. (App. G): which sources of variation the QR exponents forget.
    Baseline trial: Poisson inputs at rate 1 on [0,60] over l=2 synapses, weights
    U(150,350), V0~U(-0.5,0.5), b0~U(0,3).
    Four seeds, varied one at a time with the other three at the baseline (as in
    Toolkit 2, Sec. 6):
    input times (seed 1 / 101..119), weights (2 / 201..219), initial state (3 /
    301..319), initial QR basis (4 / 401..419); 20 runs each.
    QR after every event, no events discarded, exponents per event (natural log).
    (a) sample SD (ddof=1) of lambda_1 over the 20 runs against the number of
    events.  (b) mean +/- SD of lambda_1 at 200 events.
    """
    plt = configure_style()
    RATE, T, L, KMAX = 1.0, 60.0, 2, 170

    def run(seed_in, seed_w, seed_ic, seed_ons):
        ri = np.random.default_rng(seed_in)
        n = ri.poisson(RATE * T)
        t = np.sort(ri.uniform(0, T, n))
        syn = ri.integers(0, L, n)
        w = np.random.default_rng(seed_w).uniform(150, 350, L)
        ic = np.random.default_rng(seed_ic)
        V0, b0 = ic.uniform(-0.5, 0.5), ic.uniform(0, 3)
        ev = simulate(t, syn, w, V0=V0, b0=b0, T_end=T)
        h = qr_accumulate(ev["J"], initial_basis(seed_ons))
        K = np.arange(1, len(h) + 1)
        return h / K[:, None]

    base = (1, 2, 3, 4)
    groups = [
        ("input times", 0, [1] + list(range(101, 120)), BLUE),
        ("weights", 1, [2] + list(range(201, 220)), ORANGE),
        ("initial state", 2, [3] + list(range(301, 320)), GREEN),
        ("QR basis", 3, [4] + list(range(401, 420)), VERM),
    ]
    fig, ax = plt.subplots(
        1, 2, figsize=(WIDTH, 2.1), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    K = np.arange(1, KMAX + 1)
    ends = []
    print("source          events   lambda_1 mean +/- SD      lambda_2 mean +/- SD")
    for j, (name, pos, seeds, col) in enumerate(groups):
        runs = []
        for sd in seeds:
            sv = list(base)
            sv[pos] = sd
            runs.append(run(*sv)[:KMAX])
        R = np.array(runs)  # (runs, KMAX, 2)
        sdev = R[:, :, 0].std(axis=0, ddof=1)
        ax[0].loglog(K[4:], sdev[4:], color=col)
        ends.append((sdev[-1], name, col))
        m, s = R[:, -1, 0].mean(), R[:, -1, 0].std(ddof=1)
        ax[1].errorbar(j, m, yerr=s, fmt="o", color=col, capsize=3)
        for c in [10, 50, KMAX]:
            print(
                "%-14s  %4d     %.4f +/- %.4f        %.4f +/- %.4f"
                % (
                    name,
                    c,
                    R[:, c - 1, 0].mean(),
                    R[:, c - 1, 0].std(ddof=1),
                    R[:, c - 1, 1].mean(),
                    R[:, c - 1, 1].std(ddof=1),
                )
            )
    ax[0].loglog(K[4:], 1.0 / K[4:], color=GREY, ls=":", lw=1)
    ends.sort()
    ys = np.geomspace(
        min(e[0] for e in ends) * 0.8, max(e[0] for e in ends) * 1.6, len(ends)
    )
    for y, (v, name, col) in zip(ys, ends):
        ax[0].text(KMAX * 1.1, y, name, color=col, va="center", fontsize=8)
    ax[0].set_xlim(5, KMAX * 4.5)
    ax[0].set_xlabel("events $K$")
    ax[0].set_ylabel(r"SD of $\lambda_1$ over 20 runs")
    ax[0].set_title("(a) initial state and basis are forgotten")
    ax[1].set_xticks(
        range(4), ["inputs", "weights", "initial\nstate", "QR\nbasis"], fontsize=8
    )
    ax[1].set_ylabel(r"$\lambda_1$ per event, $K=%d$" % KMAX)
    ax[1].set_title(r"(b) mean $\pm$ SD")
    fig.tight_layout(w_pad=1.0)
    save(fig, "qr_seeds", output_dir)


def generate_random_weights(output_dir: str | Path = OUTPUT_DIR) -> None:
    """Fig. (App. F): with random weights the amplification is usually modest, but it
    has a heavy tail at the capacity boundary.
    l=100 synapses, one input each at a time uniform on [0,1], weights w_i ~ N(0,
    g^2/sqrt(l)) (g=1 is variance 1/sqrt(l)),
    V0=b0=0, 1000 draws per gain g in {1,10,30,100,300}, trial T=40 so trains run
    out.
    (a) distribution of the number of outputs.
    (b) amplification at the last output, voltage direction c2/c_{k+1} (exact),
    against u = c2/pi-(k-1), the place of c2 in its band.
    (c) survival P(amplification > A), voltage and weight directions, all g>=10
    pooled; dashed: only outputs inside a trial of length 4.
    """
    plt = configure_style()
    L, GAINS, DRAWS, T, T_SHORT, SEED = 100, [1, 10, 30, 100, 300], 1000, 40.0, 4.0, 21
    rng = np.random.default_rng(SEED)
    kdist = {}
    U, AV, KK = [], [], []
    AW = []
    AVs, AWs = [], []
    for g in GAINS:
        ks = []
        for _ in range(DRAWS):
            t = np.sort(rng.uniform(0, 1, L))
            w = rng.normal(0, g / L**0.25, L)
            ev = simulate(t, np.arange(L), w, V0=0.0, b0=0.0, T_end=T)
            ks.append(len(ev["outputs"]))
            gl = groups(ev, t, T)[-1]
            k = gl["n_out"]
            if k >= 2 and g >= 10:
                e = gl["event"]
                Mo = [ev["M"][j][0, 1:] for j in range(e + 1, e + 1 + k)]
                to = [ev["time"][j] for j in range(e + 1, e + 1 + k)]
                c2 = gl["c2"]
                U.append(c2 / PI - (k - 1))
                KK.append(k)
                AV.append(c2 / (c2 - (k - 1) * PI))
                AW.append(np.linalg.norm(Mo[-1]) / np.linalg.norm(Mo[0]))
                ks_in = sum(
                    1 for x in to if x < T_SHORT
                )  # outputs inside the short trial
                if ks_in >= 2:
                    AVs.append(c2 / (c2 - (ks_in - 1) * PI))
                    AWs.append(np.linalg.norm(Mo[ks_in - 1]) / np.linalg.norm(Mo[0]))
        kdist[g] = np.bincount(ks, minlength=20)[:20] / DRAWS
        print(
            "g=%3d: P(k=0)=%.2f P(k=1)=%.2f P(k>=2)=%.2f  median k=%d max k=%d"
            % (
                g,
                kdist[g][0],
                kdist[g][1],
                1 - kdist[g][:2].sum(),
                np.median(ks),
                max(ks),
            )
        )
    U, AV, AW, KK = map(np.array, (U, AV, AW, KK))
    AVs, AWs = np.array(AVs), np.array(AWs)
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.1))
    med = [np.average(np.arange(20), weights=kdist[g]) for g in GAINS]
    p2 = [1 - kdist[g][:2].sum() for g in GAINS]
    ax[0].semilogx(GAINS, med, "o-", color=BLACK, ms=3)
    ax[0].set_ylabel("mean outputs $k$")
    a0 = ax[0].twinx()
    a0.spines["right"].set_visible(True)
    a0.semilogx(GAINS, p2, "s--", color=VERM, ms=3)
    a0.set_ylim(0, 1)
    a0.set_ylabel("$P(k\\geq2)$", color=VERM)
    a0.tick_params(axis="y", colors=VERM, labelsize=7)
    ax[0].set_xlabel("gain $g$")
    ax[0].set_title("(a) $g=1$: $k\\leq1$")
    ax[1].scatter(U, AV, s=2, color=GREY, alpha=0.5, lw=0)
    uu = np.linspace(0.004, 1, 200)
    for k, col in [(2, BLUE), (6, ORANGE), (12, GREEN)]:
        ax[1].plot(uu, (k - 1 + uu) / uu, color=col, lw=1)
        ax[1].text(1.03, k, f"$k={k}$", color=col, fontsize=8, va="center")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0, 1.25)
    ax[1].set_xlabel(r"$u=c_2/\pi-(k-1)$")
    ax[1].set_ylabel(r"$c_2/c_{k+1}$, last output")
    ax[1].set_title("(b) blow-up as $u\\to0$")
    for arr, col, ls, lab in [
        (AV, BLUE, "-", "voltage"),
        (AW, ORANGE, "-", "weights"),
        (AVs, BLUE, ":", None),
        (AWs, ORANGE, ":", None),
    ]:
        s = np.sort(arr)
        p = 1 - np.arange(len(s)) / len(s)
        ax[2].loglog(s, p, ls, color=col, lw=1.2)
    A = np.geomspace(5, 3e3, 50)
    ax[2].loglog(A, 4 / A, color=GREY, lw=0.8, ls="--")
    ax[2].text(1e3, 2.2e-2, r"$\propto 1/A$", color="0.4", fontsize=8)
    ax[2].text(150, 0.5, "voltage", color=BLUE, fontsize=8)
    ax[2].text(150, 0.18, "weights", color=ORANGE, fontsize=8)
    ax[2].text(150, 0.065, "dotted: $T=4$", color="0.3", fontsize=8)
    ax[2].set_xlabel("amplification $A$")
    ax[2].set_ylabel(r"$P(\mathrm{amplification}>A)$")
    ax[2].set_title("(c) heavy tail")
    print(
        (
            "trains with k>=2 (g>=10): %d; voltage amplification "
            "median %.1f, p90 %.1f, p99 %.1f, max %.1f"
        )
        % (len(AV), np.median(AV), *np.percentile(AV, [90, 99]), AV.max())
    )
    print(
        (
            "                          weights amplification median "
            "%.1f, p90 %.1f, p99 %.1f, max %.1f"
        )
        % (np.median(AW), *np.percentile(AW, [90, 99]), AW.max())
    )
    print(
        "inside a trial of length %g: trains %d; voltage max %.1f, weights max %.1f"
        % (T_SHORT, len(AVs), AVs.max(), AWs.max())
    )
    fig.tight_layout(w_pad=0.6)
    save(fig, "random_weights", output_dir)


def generate_pseudospike(
    output_dir: str | Path = OUTPUT_DIR, *, recompute: bool = False
) -> None:
    """Fig. 5: a pseudospike added after the trial carries a gradient that decays like
    e^{-Delta T}.
    Numerical probe, outside (A4). Uses Klos & Memmesheimer's spikegd
    (https://github.com/chklos/spikegd) for the simulation, the
    pseudospike time and the gradients (JAX autodiff, float64):
    one input at t=0.1 from V0=b0=0 with c2/pi = K-1/2, so the train has K outputs
    and then stops;
    the trial ends Delta T after the last output;
    the pseudospike is spikegd's QIFNeuron.t_pseudo: t_ps = T + theta(I_ps) -
    phi(V(T), I_ps), I_ps = softplus_alpha(I(T)),
    i.e. after T the neuron runs on a constant suprathreshold current set by its
    current at T (their first type of pseudodynamics).
    Plotted: |d t_ps/d w| / |d t_K/d w| against Delta T.  (a) K=5, softplus scale
    alpha in {1, 10, 100} (alpha=10 is spikegd's default).
    (b) alpha=10, K in {2, 5, 10}.  Dashed: e^{-Delta T}.
    Recomputing the cache requires JAX and spikegd.
    """
    plt = configure_style()

    def ratio(K, dT, alpha):
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from spikegd.qif import QIFNeuron

        neuron = QIFNeuron(tau=1.0, eps=1e-10, alpha=alpha)
        w0 = ((K - 0.5) * np.pi + np.pi / 2) ** 2
        tin = 0.1

        def sim(w, T):
            return neuron.event(
                jnp.zeros((2, 1)),
                jnp.zeros((1, 1)),
                jnp.array([[w]]),
                (jnp.array([tin]), jnp.array([0])),
                {"T": T, "K": 4 * K + 10},
            )

        out = sim(w0, 200.0)
        ts, isin, js, xs = out
        tK = float(np.array(ts)[(~np.array(isin)) & (np.array(js) == 0)][-1])
        T = tK + dT

        def last_out(w):
            ts, isin, js, xs = sim(w, T)
            m = (~isin) & (js == 0)
            return jnp.max(jnp.where(m, ts, -jnp.inf))

        def t_ps(w):
            ts, isin, js, xs = sim(w, T)
            xe = xs[jnp.sum(js >= 0)]
            return neuron.t_pseudo(xe[:, 0], 0.0, 1, {"T": T})

        return abs(float(jax.grad(t_ps)(w0)) / float(jax.grad(last_out)(w0)))

    DT = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6])
    CACHE = (
        Path(output_dir) / "data" / "pseudospike_data.npz"
    )  # delete it (or pass --recompute) to rerun the simulations
    if CACHE.exists() and not recompute:
        with np.load(CACHE) as cached:
            data = dict(cached)
    else:
        data = {
            f"a{al:g}": np.array([ratio(5, d, al) for d in DT])
            for al in [1.0, 10.0, 100.0]
        }
        data.update(
            {f"K{K}": np.array([ratio(K, d, 10.0) for d in DT]) for K in [2, 5, 10]}
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **data)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.1), sharey=True)
    for col, al in zip([BLUE, ORANGE, GREEN], [1.0, 10.0, 100.0]):
        r = data[f"a{al:g}"]
        ax[0].semilogy(DT, r, "o-", ms=3, color=col)
        ax[0].text(6.2, r[-1], rf"$\alpha={al:g}$", color=col, va="center", fontsize=8)
        print(
            (
                "K=5 alpha=%5.1f: ratio at dT=0.02 %.3f, at dT=6 %.2e, "
                "decay per unit time over [4,6] %.3f"
            )
            % (al, r[0], r[-1], (r[-1] / r[-3]) ** 0.5)
        )
    for col, K in zip([BLUE, ORANGE, GREEN], [2, 5, 10]):
        r = data[f"K{K}"]
        ax[1].semilogy(DT, r, "o-", ms=3, color=col)
        print("alpha=10 K=%d: ratio at dT=0.02 %.3f, at dT=6 %.2e" % (K, r[0], r[-1]))
    for a in ax:
        a.semilogy(DT, np.exp(-DT), color=GREY, ls="--", lw=0.9)
        a.set_xlim(0, 7.6)
        a.set_xlabel(r"silence before the trial end, $\Delta T$")
    ax[1].text(0.3, 2e-4, r"dashed: $e^{-\Delta T}$", color="0.4", fontsize=8)
    ax[1].text(3.0, 0.15, "$K=2,5,10$ lie\non top of each other", fontsize=8)
    ax[0].set_ylabel(
        r"$|\partial t_{\mathrm{ps}}/\partial w|\,/\,|\partial t_K/\partial w|$"
    )
    ax[0].set_title(r"(a) $K=5$, three drive scales $\alpha$")
    ax[1].set_title(r"(b) $\alpha=10$: no dependence on $K$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "pseudospike", output_dir)


# %%
############################
### Command line
############################


FIGURES = {
    "r1_train": generate_r1_train,
    "qr_train": generate_qr_train,
    "capacity": generate_capacity,
    "qr_seeds": generate_qr_seeds,
    "random_weights": generate_random_weights,
    "pseudospike": generate_pseudospike,
}


def generate_figures(
    names: Sequence[str] | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    recompute: bool = False,
) -> list[Path]:
    """Generate the selected SVGs and update their sections in the numerical report."""
    import matplotlib.pyplot as plt

    selected = tuple(FIGURES) if names is None else tuple(names)
    unknown = [name for name in selected if name not in FIGURES]
    if unknown:
        raise ValueError(f"Unknown figure {unknown[0]!r}. Choose from {', '.join(FIGURES)}.")

    output_dir = Path(output_dir)
    paths = []
    with plt.rc_context():
        for name in selected:
            kwargs = {"recompute": recompute} if name == "pseudospike" else {}
            with capture_results(f"FIGURE_{name.upper()}", f"Figure: {name}", output_dir / "RUN_RESULTS.md"):
                FIGURES[name](output_dir, **kwargs)
            paths.append(output_dir / f"{name}.svg")
    return paths


def run_checks(output_dir: str | Path = OUTPUT_DIR, *, full: bool = False) -> list[dict]:
    """Update the report with simulator and bound checks, and optionally the full gradient suite."""
    report = Path(output_dir) / "RUN_RESULTS.md"
    with capture_results("NUMERICS", "Simulator checks", report):
        check_simulator()
    with capture_results("WEIGHT_BOUNDS", "Weight-bound checks", report):
        check_weight_bounds()

    records = run_gradient_checks(report) if full else []
    if records:
        passed = sum(row["ok"] for row in records)
        errata = sum(row["expected_fail"] and not row["ok"] for row in records)
        unexpected = len(records) - passed - errata
        print(f"Gradient checks: {passed} passed, {errata} known errata, {unexpected} unexpected failures.")
    print(f"Validation report: {report}")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    figures = commands.add_parser("figures", help="Generate the paper's SVG figures.")
    figures.add_argument(
        "names", nargs="*", metavar="FIGURE",
        help=f"Figures to render (default: all). Choices: {', '.join(FIGURES)}.",
    )
    figures.add_argument("--recompute", action="store_true", help="Recompute the pseudospike data cache.")
    figures.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output folder.")
    checks = commands.add_parser("check", help="Run the simulator and capacity-bound checks.")
    checks.add_argument("--full", action="store_true", help="Include the full gradient validation.")
    checks.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output folder.")
    args = parser.parse_args(argv)

    if args.command == "figures":
        import matplotlib

        matplotlib.use("Agg")
        try:
            paths = generate_figures(args.names or None, args.output, recompute=args.recompute)
        except ValueError as error:
            if any(name not in FIGURES for name in args.names):
                figures.error(str(error))
            raise
        for path in paths:
            print(path)
        return 0

    records = run_checks(args.output, full=args.full)
    return int(any(not row["ok"] and not row["expected_fail"] for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
