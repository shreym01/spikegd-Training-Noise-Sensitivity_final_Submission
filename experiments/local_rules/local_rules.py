"""Notebook companion and command-line workflows for the local learning rules."""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from spikegd.local_rules import config as C
from spikegd.local_rules import paper, rflo

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

MODULE = "experiments.local_rules.local_rules"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = C.HERE / "bench_results"


def run_demo(system, initial, rules, n_trials):
    """Compare rules on one system and seed, returning the records by name."""
    runs = {}
    for rule in rules:
        start = time.monotonic()
        runs[rule.name] = rflo.train(system, initial, rule, n_trials=n_trials)
        rflo.explain(system, runs[rule.name], rule, runs)
        print(f"  ({time.monotonic() - start:.0f}s)\n")
    return runs


def plot_demo_loss(runs):
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    for name, run in runs.items():
        ax.plot(np.arange(1, len(run["loss"]) + 1), run["loss"], label=name)
    ax.set(xlabel="Trial", ylabel="Loss")
    ax.legend(frameon=False)
    return fig


def plot_feedback(system, runs, rules):
    teacher_rules = [rule for rule in rules if rule.teacher]
    fig, axes = plt.subplots(
        1, len(teacher_rules), figsize=(5 * len(teacher_rules), 3), squeeze=False, layout="constrained",
    )
    for ax, rule in zip(axes[0], teacher_rules):
        history = runs[rule.name]["B_hist"]
        for neuron in system.TARGETS:
            ax.plot(np.arange(len(history)), history[:, neuron], "o-", label=f"B {system.NAMES[neuron]}")
        ax.set(xlabel="Teacher correction", title=rule.name)
        ax.legend(frameon=False, fontsize=7)
    return fig


def inspect_gradients(system, initial, events):
    """Check spike-time credit and compare each local trace with autodiff at initialization."""
    times, _ = system.target_times(events)
    _, (exact, _) = system.exact((system.w0, initial.W_out0), initial.Y_star)
    ell = system.ell_at(times, initial.W_out0, times, initial.Y_star)
    dloss_dt = jnp.where(times < system.PAD, initial.W_out0[:, None] * ell, 0.0)
    dt_dw = jax.jacobian(lambda w: system.target_times(system.simulate(w))[0])(system.w0)
    err = float(jnp.abs(jnp.einsum("ok,okij->ij", dloss_dt, dt_dw) - exact).max())
    scale = float(jnp.abs(exact).max())
    status = "ok" if err < 1e-6 * max(1.0, scale) else "mismatch"
    print(f"Max error between chain rule and autodiff {err:.2e} (max exact gradient {scale:.2e}, {status})")

    event_list = system.event_list(events)
    node = np.asarray(event_list["node"])
    outputs = np.asarray(event_list["valid"] & ~event_list["is_input"])
    for jac, form in rflo.JACOBIANS.items():
        trace, _ = jax.jit(lambda ev, alpha: form["fn"](system, ev, alpha))(event_list, initial.alpha0[jac])
        for row, neuron in enumerate(system.TARGETS):
            indices = np.flatnonzero(outputs & (node == neuron))[:system.NMAX]
            cosine = np.array([rflo._cos(trace[k], dt_dw[row, n, neuron]) for n, k in enumerate(indices)])
            cosine = cosine[np.isfinite(cosine)]
            if len(cosine):
                print(f"{jac:12s} {system.NAMES[neuron]} trace cosines, "
                      f"median {np.median(cosine):+.3f}, min {cosine.min():+.3f} ({len(cosine)} outputs)")

    print("\nCosine between update and exact gradient, one value per synapse row")
    print(" ".join(f"{name:>7s}" for name in system.NAMES))
    for rule in rflo.BENCH["rules"]:
        if rule.kind == "local" and not rule.teacher:
            _, (gradient, _) = rflo.rule_fn(system, rule)(
                (system.w0, initial.W_out0), initial.Y_star, initial.alpha0[rule.jacobian], initial.B0,
            )
            print(" ".join(f"{c:+7.2f}" for c in rflo._row_cos(gradient, exact)), rule.name)


# ---- Paper figure panels ----


RULES ={"exact": ("exact gradient", "#0b0b0b"), "local": ("local", "#2a78d6"),
         "fitted_base": (r"fitted ($A + C + C_{\mathrm{off}}$)", "#eb6834")}
LEVEL_NAME = {"perturb": r"input jitter $\sigma$", "init": r"initial-weight scale $w_\sigma$"}
SWITCHES = [("base", r"$A, C, C_{\mathrm{off}}$"), ("noA", r"$A=0$"), ("noCx", r"$C=0$"),
            ("noCoff", r"$C_{\mathrm{off}}=0$"), ("noC", r"$C=C_{\mathrm{off}}=0$")]
RIDGES = {"rA": (r"$r_A$", "r_A"), "rC": (r"$r_C$", "r_C"), "rCoff": (r"$r_{C,\mathrm{off}}$", "r_C_off")}
PANEL_STYLE = {"figure.figsize": (3.2, 2.4), "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200}


# ---------------- data ----------------

def load(case, name):
    """Record of one rule (exact, local) or fit configuration (fitted_<name>), or None if not run."""
    path = C.RUNS / f"train_v2_{case}_{name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    r = name if name in ("exact", "local") else "fitted"
    return {k[len(r) + 1:]: d[k] for k in d.files if k.startswith(r + "_")} | {
        "level": d["level"], "probe": d["probe_trials"]}


def final(run):
    return run["loss"][:, -C.LAST:].mean(axis=1)


def fit_name(n_inputs):
    """Run name of the base fit made on n_inputs inputs."""
    return "fitted_base" if n_inputs == 1 else f"fitted_base_in{n_inputs}"


def ratio_to_local(case, name):
    """Final loss / local per run (same input, seed and start); geometric mean and 25-75 % over level > 0."""
    run, loc = load(case, f"fitted_{name}"), load(case, "local")
    if run is None:
        return np.nan, np.nan, np.nan
    m = loc["level"] > 0
    r = final(run)[m] / final(loc)[m]
    r = r[np.isfinite(r) & (r > 0)]
    return np.exp(np.mean(np.log(r))), np.percentile(r, 25), np.percentile(r, 75)


def smooth(y, w=50):
    return np.convolve(y, np.ones(w) / w, mode="valid")


def save(fig, name):
    C.PANELS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(C.PANELS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------- main-text panels ----------------

def loss_level0(case):
    fig, ax = plt.subplots()
    for name, (label, color) in RULES.items():
        run = load(case, name)
        y = smooth(run["loss"][run["level"] == 0][0])
        ax.plot(np.arange(len(y)) + 50, y, color=color, lw=1.2, label=label)
    ax.set(xscale="log", yscale="log", xlabel="trial", ylabel="training loss")
    ax.legend(fontsize=7)
    save(fig, f"loss_{case}_level0")


def final_vs_level(case, rules=RULES, tag=""):
    """Final loss against the level, mean +- sd over seeds, for every rule in `rules` that has been run."""
    fig, ax = plt.subplots(figsize=(4.8, 2.4) if tag else None)
    present = [(name, style) for name, style in rules.items() if load(case, name) is not None]
    for k, (name, (label, color)) in enumerate(present):
        run = load(case, name)
        levels = sorted(set(run["level"]))
        f = final(run)
        mean = np.array([f[run["level"] == v].mean() for v in levels])
        sd = np.array([f[run["level"] == v].std(ddof=1) if (run["level"] == v).sum() > 1 else 0 for v in levels])
        x = np.arange(len(levels)) + (k - (len(present) - 1) / 2) * 0.3 / max(len(present) - 1, 1)
        ax.errorbar(x, mean, yerr=[np.minimum(sd, 0.999 * mean), sd], color=color, marker="o", ms=3, lw=1,
                    capsize=2, label=label)
    ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
    ax.set(yscale="log", xlabel=LEVEL_NAME[case], ylabel="final loss (mean $\\pm$ sd)")
    if tag:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, f"final_{case}{tag}")


def load_rates(case, name):
    path = C.DATA / "rates" / f"rates_{case}_{name}.npz"
    return dict(np.load(path)) if path.exists() else None


def rate_profile(case, name):
    """Population rate over the trial for every level (mean over seeds), the exact run at level 0 dashed."""
    d, ref = load_rates(case, name), load_rates(case, "exact")
    if d is None or ref is None:
        return
    t = 0.5 * (d["edges"][1:] + d["edges"][:-1])
    levels = sorted(set(d["level"]))
    cmap = plt.get_cmap("Blues" if name == "exact" else "Oranges")
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    for k, v in enumerate(levels):
        y = d["rate"][d["level"] == v].mean(axis=(0, 2))
        ax.plot(t, y, color=cmap(0.35 + 0.65 * k / max(len(levels) - 1, 1)), lw=1.1, label=f"{v:g}")
    ax.plot(t, ref["rate"][ref["level"] == 0][0].mean(axis=1), color="k", ls="--", lw=1.0, label="exact, 0")
    ax.set(xlabel="time in trial", ylabel="rate (mean over neurons)")
    ax.legend(fontsize=6, title=LEVEL_NAME[case], title_fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, f"rate_{case}_{name}")


def rate_match(case):
    """Per-neuron correlation of the rate within the trial with the exact run at level 0, mean +- sd over seeds."""
    ref = load_rates(case, "exact")
    if ref is None:
        return
    r0 = ref["rate"][ref["level"] == 0][0]
    fig, ax = plt.subplots()
    for k, name in enumerate(("exact", "fitted_base")):
        d = load_rates(case, name)
        if d is None:
            continue
        levels = sorted(set(d["level"]))
        corr = np.array([np.nanmean([np.corrcoef(R[:, i], r0[:, i])[0, 1] if R[:, i].std() > 0 else np.nan
                                     for i in range(r0.shape[1])]) for R in d["rate"]])
        mean = np.array([corr[d["level"] == v].mean() for v in levels])
        sd = np.array([corr[d["level"] == v].std() for v in levels])
        label, color = RULES[name]
        ax.errorbar(np.arange(len(levels)) + (k - 0.5) * 0.1, mean, yerr=sd, color=color, marker="o", ms=3, lw=1,
                    capsize=2, label=label)
    ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
    ax.set(ylim=(-0.05, 1.05), xlabel=LEVEL_NAME[case], ylabel="corr. with exact rate at level 0")
    ax.legend(fontsize=6)
    save(fig, f"rate_match_{case}")


def input_rules():
    """Exact, local and the base fit made on n inputs for every n in FIT_INPUTS_SWEEP (light to dark)."""
    shades = plt.get_cmap("Oranges")(np.linspace(0.45, 1.0, len(C.FIT_INPUTS_SWEEP)))
    out = {"exact": RULES["exact"], "local": RULES["local"]}
    for n, shade in zip(C.FIT_INPUTS_SWEEP, shades):
        out[fit_name(n)] = (f"fitted, {n} input" + ("s" if n > 1 else ""), matplotlib.colors.to_hex(shade))
    return out


def cosine(case, neuron, tag):
    fig, ax = plt.subplots()
    for name in ("local", "fitted_base"):
        run = load(case, name)
        y = run["cos"][run["level"] > 0][:, :, neuron]
        if not np.any(np.isfinite(y)):           # local: off-target rows are never updated
            continue
        label, color = RULES[name]
        t = np.maximum(run["probe"], 1)
        mean, sd = np.nanmean(y, 0), np.nanstd(y, 0)
        ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.2, lw=0)
        ax.plot(t, mean, color=color, lw=1.2, label=label)
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set(xscale="log", ylim=(-1.05, 1.05), xlabel="trial", ylabel="cosine with exact gradient")
    ax.legend(fontsize=7)
    save(fig, f"cos_{case}_{tag}")


# ---------------- appendix panels ----------------

def lyapunov(case):
    fig, ax = plt.subplots()
    for name, (label, color) in RULES.items():
        run = load(case, name)
        m, t = run["level"] > 0, np.maximum(run["probe"], 1)
        for key, ls in (("lyap", "-"), ("lyap_block", "--")):
            y = run[key][m][:, :, 0]
            mean, sd = np.nanmean(y, 0), np.nanstd(y, 0)
            ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.12, lw=0)
            ax.plot(t, mean, color=color, ls=ls, lw=1.1, label=label if ls == "-" else None)
    ax.plot([], [], color="0.4", ls="-", label="full network")
    ax.plot([], [], color="0.4", ls="--", label="block diagonal")
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set(xscale="log", xlabel="trial", ylabel=r"$\lambda_{\max}$ per event")
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, f"lyap_{case}")


def ridge(case, tag):
    label, key = RIDGES[tag]
    names = [p.stem.split(f"train_v2_{case}_fitted_")[1] for p in C.RUNS.glob(f"train_v2_{case}_fitted_{tag}_*.npz")]
    values = sorted([(float(re.fullmatch(rf"{tag}_(.+)", n).group(1)), n) for n in names])
    values = sorted(values + [(C.BASE_FIT[key], "base")])
    r = np.array([ratio_to_local(case, n) for _, n in values])
    x = [v for v, _ in values]
    fig, ax = plt.subplots()
    ax.fill_between(x, r[:, 1], r[:, 2], color="#eb6834", alpha=0.2, lw=0)
    ax.plot(x, r[:, 0], color="#eb6834", marker="o", ms=3, lw=1.1)
    ax.axhline(1, color="#2a78d6", ls="--", lw=0.9)
    ax.axvline(C.BASE_FIT[key], color="0.6", ls=":", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel=label, ylabel="final loss / local")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    save(fig, f"ridge_{case}_{tag}")


def switches(case):
    rows = [(label, ratio_to_local(case, name)) for name, label in SWITCHES]
    rows = [(label, r) for label, r in rows if np.isfinite(r[0])]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    x = np.arange(len(rows))
    g = np.array([r for _, r in rows])
    ax.bar(x, g[:, 0], color=["#2a78d6" if v < 1.2 else "#eb6834" for v in g[:, 0]], width=0.6)
    ax.errorbar(x, g[:, 0], yerr=[np.maximum(g[:, 0] - g[:, 1], 0), np.maximum(g[:, 2] - g[:, 0], 0)], fmt="none", color="k", lw=0.8, capsize=2)
    ax.axhline(1, color="#2a78d6", ls="--", lw=0.9)
    ax.set_xticks(x, [label for label, _ in rows], fontsize=7)
    ax.set(ylabel="final loss / local")
    save(fig, f"switches_{case}")




def render_panels():
    required = (paper.out_path(case, rule, fit)
                for case in C.CASES
                for rule, fit in (("exact", None), ("local", None), ("fitted", "base")))
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run the paper training workflow before generating panels. Missing runs:\n"
            + "\n".join(str(path) for path in missing)
        )
    with plt.rc_context(PANEL_STYLE):
        for case in C.CASES:
            loss_level0(case)
            final_vs_level(case)
            cosine(case, C.TARGET_NEURON, "target")
            cosine(case, C.OFFTARGET_NEURON, "offtarget")
            lyapunov(case)
            for tag in RIDGES:
                ridge(case, tag)
            switches(case)
            final_vs_level(case, input_rules(), "_inputs")
            for name in ("exact", "fitted_base"):
                rate_profile(case, name)
            rate_match(case)
        print("panels ->", C.PANELS)


# ---- Paper tables ----


SCALE = 1e-3
CASE_NAME = {"perturb": r"input jitter $\sigma$", "init": r"initial-weight scale $w_\sigma$"}
RULE_NAME = {"exact": "exact", "local": "local", "fitted_base": "fitted"}


WORST, BEST = "red", "green!50!black"


def cell(f, color=None):
    f = f / SCALE
    text = f"{f.mean():.2f}" if len(f) == 1 else f"{f.mean():.2f} $\\pm$ {f.std(ddof=1):.2f}"
    return text if color is None else f"\\textcolor{{{color}}}{{\\textbf{{{text}}}}}"


def rows(case, names=RULE_NAME):
    """One row per rule that has been run; in each column the largest mean in red, the smallest in green."""
    levels = (0.0,) + C.LEVELS
    names = {name: label for name, label in names.items() if load(case, name) is not None}
    finals = {}
    for name in names:
        run = load(case, name)
        finals[name] = [final(run)[run["level"] == v] for v in levels]
    means = np.array([[f.mean() for f in finals[name]] for name in names])
    out = []
    for r, name in enumerate(names):
        cells = []
        for c in range(len(levels)):
            color = WORST if r == means[:, c].argmax() else BEST if r == means[:, c].argmin() else None
            cells.append(cell(finals[name][c], color))
        out.append(names[name] + " & " + " & ".join(cells) + r" \\")
    return out


def input_names():
    """Exact, local and the fits on n = FIT_INPUTS_SWEEP inputs, as table rows."""
    return {"exact": "exact", "local": "local"} | {fit_name(n): f"fitted, $n={n}$" for n in C.FIT_INPUTS_SWEEP}


def table(names=RULE_NAME):
    levels = " & ".join(["0"] + [f"{v:g}" for v in C.LEVELS])
    lines = [r"\begin{tabular}{@{}l" + "c" * (len(C.LEVELS) + 1) + "@{}}", r"\toprule"]
    for case in C.CASES:
        lines += [CASE_NAME[case] + " & " + levels + r" \\", r"\midrule"] + rows(case, names) + [r"\midrule"]
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"




def render_tables():
    path = C.FIGS / "table_final_loss.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"% final loss x {1 / SCALE:g}, mean of the last {C.LAST} trials, mean $\\pm$ sd over seeds\n" + table())
    print(path.read_text())
    path = C.FIGS / "table_fit_inputs.tex"
    path.write_text(f"% final loss x {1 / SCALE:g} for fits made on n inputs, mean $\\pm$ sd over seeds\n"
                    + table(input_names()))
    print(path.read_text())


# ---- Batch jobs and command line ----


def rflo_jobs(filter_text=""):
    """Enumerate matching jobs by their original index, longest first."""
    matches = []
    for index, (config, _, task, rule, seed) in enumerate(rflo.jobs()):
        key = f"{config}/{rule.name}/seed{seed}"
        if filter_text in key:
            matches.append((index, key, rflo.TASKS[task]["n_trials"], rule.teacher))
    return sorted(matches, key=lambda job: -job[2] * (2 if job[3] else 1))


def format_job(job):
    index, key, trials, _ = job
    return f"{index}\t{key}\t{trials} trials"


def run_workers(commands, parallel):
    """Execute independent jobs in fresh Python processes with one log per job."""
    if not commands:
        return

    def run(command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode

    print(f"Running {len(commands)} jobs, up to {parallel} at a time.", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run, command, log): log for command, log in commands}
        try:
            for future in as_completed(futures):
                log = futures[future]
                if future.result():
                    failed.append(log)
                    print(f"Failed {log}", flush=True)
                else:
                    print(f"Finished {log.stem}", flush=True)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed. Logs are listed above.")


def run_rflo_batch(out, filter_text, parallel):
    jobs = rflo_jobs(filter_text)
    if not jobs:
        print(f"No RFLO jobs match {filter_text!r}.")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.txt").write_text("".join(format_job(job) + "\n" for job in jobs))
    commands = [
        ([sys.executable, "-m", MODULE, "rflo", "--job", str(index), "--out", str(out.resolve())],
         out / "logs" / f"job_{index}.log")
        for index, key, _, _ in jobs
        if not (out / f"{key}.npz").is_file()
    ]
    print(f"{len(jobs) - len(commands)} of {len(jobs)} jobs already saved.")
    run_workers(commands, parallel)
    rflo.summarize(out)


def train_paper_batch(only, parallel):
    fits = sorted(path.stem.removeprefix("fit_") for path in C.FITS.glob("fit_*.json"))
    if only:
        fits = [name for name in fits if name in only]
    rules = [("exact", None), ("local", None)] + [("fitted", name) for name in fits]
    commands = []
    for case in C.CASES:
        for rule, fit in rules:
            result = paper.out_path(case, rule, fit)
            if result.is_file():
                continue
            command = [sys.executable, "-m", MODULE, "paper", "train-case", case, rule]
            if fit:
                command.append(fit)
            if (fit or rule) not in ("exact", "local", "base"):
                command.append("--no-lyap")
            commands.append((command, C.LOGS / f"{result.stem}.log"))
    run_workers(commands, parallel)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_paper_commands(command):
    steps = command.add_subparsers(dest="step", required=True)
    all_steps = steps.add_parser("all", help="Record, fit, train, and generate figures and tables.")
    all_steps.add_argument("-j", "--parallel", type=positive_int, default=8)
    record = steps.add_parser("record", help="Record exact-gradient runs.")
    record.add_argument("--inputs", type=positive_int, default=C.N_FIT_INPUTS)
    record.add_argument("--input", type=int, help="Record just this input index.")
    fit = steps.add_parser("fit", help="Fit corrections from saved recordings.")
    fit.add_argument("--inputs", type=positive_int, default=C.N_FIT_INPUTS)
    fit.add_argument("--force", action="store_true", help="Recompute existing fit files.")
    train = steps.add_parser("train", help="Run exact, local, and fitted cases in parallel.")
    train.add_argument("fits", nargs="*", help="Only use these fitted configurations.")
    train.add_argument("-j", "--parallel", type=positive_int, default=8)
    for step, help_text in (("train-case", "Train one rule and case."), ("rates", "Compute firing rates.")):
        sub = steps.add_parser(step, help=help_text)
        sub.add_argument("case", choices=C.CASES)
        sub.add_argument("rule", choices=("exact", "local", "fitted"))
        sub.add_argument("fit", nargs="?", help="Fit configuration for the fitted rule.")
        if step == "train-case":
            sub.add_argument("--no-lyap", action="store_true", help="Skip Lyapunov probes.")
    steps.add_parser("figures", help="Generate panels and tables from saved training runs.")
    steps.add_parser("panels", help="Generate only the figure panels.")
    steps.add_parser("tables", help="Generate only the LaTeX tables.")
    steps.add_parser("check", help="Check reproduced losses against saved training runs.")


def run_paper_command(parser, args):
    if args.step == "all":
        paper.record(range(C.N_FIT_INPUTS))
        paper.fit_all()
        train_paper_batch(None, args.parallel)
        render_panels()
        render_tables()
    elif args.step == "record":
        if args.input is not None and args.input < 0:
            parser.error("--input must be nonnegative")
        paper.record(range(args.inputs) if args.input is None else [args.input])
    elif args.step == "fit":
        paper.fit_all(args.inputs, force=args.force)
    elif args.step == "train":
        train_paper_batch(args.fits, args.parallel)
    elif args.step in ("train-case", "rates"):
        if (args.rule == "fitted") != bool(args.fit):
            parser.error("provide a fit name when using the fitted rule")
        if args.step == "train-case":
            paper.train_case(args.case, args.rule, args.fit, with_lyapunov=not args.no_lyap)
        else:
            paper.rates(args.case, args.rule, args.fit)
    elif args.step in ("figures", "panels", "tables"):
        if args.step in ("figures", "panels"):
            render_panels()
        if args.step in ("figures", "tables"):
            render_tables()
    elif args.step == "check":
        for case in C.CASES:
            paper.check(case, "exact")
            paper.check(case, "local")
            paper.check(case, "fitted", "base")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Run the short RFLO comparison.")
    demo.add_argument("--trials", type=positive_int, default=10)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--teacher", action="store_true", help="Include one teacher correction.")

    bench = commands.add_parser("rflo", help="List jobs, run one job, or summarize RFLO results.")
    action = bench.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="Print matching job indices.")
    action.add_argument("--describe", action="store_true", help="Print matching jobs with settings.")
    action.add_argument("--job", type=int, help="Run a job by its index.")
    action.add_argument("--summarize", action="store_true", help="Write summary.csv in the results folder.")
    bench.add_argument("--filter", default="", help="Match part of config/rule/seed.")
    bench.add_argument("--out", type=Path, default=RESULTS)

    batch = commands.add_parser("rflo-batch", help="Run matching RFLO jobs in parallel.")
    batch.add_argument("-j", "--parallel", type=positive_int, default=max(1, (os.cpu_count() or 2) - 1))
    batch.add_argument("-f", "--filter", default="", help="Match part of config/rule/seed.")
    batch.add_argument("-o", "--out", type=Path, default=RESULTS)

    add_paper_commands(commands.add_parser("paper", help="Run the published spike-time ablations."))
    args = parser.parse_args()
    if args.command == "demo":
        system = rflo.build()
        initial = rflo.make_init(system, args.seed)
        rules = [rflo.EXACT, rflo.READOUT_ONLY, rflo.rflo("murray")]
        if args.teacher:
            rules.append(rflo.rflo("murray", "teacher", 1, M=1, R=1))
        run_demo(system, initial, rules, args.trials)
    elif args.command == "paper":
        try:
            run_paper_command(parser, args)
        except FileNotFoundError as error:
            parser.exit(1, f"{error}\n")
    elif args.command == "rflo-batch":
        run_rflo_batch(args.out, args.filter, args.parallel)
    elif args.list or args.describe:
        for job in rflo_jobs(args.filter):
            print(format_job(job) if args.describe else job[0])
    elif args.job is not None:
        if not 0 <= args.job < len(rflo.jobs()):
            parser.error("--job is outside the range of RFLO job indices")
        rflo.run_job(args.job, args.out)
    else:
        rflo.summarize(args.out)


if __name__ == "__main__":
    main()
