# Local learning rules

This folder contains the QIF RFLO benchmark and the spike-time learning-rule ablations for *Exact Gradients of QIF Neurons and Online Learning Rules*. The RFLO benchmark compares exact and readout-only controls with local rules that use random or teacher-fitted feedback. The published spike-time workflow fits corrections to recorded exact gradients, then compares exact, local, and fitted rules under input jitter and perturbed initial weights.

```text
experiments/local_rules/
  rflo_bench.ipynb  Short RFLO runs, gradient diagnostics, and a paper-task preview
  local_rules.py   Notebook helpers, job runner, paper plots, tables, and CLI
  bench_results/   Saved RFLO runs, job lists, and logs
  README.md        Setup and reproduction instructions
spikegd/local_rules/
  config.py        Reproducibility settings, constants, and output paths
  gradients.py     Eligibility traces, spike counts, and Lyapunov probes
  rflo.py          RFLO system, rule definitions, teacher, and ablation grid
  paper.py         Spike-time tasks, training, fitting, rates, and reproduction checks
```

## Notebook

From the repository root, install the package and open the notebook in the same Python environment.

```sh
python -m pip install -e .
python -m jupyter notebook experiments/local_rules/rflo_bench.ipynb
```

Use **Run All** in a fresh Python kernel. Setup enables CPU, float64, and the original JAX Threefry behavior before importing JAX. The default demonstration runs ten trials per rule and includes one teacher correction. The notebook plots the readout at initialization, loss curves, and the teacher's feedback values. It also lists the full grid and has an optional four-trial preview of the published spike-time task.

## RFLO benchmark

Run commands from the repository root.

```sh
python -m experiments.local_rules.local_rules demo --trials 10 --teacher
python -m experiments.local_rules.local_rules rflo --describe --filter mmm.cyc_per_k=1.
python -m experiments.local_rules.local_rules rflo --job 0
python -m experiments.local_rules.local_rules rflo-batch -j 8 -f mmm.cyc_per_k=1.
python -m experiments.local_rules.local_rules rflo --summarize
```

The full RFLO sweep contains thousands of jobs. Use `rflo --describe` to inspect settings and job indices. Add `--out DIR` to write results elsewhere. The parallel runner preserves existing seed archives, writes `jobs.txt` and a log per launched job, then summarizes the results in `summary.csv`. See `BENCH` in [rflo.py](../../spikegd/local_rules/rflo.py) for the grid and rule settings.

## Published spike-time ablations

`paper all` records the baseline exact run, fits corrections, trains the rules and ablations, then generates panels and tables. Recording and batch training skip finished outputs. A full run uses 10,000 trials per condition.

```sh
python -m experiments.local_rules.local_rules paper all -j 8
```

Individual steps are available through the same runner.

```sh
python -m experiments.local_rules.local_rules paper record
python -m experiments.local_rules.local_rules paper fit
python -m experiments.local_rules.local_rules paper train -j 8
python -m experiments.local_rules.local_rules paper train-case init fitted noCx --no-lyap
python -m experiments.local_rules.local_rules paper rates perturb exact
python -m experiments.local_rules.local_rules paper rates perturb fitted base
python -m experiments.local_rules.local_rules paper figures
python -m experiments.local_rules.local_rules paper check
```

`paper check` retrains the first three runs per case for 400 trials and reports the largest loss difference against the saved runs. It requires the exact, local, and base-fit outputs. Small differences can accumulate across machines.

For fits on multiple inputs, run `paper record --inputs 20` once, then `paper fit --inputs N` for each of 5, 10, and 20. Use `paper train base_in5 base_in10 base_in20` to train those fits. For the rate panels for both cases, run `paper rates CASE exact` and `paper rates CASE fitted base` for `perturb` and `init` before regenerating the figures.

### Task and fitting parameters

The paper constants are in [config.py](../../spikegd/local_rules/config.py).

`task` The task of Klos & Memmesheimer: N = 10 neurons, T = 10, two target neurons with 7 and 2 target spikes, one
excitatory and one inhibitory Poisson input per neuron (rate 1, weights ±5), seed 7, zero start.

`perturb` Input jitter: every input time shifted by sigma xi, xi ~ N(0, 1), drawn once per run; sigma = 0.25, 0.5,
0.75, 1, 1.5, 2; three seeds each; level 0 is the task input (one run).

`init` Perturbed initial weights: w0 = (w_sigma / sqrt N) xi on the task input; same levels and seeds.

`training` 10,000 trials, AdaBelief, learning rate 1e-2 exp(-n / 2000); cosines and Lyapunov exponents every 200
trials; final loss is the mean of the last 250 trials.

`fit` The exact run on the task input, every 10th trial recorded, alternate blocks of 50 recordings held out;
ridges r_A = 1, r_C = 1e-3, r_C,off = 1; the first 8 outputs of each hidden neuron.

`fit configurations` base; A = 0 (`noA`); C = 0 (`noCx`); C_off = 0 (`noCoff`); C = C_off = 0 (`noC`); each ridge
swept over 1e-5 to 2 with the other two at base (`rA_*`, `rC_*`, `rCoff_*`).

`fit inputs` The base fit made on n = 1, 5, 10, 20 inputs; input 0 is the task input, input k > 0 a new Poisson
sample with key 3000 + k and the same targets.

`rates` Spikes of each neuron at every checkpoint of every run, binned in bins of 0.5 over the trial and averaged
over checkpoints.

## Paper figures and tables

| paper | files in `output/figs` | made by |
|---|---|---|
| Fig. 2 (a) loss curves at level 0, (b) final loss vs sigma, (c) final loss vs w_sigma | `panels/loss_perturb_level0`, `final_perturb`, `final_init` | `paper panels` |
| Table 2, final loss x 1e3 per rule and level | `table_final_loss.tex` | `paper tables` |
| Fig. 13, cosine with the exact gradient, target / hidden neuron | `panels/cos_<case>_target`, `cos_<case>_offtarget` | `paper panels` |
| Figs. 14 and 15 (a) largest Lyapunov exponent, (b) switches, (c-e) ridge sweeps | `panels/lyap_<case>`, `switches_<case>`, `ridge_<case>_<rA, rC, rCoff>` | `paper panels` |
| Fig. 16 (a, b) final loss for fits on n inputs, (c, d) rate correlation with the teacher's run | `panels/final_<case>_inputs`, `rate_match_<case>` | `paper panels` |
| Table 3, final loss for fits on n inputs | `table_fit_inputs.tex` | `paper tables` |

Fig. 14 is input jitter (`<case>` = `perturb`) and Fig. 15 perturbed initial weights (`init`). Every panel is saved
as svg and pdf; the paper includes the pdf files from `figs/abl/`, so copy the panels and both tables there after
new runs. `paper panels` also writes `loss_init_level0` (the same run as `loss_perturb_level0`) and
`rate_<case>_<rule>`, which are not in the paper.

## Data

`output/data` holds `fit_data.npz` (the recording on the task input), `fit_inputs/` (inputs 1-19), `fits/` (one
json per fit configuration), `ablation/` (one `train_v2_<case>_<rule>.npz` per rule and case) and `rates/`. The saved
runs keep `level`, `seed`, `probe_trials` and, with `<r>` = exact, local or fitted, `<r>_loss` per trial and
`<r>_theta`, `<r>_cos`, `<r>_gnorm_exact`, `<r>_lyap`, `<r>_lyap_block`, `<r>_n_events` every 200 trials. The
per-trial update norms and spike counts (`<r>_gnorm`, `<r>_counts`) were removed to keep the data small; `paper train-case`
still writes them.
