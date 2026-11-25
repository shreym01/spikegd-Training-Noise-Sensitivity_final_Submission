# QIF capacity bound

This experiment reproduces six figures for *Exact Gradients of QIF Neurons and Online Learning Rules* and checks the event-driven simulator, gradients, and capacity bounds.

```text
experiments/capacity_bound/
  capacity_bound.ipynb  Run and display selected figures and validate the simulator
  capacity_bound.py     Figure functions and command-line runner
  README.md             Setup and run instructions
  output/               Generated SVGs, report, and pseudospike cache
spikegd/capacity_bound/ Reusable numerical code and validation
```

## Run the notebook

From the repository root, install the package and open the notebook in the same Python environment.

```sh
python -m pip install -e .
python -m jupyter notebook experiments/capacity_bound/capacity_bound.ipynb
```

Use **Run All** to display the capacity and recurrent-current figures and save the simulator and weight-bound checks. Set `FIGURE_NAMES = tuple(bench.FIGURES)` to generate all six figures. For full gradient validation, install SciPy and mpmath with `python -m pip install scipy mpmath` and set `RUN_FULL_VALIDATION = True`.

## Run from the command line

Use these commands from the repository root.

```sh
python -m experiments.capacity_bound.capacity_bound figures
python -m experiments.capacity_bound.capacity_bound figures capacity r1_train
python -m experiments.capacity_bound.capacity_bound figures pseudospike --recompute
python -m experiments.capacity_bound.capacity_bound check
python -m experiments.capacity_bound.capacity_bound check --full
```

Add `--output DIR` to either command to change the destination. Report sections are updated independently, so rerunning one figure or check preserves the other sections. The pseudospike figure reuses `output/data/pseudospike_data.npz` when available. Use `--recompute` to refresh it with the float64 JAX simulation.

| Figure | Published settings |
| --- | --- |
| `r1_train` | Recurrent current with and without delayed re-entry. Recurrent weight 12, delay 0.15, external weight 10 at time 0.3, duration 6.5. |
| `pseudospike` | Gradient decay for spike counts 2, 5, and 10 and softplus scales 1, 10, and 100. Time after the last spike ranges from 0.02 to 6, with float64, tau 1, and eps 1e-10. |
| `qr_train` | QR exponents and gradient growth after one input at time 0.1. The three values of c2/pi are 5.5, 10.5, and 20.5, with duration 200. |
| `capacity` | Output counts for weight 250, input counts 1, 2, 4, and 8, and trial lengths 1, 2, 4, 8, and 16. There are 300 draws per pair, with seed 11. |
| `qr_seeds` | QR variability at input rate 1, duration 60, and two synapses. Input, weight, initial-state, and QR-basis seeds vary over 20 runs each. |
| `random_weights` | Gaussian weights across 100 synapses, with gains 1, 10, 30, 100, and 300. There are 1,000 draws per gain, with duration 40 and seed 21. |

SVG figures and the numerical report `RUN_RESULTS.md` are saved in `output/` by default. The [package README](../../spikegd/capacity_bound/README.md) describes the numerical modules and direct Python usage.
