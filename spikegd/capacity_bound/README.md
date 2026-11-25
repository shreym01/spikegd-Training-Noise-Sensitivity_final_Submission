# QIF capacity-bound components

The [experiment README](../../experiments/capacity_bound/README.md) covers the notebook, figure generation, and validation commands.

| File | Contents |
| --- | --- |
| `simulation.py` | NumPy closed-form flow, complex-step event Jacobians, event simulation with eligibility traces, QR accumulation, and a values-only delayed-recurrence reference |
| `paper_forms.py` | Analytic voltage and spike-time formulas, event derivatives, capacity bounds, and RTRL for single, recurrent, and multi-neuron systems |
| `jax_reference.py` | Float64 wrappers around SpikeGD's QIF neuron for independent comparisons with JAX autodiff |
| `validation.py` | Simulator and weight-bound checks, the complete gradient validation, and Markdown report updates |

## Simulate one trial

```python
from spikegd.capacity_bound.simulation import groups, simulate

events = simulate([0.1], [0], [250.0], T_end=4.0)
print(events["outputs"])
print(groups(events, [0.1], 4.0)[0]["n_out"])
```

The NumPy simulator and the basic numerical checks can run without JAX. The full gradient validation also uses SciPy, mpmath, and float64 JAX.

```python
from spikegd.capacity_bound.validation import run_gradient_checks

records = run_gradient_checks("RUN_RESULTS.md")
print(sum(row["ok"] for row in records), "/", len(records))
```

Pass no report path to return the records without writing a file. Each record reports the reference implementation, tolerance, and whether a mismatch is a known erratum.
