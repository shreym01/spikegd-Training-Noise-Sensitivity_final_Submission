# Local learning-rule components

Install the package from the repository root with `python -m pip install -e .`. See the [experiment README](../../experiments/local_rules/README.md) for the notebook, batch commands, and published outputs.

| File | Contents |
| --- | --- |
| `config.py` | CPU, float64, and Threefry defaults, published task parameters, and output paths |
| `gradients.py` | QIF eligibility traces, frozen-surround and fitted gradients, spike counts, and Lyapunov probes |
| `rflo.py` | QIF simulation and kernel readout, `LocalRule`, `Init`, teacher phases, ablation grid, and result archives |
| `paper.py` | Published spike-time task, perturbations, exact-gradient recording, ridge fitting, training, and rate and reproduction checks |

The paper task and the RNN experiment share their input, target, and output-spike helpers in [spikegd/utils/rnn.py](../utils/rnn.py).

## Run one RFLO configuration

Start Python or restart the notebook kernel before importing the local-rule package so it can apply its JAX startup settings.

```python
from spikegd.local_rules import rflo

system = rflo.build()
initial = rflo.make_init(system, seed=0)
rule = rflo.rflo("murray")
record = rflo.train(system, initial, rule, n_trials=10)
print(record["loss"])
```

Use `rflo.rflo("murray", "teacher", 1, M=1, R=1)` for a short comparison with one teacher correction and one perturbed copy. Use `rflo.jobs()` to enumerate the full grid and `rflo.load_results(path)` to read saved runs.

## Run a short spike-time comparison

```python
from spikegd.local_rules import paper

task, neuron = paper.make_task(), paper.make_neuron()
step = paper.make_trainer(
    task, neuron, paper.local(task, neuron),
    with_lyapunov=False, n_trials=4, probe_every=2,
)
result = step(task["spikes_in"], task["w0"], task["x0"])
print(result["loss"])
```

Use a trial count divisible by the probe interval to retain all requested trials. The full reproduction commands use the configured 10,000 trials and 200-trial probe interval.
