# QIF test bench

This experiment uses the same notebook and Python companion layout as [RNN](../rnn/). The notebook runs the QIF network, task, loss, RFLO rule, and trainer described in the refactored test bench specification.

```text
experiments/
  test_bench/
    bench.ipynb    Configure, train, plot results, and optionally export a network
    bench.py       Experiment presets, plots, and command-line entry point
    README.md      Setup and run instructions
spikegd/
  test_bench/      Reusable network, tasks, rules, and training classes
```

## Run the notebook

From the repository root, install the package and start Jupyter in the same Python environment.

```sh
python -m pip install -e .
python -m jupyter notebook experiments/test_bench/bench.ipynb
```

Use **Run All** in a fresh kernel. The default cyclic run uses CPU float64, takes 10 training trials, and needs no external data. The notebook shows the training loss and a sample rollout with the final readout, target, and network spike raster. Set `export_path` to save the trained network and verify the reload. Restart the kernel before changing JAX startup settings.

## Run from the command line

Run these commands from the repository root.

```sh
python -m experiments.test_bench.bench
python -m experiments.test_bench.bench --task copy --trials 10 --output experiments/test_bench/output
python -m experiments.test_bench.bench --help
```

`--output` writes `loss.png`, `rollout.png`, and `network.npz` to that folder. The trainer records metric checkpoints in memory. Increase `--event-cap` or set `TrainerConfig(K=...)` if the automatic cap is too small for a later sample.

| Task | Inputs |
| --- | --- |
| `cyclic` | Seeded Poisson background drive, with a continuous target derived from the network's readout kernel |
| `copy` | Seeded symbol sequences, with one class label per step |
| `add` | Seeded value and marker sequences, with a numeric target |
| `mnist` | Local MNIST IDX files, with one class label per image |
| `timit` | Local TIMIT corpus, with one phone label per frame |

For MNIST and TIMIT, set `task_options = {"data_root": "/path/to/dataset"}` in the notebook or pass `--data-root /path/to/dataset` on the command line. See the [package README](../../spikegd/test_bench/README.md) for the expected files, optional dependencies, and library usage.

The runner uses the online RFLO variants `murray`, `e`, `exact_decay`, and `exact_own`. The original and reorganized `exact_own` rule both produced nonfinite updates on a synthetic TIMIT sequence. The default `murray` rule passed that check. Exact RTRL, teacher fitting, and the pseudospike readout are listed as future work in the specification.
