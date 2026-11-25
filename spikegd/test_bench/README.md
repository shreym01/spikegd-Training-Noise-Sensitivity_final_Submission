# QIF test bench components

Install the package with `python -m pip install -e .` from the repository root. Run the [experiment notebook or CLI](../../experiments/test_bench/README.md) to use the preconfigured CPU float64 bench.

| File | Contents |
| --- | --- |
| `network.py` | `NetConfig` and its settings, `QIFNet` using SpikeGD's event loop, the alpha-kernel head, and NPZ export and loading |
| `tasks.py` | `Sample`, `NetRequirements`, task interfaces, spike encodings, and the cyclic, copy, adding, MNIST, and TIMIT tasks |
| `rules.py` | Closed-form QIF event maps and Jacobians, rule templates, and RFLO presets |
| `training.py` | Four loss functions and `LOSSES`, `Trainer`, `TrainerConfig`, `Checkpoint`, `Record`, and metric and figure interfaces |
| `checks.py` | Argument validation shared by the benchmark components |

The trainer binds custom tasks or validates standard tasks against the network, simulates each sample, and evaluates the loss and its derivative with respect to the readout. The rule receives the simulated events and that derivative. The trainer applies the optimizer update and projects parameters to enforce connectivity and Dale's law. It returns the final parameters and metrics in a `Record`.

## Use the library

Set the JAX environment before Python starts if you want the experiment's precision and device settings, for example `JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python`.

```python
from spikegd.test_bench.network import InputChannels, NetConfig, build_net
from spikegd.test_bench.rules import rflo
from spikegd.test_bench.tasks import CyclicTask
from spikegd.test_bench.training import Trainer, TrainerConfig

net = build_net(NetConfig(n_input_neurons=0, input_channels=InputChannels(0)))
task = CyclicTask(dt=0.1)
rule = rflo("murray", signal="W_out_targets")
record = Trainer(net, task, rule, TrainerConfig(n_trials=10)).train()
print(record.trials, record.metrics["loss"])
```

`net.export(path, record.params)` saves the trained parameters and network structure. Use `load_net(path)` to get the network and parameters for simulation. For a single trial, the same task interface supplies the spikes, read points, duration, and target.

```python
from spikegd.test_bench.training import LOSSES

sample = task.sample(0, "train")
y, events = net.forward(record.params, sample.spikes_in, sample.T, K=256, read_at=sample.read_at)
loss = LOSSES[task.loss](y, sample.y_star, **task.loss_kwargs)
```

## Dataset setup

- **MNIST** uses `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte`, with optional `.gz` compression. Place them in the supplied `data_root` or its `MNIST/raw` subfolder. Set `download=True` in `MnistTask` to use the optional TorchVision downloader.
- **TIMIT** expects `TRAIN` and `TEST` directories containing the dialect and speaker subfolders with their `.PHN` and `.WAV` files. Install the feature dependencies with `python -m pip install soundfile python_speech_features`. Features are cached as `timit_mfcc39.npz` under `data_root` by default. Set `cache_file` to change the destination.

The specification's roadmap still covers exact and masked RTRL, teacher phases, cross-neuron credit extensions, pseudospike readout, gradient alignment, Lyapunov diagnostics, and network topology figures. The implemented head is the alpha-kernel readout. The Klos loss is available for caller-supplied spike times.
