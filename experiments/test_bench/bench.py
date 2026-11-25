"""Configure, train, and inspect a QIF benchmark from the notebook or command line."""

import argparse
import os
from pathlib import Path

# Set the experiment defaults before importing JAX. Shell overrides remain available.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np  # noqa: E402

from spikegd.test_bench.network import InputChannels, NetConfig, build_net  # noqa: E402
from spikegd.test_bench.rules import MAPS, SIGNALS, rflo  # noqa: E402
from spikegd.test_bench.tasks import AddTask, CopyTask, CyclicTask, MnistTask, TimitTask  # noqa: E402
from spikegd.test_bench.training import LOSSES, Figure, Trainer, TrainerConfig  # noqa: E402


TASKS = {
    "cyclic": (CyclicTask, dict(dt=0.1)),
    "copy": (CopyTask, dict(n_symbols=3, sequence_length=3, delay=3, n_train=16, n_test=4)),
    "add": (AddTask, dict(sequence_length=4, n_train=16, n_test=4)),
    "mnist": (MnistTask, dict(n_train=16, n_test=4)),
    "timit": (TimitTask, dict(max_sentences=4)),
}


def build_task(name, **options):
    """Build a small task for the example, with overrides for its constructor."""
    task_type, defaults = TASKS[name]
    return task_type(**(defaults | options))


def build_task_network(task):
    """Build a small recurrent network with the channel counts the task needs."""
    if isinstance(task, CyclicTask):
        config = NetConfig(
            neurons_per_layer=[4], n_input_neurons=0, n_readout_neurons=2,
            input_channels=InputChannels(0),
        )
    else:
        config = NetConfig(
            neurons_per_layer=[6], n_input_neurons=2, n_readout_neurons=2,
            input_channels=InputChannels(task.n_input_channels, mode="shared"),
            n_output_channels=task.n_outputs,
        )
    return build_net(config)


def evaluate(trainer, record, index=0, split=None):
    """Run a sample with the final weights and return its readout, events, and loss."""
    split = trainer.config.split if split is None else split
    sample = trainer.task.sample(index, split)
    y, events = trainer.net.forward(
        record.params, sample.spikes_in, sample.T, trainer.K, read_at=sample.read_at,
    )
    loss = LOSSES[trainer.task.loss](y, sample.y_star, **trainer.task.loss_kwargs)
    return sample, y, events, float(loss)


class LossFigure(Figure):
    name = "loss"
    needs = ("loss",)

    def draw(self, record, ax=None):
        import matplotlib.pyplot as plt

        self.check(record)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.plot(np.asarray(record.trials) + 1, record.metrics["loss"], marker=".")
        ax.set(xlabel="Training trial", ylabel="Loss", title=record.task)
        return ax.figure


def plot_rollout(task, sample, y, events):
    """Plot the final readout against the target, and the network spike raster."""
    import matplotlib.pyplot as plt

    fig, (ax, raster) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5), layout="constrained",
    )
    read_at = np.asarray(sample.read_at)
    time = read_at if read_at.ndim == 1 else read_at.mean(axis=1)
    if task.loss == "softmax":
        ax.plot(time, sample.y_star, "o-", label="Target")
        ax.plot(time, np.argmax(y, axis=0), "x--", label="Prediction")
        ax.set_ylabel("Class index")
    else:
        for channel, (target, prediction) in enumerate(zip(sample.y_star, y), start=1):
            line, = ax.plot(time, target, ".-", label=f"Target {channel}")
            ax.plot(time, prediction, "x--", color=line.get_color(), label=f"Readout {channel}")
        ax.set_ylabel("Readout")
    ax.legend()

    times, is_input, neuron, _ = map(np.asarray, events)
    spikes = ~is_input & (neuron >= 0)
    raster.scatter(times[spikes], neuron[spikes], marker="|")
    raster.set(xlim=(0, sample.T), xlabel="Time", ylabel="Firing neuron")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="cyclic")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rule", choices=MAPS, default="murray")
    parser.add_argument("--signal", choices=SIGNALS, default="W_out_targets")
    parser.add_argument("--event-cap", type=int, help="Maximum number of events per trial (default: automatic).")
    parser.add_argument("--data-root", help="Local MNIST or TIMIT dataset folder.")
    parser.add_argument("--output", type=Path, help="Optional folder for plots and the trained network.")
    args = parser.parse_args()
    if args.task in ("mnist", "timit") and not args.data_root:
        parser.error("--data-root is required for MNIST and TIMIT")
    if args.data_root and args.task not in ("mnist", "timit"):
        parser.error("--data-root is only used by MNIST and TIMIT")
    options = {"data_root": args.data_root} if args.data_root else {}
    task = build_task(args.task, **options)
    net = build_task_network(task)
    config = TrainerConfig(n_trials=args.trials, K="auto" if args.event_cap is None else args.event_cap)
    trainer = Trainer(net, task, rflo(args.rule, signal=args.signal), config)
    record = trainer.train()
    sample, y, events, loss = evaluate(trainer, record)
    print(f"{record.task} with {record.rule} | {args.trials} trials | final sample loss {loss:.6g}")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        LossFigure().draw(record).savefig(args.output / "loss.png", dpi=150)
        plot_rollout(task, sample, y, events).savefig(args.output / "rollout.png", dpi=150)
        net.export(args.output / "network.npz", record.params)


if __name__ == "__main__":
    main()
