"""Task interfaces, spike encodings, and the five QIF benchmarks.

Custom tasks bind to NetRequirements. Standard tasks validate the network against a fixed dataset.
Every task produces a Sample and names a loss from training.LOSSES.
"""

import gzip
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .checks import check_bool, check_choice, check_int, check_number, check_type


# =====================================================================================================================
# A sample, and what a task can need from the network
# =====================================================================================================================

@dataclass
class Sample:
    """One trial.

    spikes_in   (times, channels) of the task's input spikes, sorted by time (both empty if the task has no input)
    y_star      the target, shaped against the head's y (n_out, n_read) as the task's loss expects
    read_at     where the head reads y: times (n_read,) or windows (n_read, 2) of (start, end)
    T           trial length
    """
    spikes_in: tuple
    y_star: object
    read_at: object
    T: float


@dataclass
class NetRequirements:
    """What a task takes from the network.

    kernel            the read-out kernel of the network's output head, kappa(s) (s = t - spike time)
    w_out_shape       (n_out, k): output channels by read-out neurons
    n_neurons         N
    n_layers          number of layers (the cyclic task needs 1)
    n_input_channels  the network's task channels C
    """
    kernel: object | None
    w_out_shape: tuple
    n_neurons: int
    n_layers: int = 1
    n_input_channels: int | None = None

    @classmethod
    def from_net(cls, net):
        return cls(kernel=net.kernel, w_out_shape=(net.config.n_output_channels, len(net.readout_neurons)),
                   n_neurons=net.n_neurons, n_layers=len(net.layer_sizes), n_input_channels=net.n_input_channels)


# =====================================================================================================================
# Base classes
# =====================================================================================================================

class Task:
    """Spec fields are set at construction (no network needed)."""

    loss = None              # name of the loss in training.LOSSES that fits the task
    splits = ("train",)

    @property
    def loss_kwargs(self):
        """Extra keyword arguments of the loss (e.g. dt and T for "continuous", class_axis for "softmax")."""
        return {}

    def sample(self, index=0, split="train"):
        raise NotImplementedError

    def n_samples(self, split="train"):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError


class CustomAnalysisTask(Task):
    """A task whose target depends on the network. bind(requirements) builds everything that needs the network;
    sample() can only be used after it."""

    requirements = None

    def bind(self, requirements):
        raise NotImplementedError

    def _check_bound(self):
        if self.requirements is None:
            raise RuntimeError(f"{type(self).__name__}: call bind(NetRequirements.from_net(net)) first")


class StandardAnalysisTask(Task):
    """A standard benchmark with a fixed dataset.

    A child class sets n_input_channels, n_outputs and splits, fills self._data = {split: list of items} (an item is
    whatever the child keeps per sample: a ready Sample, or raw data that _to_sample turns into one), and overrides
    _to_sample(item, split, index) if its items are not Samples already.
    """

    splits = ("train", "test")

    def check(self, requirements):
        """Raises if the network does not fit the task: input channels and output channels must match."""
        check_type("requirements", requirements, NetRequirements)
        if requirements.n_input_channels != self.n_input_channels:
            raise ValueError(f"{type(self).__name__} has {self.n_input_channels} input channels; the network has "
                             f"n_input_channels = {requirements.n_input_channels}")
        if requirements.w_out_shape[0] != self.n_outputs:
            raise ValueError(f"{type(self).__name__} needs n_output_channels = {self.n_outputs}; the network has "
                             f"{requirements.w_out_shape[0]}")

    def sample(self, index=0, split="train"):
        check_choice("split", split, self.splits)
        items = self._data[split]
        check_int("index", index, low=0, high=len(items) - 1)
        return self._to_sample(items[index], split, index)

    def n_samples(self, split="train"):
        check_choice("split", split, self.splits)
        return len(self._data[split])

    # ---- for the child classes ----

    def _to_sample(self, item, split, index):
        return item

    def _check_dataset_fields(self):
        check_number("step_length", self.step_length, low=0.0, low_open=True)
        check_int("n_train", self.n_train, low=1)
        check_int("n_test", self.n_test, low=0)
        check_int("data_seed", self.data_seed, low=0)

    def _build_dataset(self):
        """For tasks that draw their samples: n_train and n_test calls of _draw_sample(rng) -> Sample."""
        rng = np.random.default_rng(self.data_seed)
        self._data = dict(train=[self._draw_sample(rng) for _ in range(self.n_train)],
                          test=[self._draw_sample(rng) for _ in range(self.n_test)])

    def _step_windows(self, steps):
        """(len(steps), 2): start and end time of each step in steps."""
        steps = np.asarray(steps)
        return np.stack([steps * self.step_length, (steps + 1) * self.step_length], 1)


def make_sample(times, channels, y_star, read_at, T):
    """A Sample with the spikes sorted by time and everything as jax arrays."""
    times, channels = np.asarray(times, float).ravel(), np.asarray(channels, int).ravel()
    order = np.argsort(times, kind="stable")
    return Sample(spikes_in=(jnp.asarray(times[order]), jnp.asarray(channels[order])), y_star=jnp.asarray(y_star),
                  read_at=jnp.asarray(read_at, float), T=float(T))


def no_spikes():
    return jnp.zeros(0), jnp.zeros(0, int)


# =====================================================================================================================
# Spike encodings used by the standard tasks
# =====================================================================================================================

def event_spike(step, step_length):
    """One spike at the start of the step (a symbol, a marker, a delimiter)."""
    return np.array([step * step_length])


def value_spikes(value, step, step_length, encoding, latency_span=0.9, max_rate=20.0, rng=None):
    """Spikes coding a value in [0, 1] inside one step.

    latency   one spike at step_start + (1 - value) * latency_span * step_length: earlier for a larger value
    rate      Poisson spikes at rate value * max_rate, uniform over the step
    """
    start = step * step_length
    if encoding == "latency":
        return np.array([start + (1.0 - value) * latency_span * step_length])
    count = rng.poisson(value * max_rate * step_length)
    return np.sort(rng.uniform(start, start + step_length, count))


def frame_spikes(values, step_length, encoding, latency_span=0.9, max_rate=20.0, rng=None):
    """Spikes for a sequence of frames: values (n_frames, n_channels) in [0, 1], frame f in step f, channel c of a
    frame coded with value_spikes. Returns (times, channels)."""
    values = np.asarray(values, float)
    n_frames, n_channels = values.shape
    starts = np.arange(n_frames)[:, None] * step_length
    channel_grid = np.broadcast_to(np.arange(n_channels)[None, :], values.shape)
    if encoding == "latency":
        times = starts + (1.0 - values) * latency_span * step_length
        return times.ravel(), channel_grid.ravel()
    counts = rng.poisson(values * max_rate * step_length).ravel()
    frame_of_spike = np.repeat(np.repeat(np.arange(n_frames), n_channels), counts)
    times = frame_of_spike * step_length + rng.uniform(0.0, step_length, counts.sum())
    return times, np.repeat(channel_grid.ravel(), counts)


def check_encoding(task):
    check_choice("encoding", task.encoding, ("latency", "rate"))
    check_number("latency_span", task.latency_span, low=0.0, low_open=True, high=1.0)
    check_number("max_rate", task.max_rate, low=0.0, low_open=True)


# ---- Cyclic task ----

@dataclass
class CyclicTask(CustomAnalysisTask):
    """The k read-out neurons fire in turn, n_cycles = cycles_per_readout * k times, at jittered times in [0, T].
    Only for flat (one-layer) networks.

    Spec (set before the network; defaults = the current bench, base task cycle2 with k = 2):
      cycles_per_readout   n_cycles / k (1.0: n_cycles = k; n_slots = round(k * n_cycles) target spikes in all)
      jitter               the gaps between target spikes are U(1 - jitter, 1 + jitter), rescaled to fit [0, T]
      target_seed          seed of the target times
      T                    trial length
      w_star_max           W* ~ integers in [1, w_star_max); "2k" = 2k
      w_star_seed          seed of W*
      dt                   spacing of the time grid y and y* live on

    From the network (bind):
      kernel               y*(t) = sum_o W*_o sum_q kernel(t - t*_oq), the kernel of the network's output head, so
                           W_out = W* and spikes at t* give zero loss (y is read with read_at = grid)
      w_out_shape (n_out, k)   number of read-out neurons k, and W* of the same shape as W_out

    After bind:
      t_star               list of k arrays: target spike times of read-out neuron o (slot i goes to neuron i mod k)
      w_star               (n_out, k)
      grid                 (n_grid,) midpoints of [0, T] in steps of dt
      y_star               (n_out, n_grid)
      loss                 "continuous", with loss_kwargs dt and T
    """
    cycles_per_readout: float = 1.0
    jitter: float = 0.5
    target_seed: int = 0
    T: float = 10.0
    w_star_max: int | str = "2k"
    w_star_seed: int = 0
    dt: float = 0.01

    loss = "continuous"

    def __post_init__(self):
        check_number("cycles_per_readout", self.cycles_per_readout, low=0.0, low_open=True)
        check_number("jitter", self.jitter, low=0.0)
        if self.jitter >= 1.0:
            raise ValueError(f"jitter = {self.jitter} must be < 1 (gaps are U(1 - jitter, 1 + jitter))")
        check_number("T", self.T, low=0.0, low_open=True)
        check_number("dt", self.dt, low=0.0, low_open=True)
        if self.dt >= self.T:
            raise ValueError(f"dt = {self.dt} must be smaller than T = {self.T}")
        for name in ("target_seed", "w_star_seed"):
            check_int(name, getattr(self, name), low=0)
        if self.w_star_max != "2k":
            check_int("w_star_max", self.w_star_max, low=2)

    # ---- binding to a network ----

    def bind(self, requirements):
        check_type("requirements", requirements, NetRequirements)
        if requirements.n_layers != 1:
            raise ValueError(f"CyclicTask is only for flat networks, got {requirements.n_layers} layers")
        if not callable(requirements.kernel):
            raise TypeError("requirements.kernel must be a function kappa(s)")
        if len(requirements.w_out_shape) != 2 or min(requirements.w_out_shape) < 1:
            raise ValueError(f"requirements.w_out_shape must be (n_out, k) with n_out, k >= 1, "
                             f"got {requirements.w_out_shape}")
        check_int("requirements.n_neurons", requirements.n_neurons, low=1)
        n_out, k = requirements.w_out_shape
        if k > requirements.n_neurons:
            raise ValueError(f"k = {k} read-out neurons but only {requirements.n_neurons} neurons")

        self.requirements = requirements
        self.k, self.n_out = k, n_out
        self.n_cycles = self.cycles_per_readout * k
        self.n_slots = max(1, int(round(k * self.n_cycles)))
        self.t_star = self._target_times()
        self.w_star = self._draw_w_star()
        self.grid = jnp.arange(0.0, self.T, self.dt) + self.dt / 2
        self.y_star = self.kernel_readout(self.w_star, self.t_star)
        return self

    # ---- target ----

    def _target_times(self):
        rng = np.random.default_rng(self.target_seed)
        gaps = rng.uniform(1 - self.jitter, 1 + self.jitter, self.n_slots + 1)
        times = np.cumsum(gaps)[:self.n_slots] / gaps.sum() * self.T
        return [times[readout::self.k] for readout in range(self.k)]

    def _draw_w_star(self):
        high = 2 * self.k if self.w_star_max == "2k" else self.w_star_max
        rng = np.random.default_rng(self.w_star_seed)
        return jnp.asarray(rng.integers(1, max(high, 2), size=(self.n_out, self.k)).astype(float))

    def kernel_readout(self, weights, spike_times):
        """sum_o w[:, o] sum_q kernel(t - t_oq) on the grid, for spike_times = list of k arrays. Used for y* only;
        the model's head computes y the same way from the network's read-out spikes."""
        kernel = self.requirements.kernel
        traces = []
        for times in spike_times:
            if len(times):
                traces.append(kernel(self.grid[:, None] - jnp.asarray(times)[None, :]).sum(1))
            else:
                traces.append(jnp.zeros_like(self.grid))
        return jnp.asarray(weights) @ jnp.stack(traces)

    # ---- samples and printout ----

    def sample(self, index=0, split="train"):
        """The one trial of the task (the same for every index): no input spikes, y_star on the grid."""
        self._check_bound()
        return Sample(spikes_in=no_spikes(), y_star=self.y_star, read_at=self.grid, T=self.T)

    def n_samples(self, split="train"):
        return 1

    @property
    def loss_kwargs(self):
        return dict(dt=self.dt, T=self.T)

    def describe(self):
        print(f"CyclicTask: cycles/k = {self.cycles_per_readout:g}, jitter {self.jitter}, T {self.T}, dt {self.dt}, "
              f"no task input (the network's background drives it), loss '{self.loss}'")
        if self.requirements is None:
            print("  not bound to a network yet")
            return
        print(f"  k = {self.k}, n_cycles = {self.n_cycles:g}, {self.n_slots} target spikes, "
              f"W* = {np.asarray(self.w_star).tolist()}")
        for readout, times in enumerate(self.t_star):
            print(f"  read-out {readout}: t* = {np.round(times, 3).tolist()}")


# ---- Copy task ----

@dataclass
class CopyTask(StandardAnalysisTask):
    """
    n_symbols        A, the number of symbols (8 in Arjovsky et al.)
    sequence_length  S, the number of symbols to copy (10 in Arjovsky et al.)
    delay            D, the steps between the last symbol and the first answer (the delimiter is the last of them)
    step_length      length of one step in the network's time units (tau = 1)
    n_train, n_test  samples in the fixed train and test sets
    data_seed        seed of the dataset
    """
    n_symbols: int = 8
    sequence_length: int = 10
    delay: int = 10
    step_length: float = 1.0
    n_train: int = 1000
    n_test: int = 200
    data_seed: int = 0

    loss = "softmax"

    def __post_init__(self):
        check_int("n_symbols", self.n_symbols, low=1)
        check_int("sequence_length", self.sequence_length, low=1)
        check_int("delay", self.delay, low=1)
        self._check_dataset_fields()

        self.blank_channel, self.delimiter_channel = 0, self.n_symbols + 1
        self.n_input_channels = self.n_symbols + 2
        self.n_outputs = self.n_symbols + 1
        self.n_steps = 2 * self.sequence_length + self.delay
        self.T = self.n_steps * self.step_length
        self.read_windows = self._step_windows(np.arange(self.n_steps))
        self.baseline_loss = self.sequence_length * np.log(self.n_symbols) / self.n_steps
        self._build_dataset()

    def _draw_sample(self, rng):
        symbols = rng.integers(1, self.n_symbols + 1, self.sequence_length)
        delimiter_step = self.sequence_length + self.delay - 1
        input_channels = np.full(self.n_steps, self.blank_channel)
        input_channels[:self.sequence_length] = symbols
        input_channels[delimiter_step] = self.delimiter_channel

        times, channels = [], []
        for step, channel in enumerate(input_channels):
            spike_times = event_spike(step, self.step_length)
            times.append(spike_times)
            channels.append(np.full(len(spike_times), channel))

        labels = np.zeros(self.n_steps, int)
        labels[self.n_steps - self.sequence_length:] = symbols
        return make_sample(np.concatenate(times), np.concatenate(channels), labels, self.read_windows, self.T)

    @property
    def loss_kwargs(self):
        return dict(class_axis=0)

    def describe(self):
        print(f"CopyTask: A = {self.n_symbols} symbols, S = {self.sequence_length}, D = {self.delay}, "
              f"{self.n_steps} steps of {self.step_length} (T = {self.T}), {self.n_input_channels} input channels, "
              f"{self.n_outputs} output classes, loss '{self.loss}' on every step "
              f"(random-guess baseline {self.baseline_loss:.3f}), {self.n_train} train / {self.n_test} test samples")


# ---- Adding task ----

VALUE_CHANNEL, MARKER_CHANNEL = 0, 1


@dataclass
class AddTask(StandardAnalysisTask):
    """Predict the sum of two marked values after a sequence of encoded inputs.

    sequence_length  L, the number of values (100 and more in the literature; short by default for small networks)
    encoding         "latency" or "rate" (how a value is sent in its step)
    latency_span     latency: fraction of the step the spike time ranges over
    max_rate         rate: spikes per unit time for value 1
    answer_steps     steps after the sequence in which the output is read
    step_length      length of one step in the network's time units (tau = 1)
    n_train, n_test  samples in the fixed train and test sets
    data_seed        seed of the dataset (values, marker positions and, for "rate", the Poisson spikes)
    """
    sequence_length: int = 20
    encoding: str = "latency"
    latency_span: float = 0.9
    max_rate: float = 20.0
    answer_steps: int = 1
    step_length: float = 1.0
    n_train: int = 1000
    n_test: int = 200
    data_seed: int = 0

    loss = "mse"

    def __post_init__(self):
        check_int("sequence_length", self.sequence_length, low=2)
        check_int("answer_steps", self.answer_steps, low=1)
        check_encoding(self)
        self._check_dataset_fields()

        self.n_input_channels = 2
        self.n_outputs = 1
        self.n_steps = self.sequence_length + self.answer_steps
        self.T = self.n_steps * self.step_length
        self.read_windows = self._step_windows(np.arange(self.sequence_length, self.n_steps))
        self.baseline_loss = 1.0 / 6.0
        self._build_dataset()

    def _draw_sample(self, rng):
        values = rng.uniform(0.0, 1.0, self.sequence_length)
        half = self.sequence_length // 2
        marked_steps = [rng.integers(0, half), rng.integers(half, self.sequence_length)]

        times, channels = [], []
        for step, value in enumerate(values):
            spike_times = value_spikes(value, step, self.step_length, self.encoding, self.latency_span,
                                       self.max_rate, rng)
            times.append(spike_times)
            channels.append(np.full(len(spike_times), VALUE_CHANNEL))
        for step in marked_steps:
            spike_times = event_spike(step, self.step_length)
            times.append(spike_times)
            channels.append(np.full(len(spike_times), MARKER_CHANNEL))

        y_star = np.full((1, self.answer_steps), values[marked_steps].sum())
        return make_sample(np.concatenate(times), np.concatenate(channels), y_star, self.read_windows, self.T)

    def describe(self):
        print(f"AddTask: L = {self.sequence_length} values ({self.encoding} coded), {self.answer_steps} answer "
              f"step(s), {self.n_steps} steps of {self.step_length} (T = {self.T}), 2 input channels, 1 output, "
              f"loss '{self.loss}' (baseline {self.baseline_loss:.3f}), {self.n_train} train / {self.n_test} test "
              f"samples")


# ---- MNIST ----

N_PIXELS, N_CLASSES = 784, 10
FILES = dict(train=("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
             test=("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"))


@dataclass
class MnistTask(StandardAnalysisTask):
    """Classify an image from one latency- or rate-coded channel per pixel.

    data_root        folder with the MNIST IDX files
    download         download with torchvision if the files are not found
    encoding         "latency" (Klos & Memmesheimer) or "rate"
    n_bins           latency: brightness bins that spike (Klos & Memmesheimer's Nin_virtual; 1 in their runs)
    max_rate         rate: spikes per unit time of a white pixel
    T                trial length (2.0 in Klos & Memmesheimer)
    read_from        the read window is [read_from * T, T]
    n_train, n_test  images used from the train and test sets (the first ones; at most 60000 and 10000)
    data_seed        seed of the rate-coded spikes
    """
    data_root: str = "data/mnist"
    download: bool = False
    encoding: str = "latency"
    n_bins: int = 1
    max_rate: float = 20.0
    T: float = 2.0
    read_from: float = 0.0
    n_train: int = 60000
    n_test: int = 10000
    data_seed: int = 0

    loss = "softmax"

    def __post_init__(self):
        check_bool("download", self.download)
        check_choice("encoding", self.encoding, ("latency", "rate"))
        check_int("n_bins", self.n_bins, low=1, high=255)
        check_number("max_rate", self.max_rate, low=0.0, low_open=True)
        check_number("T", self.T, low=0.0, low_open=True)
        check_number("read_from", self.read_from, low=0.0)
        if self.read_from >= 1.0:
            raise ValueError(f"read_from = {self.read_from} must be < 1 (the read window is [read_from T, T])")
        check_int("n_train", self.n_train, low=1, high=60000)
        check_int("n_test", self.n_test, low=0, high=10000)
        check_int("data_seed", self.data_seed, low=0)

        self.n_input_channels, self.n_outputs = N_PIXELS, N_CLASSES
        self.read_window = np.array([[self.read_from * self.T, self.T]])
        images, labels = load_mnist(self.data_root, self.download)
        self._data = dict(train=list(zip(images["train"][:self.n_train], labels["train"][:self.n_train])),
                          test=list(zip(images["test"][:self.n_test], labels["test"][:self.n_test])))

    def _to_sample(self, item, split, index):
        image, label = item
        pixels = np.asarray(image, int).ravel()
        if self.encoding == "latency":
            times, channels = self.latency_spikes(pixels)
        else:
            rng = np.random.default_rng([self.data_seed, self.splits.index(split), index])
            counts = rng.poisson(pixels / 255.0 * self.max_rate * self.T)
            channels = np.repeat(np.arange(N_PIXELS), counts)
            times = rng.uniform(0.0, self.T, counts.sum())
        return make_sample(times, channels, np.array([label]), self.read_window, self.T)

    def latency_spikes(self, pixels):
        """Klos & Memmesheimer's binned latency code: (times, channels) of the pixels that spike."""
        bins = (255 - pixels) // (256 // (self.n_bins + 1))
        spiking = bins < self.n_bins
        drive = self.n_bins - bins[spiking]
        threshold = 0.01 * self.n_bins
        return self.T * np.log(drive / (drive - threshold)), np.where(spiking)[0]

    @property
    def loss_kwargs(self):
        return dict(class_axis=0)

    def describe(self):
        coding = f"latency, {self.n_bins} bin(s)" if self.encoding == "latency" else f"rate, max {self.max_rate}"
        print(f"MnistTask: {N_PIXELS} input channels ({coding}), {N_CLASSES} classes, T = {self.T}, read window "
              f"{self.read_window[0].tolist()}, loss '{self.loss}', {len(self._data['train'])} train / "
              f"{len(self._data['test'])} test images")


# =====================================================================================================================
# Loading
# =====================================================================================================================

def load_mnist(data_root, download=False):
    """({"train": (60000, 28, 28) uint8, "test": ...}, {"train": (60000,) int, "test": ...}) from the IDX files."""
    folder = find_idx_folder(Path(data_root))
    if folder is None and download:
        try:
            from torchvision.datasets import MNIST
        except ImportError as error:
            raise ImportError("MnistTask: download=True needs torchvision; or put the IDX files in "
                              "data_root") from error
        MNIST(str(data_root), train=True, download=True)
        MNIST(str(data_root), train=False, download=True)
        folder = find_idx_folder(Path(data_root))
    if folder is None:
        raise FileNotFoundError(f"MnistTask: no MNIST IDX files in {data_root} or {data_root}/MNIST/raw; set "
                                f"download=True or put them there")
    images, labels = {}, {}
    for split, (image_file, label_file) in FILES.items():
        images[split] = read_idx(idx_path(folder, image_file))
        labels[split] = read_idx(idx_path(folder, label_file)).astype(int)
        if len(images[split]) != len(labels[split]):
            raise ValueError(f"MnistTask: {split} has {len(images[split])} images but {len(labels[split])} labels")
    return images, labels


def find_idx_folder(root):
    for folder in (root, root / "MNIST" / "raw"):
        if all(idx_path(folder, name) is not None for pair in FILES.values() for name in pair):
            return folder
    return None


def idx_path(folder, name):
    for candidate in (folder / name, folder / f"{name}.gz"):
        if candidate.is_file():
            return candidate
    return None


def read_idx(path):
    """An IDX file (the MNIST format): magic number, dimensions, then the uint8 data."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        data = handle.read()
    zero, dtype_code, n_dims = data[0] * 256 + data[1], data[2], data[3]
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file")
    shape = tuple(int.from_bytes(data[4 + 4 * axis: 8 + 4 * axis], "big") for axis in range(n_dims))
    return np.frombuffer(data, np.uint8, offset=4 + 4 * n_dims).reshape(shape)


# ---- TIMIT ----

PHONES_61 = ("aa ae ah ao aw ax ax-h axr ay b bcl ch d dcl dh dx eh el em en eng epi er ey f g gcl h# hh hv ih ix iy "
             "jh k kcl l m n ng nx ow oy p pau pcl q r s sh t tcl th uh uw ux v w y z zh").split()
PHONE_INDEX = {phone: index for index, phone in enumerate(PHONES_61)}
assert len(PHONES_61) == 61

# Lee & Hon (1989): 61 -> 39 classes; "q" is dropped (mapped to None)
FOLD_39 = dict(ao="aa", ax="ah", **{"ax-h": "ah"}, axr="er", hv="hh", ix="ih", el="l", em="m", en="n", nx="n",
               eng="ng", zh="sh", ux="uw", pcl="sil", tcl="sil", kcl="sil", bcl="sil", dcl="sil", gcl="sil",
               **{"h#": "sil"}, pau="sil", epi="sil", q=None)

CORE_TEST_SPEAKERS = ("mdab0 mwbt0 felc0 mtas1 mwew0 fpas0 mjmp0 mlnt0 fpkt0 mlll0 mtls0 fjlm0 mbpm0 mklt0 fnlp0 "
                      "mcmj0 mjdh0 fmgd0 mgrt0 mnjm0 fdhc0 mjln0 mpam0 fmld0").split()

SAMPLE_RATE, WINDOW, HOP = 16000, 400, 160         # 25 ms window, 10 ms frames


@dataclass
class TimitTask(StandardAnalysisTask):
    """Classify each TIMIT frame from its 13 MFCCs and their first and second derivatives.

    Features are normalized with the training split. The target at each frame is one of
    the 61 phone labels. See the package README for corpus layout and feature dependencies.

    data_root         the TIMIT corpus (folders TRAIN and TEST, any letter case)
    cache_file        where the features are kept (default: data_root/timit_mfcc39.npz)
    encoding          "latency" or "rate" (how a feature value in [0, 1] becomes spikes in its frame)
    latency_span      latency: fraction of the frame the spike time ranges over
    max_rate          rate: spikes per unit time for value 1
    step_length       length of one frame in the network's time units (tau = 1)
    n_validation      validation sentences (drawn from the non-core TEST speakers)
    max_sentences     keep only the first max_sentences of each split (None: all)
    data_seed         seed of the validation draw and of the rate-coded spikes
    """
    data_root: str = "data/TIMIT"
    cache_file: str | None = None
    encoding: str = "latency"
    latency_span: float = 0.9
    max_rate: float = 20.0
    step_length: float = 0.25
    n_validation: int = 400
    max_sentences: int | None = None
    data_seed: int = 0

    loss = "softmax"
    splits = ("train", "validation", "test")

    def __post_init__(self):
        check_encoding(self)
        check_number("step_length", self.step_length, low=0.0, low_open=True)
        check_int("n_validation", self.n_validation, low=0)
        if self.max_sentences is not None:
            check_int("max_sentences", self.max_sentences, low=1)
        check_int("data_seed", self.data_seed, low=0)

        self.n_input_channels, self.n_outputs = 39, len(PHONES_61)
        root = Path(self.data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"TimitTask: no folder {root}")
        cache = Path(self.cache_file) if self.cache_file else root / "timit_mfcc39.npz"
        sentences = split_sentences(root, self.n_validation, self.data_seed)
        features, labels = load_or_compute_features(sentences, cache)

        low = np.min([frames.min(0) for frames in features["train"]], 0)
        high = np.max([frames.max(0) for frames in features["train"]], 0)
        scale = np.where(high > low, high - low, 1.0)
        self._data = {}
        for split in self.splits:
            pairs = [(np.clip((frames - low) / scale, 0.0, 1.0), frame_labels)
                     for frames, frame_labels in zip(features[split], labels[split])]
            self._data[split] = pairs[:self.max_sentences] if self.max_sentences else pairs

    def _to_sample(self, item, split, index):
        values, labels = item
        rng = np.random.default_rng([self.data_seed, self.splits.index(split), index])
        times, channels = frame_spikes(values, self.step_length, self.encoding, self.latency_span, self.max_rate, rng)
        n_frames = len(labels)
        return make_sample(times, channels, labels, self._step_windows(np.arange(n_frames)),
                           n_frames * self.step_length)

    @property
    def loss_kwargs(self):
        return dict(class_axis=0)

    def describe(self):
        sizes = ", ".join(f"{len(self._data[split])} {split}" for split in self.splits)
        frames = np.mean([len(frame_labels) for _, frame_labels in self._data["train"]])
        print(f"TimitTask: 39 input channels ({self.encoding} coded), {self.n_outputs} phone classes, frames of "
              f"{self.step_length} (mean {frames:.0f} frames per training sentence), loss '{self.loss}' every "
              f"frame; {sizes} sentences")


def fold_to_39(labels):
    """61-phone labels -> 39-class names (Lee & Hon 1989); "q" frames give None (left out of the scoring)."""
    names = [PHONES_61[label] for label in np.asarray(labels).ravel()]
    return [FOLD_39.get(name, name) for name in names]


# =====================================================================================================================
# Sentences, features and labels
# =====================================================================================================================

def split_sentences(root, n_validation, seed):
    """{split: sorted list of sentence paths without extension}: train = SI/SX of TRAIN, test = SI/SX of the core
    test speakers, validation = n_validation SI/SX sentences of the other TEST speakers, drawn with seed."""
    def si_sx(folder):
        paths = [path.with_suffix("") for path in folder.rglob("*") if path.suffix.lower() == ".phn"]
        return sorted(path for path in paths if not path.name.lower().startswith("sa"))

    folders = {path.name.lower(): path for path in root.iterdir() if path.is_dir()}
    if "train" not in folders or "test" not in folders:
        raise FileNotFoundError(f"TimitTask: {root} needs the folders TRAIN and TEST")
    train = si_sx(folders["train"])
    test_all = si_sx(folders["test"])
    core = [path for path in test_all if path.parent.name.lower() in CORE_TEST_SPEAKERS]
    others = [path for path in test_all if path.parent.name.lower() not in CORE_TEST_SPEAKERS]
    if n_validation > len(others):
        raise ValueError(f"TimitTask: n_validation = {n_validation}, but only {len(others)} non-core test sentences")
    chosen = np.random.default_rng(seed).choice(len(others), size=n_validation, replace=False)
    validation = sorted(others[index] for index in chosen)
    if not train or not core:
        raise FileNotFoundError(f"TimitTask: found {len(train)} training and {len(core)} core test sentences in {root}")
    return dict(train=train, validation=validation, test=core)


def load_or_compute_features(sentences, cache):
    """{split: list of (n_frames, 39) arrays}, {split: list of (n_frames,) label arrays}, cached in cache (.npz)."""
    keys = {split: [str(path) for path in paths] for split, paths in sentences.items()}
    if cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if all(np.array_equal(stored[f"{split}_paths"], np.array(paths)) for split, paths in keys.items()):
            return unpack(stored, list(keys))
    features, labels = {}, {}
    for split, paths in sentences.items():
        pairs = [sentence_features(path) for path in paths]
        features[split] = [pair[0] for pair in pairs]
        labels[split] = [pair[1] for pair in pairs]
    pack(cache, keys, features, labels)
    return features, labels


def sentence_features(path):
    """(n_frames, 39) MFCCs with derivatives and (n_frames,) phone labels of one sentence (path without extension)."""
    signal = read_audio(find_file(path, ".wav"))
    features = mfcc_39(signal)
    starts, ends, phones = read_phn(find_file(path, ".phn"))
    centres = np.arange(len(features)) * HOP + WINDOW // 2
    segment = np.clip(np.searchsorted(starts, centres, side="right") - 1, 0, len(phones) - 1)
    labels = np.array([PHONE_INDEX[phones[index]] for index in segment])
    return features, labels


def find_file(path, extension):
    """path + extension in either letter case (and the .WAV.wav naming of some copies)."""
    for suffix in (extension, extension.upper(), extension.upper() + extension):
        candidate = path.parent / (path.name + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"TimitTask: no {extension} file for {path}")


def read_audio(path):
    try:
        import soundfile
    except ImportError as error:
        raise ImportError("TimitTask: reading TIMIT audio needs soundfile (pip install soundfile)") from error
    signal, rate = soundfile.read(str(path))
    if rate != SAMPLE_RATE:
        raise ValueError(f"TimitTask: {path} has sample rate {rate}, expected {SAMPLE_RATE}")
    return np.asarray(signal, float)


def read_phn(path):
    starts, ends, phones = [], [], []
    for line in Path(path).read_text().split("\n"):
        parts = line.split()
        if len(parts) == 3:
            starts.append(int(parts[0]))
            ends.append(int(parts[1]))
            phones.append(parts[2].lower())
    unknown = set(phones) - set(PHONE_INDEX)
    if unknown:
        raise ValueError(f"TimitTask: {path} has phones outside the 61-phone set: {sorted(unknown)}")
    return np.array(starts), np.array(ends), phones


def mfcc_39(signal):
    """13 MFCCs (25 ms window, 10 ms step) with first and second derivatives: (n_frames, 39)."""
    try:
        from python_speech_features import delta, mfcc
        coefficients = mfcc(signal, SAMPLE_RATE, winlen=WINDOW / SAMPLE_RATE, winstep=HOP / SAMPLE_RATE, numcep=13,
                            nfft=512)
        first = delta(coefficients, 2)
        second = delta(first, 2)
    except ImportError:
        try:
            import librosa
        except ImportError as error:
            raise ImportError("TimitTask: the MFCCs need python_speech_features or librosa") from error
        coefficients = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, win_length=WINDOW,
                                            hop_length=HOP, center=False).T
        first = librosa.feature.delta(coefficients.T, order=1).T
        second = librosa.feature.delta(coefficients.T, order=2).T
    return np.concatenate([coefficients, first, second], 1)


def pack(cache, keys, features, labels):
    arrays = {}
    for split, paths in keys.items():
        lengths = np.array([len(frame_labels) for frame_labels in labels[split]])
        arrays[f"{split}_paths"] = np.array(paths)
        arrays[f"{split}_lengths"] = lengths
        arrays[f"{split}_features"] = np.concatenate(features[split]) if paths else np.zeros((0, 39))
        arrays[f"{split}_labels"] = np.concatenate(labels[split]) if paths else np.zeros(0, int)
    np.savez(cache, **arrays)


def unpack(stored, splits):
    features, labels = {}, {}
    for split in splits:
        bounds = np.concatenate([[0], np.cumsum(stored[f"{split}_lengths"])])
        pieces = list(zip(bounds[:-1], bounds[1:]))
        features[split] = [stored[f"{split}_features"][start:end] for start, end in pieces]
        labels[split] = [stored[f"{split}_labels"][start:end] for start, end in pieces]
    return features, labels
