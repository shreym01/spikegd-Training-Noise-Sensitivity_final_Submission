"""Losses, diagnostics, and training for the QIF test bench.

The trainer obtains a sample, runs the network, differentiates the loss with respect to
the readout, and passes that signal to the rule. Metrics are collected in a Record.
"""

from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checks import check_bool, check_choice, check_int, check_number, check_positive, check_type, is_concrete
from .rules import LearningRule
from .tasks import CustomAnalysisTask, NetRequirements, StandardAnalysisTask


# ---- Loss functions ----

def klos_loss(y, y_star, mask=None):
    """Relative squared error of spike times, as in Klos & Memmesheimer.

    y, y_star   (n_out, n_slots): output spike times (with pseudospikes for missing spikes) and target times,
                one row per read-out neuron
    mask        (n_out, n_slots) bool: which slots hold a target (rows can have different numbers of targets; the
                targets of a row are its first slots). Default: all slots.

    Each slot adds ((y - y*) / (y* + 2))^2. The last target of each row is the extra target (1.1 T in their task):
    it only counts while y < y*, so it penalises an extra spike before it and nothing after it. The sum is divided
    by the number of targets (extra ones included).
    """
    y, y_star, mask = _check_pair("klos_loss", y, y_star, mask, ndim=2)
    if is_concrete(y_star) and is_concrete(mask):
        mask_values, target_values = np.asarray(mask), np.asarray(y_star)
        if (np.diff(mask_values.astype(int), axis=-1) > 0).any():
            raise ValueError("klos_loss: the targets of each row must be its first slots (mask rows like 1 1 1 0 0)")
        targets = target_values[mask_values]
        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("klos_loss: target times must be finite and >= 0")
    n_targets = mask.sum(-1, keepdims=True)
    is_last = jnp.arange(y.shape[-1]) == n_targets - 1
    y_star = jnp.where(mask, y_star, 0.0)       # padded slots (any value, inf too) give 0 and no NaN gradient
    terms = ((y - y_star) / (y_star + 2.0)) ** 2
    terms = jnp.where(is_last, terms * (y < y_star), terms)
    return jnp.sum(jnp.where(mask, terms, 0.0)) / jnp.sum(n_targets)


def mse_loss(y, y_star, mask=None):
    """mean over the (masked) entries of (y - y*)^2."""
    y, y_star, mask = _check_pair("mse_loss", y, y_star, mask)
    y_star = jnp.where(mask, y_star, y)          # padded slots give 0 and no NaN gradient
    return jnp.sum(jnp.where(mask, (y - y_star) ** 2, 0.0)) / jnp.sum(mask)


def continuous_loss(y, y_star, dt, T):
    """(1 / 2T) int_0^T |y(t) - y*(t)|^2 dt, as a Riemann sum.

    y, y_star   (..., n_grid): the output trace and the target on a grid with spacing dt over [0, T]
                (leading axes, e.g. output channels, are summed over)
    """
    y, y_star, _ = _check_pair("continuous_loss", y, y_star, None)
    check_positive("continuous_loss: dt", dt)
    check_positive("continuous_loss: T", T)
    n_grid = y.shape[-1]
    if not abs(n_grid * dt - T) <= dt * (1 + 1e-9):
        raise ValueError(f"continuous_loss: {n_grid} grid points with dt = {dt} cover {n_grid * dt:g}, not T = {T}")
    return 0.5 * jnp.sum((y - y_star) ** 2) * dt / T


def softmax_loss(y, y_star, time_coded=False, gamma=0.0, T=None, class_axis=-1):
    """Cross entropy of softmax over classes with the label y_star.

    y           (n_classes,) or (batch, n_classes): scores, or first output spike times if time_coded
                (class_axis = 0 for the head's y of shape (n_classes, n_read): one label per read point)
    y_star      () or (batch,) int labels
    time_coded  the class scores are -y (earliest spike wins), as in Klos & Memmesheimer's MNIST
    gamma, T    with time_coded: their regulariser gamma (exp(y_label / T) - 1), which pushes the correct
                neuron to spike early

    Returns the mean loss over the batch.
    """
    y, y_star = jnp.asarray(y), jnp.asarray(y_star)
    if y.ndim == 2 and class_axis in (0, -2):
        y = y.T
    elif class_axis not in (-1, y.ndim - 1):
        raise ValueError(f"softmax_loss: class_axis = {class_axis} does not fit y of shape {y.shape}")
    if y.ndim not in (1, 2):
        raise ValueError(f"softmax_loss: y must be (n_classes,) or (batch, n_classes), got shape {y.shape}")
    if y.ndim - 1 != y_star.ndim or (y.ndim == 2 and y.shape[0] != y_star.shape[0]):
        raise ValueError(f"softmax_loss: labels of shape {y_star.shape} do not match y of shape {y.shape} "
                         f"(one label per row)")
    if not jnp.issubdtype(y_star.dtype, jnp.integer):
        raise TypeError(f"softmax_loss: labels must be integers, got {y_star.dtype}")
    if y.shape[-1] < 2:
        raise ValueError("softmax_loss: needs at least two classes")
    if is_concrete(y_star):
        labels = np.asarray(y_star)
        if labels.size and (labels.min() < 0 or labels.max() >= y.shape[-1]):
            raise ValueError(f"softmax_loss: labels must be in [0, {y.shape[-1]}), got "
                             f"[{labels.min()}, {labels.max()}]")
    check_bool("softmax_loss: time_coded", time_coded)
    check_number("softmax_loss: gamma", gamma, low=0.0)
    if gamma and not time_coded:
        raise ValueError("softmax_loss: gamma (the spike-time regulariser) needs time_coded=True")
    if gamma:
        if T is None:
            raise ValueError("softmax_loss: gamma > 0 needs T")
        check_positive("softmax_loss: T", T)

    y, y_star = jnp.atleast_2d(y), jnp.atleast_1d(y_star)
    scores = -y if time_coded else y
    log_p = jax.nn.log_softmax(scores, axis=-1)
    loss = -jnp.take_along_axis(log_p, y_star[:, None], axis=-1)[:, 0]
    if gamma:
        loss = loss + gamma * (jnp.exp(jnp.take_along_axis(y, y_star[:, None], axis=-1)[:, 0] / T) - 1.0)
    return jnp.mean(loss)


LOSSES = dict(klos=klos_loss, mse=mse_loss, continuous=continuous_loss, softmax=softmax_loss)


# =====================================================================================================================
# Checks
# =====================================================================================================================

def _check_pair(name, y, y_star, mask, ndim=None):
    """y and y_star of the same shape (ndim if given), mask of that shape with at least one True entry."""
    y, y_star = jnp.asarray(y), jnp.asarray(y_star)
    if y.shape != y_star.shape:
        raise ValueError(f"{name}: y has shape {y.shape} but y_star has shape {y_star.shape}")
    if ndim is not None and y.ndim != ndim:
        raise ValueError(f"{name}: y and y_star must have {ndim} axes, got shape {y.shape}")
    if y.size == 0:
        raise ValueError(f"{name}: y is empty")
    if mask is None:
        mask = jnp.ones(y.shape, bool)
    else:
        mask = jnp.asarray(mask)
        if mask.shape != y.shape:
            raise ValueError(f"{name}: mask has shape {mask.shape}, y has shape {y.shape}")
        if mask.dtype != jnp.bool_:
            raise TypeError(f"{name}: mask must be boolean, got {mask.dtype}")
    if is_concrete(mask) and not np.asarray(mask).any():
        raise ValueError(f"{name}: mask has no True entry (nothing to average over)")
    return y, y_star, mask


# ---- Metrics ----

@dataclass
class Checkpoint:
    """What a metric can read at a checkpoint.

    net, params, task, rule, rule_state    the parts being trained, as they are now
    trial                                   index of the trial (across trials)
    time                                    time inside the trial (None for a checkpoint at the end of a trial)
    sample, y, events, loss, grads          the trial's sample, the head's output, the events, the loss and the rule's
                                            update (grads), when the checkpoint is inside or at the end of a trial
    """
    net: object
    params: dict
    task: object
    rule: object
    rule_state: dict
    trial: int
    time: float | None = None
    sample: object = None
    y: object = None
    events: object = None
    loss: float | None = None
    grads: dict | None = None


class Metric:
    """name     key of the metric in the trainer's record
    within_trial   True: computed at the checkpoints inside a trial; False: once at the end of a checkpoint trial
    compute(checkpoint) -> a value (number or array) stored in the record"""

    name = "metric"
    within_trial = False

    def compute(self, checkpoint):
        raise NotImplementedError


class Loss(Metric):
    """The trial's loss (the one metric every run records)."""

    name = "loss"

    def compute(self, checkpoint):
        return checkpoint.loss


# ---- Trainer ----

@dataclass
class TrainerConfig:
    """
    n_trials                    trials of training
    learning_rate, optimizer    the optimizer ("adabelief", as in the bench, "adam" or "sgd") and its rate
    decay_steps                 learning rate * exp(-trial / decay_steps); None = constant
    K                           maximum events per trial; "auto" = 3 x the events of sample 0 at init + 100
    split                       the task split trained on
    sample_order                "cycle" (0, 1, 2, ...) or "shuffle" (a new permutation every pass)
    checkpoints_across_trials   trials at which the metrics are computed, spread evenly over training (plus the
                                last trial)
    checkpoints_per_trial       points inside a checkpoint trial (evenly spaced in time) at which the
                                within-trial metrics are computed; 0 = none
    seed                        seed of the sample order
    """
    n_trials: int = 1000
    learning_rate: float = 0.01
    optimizer: str = "adabelief"
    decay_steps: float | None = 2000
    K: int | str = "auto"
    split: str = "train"
    sample_order: str = "cycle"
    checkpoints_across_trials: int = 50
    checkpoints_per_trial: int = 0
    seed: int = 0

    def __post_init__(self):
        check_int("n_trials", self.n_trials, low=1)
        check_number("learning_rate", self.learning_rate, low=0.0, low_open=True)
        check_choice("optimizer", self.optimizer, ("adabelief", "adam", "sgd"))
        if self.decay_steps is not None:
            check_number("decay_steps", self.decay_steps, low=0.0, low_open=True)
        if self.K != "auto":
            check_int("K", self.K, low=1)
        check_choice("sample_order", self.sample_order, ("cycle", "shuffle"))
        check_int("checkpoints_across_trials", self.checkpoints_across_trials, low=1)
        check_int("checkpoints_per_trial", self.checkpoints_per_trial, low=0)
        check_int("seed", self.seed, low=0)


@dataclass
class Record:
    """What train() returns and the figures read.

    metrics       name -> list of values, one per checkpoint trial (within-trial metrics: one list per trial)
    trials        the checkpoint trials
    times         per checkpoint trial, the times of the within-trial checkpoints
    params        the final params
    config, rule, task, net_config    descriptions of the run
    """
    metrics: dict = field(default_factory=dict)
    trials: list = field(default_factory=list)
    times: list = field(default_factory=list)
    params: dict | None = None
    config: dict | None = None
    rule: str = ""
    task: str = ""
    net_config: object = None


class Trainer:
    def __init__(self, net, task, rule, config=None, metrics=()):
        check_type("rule", rule, LearningRule)
        self.config = TrainerConfig() if config is None else config
        check_type("config", self.config, TrainerConfig)
        self.net, self.task, self.rule = net, task, rule
        self.metrics = [Loss()] + [metric for metric in metrics if not isinstance(metric, Loss)]
        for metric in self.metrics:
            check_type("metric", metric, Metric)
        self.setup()

    # ---- setup ----

    def setup(self):
        """Bind or check the task against the network, pick K, build the optimizer and the rule's state."""
        requirements = NetRequirements.from_net(self.net)
        if isinstance(self.task, CustomAnalysisTask):
            self.task.bind(requirements)
        elif isinstance(self.task, StandardAnalysisTask):
            self.task.check(requirements)
        check_choice("split", self.config.split, self.task.splits)
        self.params = self.net.init_params()
        self.K = self.auto_K() if self.config.K == "auto" else self.config.K
        self.optimizer = self.make_optimizer()
        self.optimizer_state = self.optimizer.init(self.params)
        self.rule_state = self.rule.init_state(self.net, self.params, self.task)

    def auto_K(self):
        """3 x (events of sample 0 at init) + 100, as in the bench; simulated with a generous cap first."""
        sample = self.task.sample(0, self.config.split)
        cap = 10 * (len(sample.spikes_in[0]) + 1000)
        _, _, neuron, _ = self.net.simulate(self.params, sample.spikes_in, sample.T, cap)
        return 3 * int(np.sum(np.asarray(neuron) >= 0)) + 100

    def make_optimizer(self):
        config = self.config
        rate = config.learning_rate
        if config.decay_steps is not None:
            rate = optax.exponential_decay(config.learning_rate, transition_steps=1, decay_rate=np.exp(
                -1.0 / config.decay_steps))
        return dict(adabelief=optax.adabelief, adam=optax.adam, sgd=optax.sgd)[config.optimizer](rate)

    # ---- checkpoints ----

    def checkpoint_trials(self):
        """Trials at which the metrics are computed: checkpoints_across_trials evenly spread, and the last trial."""
        count = min(self.config.checkpoints_across_trials, self.config.n_trials)
        if count == 1:
            return [self.config.n_trials - 1]
        trials = np.unique(np.round(np.linspace(0, self.config.n_trials - 1, count)).astype(int))
        return trials.tolist()

    def checkpoint_times(self, T):
        """Times inside a trial of length T at which the within-trial metrics are computed."""
        count = self.config.checkpoints_per_trial
        return ((np.arange(count) + 1) / (count + 1) * T).tolist() if count else []

    def sample_index(self, trial, rng, order):
        """Index of the sample used at this trial."""
        n_samples = self.task.n_samples(self.config.split)
        if self.config.sample_order == "cycle":
            return trial % n_samples, order
        if trial % n_samples == 0:
            order = rng.permutation(n_samples)
        return int(order[trial % n_samples]), order

    # ---- training ----

    def train(self):
        """Run the rule's algorithm and return the Record."""
        if self.rule.algorithm == "online":
            record = self._train_online()
        elif self.rule.algorithm == "teacher_phases":
            record = self._train_with_teacher()
        else:
            raise ValueError(f"unknown training algorithm {self.rule.algorithm!r}")
        record.params = self.params
        record.config = asdict(self.config)
        record.rule, record.task, record.net_config = self.rule.name, type(self.task).__name__, self.net.config
        return record

    def trial(self, index):
        """One trial: forward, loss, the rule's update, the optimizer step, the projection. Returns what the
        metrics may need."""
        sample = self.task.sample(index, self.config.split)
        y, events = self.net.forward(self.params, sample.spikes_in, sample.T, self.K, read_at=sample.read_at)
        loss_fn = LOSSES[self.task.loss]
        loss, dloss_dy = jax.value_and_grad(lambda out: loss_fn(out, sample.y_star, **self.task.loss_kwargs))(y)
        grads, self.rule_state, info = self.rule.gradient(self.net, self.params, sample, y, events, dloss_dy,
                                                          self.rule_state)
        mask = self.net.trainable_mask(self.params)
        grads = jax.tree_util.tree_map(lambda grad, keep: grad * keep, grads, mask)
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state, self.params)
        self.params = self.net.project_params(optax.apply_updates(self.params, updates))
        return dict(sample=sample, y=y, events=events, loss=float(loss), grads=grads, info=info)

    def _train_online(self):
        """One rule update per trial; metrics at the checkpoint trials (and inside them)."""
        record = Record()
        checkpoints = set(self.checkpoint_trials())
        rng, order = np.random.default_rng(self.config.seed), None
        for trial in range(self.config.n_trials):
            index, order = self.sample_index(trial, rng, order)
            outcome = self.trial(index)
            if trial in checkpoints:
                self.record_checkpoint(record, trial, outcome)
        return record

    def _train_with_teacher(self):
        """The phased algorithm (rule.teacher is set): the trials are split into P = teacher.n_corrections phases.
        In each phase
          1. the teacher runs its trials of the phase with the exact gradient (on its own weights, or on a copy of
             the student's if teacher_on = "student"), branching into perturbed copies at the chosen trials;
          2. from the branches it draws the corrections it is responsible for (B for "c", alpha / A for "A");
          3. the student runs the same trials with the corrected rule.
        Metrics are recorded at the checkpoint trials of the student, as in _train_online."""
        raise NotImplementedError

    def record_checkpoint(self, record, trial, outcome):
        """Compute every metric at this trial: the end-of-trial metrics once, the within-trial ones at
        checkpoint_times(T)."""
        times = self.checkpoint_times(outcome["sample"].T)
        record.trials.append(trial)
        record.times.append(times)
        for metric in self.metrics:
            base = Checkpoint(net=self.net, params=self.params, task=self.task, rule=self.rule,
                              rule_state=self.rule_state, trial=trial, sample=outcome["sample"], y=outcome["y"],
                              events=outcome["events"], loss=outcome["loss"], grads=outcome["grads"])
            if metric.within_trial:
                value = [metric.compute(Checkpoint(**{**base.__dict__, "time": time})) for time in times]
            else:
                value = metric.compute(base)
            record.metrics.setdefault(metric.name, []).append(value)


# ---- Figure interface ----

class Figure:
    """draw(record, ax=None) -> the matplotlib figure (or axes) drawn from a trainer Record."""

    name = "figure"
    needs = ()             # metric names the figure reads from the record

    def draw(self, record, ax=None):
        raise NotImplementedError

    def check(self, record):
        missing = [name for name in self.needs if name not in record.metrics]
        if missing:
            raise KeyError(f"{type(self).__name__} needs the metrics {missing}, which the record does not have")
