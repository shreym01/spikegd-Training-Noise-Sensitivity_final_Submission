"""Constants shared by every script: task, seeds, perturbation levels, training, fit and paths."""
import os
from pathlib import Path

# Before importing JAX anywhere: CPU, float64, one thread per process.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_THREEFRY_PARTITIONABLE", "False")   # JAX >= 0.5 changed the default; keeps the task
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

HERE = Path(__file__).resolve().parents[2] / "experiments" / "local_rules"
DATA = HERE / "output" / "data"
FITS = DATA / "fits"
RUNS = DATA / "ablation"
RECORDING = DATA / "fit_data.npz"         # exact run on input 0 (the task input)
FIT_INPUTS = DATA / "fit_inputs"          # exact runs on inputs 1, 2, ... (input_<k>.npz)
FIGS = HERE / "output" / "figs"
PANELS = FIGS / "panels"
LOGS = HERE / "logs"

# Published task of Klos & Memmesheimer (rnn.ipynb).
TASK = {"seed": 7, "tau": 1.0, "eps": 1e-6, "alpha": 10.0, "N": 10, "w_in_exc": 5.0, "w_in_inh": -5.0,
        "r_in": 1.0, "T": 10.0, "K": 500, "dt": 0.01, "Ntar": 2, "r_targets": [1.0, 0.5]}

# Perturbations: level 0 is the input and start the fit was made on (one run).
CASES = ("perturb", "init")                # input jitter sigma / initial-weight scale w_sigma
LEVELS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
N_SEEDS = 3
SEED_INPUT, SEED_WEIGHTS = 1000, 2000

# Training.
N_TRIALS = 10_000
PROBE_EVERY = 200                          # trials between probes (cosine, Lyapunov)
LR, DECAY_STEPS = 1e-2, 2_000              # AdaBelief, lr * exp(-n / DECAY_STEPS)
LAST = 250                                 # final loss = mean of the last LAST trials

# Fit.
RECORD_STRIDE = 10                         # record every RECORD_STRIDE-th trial of the exact run
N_HIDDEN_SPIKES = 8                        # off-target outputs used per neuron
BLOCK = 50                                 # alternating train / held-out blocks of recordings
BASE_FIT = {"q": True, "A": True, "C": True, "Cx": True, "Coff": True, "off": "C",
            "r_A": 1.0, "r_C": 1e-3, "r_C_off": 1.0}
N_FIT_INPUTS = 1                           # inputs the fit is made on (--inputs n); input 0 is the task input
SEED_FIT_INPUTS = 3000                     # input k > 0 is a new Poisson sample drawn with key SEED_FIT_INPUTS + k
FIT_INPUTS_SWEEP = (1, 5, 10, 20)          # fits compared in table_fit_inputs and final_<case>_inputs
RIDGES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1.5, 2.0)

# Figures.
TARGET_NEURON, OFFTARGET_NEURON = 0, 7
