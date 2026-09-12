# M-RWGAN on MNIST — three demo notebooks

Three fully self-contained Jupyter notebooks that build the M-RWGAN idea up from
a plain Wasserstein GAN, on MNIST, in TensorFlow / Keras **eager** mode (TF 2.15,
gradient penalty via `tf.GradientTape`).

| Notebook | Demonstrates |
|---|---|
| `01_wgan_gp.ipynb` | Plain **WGAN-GP** — CNN generator + CNN critic, gradient penalty, no reward. Learns all ten digits. |
| `02_rwgan_single_reward.ipynb` | **RWGAN** — WGAN-GP + **one** frozen CNN reward (`P(digit == 5)`). Generator converges to the digit "5". |
| `03_mrwgan_two_rewards.ipynb` | **M-RWGAN** — WGAN-GP + **two** independent frozen CNN rewards ("5" and "1"). Generator produces a two-mode 1/5 mix. |

The progression is **WGAN-GP → RWGAN → M-RWGAN**: each notebook keeps the exact
WGAN-GP core of the previous one and adds frozen reward networks that pull the
generator toward chosen digits.

Every model, training and plotting block is **inline in each notebook** — nothing
is imported from the repo's `mrwgan` package (that code is graph-shaped and
graph-mode; nothing transfers cleanly). The three notebooks deliberately repeat
their shared blocks so each reads top-to-bottom on its own. The reward loss and
its lambda annealing mirror the semantics of `src/mrwgan/losses.py`
(`LambdaSchedule`).

## What each notebook contains

1. A title markdown cell — what it demonstrates and where it sits in the
   WGAN-GP → RWGAN → M-RWGAN progression.
2. Imports → data → (reward nets, nb2/nb3) → models → WGAN-GP core → training
   loop → evaluation plots.
3. A closing markdown cell interpreting the result.

Shared building blocks:

- **Data** — `tf.keras.datasets.mnist`, scaled to `[-1, 1]`, shape `(28, 28, 1)`,
  `tf.data.Dataset` shuffled and batched (128).
- **Generator (CNN)** — `z(100) → Dense 7·7·256 → reshape → Conv2DTranspose 128
  (stride 2) → BN → ReLU → Conv2DTranspose 64 (stride 2) → BN → ReLU → Conv2D 1
  (kernel 7, tanh)`.
- **Critic (CNN)** — `Conv2D 64 (stride 2) → LayerNorm → LeakyReLU → Conv2D 128
  (stride 2) → LayerNorm → LeakyReLU → Flatten → Dense 1`. No BatchNorm (WGAN-GP
  requirement).
- **WGAN-GP** — `n_critic = 5`; gradient-penalty coefficient 10 on
  `x̂ = ε·real + (1−ε)·fake`; `Adam(2e-4, β1 0.5, β2 0.9)` for both networks.
- **Reward loss** (nb2/nb3) —
  `L_G = λ·(−E[critic(G(z))]) + (1−λ)·REWARD_WEIGHT·reward_term`, with
  `REWARD_WEIGHT` (≈ 8) the paper's `β` coefficient that lifts the bounded `[0, 1]`
  reward term to the magnitude of the unbounded critic term, and `λ` annealed by
  an inline `LambdaSchedule` (1.0 → floor 0.15, `Δ = −0.025` per epoch, starting
  epoch 20, floor reached ~epoch 54). nb2 uses `reward_term = mean((1 − R5)²)`;
  nb3 uses a **soft-OR**, `reward_term = mean((1 − max(R5, R1))²)` — nb3's
  markdown explains why the literal weighted sum collapses to ambiguous strokes.

## Environment

```
conda env create -f environment.yml
conda activate mrwgan-mnist
```

This env is independent of the repo's main `mrwgan` env — it pins
`tensorflow[and-cuda]==2.15.1` (eager, Keras 2), not the TF 2.2 legacy stack.
The `[and-cuda]` extra bundles the CUDA/cuDNN pip wheels, so GPU works with
only an NVIDIA driver installed; it falls back to CPU when no GPU is present.

## Running

Interactively:

```
jupyter lab        # or: jupyter notebook
```

Headless / from the command line:

```
jupyter nbconvert --to notebook --execute --inplace 01_wgan_gp.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_rwgan_single_reward.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_mrwgan_two_rewards.ipynb
```

The notebooks are shipped **unexecuted** (no output cells).

## Fast smoke run

Near the top of every notebook:

```
EPOCHS = 100           # set to 2-3 for a quick smoke run  (nb2/nb3; nb1 uses 40)
N_EVAL_SAMPLES = 2000  # generated images used for the digit-class histogram
```

Drop `EPOCHS` to 2–3 (and optionally `N_EVAL_SAMPLES` to a few hundred) to check
that a notebook runs end to end. Training is CPU-feasible but a GPU is strongly
recommended for the full run (nb2/nb3 `EPOCHS = 100` — the reward anneal only
reaches its floor around epoch 54, so the reward-shaping effect needs the back
half of the run to become visually obvious).

## Expected outputs (`output/`)

| File | Expectation |
|---|---|
| `01_wgan_gp_samples.png` | 8×8 fixed-noise grid: varied, legible digits spanning all ten classes. |
| `02_rwgan_single_reward_samples.png` | 8×8 grid with 5s as the most common digit, mixed with other digit-like strokes (not a pure "all 5s" grid — see note below). |
| `02_rwgan_single_reward_digit_hist.png` | Predicted-class histogram over `N_EVAL_SAMPLES` samples: class 5 is the mode at roughly **25–35 %** of samples, with a broad tail over the other digits — not a sharp concentration. A single frozen differentiable reward is not enough to force a >90 % single-digit collapse: once `λ` reaches its floor the generator minimises `mean((1 − R5)²)` by drifting off the real-image manifold into `R5`'s blind spots (`R5 → ~0.99`, `d_loss` collapses to ≈ −5) rather than by producing only clean 5s, so an independent classifier still sees a spread of digits. |
| `03_mrwgan_two_rewards_samples.png` | 8×8 grid of 1s and 5s. |
| `03_mrwgan_two_rewards_digit_hist.png` | Histogram concentrated on classes 1 and 5, roughly balanced (50/50 typical, up to ~90/10 fine). |

The nb2 numbers above are the shipped config: `EPOCHS=100`, `BATCH_SIZE=128`,
`LATENT_DIM=100`, `N_CRITIC=5`, `GP_WEIGHT=10`, `REWARD_WEIGHT=8`,
`LambdaSchedule(Δ=−0.025, start 20, every 1, floor 0.15)`, `Adam(2e-4, β1 0.5,
β2 0.9)`, binary CNN reward. It was the best of a ~26-config sweep (reward weight
3–20, λ-floor 0–0.55, anneal shape, latent 100–512, reward arch
binary / ensemble / 10-way-softmax, longer training); none exceeded a sustained
~35 % verified five-share, because a single frozen differentiable reward gets
gamed rather than obeyed. See nb2's closing **Result** cell.

Intermediate `*_samples_epoch###.png` grids are also written every 10 epochs.
`output/` is git-ignored (kept by `.gitkeep`).
