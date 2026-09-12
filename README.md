# M-RWGAN — Multi-Reward Wasserstein GAN

**M-RWGAN** is a Wasserstein GAN (WGAN-GP) whose generator is steered, during
training, by one or more independently pretrained, frozen **reward networks**.
Each reward network scores a generated sample against an objective; the scores
are folded into the generator loss with user-adjustable weights and a schedule
that hands control from the critic to the rewards as training progresses. The
result is a generator that produces realistic samples *and* converges toward a
chosen region of the output distribution.

This repository provides the tools to build and train an M-RWGAN, with two
worked applications:

1. **Heterogeneous Network-on-Chip design** (`src/mrwgan/` + `scripts/`) — the
   original application from **DATE2022** (*"A Generative AI for Heterogeneous
   Network-on-Chip Design Space Pruning"*) and **PhD thesis, Chapter 6**
   (Maxime Mirka). A fixed 8×8-mesh generator assigns each of 64 routers a
   class (Big / Medium / Small buffer size), steered by three frozen GNN/CNN
   reward networks for throughput, power, and area.
2. **MNIST digit generation** (`demos/mnist/`) — three self-contained notebooks
   that build the method up in stages on a familiar dataset: plain WGAN-GP → one
   reward → two rewards.

`figures/` reproduces the Chapter 6 figures from bundled historical data.

## Repository layout

```
src/mrwgan/            importable package: generator/critic model, reward networks, losses, training loops
scripts/               CLI entry points for the NoC application (train the GAN, train a reward network)
demos/mnist/           three standalone Jupyter notebooks — M-RWGAN on MNIST (own conda env)
figures/               self-contained: figure scripts + bundled historical data to reproduce the paper figures
data/reference/        small (~11 MB), git-tracked: pretrained reward-network checkpoints + baseline experimental data
data/raw/              large NoC training data (gitignored, ~924 MB) — availability + docs in data/raw/README.md
reference/hnocs_pipeline/   HNOCS/OMNeT++ + Orion3.0 glue code — documents how the NoC training data was simulated
results/               training output (gitignored), except results/figures/ — committed reference renders
```

## Environment setup (NoC application)

```bash
conda env create -f environment.yml
conda activate mrwgan
```

This pins the TensorFlow 2.2 stack the reward-network checkpoints in
`data/reference/` were trained and saved with — see the comments in
`environment.yml` for why each pin matters (in particular: spektral needs the
`GCNConv` layer name the checkpoints were saved with, and protobuf 3.20+ breaks
TF 2.2's import entirely).

### GPU training

The TF 2.2 stack is **CPU-only on modern hardware**: TF 2.2 / CUDA 10.1 have no
kernels for Ampere-or-newer GPUs (compute capability ≥ 8.0, e.g. an RTX
30-series card). For GPU training use the separate environment:

```bash
conda env create -f environment-gpu.yml
conda activate mrwgan-gpu
# let TF find the conda-provided CUDA libraries:
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' \
  > "$CONDA_PREFIX/etc/conda/activate.d/cuda_ld.sh"
conda deactivate && conda activate mrwgan-gpu
```

This retargets the stack to TensorFlow 2.10 + CUDA 11.2 / cuDNN 8.1 (see the
comments in `environment-gpu.yml`). `scripts/train_mrwgan.py` prints which device
TensorFlow selected on startup and enables GPU memory growth; pass `--cpu` to
force CPU even when a GPU is present.

> The MNIST demos use their own environment — see `demos/mnist/README.md`.

## Application 1 — heterogeneous NoC design

```bash
# M-RWGAN itself: the reward-weighted WGAN-GP generator
python scripts/train_mrwgan.py --help
python scripts/train_mrwgan.py --dataset <your_dataset.pkl> --reward-weights 0.5 0.1 0.4 --epochs 300

# One of the three frozen reward networks (saturation/power = GCN, area = CNN)
python scripts/train_reward_module.py --help
python scripts/train_reward_module.py --target saturation --dataset <your_dataset.pkl>
```

Both expect a pickled `list[mrwgan.data_types.Data_AX]` for the 8×8 mesh (see
that module's docstring). Checkpoints and logs land under
`results/checkpoints/<run-name>/` and `results/logs/<run-name>/`.

Pretrained reward-network checkpoints (hotspot30 and uniform traffic) ship in
`data/reference/reward_networks/checkpoints/` — you only need
`train_reward_module.py` to retrain on new data. For training data, see
[Data provenance](#data-provenance) for the datasets used in the paper, or
`reference/hnocs_pipeline/REPRODUCTION.md` to regenerate it from simulation.

## Application 2 — MNIST demos

Three standalone Jupyter notebooks in `demos/mnist/`, in eager TensorFlow 2.10,
building the method up in stages:

- `01_wgan_gp.ipynb` — plain WGAN-GP baseline (CNN generator + CNN critic);
  learns all ten digits.
- `02_rwgan_single_reward.ipynb` — WGAN-GP plus one frozen CNN reward
  (`P(digit == 5)`); the generator converges to the digit 5.
- `03_mrwgan_two_rewards.ipynb` — two independent frozen rewards (5 and 1) with
  a soft-OR aggregation; the generator produces a roughly balanced 1/5 mix.

They import nothing from `mrwgan` and use their own conda env
(`demos/mnist/environment.yml`, `mrwgan-mnist`). See `demos/mnist/README.md` for
setup, the fast smoke-run knobs, and the expected output figures.

## Reproducing the paper figures

`figures/` is a self-contained folder — the figure scripts plus the historical
data they read, all under `figures/data/` (~110 MB git-tracked). **A plain clone
rebuilds all 34 figures with nothing downloaded:** the four oversized inputs
(~1.2 GB, git-ignored) are reduced by `figures/bake/` to ~3.3 MB of committed
artifacts under `figures/data/derived/`, each holding exactly the arrays its
figures consume, and `figures/bake/verify.sh` checks that the PNGs come out
byte-identical either way. No figure script resolves a path outside `figures/`.
It reproduces the Chapter 6 figures (router-class/router-size evolution Fig 6.9,
6.11, 6.12, 6.14–6.17; loss curves Fig 6.10; Pareto comparisons Fig
6.13/6.18/6.19; the IGD bar chart Fig 6.20) and DATE2022 paper figures with only numpy + matplotlib — no
`mrwgan` import.

See `figures/README.md` for the figure → script → data table, the exact
regeneration commands, and how to point the scripts at your own training runs
instead of the shipped data; `figures/bake/README.md` for what each derived
artifact holds and why the reduction is lossless.

## Data provenance

- **`data/reference/`** (~11 MB, git-tracked): the pretrained reward-network
  checkpoints and small baseline experimental data. See `data/reference/README.md`.
- **`data/raw/`** (~924 MB, gitignored — exceeds GitHub's file-size limits):
  the 10k-sample `Data_AX` training datasets (hotspot30 and uniform traffic),
  generated-sample sets, and weight-sweep run artifacts.
  The archives are **available on request** — open an issue on this repository and ask.
  See `data/raw/README.md`.

`reference/hnocs_pipeline/` documents how the NoC training data was originally
produced (HNOCS + Orion3.0 simulation); `reference/hnocs_pipeline/REPRODUCTION.md`
covers building and running that simulator to regenerate data from scratch. The
12-router `perfs_<weights>` design-space study (`data/raw/dse_12r3c/`) is out of
scope here and kept only as provenance; the parts of it that thesis Fig 6.20
still needs live under `figures/data/{dse_12r3c,igd_reference}/`.

## Lineage and references

M-RWGAN generalises **RWGAN** (Reward-Wasserstein GAN) from a single frozen
reward network to several simultaneous ones combined by multi-objective descent.
RWGAN is the earlier, single-objective design: it generates NoC *topologies* —
9×9 router adjacency matrices — rather than router-class assignments on a fixed
mesh. It has its own reproducibility repository:

- **GANNoC** — <https://github.com/mmirka/GANNoC> — RWGAN / RAPIDO 2021
  (PhD thesis, Chapter 5).

### Papers

- **RWGAN / GANNoC** (thesis Chapter 5). Maxime Mirka, Maxime France-Pillois,
  Gilles Sassatelli, and Abdoulaye Gamatié. 2021. GANNoC: A Framework for
  Automatic Generation of NoC Topologies using Generative Adversarial Networks.
  In *Proceedings of the 2021 Drone Systems Engineering and Rapid Simulation and
  Performance Evaluation: Methods and Tools (DroneSE and RAPIDO '21)*. ACM,
  51–58. <https://doi.org/10.1145/3444950.3447283>
- **M-RWGAN** (thesis Chapter 6). Maxime Mirka, Maxime France-Pillois, Gilles
  Sassatelli, and Abdoulaye Gamatié. 2022. A Generative AI for Heterogeneous
  Network-on-Chip Design Space Pruning. In *2022 Design, Automation & Test in
  Europe Conference & Exhibition (DATE)*. IEEE, 1135–1138.
  <https://doi.org/10.23919/DATE54114.2022.9774721>

## Licence

Source code — `src/`, `scripts/`, `figures/*.py`, `figures/bake/`, `demos/` — is
licensed under the **Apache License 2.0**; see [`LICENSE`](LICENSE).

Prose and documentation are **CC BY 4.0**. Simulated datasets and derived
artefacts — `figures/data/`, `data/reference/`, `results/figures/` — are
**CC BY 4.0**; they are outputs of the simulation pipeline and of the models in
this repository.

**Excluded from both grants** (see [`NOTICE`](NOTICE)):

- `reference/hnocs_pipeline/HNOCS_README.md` — reproduced from the upstream
  HNOCS project README, © Ben-Itzhak, Zahavi, Cidon & Kolodny (Technion),
  SAMOS XII, 2012.
- The `Docker.memo` block quoted verbatim in
  `reference/hnocs_pipeline/REPRODUCTION.md`, © the HNOCS authors.

Neither HNOCS nor Orion 3.0 is vendored here. Orion 3.0 in particular carries a
non-commercial UC San Diego academic licence, which is why it is not.

**Data format and trust.** The `Data_AX` pickles and the TensorFlow SavedModels
under `data/reference/reward_networks/` execute code when deserialised — a
property of those formats, not of these files. Load only copies obtained from
this repository or from the archive it cites.
