# Raw training data

Everything under this directory is large raw material used to train
M-RWGAN and its reward networks and to reproduce the paper figures. It is
kept separate from `data/reference/` (small, curated data tracked in git).

**Availability:** this directory (~924MB) is gitignored — too large for a
normal git push (two files exceed GitHub's 100MB hard limit).
The archives are **available on request** — open an issue on this repository and ask.
Place the contents under `data/raw/` (preserving the subfolder structure
below) to reproduce training locally.

None of this is needed to rebuild the figures: `figures/` ships baked
artifacts and resolves every input from inside itself.

## `dataset_10k_hotspot30_3c`, `dataset_10k_uniform_3c`

The raw training datasets for M-RWGAN and its three reward networks.

Format: pickled `list[mrwgan.data_types.Data_AX]` on the 8×8 mesh (64
routers). Verified by loading both with this repo's own `Data_AX` class:

- `dataset_10k_hotspot30_3c` — 9,592 samples
- `dataset_10k_uniform_3c` — 10,458 samples

Each sample holds the adjacency matrix (`A`), router-class assignment
(`X`), and simulated performance curves (`latency`, `total_power`/`powers`,
`total_area`/`areas`, `total_jouls`) from a real HNOCS+Orion3.0 run.

Use directly with `--dataset` on `scripts/train_mrwgan.py` and
`scripts/train_reward_module.py`.

These are the 3-class ("Big"/"Medium"/"Small" homogeneous routers) datasets
that match the reward-net checkpoints in this repo. 8-class variants exist
but have no matching reward-net checkpoints here, so they are outside the
documented scope.

## `reward_weight_sweep_8x8_3reward/`

Dated 3-reward (and one 2-reward) M-RWGAN training-run artifacts on the 8×8
mesh — see its own README for exactly which runs and which figures they
reproduce.

## `generated/`

Already-simulated GAN-generated NoC configurations (`Data_AX` pickles with
actual latency/power/area curves, not just raw generator output) — see its
own README for which figures use them.

## `dse_12r3c/`

The 12-router/3-class design-space-exploration data behind the
`perfs_<weights>` sweep files in `data/reference/`. See its own README.

Its three `dataset_all_{X,sat,pow}` files also live under
`figures/data/dse_12r3c/`, where `figures/igd_compare.py` reads them (thesis
Fig 6.20) — `figures/` resolves every input from inside itself. If you
obtain this directory, copy those three files there as well.
