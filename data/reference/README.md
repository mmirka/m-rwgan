# Reference data

Small, curated data committed directly with the repo (~1 MB) so the
reward-network checkpoints and the 12-router sweep artifacts need no
separate download.

## `reward_networks/checkpoints/`

Six trained TensorFlow `SavedModel` reward networks (frozen, `trainable=False`
when used inside M-RWGAN), one triple (saturation/throughput, power, area)
per traffic pattern:

- `hotspot30_3classes/` — trained on hotspot30 traffic. Used for the paper's
  main (8×8, hotspot) results.
- `uniform_3classes/` — trained on uniform traffic.

Each reward net predicts a single scalar (MAE ~0.003–0.1 depending on target,
per the directory names) from a router-class assignment + adjacency matrix
(GCN for saturation/power) or a router-class grid image (CNN for area).

These are loaded directly by `scripts/train_mrwgan.py` — no retraining
needed to run the GAN in inference/fine-tune mode against them.

## `experimental_data/reward_weight_sweep/`

Results of a 14-point sweep over (saturation, power, area) reward-weight
triples (e.g. `50-0-50`, `34-33-33`), run on a **3×4 mesh** (12 routers) —
smaller than the 8×8 mesh used for the paper's headline figures. Per combo:

- `..._reward{1,2,3}_loss` — per-epoch reward loss curve (301 values: 300
  training epochs + 1 final), `numpy.ndarray`, plain `pickle`.
- `..._historyFake_fix` — generator output on a fixed noise vector, sampled
  once per epoch: `list` of 301 arrays, each `(10, 3, 4, 3)` = (10 samples,
  3×4 mesh, 3 router classes).
- `perfs_<weights>` — `(100, 3)` numpy arrays, one row per generated sample,
  columns [saturation, power, area]. Kept as provenance: this is the
  3×4/12-router mesh, not the paper's 8×8 headline mesh (see
  `data/raw/dse_12r3c/README.md`).
- `perfs_<weights>_norm` — the min-max-normalized counterparts, **moved to
  `figures/data/igd_reference/`**, where `figures/igd_compare.py` reads them
  for thesis Fig 6.20 (`figures/` resolves every input from inside itself).

## `experimental_data/baseline_topology_dataset/` — moved

The seven hand-tuned reference 8×8-mesh topologies (`dataset_refNoC_*`) that
back the Pareto figures now live at
**`figures/data/baseline_topology_dataset/`**, alongside the figure scripts
that consume them. See `figures/README.md`.

## Larger raw material

The 10k-sample training sets, the dated 8×8 M-RWGAN training-run artifacts,
and the GAN-generated comparison series are large and kept out of git — see
`data/raw/README.md` for what they are and where to get them. A curated
subset needed for the paper figures is committed under `figures/data/`.
