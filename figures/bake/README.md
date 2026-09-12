# `bake/` — reducing the oversized figure inputs

Four of the archived inputs the figure scripts read total ~1.2 GB and are too
large for git. Each is consumed by a *summarising* step, so what the figures
actually take from them is small: a few arrays. These scripts compute that
reduction once and write it to `figures/data/derived/`, which is committed.

The result is that **a plain clone rebuilds all 34 figures with no download**,
while the raw archives stay usable for re-baking or for deriving something new.

| Script | Reads | Writes | Reduction |
|---|---|---|--:|
| `bake_dataset_summary.py` | `data/dataset_10k_{uniform,hotspot30}_3c` | `*_summary.npz` — per-sample saturation / power / normalized area | 308 MB → 103 KB · 242 MB → 94 KB |
| `bake_dse_front.py` | `data/dse_12r3c/dataset_all_{X,sat,pow}` | `dse_12r3c_front.npz` — the marginal true Pareto front | 57 MB → 2 KB |
| `bake_history_classes.py` | every `*_historyFake[_fix]` under `data/reward_weight_sweep_8x8_3reward/` | `*_classes.npz` — `(n_epochs, batch, 64)` int8 per-router class indices | 492 MB → 210 KB (and 11 more) |
| `bake_loss_curves.py` | one run's `*_{generator,discriminator,reward N}_loss` | `*_losses.npz` — the two plotted loss columns + the reward arrays | 12 MB → 1.4 MB |

**1.3 GB → 3.3 MB.** `bake_common.py` holds the shared artifact format; run it
to describe what an artifact contains and which archive it came from.

## Re-baking

Needs the raw archives — see [`../data/README.md`](../data/README.md).

```bash
bake/bake_all.sh     # re-derive every artifact
bake/verify.sh       # prove the artifacts still reproduce the figures exactly
```

## Why this is not lossy

Each reduction is the *whole* of what its figures read, not a downsample:

- The 10k datasets reach every figure through one call,
  `noc_data.summarize_samples`, which returns three numbers per sample.
- The 531,441-NoC enumeration reaches Fig 6.20 only as the marginal front
  derived from it.
- A `historyFake` batch reaches every reader through
  `noc_data.per_router_class_fraction`, whose first act is an `argmax` over the
  class columns. Baking takes that `argmax` once.
- Fig 6.10 plots column 0 of the generator log and column 2 of the critic log.

Values are kept at their original precision (float64, or exact integers for the
argmax), so this is not a tolerance argument. `verify.sh` builds every figure
twice — once from the raw archives, once in a copy of `figures/` holding only
the git-tracked files — and compares the PNGs byte for byte. All 34 match.

## Provenance

Every artifact records the filename, byte size and SHA-256 of each archive it
was derived from:

```bash
python bake/bake_common.py                                # all of them
python bake/bake_common.py data/derived/dse_12r3c_front.npz
```

Re-baking from a different archive therefore produces a visibly different
artifact, rather than a silent substitution.
