# `figures/data/` — bundled inputs for the paper-figure scripts

Everything the `figures/*.py` scripts read lives here — no script resolves a
path outside `figures/`. **Every figure rebuilds from a plain clone with
nothing downloaded.**

Four of the original inputs total ~1.2 GB and are git-ignored. What the figures
actually take from them — a few arrays each — is baked into
[`derived/`](#derived--the-baked-inputs) and committed, ~3.3 MB in all. The
scripts read the baked artifacts by default and the raw archives when you
point them at one.

## Layout

| Path | Used by | What it is |
|---|---|---|
| `baseline_topology_dataset/dataset_refNoC_*/` | `best_gene_compare.py`, `moustache_boxplots.py`, `results_analysis_date2022.py` | 7 hand-tuned reference 8×8-mesh topologies; each dir holds a pickled list of `Data_AX` (`dataset`) plus its `X_set`. Not used by `pareto_figures.py` — its "dataset" curve is the 10k-sample set below, not these 7 points. |
| `generated/`, `generated/10-50/` | `pareto_figures.py`, `router_evolution.py`, `best_gene_compare.py`, `moustache_boxplots.py`, `results_analysis_date2022.py` | Already-simulated `Data_AX` pickles for M-RWGAN-generated router-class configs (single combos at top level; the 7-point weight-ratio sweep under `10-50/`, which is what `pareto_figures.py` and `router_evolution.py --type reward-ratio-compare` read). |
| `reward_weight_sweep_8x8_3reward/{uniform_3c,hotspot30_3c}/` | `loss_curves.py`, `router_evolution.py --type training-progress`, `taux_routeurs_date2022.py`, `router_size_maps.py` | Per-run training artifacts: `*_generator_loss` / `*_discriminator_loss` (per-batch), `*_reward{1,2,3}_loss` (per-epoch), `*_historyFake_fix` (fixed-noise generator sample, 10/epoch) and `*_historyFake` (full batch, 100/epoch) — a `list` of per-epoch arrays. Git-ignored in full; the scripts read the baked `derived/` counterparts. |
| `igd_reference/perfs_<weights>_norm` | `igd_compare.py` | The 14 reward-weight combos of the 12-router sweep, one file each: already-normalized `(100, 3)` [saturation, power, area] arrays, one row per generated NoC. 36 KB total, git-tracked. Provenance: `../../data/reference/README.md`. |
| `dse_12r3c/dataset_all_{X,sat,pow}` | `igd_compare.py` | The full 3¹² = 531,441-NoC enumeration of the smaller 12-router/3-class mesh, with simulated saturation/power per NoC — Fig 6.20's "true Pareto front" ground truth. Git-ignored; `igd_compare.py` reads `derived/dse_12r3c_front.npz` instead. Provenance in `../../data/raw/dse_12r3c/README.md`. |
| `traffic_maps/{uni,hot}_img`, `traffic_maps/norm_{uni,hot}_traf` | `router_size_maps.py` | The 8×8 per-router traffic-load map (`*_img`) and its `(5, 64)` normalized-load array (`norm_*_traf`; row 0 is used). Pickles, no extension. |
| `moustache/dataset_{uni,hot}{100,501040}_e{100,200}` | `moustache_boxplots.py` | Generator output re-simulated at training epochs 100 and 200, for the T₁₀₀ (`…100…`) and T₅₀P₁₀A₄₀ (`…501040…`) reward weightings, uniform (`uni`) and hotspot (`hot`). 100 `Data_AX` each. The epoch-300 sets are **not** copied here — they are byte-identical to `generated/10-50/{…}_SatAndPow_GCN-GCN_100-0_dataset` and `generated/{…}_SatAndPowAndArea_50-10-40_dataset`, which the script reads directly. |

All data files are extension-less Python pickles. `Data_AX` is defined in
`figures/noc_data.py` and registered under `__main__` by
`load_data_ax_dataset()` so the pickles load regardless of how a script is run.

## `derived/` — the baked inputs

Committed, ~3.3 MB, written by [`../bake/`](../bake/). Each artifact holds
exactly the reduction its figures consume, and records the byte size and
SHA-256 of the archive it came from, so the reduction is auditable rather than
magic:

```
python bake/bake_common.py                 # describe every artifact
python bake/bake_common.py data/derived/dse_12r3c_front.npz
```

| Artifact | Baked from | The reduction |
|---|---|---|
| `dataset_10k_{uniform,hotspot30}_3c_summary.npz` | the 10k simulated datasets (309 / 243 MB) | per-sample saturation, power and normalized area — the three arrays `noc_data.summarize_samples` returns, which is all these figures read |
| `dse_12r3c_front.npz` | the 531,441-NoC enumeration (59 MB) | the marginal "true Pareto front": per distinct saturation value, the minimum power and area |
| `reward_weight_sweep_8x8_3reward/**/*_historyFake[_fix]_classes.npz` | the per-epoch generator logs (23 MB each; one is 493 MB) | `(n_epochs, batch, 64)` int8 per-router class indices — the `argmax` every reader takes first |
| `reward_weight_sweep_8x8_3reward/**/*_losses.npz` | `*_{generator,discriminator}_loss` (6–8 MB each) | the two columns Fig 6.10 plots (`generator_loss[:, 0]`, `discriminator_loss[:, 2]`), plus the per-epoch reward arrays in full |

Artifacts mirror their source's path under `data/`, not just its filename: two
runs share a basename across `uniform_3c` and `hotspot30_3c`.

[`../bake/verify.sh`](../bake/verify.sh) is the check that this is lossless. It
builds all 34 figures from the raw archives, builds them again in a copy of
`figures/` holding only the git-tracked files, and compares byte for byte —
34/34 identical.

## The raw archives — optional

Not needed to rebuild any figure. Fetch them only to re-bake, to derive
something the current artifacts do not carry, or to check the bake yourself.
They exceed GitHub's file-size limits and are listed in the repo `.gitignore`.

| Path (under `figures/data/`) | Size | Baked into |
|---|---|---|
| `dataset_10k_uniform_3c` | ~309 MB | `derived/dataset_10k_uniform_3c_summary.npz` |
| `dataset_10k_hotspot30_3c` | ~243 MB | `derived/dataset_10k_hotspot30_3c_summary.npz` |
| `reward_weight_sweep_8x8_3reward/*` | >500 MB (the uniform T₁₀₀ history alone is ~493 MB) | the `*_classes.npz` and `*_losses.npz` artifacts above |
| `dse_12r3c/dataset_all_{X,sat,pow}` | ~59 MB | `derived/dse_12r3c_front.npz` |

**Availability** (first three): The archives are **available on request** — open an issue on this repository and ask.

`dse_12r3c` comes from the separate `data/raw/` archive instead — see
[`../../data/raw/README.md`](../../data/raw/README.md), then copy
`dse_12r3c/dataset_all_{X,sat,pow}` here.

Drop them at the paths in the table and re-run:

```
bake/bake_all.sh        # re-derive every artifact from the raw archives
bake/verify.sh          # prove the artifacts still reproduce the figures exactly
```

Every script that reads one of these also takes a per-input override
(`--history-path`, `--history-{sat100,801010,501040}`,
`--epoch0-{uniform,hotspot}`, `--dataset-10k`, `--tenk-path`, `--dse-dir`,
`--run-prefix`), which reads the raw archive directly and takes precedence over
the baked artifact.

