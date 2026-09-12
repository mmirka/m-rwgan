# Paper figures

Self-contained reproduction of the M-RWGAN figures from DATE2022 / the PhD
thesis Chapter 6. Every input these scripts read lives under `figures/data/` —
no `src/mrwgan` import, no path outside this folder, **and no download**: the
four oversized inputs (~1.2 GB, git-ignored) are reduced to ~3.3 MB of
committed artifacts under `data/derived/` by [`bake/`](bake/), holding exactly
what the figures consume. A plain clone rebuilds all 34 PNGs. See
[`data/README.md`](data/README.md) for what each artifact holds, and
[`bake/verify.sh`](bake/verify.sh) for the check that it is lossless.

The scripts use only `numpy` + `matplotlib` (and a small vendored
`noc_data.py` for the pickled simulation records). Any environment with
those two packages runs them; the repo's `mrwgan` conda env works out of
the box.

Run everything from this directory:

```
cd figures
```

Figures are written to `figures/output/` by default (`--output-dir` to
change).

## Figure → script → data

Figure numbers below are verified against the thesis PDF itself (list of
figures + body text), not just the source notebooks — a previous version of
this table had several wrong (e.g. it labeled the uniform-traffic Pareto
figure "6.21/6.22"; it is actually 6.13).

| Figure | Script (from `figures/`) | Data consumed |
|---|---|---|
| **Fig 6.9** — 5-panel spatial snapshot of router size over training (epochs 0/50/100/200/last), uniform traffic, Sat-only reward | `router_evolution.py --config uniform --type training-progress --reward-ratio Sat100` | the baked `…_SatAndArea_100-0_date08072021_historyFake_classes.npz` (from the ~493 MB `data/reward_weight_sweep_8x8_3reward/uniform_3c/multiRWGAN_R-SatGCN_…_historyFake`) |
| **Fig 6.10** — generator/critic loss + reward scores | `loss_curves.py` | the baked `…_SatAndArea_100-0_date08072021_losses.npz` (from `data/reward_weight_sweep_8x8_3reward/uniform_3c/multiRWGAN_R-SatGCN_…_{generator,discriminator,reward1,reward2}_loss`) |
| **Fig 6.11** — router size vs Sat/Pow reward-weight sweep (7-panel grid), uniform | `router_evolution.py --config uniform --type reward-ratio-compare --reward-pair SatAndPow` | `data/generated/10-50/uniform_SatAndPow_GCN-GCN_*-*_dataset` |
| **Fig 6.12** — router size vs Sat/Area reward-weight sweep (7-panel grid), uniform | `router_evolution.py --config uniform --type reward-ratio-compare --reward-pair SatAndArea` | `data/generated/10-50/uniform_SatAndArea_GCN-CNN_*-*_dataset` (6 of 7 points bundled; the missing `100-0` panel falls back to the Sat/Pow sweep's `100-0` point, see the script's docstring) |
| **Fig 6.13(a)/(b)** — saturation vs power / vs area, generated vs dataset, uniform | `pareto_figures.py Power uniform` / `pareto_figures.py Area uniform` | `data/derived/dataset_10k_uniform_3c_summary.npz` (baked from the git-ignored `data/dataset_10k_uniform_3c`) + `data/generated/10-50/uniform_SatAndPow_GCN-GCN_*` and `uniform_SatAndArea_GCN-CNN_*` |
| **Fig 6.14** — router size vs Sat/Pow reward-weight sweep, hotspot30 (GCN rewards) | `router_evolution.py --config hotspot --type reward-ratio-compare --reward-pair SatAndPow` | `data/generated/10-50/hotspot30_SatAndPow_GCN-GCN_*-*_dataset` |
| **Fig 6.15** — router size vs Sat/Area reward-weight sweep, hotspot30 (GCN rewards) | `router_evolution.py --config hotspot --type reward-ratio-compare --reward-pair SatAndArea` | `data/generated/10-50/hotspot30_SatAndArea_GCN-CNN_*-*_dataset` (6 of 7 points bundled, same `100-0` fallback as Fig 6.12) |
| **Fig 6.16** — router size vs Sat/Pow reward-weight sweep, hotspot30 (CNN rewards) | `router_evolution.py --config hotspot-cnn --type reward-ratio-compare --reward-pair SatAndPow` | `data/generated/10-50/hotspot30_SatAndPow_CNN-CNN_*-*_dataset` |
| **Fig 6.17** — router size vs Sat/Area reward-weight sweep, hotspot30 (CNN rewards) | `router_evolution.py --config hotspot-cnn --type reward-ratio-compare --reward-pair SatAndArea` | `data/generated/10-50/hotspot30_SatAndArea_CNN-CNN_*-*_dataset` |
| **Fig 6.18(a)/(b)** — saturation vs power / vs area, generated vs dataset, hotspot30 (GCN rewards) | `pareto_figures.py Power hotspot30-gcn` / `pareto_figures.py Area hotspot30-gcn` | `data/derived/dataset_10k_hotspot30_3c_summary.npz` (baked from the git-ignored `data/dataset_10k_hotspot30_3c`) + `data/generated/10-50/hotspot30_SatAndPow_GCN-GCN_*` and `hotspot30_SatAndArea_GCN-CNN_*` |
| **Fig 6.19(a)/(b)** — saturation vs power / vs area, generated vs dataset, hotspot30 (CNN rewards) | `pareto_figures.py Power hotspot30-cnn` / `pareto_figures.py Area hotspot30-cnn` | `data/derived/dataset_10k_hotspot30_3c_summary.npz` (baked from the git-ignored `data/dataset_10k_hotspot30_3c`) + `data/generated/10-50/hotspot30_SatAndPow_CNN-CNN_*` and `hotspot30_SatAndArea_CNN-CNN_*` |
| **Fig 6.20** — IGD / distance-to-true-Pareto-front bar chart | `igd_compare.py` (no arguments) | `data/derived/dse_12r3c_front.npz` (baked from the git-ignored 12-router/3-class full design-space enumeration `data/dse_12r3c/dataset_all_{X,sat,pow}`) + `data/igd_reference/perfs_*_norm` (git-tracked) |

`pareto_figures.py`'s two positional arguments (`Power`/`Area`, then
`uniform`/`hotspot30-gcn`/`hotspot30-cnn`) are the only required arguments —
each of the 6 combinations reproduces exactly one panel above.
`router_evolution.py --type reward-ratio-compare` defaults to
`--reward-pair both`, writing both the Sat/Pow and Sat/Area PNGs for the
given `--config` in one invocation.

### DATE2022 conference-paper figures

Five scripts reproduce the figures from the `picture_DATE2022_*` notebooks
(distinct from the thesis Chapter 6 variants above). They share the same
conventions (`numpy` + `matplotlib`, `--output-dir`, PNG at `dpi=150`). Output
filenames keep the notebooks' original `savefig` basenames;
`results_analysis_date2022.py` additionally prefixes the traffic because the
three `resultat_*` notebooks reused the same basenames.

| Figure(s) | Script (from `figures/`) | Data |
|---|---|---|
| router-rate curve over training — `BDtime3`, `BDtimeBW`, `BDtimeBWr`, `BDtimeBW2`, `BDtimeBW3` | `taux_routeurs_date2022.py --style all` | the baked `data/derived/reward_weight_sweep_8x8_3reward/uniform_3c/…SatAndPowAndArea_50-10-40_date16072021_historyFake_classes.npz` |
| 5-panel router-size maps — `4uni_t2`, `4hot_t2` | `router_size_maps.py --traffic {uniform,hotspot}` | `data/traffic_maps/*` + three baked `data/derived/reward_weight_sweep_8x8_3reward/**/*_historyFake_classes.npz` per traffic |
| best-config latency/power comparison — `gene_compare4` | `best_gene_compare.py` | `baseline_topology_dataset/*` + `generated/10-50/hotspot30_SatAndPow_GCN-GCN_10-90_dataset` + `generated/hotspot30_SatAndPowAndArea_50-10-40_dataset` |
| box-and-whisker training evolution — `moustache_hotzoom2_moy3`, `moustache_uni3`, `moustache_all` | `moustache_boxplots.py --figure all` | `data/moustache/*` + `baseline_topology_dataset/*` + `generated/*` + the baked `data/derived/dataset_10k_*_summary.npz` (epoch 0) |
| results-analysis Pareto/scatter — `gene_uniSurf_isca4`, `geneSurf_hot_isca4`, `geneSurf_hot_isca`, `gene_uni_pareto_satar` | `results_analysis_date2022.py --traffic {uniform,hotspot,hotspot-area}` | `baseline_topology_dataset/*` + `generated/10-50/*` + the baked `data/derived/dataset_10k_*_summary.npz` (for the `_pareto_satar` dataset curve) |

The inputs marked **git-ignored** above are read through their baked
`data/derived/` counterparts, so nothing needs fetching. To read a raw archive
directly instead — you fetched one per [`data/README.md`](data/README.md), or
you keep a copy elsewhere — every script that reads one takes a per-input
override, which wins over the baked artifact: `--history-path`
(`taux_routeurs_date2022.py`, `router_evolution.py`),
`--history-{sat100,801010,501040}` (`router_size_maps.py`),
`--epoch0-{uniform,hotspot}` (`moustache_boxplots.py`), `--dataset-10k`
(`results_analysis_date2022.py`), `--tenk-path` (`pareto_figures.py`),
`--dse-dir` (`igd_compare.py`), `--run-prefix` (`loss_curves.py`).

## Regenerate each PNG

```
cd figures

# Fig 6.13(a)/(b) — uniform traffic, saturation vs power / vs area
#                   (dataset curve from data/derived/…uniform…_summary.npz)
python pareto_figures.py Power uniform
python pareto_figures.py Area uniform

# Fig 6.18(a)/(b) — hotspot30 traffic, GCN rewards
#                   (dataset curve from data/derived/…hotspot30…_summary.npz)
python pareto_figures.py Power hotspot30-gcn
python pareto_figures.py Area hotspot30-gcn

# Fig 6.19(a)/(b) — hotspot30 traffic, CNN rewards
python pareto_figures.py Power hotspot30-cnn
python pareto_figures.py Area hotspot30-cnn

# Fig 6.10 — generator/critic loss and reward scores
#            (loss series from data/derived/…_losses.npz)
python loss_curves.py --run-prefix \
    "data/reward_weight_sweep_8x8_3reward/uniform_3c/multiRWGAN_R-SatGCN_300e_L0.2-d0.05_beta10_SatAndArea_100-0_date08072021_"

# Fig 6.9 — 5-panel router-size-over-training snapshot, uniform, Sat-only reward
#           (class indices from data/derived/…_historyFake_classes.npz)
python router_evolution.py --config uniform --type training-progress --reward-ratio Sat100

# Fig 6.11/6.12 — router-size vs Sat/Pow and Sat/Area reward-weight sweep, uniform
python router_evolution.py --config uniform --type reward-ratio-compare

# Fig 6.14/6.15 — same, hotspot30 traffic, GCN rewards
python router_evolution.py --config hotspot --type reward-ratio-compare

# Fig 6.16/6.17 — same, hotspot30 traffic, CNN rewards
python router_evolution.py --config hotspot-cnn --type reward-ratio-compare

# Fig 6.20 — IGD / distance-to-true-Pareto-front bar chart (no arguments)
#            (Pareto front from data/derived/dse_12r3c_front.npz)
python igd_compare.py

# --- DATE2022 conference-paper figures ---
python taux_routeurs_date2022.py --style all   # baked history
python router_size_maps.py --traffic hotspot   # baked histories
python router_size_maps.py --traffic uniform   # idem — the ~493 MB T100 history bakes to 210 KB
python best_gene_compare.py                    # fully git-tracked, no download
python moustache_boxplots.py --figure all      # baked dataset_10k_* summaries (epoch 0)
python results_analysis_date2022.py --traffic uniform        # baked uniform summary
python results_analysis_date2022.py --traffic hotspot        # baked hotspot30 summary
python results_analysis_date2022.py --traffic hotspot-area   # idem
```

Committed reference renders of the thesis-Chapter-6 PNGs are in
`../results/figures/`. There are no committed DATE2022 reference renders — verify
those by eye against the rendered notebooks / the DATE2022 paper.

## What the data under `data/` is

- **`data/baseline_topology_dataset/`** — seven hand-tuned reference
  8×8-mesh topologies (`dataset_refNoC_*/dataset`), each a pickled list of
  `Data_AX` objects: adjacency matrix, router-class assignment, and
  simulated latency/power/area curves. Used by `best_gene_compare.py`,
  `moustache_boxplots.py` and `results_analysis_date2022.py` (the DATE2022
  scripts). **Not** used by `pareto_figures.py`: the thesis Fig
  6.13/6.18/6.19 "dataset" curve is the full 10k-sample dataset below, not
  these seven hand-tuned points — an earlier version of this script
  (incorrectly) used `baseline_topology_dataset` for that curve.
- **`data/reward_weight_sweep_8x8_3reward/{hotspot30_3c,uniform_3c}/`**
  (**git-ignored in full**, >500 MB; read via the baked
  `data/derived/reward_weight_sweep_8x8_3reward/**`) — per-run
  training artifacts for a handful of dated M-RWGAN runs on the 8×8 mesh:
  - `*_generator_loss` / `*_discriminator_loss` — per-batch loss arrays
    (`(n_batches, 4)`), used by `loss_curves.py`.
  - `*_reward{1,2,3}_loss` — per-epoch reward score arrays (301 values:
    300 epochs + 1 final); a 2-reward run only has `reward1`/`reward2`.
  - `*_historyFake_fix` — the generator's output on a fixed noise vector,
    sampled once per epoch (a `list` of per-epoch arrays); used as a
    fallback by `router_evolution.py --type training-progress` when no
    `_historyFake` match exists.
  - `*_historyFake` (no `_fix`) — the full per-epoch generator batch
    (100 samples/epoch, vs 10 for `_fix`); used by
    `taux_routeurs_date2022.py`, `router_size_maps.py`, and
    `router_evolution.py --type training-progress` (which prefers a `_fix`
    match but falls back to this). Five run ~23 MB each; the sixth,
    `uniform_3c/multiRWGAN_R-SatGCN_…SatAndArea_100-0_date08072021_historyFake`,
    is ~493 MB — it is the exact run Fig 6.9 itself needs (`--config uniform
    --reward-ratio Sat100`).
- **`data/traffic_maps/`** — `{uni,hot}_img` (the 8×8 per-router traffic-load
  map) and `norm_{uni,hot}_traf` (its `(5, 64)` normalized-load array; row 0 is
  used). Consumed by `router_size_maps.py` for the "Traffic" panel.
- **`data/moustache/`** — generator output re-simulated at training epochs 100
  and 200 (`dataset_{uni,hot}{100,501040}_e{100,200}`, 100 `Data_AX` each), for
  the two reward weightings T₁₀₀ (`…100…`) and T₅₀P₁₀A₄₀ (`…501040…`), uniform
  and hotspot. Used by `moustache_boxplots.py`; the epoch-300 sets it needs are
  the (byte-identical) files already under `data/generated/`.
- **`data/dataset_10k_{uniform,hotspot30}_3c`** (**git-ignored**, ~309 / 243 MB;
  read via the baked `data/derived/dataset_10k_*_summary.npz`) — the
  10k-sample simulated datasets: the
  "dataset" curve in `pareto_figures.py`'s Fig 6.13/6.18/6.19, the moustache
  epoch-0 box, and the DATE2022 results-analysis dataset curve.
- **`data/igd_reference/`** — the 14 `perfs_<weights>_norm` files (git-tracked,
  36 KB): already-normalized `(100, 3)` [saturation, power, area] arrays, one
  row per generated NoC, for each reward-weight combo of the 12-router sweep.
  Read by `igd_compare.py`.
- **`data/dse_12r3c/`** (**git-ignored**, ~59 MB; read via the baked
  `data/derived/dse_12r3c_front.npz`) —
  `dataset_all_{X,sat,pow}`, the full 3¹² = 531,441-NoC enumeration of the
  smaller 12-router/3-class mesh. `igd_compare.py` derives Fig 6.20's "true
  Pareto front" ground truth from it; nothing else uses it. This is a separate,
  earlier study from the 8×8 headline results — `../data/raw/dse_12r3c/README.md`
  documents its provenance.
- **`data/generated/`** — already-simulated `Data_AX` pickles for
  GAN-generated router-class configurations: real adjacency matrices,
  router-class assignments, and real per-sample latency/power/area curves
  (not raw generator output).
  - Top level: `{hotspot30,uniform}_SatAndPowAndArea_<ratio>_dataset` —
    100-sample, 3-reward (Saturation + Power + Area) generated sets.
  - `10-50/<traffic>_<rewards>_<model-pairing>_<ratio>_dataset` —
    40-sample batches across a 7-point weight-ratio sweep (`0-100` …
    `100-0`), for `<rewards>` in `{SatAndPow, SatAndArea}` and
    `<model-pairing>` in `{CNN-CNN, GCN-CNN, GCN-GCN}`, plus two
    `SatAndPowAnDArea_GNN-GNN-CNN` 3-reward uniform sets (the `AnD` in
    those filenames is an original typo, kept verbatim so the paths match).
    Consumed by `pareto_figures.py` (Fig 6.13/6.18/6.19) and
    `router_evolution.py --type reward-ratio-compare` (Fig
    6.11/6.12/6.14–6.17), which average each sweep point's 40-100 samples'
    router-class assignment into one spatial "router size" map per panel.

Every path above is under `figures/data/`; no figure script reads anything
outside this folder.

## Making new figures from your own training runs

`scripts/train_mrwgan.py` writes per-run pickles under
`results/logs/<run-name>/` (with `--output-dir results/`, the default):
`history_fake_fix.pkl`, `generator_loss.pkl`, `discriminator_loss.pkl`,
`reward_sat_loss.pkl`, `reward_pow_loss.pkl`, `reward_area_loss.pkl`.
These are the same kinds of arrays the bundled data holds — you can point
the figure scripts at them to document *your* run's evolution rather than
reproduce a published figure.

- **`router_evolution.py --type training-progress`** accepts an explicit
  `--history-path`, which overrides the `--config`/`--reward-ratio`-based
  lookup into the bundled data (`--config` is still required, for the output
  filename only):

  ```
  python router_evolution.py --config uniform --type training-progress \
      --history-path ../results/logs/<run-name>/history_fake_fix.pkl
  ```

  `--type reward-ratio-compare` has no such override — it needs a full
  7-point reward-weight sweep of already-simulated datasets (see "What the
  data under `data/` is" above), which a single training run doesn't produce.

- **`loss_curves.py`** builds filenames from `--run-prefix` + the suffixes
  `generator_loss`, `discriminator_loss`, `reward{1,2,3}_loss` (no
  extension). Your training run uses different names —
  `generator_loss.pkl` vs the bundled extensionless `*_generator_loss`,
  and `reward_sat_loss.pkl` / `reward_pow_loss.pkl` / `reward_area_loss.pkl`
  vs `*_reward1_loss` / `*_reward2_loss` / `*_reward3_loss`. Rename or
  symlink your run's files to that scheme first, e.g.:

  ```
  cd ../results/logs/<run-name>
  ln -s generator_loss.pkl     run_generator_loss
  ln -s discriminator_loss.pkl run_discriminator_loss
  ln -s reward_sat_loss.pkl    run_reward1_loss
  ln -s reward_pow_loss.pkl    run_reward2_loss
  ln -s reward_area_loss.pkl   run_reward3_loss
  cd -
  python loss_curves.py --run-prefix ../results/logs/<run-name>/run_
  ```
