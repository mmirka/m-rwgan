# GAN-generated NoC configurations — simulated performance data

Already-simulated `Data_AX` pickles for GAN-generated router-class
configurations. Each holds real adjacency matrices, router-class
assignments, and real per-sample latency/power/area curves (not raw
generator output — that is a different, earlier-stage artifact).

These back the "generated" comparison series in the Pareto figures
(Fig 6.13/6.14/6.19–6.22).

## Layout (scripts load these by exact relative path)

- `hotspot30_SatAndPowAndArea_{50-10-40,80-10-10}_dataset`,
  `uniform_SatAndPowAndArea_{50-10-40,70-10-20}_dataset` — 100-sample,
  3-reward (Saturation+Power+Area) generated sets.
- `10-50/<traffic>_<rewards>_<model-pairing>_<ratio>_dataset` — 40-sample
  batches across a 7-point weight-ratio sweep (`0-100` … `100-0`), for
  `<rewards>` in `{SatAndPow, SatAndArea}` and `<model-pairing>` in
  `{CNN-CNN, GCN-CNN, GCN-GCN}` (plus two `SatAndPowAnDArea_GNN-GNN-CNN`
  3-reward uniform sets — sic on the "AnD" typo, kept verbatim).
- `0-10/` — the same naming pattern, 10-sample batches; combined with
  `10-50/` to reach 50 total samples for one ratio.

## Which figures use this

`figures/pareto_figures.py --generated-dataset` reads these directly to
produce Fig 6.13/6.14/6.19–6.22.
