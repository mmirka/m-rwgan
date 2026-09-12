# 12-router/3-class design-space enumeration

This is an earlier design-space-exploration study on a smaller 12-router
mesh, separate from the paper's headline 8×8 results. It backs the
`perfs_<weights>` files in `data/reference/experimental_data/reward_weight_sweep/`
and their normalized counterparts in `figures/data/igd_reference/`.

- `dataset_all_X`, `dataset_all_sat`, `dataset_all_pow` — the full
  12-router/3-class combinatorial enumeration (3^12 = 531,441 router-class
  assignments) with simulated saturation/power for each; the "true Pareto
  front" baseline the generated samples are compared against. **These now live
  at `figures/data/dse_12r3c/`** (see below), not in this directory.
- `reward_weight_sweep_results/` — per-ratio training artifacts (14 weight
  combos, dated Oct 2021): `historyFake`, `historyFake_fix`, and
  `reward{1,2,3}_loss`. Per-batch `generator_loss`/`discriminator_loss` and
  `.h5` model-weight checkpoints are not included.

`figures/igd_compare.py` reads the enumeration from `figures/data/dse_12r3c/`
as the "true Pareto front" ground truth for thesis Fig 6.20 — the only analysis
path this 12-router data still feeds, and `figures/` resolves every input from
inside itself. The already-normalized `perfs_<weights>_norm` arrays it compares
against moved to `figures/data/igd_reference/` for the same reason. What stays
here — `reward_weight_sweep_results/`, and the un-normalized `perfs_<weights>`
in `data/reference/` — is provenance only. Both directories are git-ignored and
come from the same archive (`data/raw/README.md`).
