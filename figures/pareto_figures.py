#!/usr/bin/env python3
"""Reproduce the thesis Chapter 6 saturation-vs-power/area Pareto figures.

Two required positional arguments select exactly one figure:

    python pareto_figures.py <metric> <config>

- ``metric``: ``Power`` or ``Area`` -- the y-axis (saturation threshold is
  always the x-axis, matching the thesis plots: "Taux de saturation" on x,
  "Puissance (mW)" / "Surface normalisée" on y). Case-insensitive.
- ``config``: ``uniform``, ``hotspot30-gcn`` (hotspot30 traffic, GCN-based
  rewards) or ``hotspot30-cnn`` (hotspot30 traffic, CNN-based rewards).

Figure-number mapping (verified against the thesis PDF, pages 123/129, and
the source notebooks -- see below; the repo's older docstrings/README
mislabeled these, e.g. as 6.14/6.19-6.22):

    metric=Power  config=uniform        -> Fig 6.13(a)
    metric=Area   config=uniform        -> Fig 6.13(b)
    metric=Power  config=hotspot30-gcn  -> Fig 6.18(a)
    metric=Area   config=hotspot30-gcn  -> Fig 6.18(b)
    metric=Power  config=hotspot30-cnn  -> Fig 6.19(a)
    metric=Area   config=hotspot30-cnn  -> Fig 6.19(b)

Each figure overlays four series (colors/markers match the thesis):

- **dataset** (black, "^-"): the *full* training dataset
  (``data/dataset_10k_{uniform,hotspot30}_3c``, 10k simulated samples),
  binned by its (rounded) saturation value and averaged per bin -- this
  traces the "frontier" the generated NoCs are compared against. This is
  **not** ``data/baseline_topology_dataset/`` (the 7 hand-tuned reference
  topologies): the source notebooks load the 10k-sample dataset for this
  curve, and only use ``baseline_topology_dataset`` for other,
  differently-normalized DATE2022-paper figures (see
  ``results_analysis_date2022.py``) -- the two are unrelated.
- **dataset: moyenne** (red triangle): the mean of that same 10k dataset.
- **Sat-Area** (blue "o"): the M-RWGAN weight-ratio sweep trained with
  Saturation+Area rewards (``data/generated/10-50/<prefix>SatAndArea_<pairing>_<ratio>_dataset``).
- **Sat-Pow** (green "o"): the sweep trained with Saturation+Power rewards
  (``data/generated/10-50/<prefix>SatAndPow_<pairing>_<ratio>_dataset``).

Both sweeps are plotted on every figure (Sat-Area-trained NoCs still have a
measurable power, and vice versa) -- only which metric is on the y-axis
changes between the ``Power`` and ``Area`` invocations. The GCN-CNN /
CNN-CNN model pairing and whether the Area sweep's ``100-0`` point exists
depends on ``config``; see ``CONFIGS`` below (verified against
``data/generated/10-50/`` and the ``resultat_{uniform,hotspot,hotspot_CNN}_Analysis.ipynb``
notebooks).

The dataset curve comes from the bundled ``data/derived/dataset_10k_*_summary.npz``
-- the three per-sample arrays (saturation, power, area) this figure actually
plots, baked out of the ~250-320 MB ``dataset_10k_*`` pickles, which stay
git-ignored. Nothing needs downloading. To rebuild from a raw pickle instead,
fetch it per ``data/README.md`` and pass ``--tenk-path``.

Example::

    python pareto_figures.py Power uniform
    python pareto_figures.py Area hotspot30-cnn --output-dir /tmp/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noc_data import load_data_ax_dataset, load_dataset_summary, summarize_samples  # noqa: E402

HERE = Path(__file__).resolve().parent

# coef sweep points, richest-Power-first (matches the notebooks' `coef` list).
COEFS = ("100-0", "90-10", "70-30", "50-50", "30-70", "10-90", "0-100")

# Per-config sweep sources and thesis figure number. The `tenk` sets are the
# git-ignored ~250-320 MB files (see data/README.md). `area_skip_100_0` mirrors
# the source notebooks/data: the uniform and hotspot30-gcn Area (GCN-CNN)
# sweeps have no `100-0` file (6 points); hotspot30-cnn's (CNN-CNN) has all 7.
CONFIGS = {
    "uniform": dict(
        prefix="uniform_", tenk="dataset_10k_uniform_3c",
        area_pairing="GCN-CNN", area_skip_100_0=True, pow_pairing="GCN-GCN",
        thesis_fig="6.13",
    ),
    "hotspot30-gcn": dict(
        prefix="hotspot30_", tenk="dataset_10k_hotspot30_3c",
        area_pairing="GCN-CNN", area_skip_100_0=True, pow_pairing="GCN-GCN",
        thesis_fig="6.18",
    ),
    "hotspot30-cnn": dict(
        prefix="hotspot30_", tenk="dataset_10k_hotspot30_3c",
        area_pairing="CNN-CNN", area_skip_100_0=False, pow_pairing="CNN-CNN",
        thesis_fig="6.19",
    ),
}

METRIC_LABELS = {"pow": "Power (mW)", "area": "Area (normalized)"}
METRIC_SUBLETTER = {"pow": "a", "area": "b"}


def load_sweep(generated_dir: Path, prefix: str, reward: str, pairing: str, coefs) -> dict:
    """One reward sweep (e.g. 6-7 weight-ratio points) -> per-point mean/std of sat/pow/area."""
    names, out = [], {f"{m}_{s}": [] for m in ("sat", "pow", "area") for s in ("mean", "std")}
    for coef in coefs:
        path = generated_dir / f"{prefix}{reward}_{pairing}_{coef}_dataset"
        summ = summarize_samples(load_data_ax_dataset(str(path)))
        names.append(coef)
        for m in ("sat", "pow", "area"):
            out[f"{m}_mean"].append(float(summ[m].mean()))
            out[f"{m}_std"].append(float(summ[m].std()))
    arrs = {k: np.asarray(v) for k, v in out.items()}
    arrs["names"] = names
    return arrs


def dataset_curve_and_mean(summ: dict) -> tuple[dict, dict]:
    """The 10k dataset summary -> (per-sat-bin mean curve, overall mean), for sat/pow/area.

    The curve mirrors the notebooks' "dataset" black frontier: bin every
    sample by its (rounded) saturation value and average pow/area per bin.
    The mean is the single "dataset: moyenne" point.

    ``summ`` is ``{'sat', 'pow', 'area'}`` as produced by
    ``noc_data.summarize_samples`` — the whole of what this figure takes from
    the 10k dataset, and therefore what ``bake/bake_dataset_summary.py`` bakes.
    """
    curve = {"sat": [], "pow": [], "area": []}
    for value in sorted(set(summ["sat"])):
        mask = summ["sat"] == value
        for m in ("sat", "pow", "area"):
            curve[m].append(float(summ[m][mask].mean()))
    curve = {m: np.asarray(v) for m, v in curve.items()}
    mean = {m: float(summ[m].mean()) for m in ("sat", "pow", "area")}
    return curve, mean


def plot_pareto(
    sweep_area: dict, sweep_pow: dict, curve: dict, mean: dict,
    metric: str, config: str, thesis_fig: str, out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        sweep_area["sat_mean"], sweep_area[f"{metric}_mean"],
        xerr=sweep_area["sat_std"], yerr=sweep_area[f"{metric}_std"],
        linestyle="None", marker="o", color="b", capsize=4, label="Sat-Area",
    )
    ax.errorbar(
        sweep_pow["sat_mean"], sweep_pow[f"{metric}_mean"],
        xerr=sweep_pow["sat_std"], yerr=sweep_pow[f"{metric}_std"],
        linestyle="None", marker="o", color="g", capsize=4, label="Sat-Pow",
    )
    ax.errorbar(
        curve["sat"], curve[metric],
        linestyle="-", marker="^", color="k", label="dataset",
    )
    ax.scatter(
        mean["sat"], mean[metric],
        marker="^", s=100, color="r", label="dataset: moyenne", zorder=3,
    )

    for sweep, color in ((sweep_area, "b"), (sweep_pow, "g")):
        for name, x, y in zip(sweep["names"], sweep["sat_mean"], sweep[f"{metric}_mean"]):
            ax.annotate(
                name, (x, y), textcoords="offset points", xytext=(5, 5),
                fontsize=9, color=color,
            )

    ax.set_xlabel("Taux de saturation")
    ax.set_ylabel(METRIC_LABELS[metric])
    subletter = METRIC_SUBLETTER[metric]
    ax.set_title(f"Fig {thesis_fig}({subletter}) -- {config}: saturation vs {METRIC_LABELS[metric]}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _normalize_metric(value: str) -> str:
    v = value.strip().capitalize()
    if v not in ("Power", "Area"):
        raise argparse.ArgumentTypeError(f"metric must be Power or Area (got {value!r})")
    return v


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metric", type=_normalize_metric, choices=["Power", "Area"],
        help="Which metric is the y-axis of the saturation scatter (case-insensitive).",
    )
    parser.add_argument(
        "config", choices=list(CONFIGS),
        help="Traffic/reward-model combination -- selects the thesis figure (see module docstring).",
    )
    parser.add_argument(
        "--generated-dir", default=str(HERE / "data" / "generated" / "10-50"),
        help="Directory holding the <traffic>_<rewards>_<pairing>_<ratio>_dataset sweep files.",
    )
    parser.add_argument(
        "--tenk-path", default=None,
        help="Raw data/dataset_10k_<traffic>_3c pickle to summarise instead of the "
        "bundled data/derived/*_summary.npz (see data/README.md).",
    )
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = CONFIGS[args.config]
    metric = "pow" if args.metric == "Power" else "area"

    generated_dir = Path(args.generated_dir)
    area_coefs = COEFS[1:] if cfg["area_skip_100_0"] else COEFS
    sweep_area = load_sweep(generated_dir, cfg["prefix"], "SatAndArea", cfg["area_pairing"], area_coefs)
    sweep_pow = load_sweep(generated_dir, cfg["prefix"], "SatAndPow", cfg["pow_pairing"], COEFS)
    print(f"Sat-Area sweep: {len(sweep_area['names'])} points {sweep_area['names']}")
    print(f"Sat-Pow  sweep: {len(sweep_pow['names'])} points {sweep_pow['names']}")

    summ = load_dataset_summary(
        args.tenk_path, HERE / "data" / cfg["tenk"], what=f"{args.config} dataset_10k",
    )
    curve, mean = dataset_curve_and_mean(summ)
    print(f"dataset curve: {len(curve['sat'])} saturation bins; dataset mean sat={mean['sat']:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pareto_{args.config}_{args.metric.lower()}.png"
    plot_pareto(sweep_area, sweep_pow, curve, mean, metric, args.config, cfg["thesis_fig"], out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
