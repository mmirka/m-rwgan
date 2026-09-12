#!/usr/bin/env python3
"""Reproduce Fig 6.20 -- IGD / distance-to-true-Pareto-front bar chart.

PhD thesis Chapter 6, p.132-133 (French): "Distance euclidienne entre le
meilleur NoC genere de chaque apprentissage et le vrai front Pareto, ainsi
que le detail pour chaque objectif. Mesure de l'IGD - i.e. moyenne."

Source notebook:
``PhD_Work/Explo_Pareto/generated_compare_pareto_norm.ipynb``, the cells
following the "# Plot Bar errors" markdown cell (specifically the
``df2_avg_t.plot.bar()`` figure -- the other bar chart built in that same
section, ``df_t``/``df3_avg_t``, is not Fig 6.20).

This is a *different*, smaller design-space-exploration study than the
paper's 8x8-mesh headline results used by the other ``figures/*.py``
scripts: a fully-enumerated 4x3 mesh (12 routers, 3 router classes ->
3**12 = 531,441 possible NoCs), simulated for hotspot30 traffic, used here
purely as ground truth for the "true Pareto front" reference set. 14
M-RWGAN training runs (one per saturation/power/area reward-weight
combination) each generated 100 NoCs; for each run we take its single best
(min-distance-to-front) generated NoC, then average those 14 "best of run"
distances into the IGD (Inverse Generational Distance, thesis eq. 6.3 --
here computed as a plain mean of the 14 per-run best distances, exactly as
the notebook does, not as a per-reference-point sum/|R| average).

Data (both under ``figures/data/``, see ``figures/data/README.md``):

- True-Pareto-front source: ``data/derived/dse_12r3c_front.npz`` -- the
  marginal front itself, three short arrays baked out of the full
  531,441-NoC enumeration ``data/dse_12r3c/dataset_all_{X,sat,pow}``. The
  enumeration stays git-ignored (~59 MB); the front is committed, so this
  figure needs no download. To recompute it from the enumeration, fetch it
  per ``data/README.md`` and point ``--dse-dir`` at it.
- Per-run generated-sample source: the 14
  ``data/igd_reference/perfs_<weights>_norm`` files -- already-normalized
  ``(100, 3)`` [saturation, power, area] arrays, small and committed directly
  to git, so no download is needed for this part.

Run from ``figures/``::

    python igd_compare.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noc_data import (  # noqa: E402
    euclidean_igd_best,
    load_true_pareto_front,
    resolve_data_path,
)

HERE = Path(__file__).resolve().parent

# (perfs-file weight suffix, x-axis label) -- order and labels match the
# notebook's ``list_all_perfs`` / ``list_name`` (cells 37 and 54).
REWARD_COMBOS = [
    ("343333", "T34_P33_A33"),
    ("10000", "T100_P0_A0"),
    ("01000", "T0_P100_A0"),
    ("00100", "T0_P0_A100"),
    ("90100", "T90_P10_A0"),
    ("90010", "T90_P0_A10"),
    ("70300", "T70_P30_A0"),
    ("70030", "T70_P0_A30"),
    ("50500", "T50_P50_A0"),
    ("50050", "T50_P0_A50"),
    ("30700", "T30_P70_A0"),
    ("30070", "T30_P0_A70"),
    ("10900", "T10_P90_A0"),
    ("10090", "T10_P0_A90"),
]

PERFS_DIR_DEFAULT = HERE / "data" / "igd_reference"
# The 531,441-NoC enumeration is git-ignored (~59 MB). The marginal front it is
# reduced to is baked into the tracked data/derived/dse_12r3c_front.npz, which
# noc_data.load_true_pareto_front prefers -- see figures/data/README.md.
DSE_DIR_DEFAULT = HERE / "data" / "dse_12r3c"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_distances(perfs_dir: Path, list_sat, list_pow_min, list_area_min):
    """Per reward-combo: best generated NoC's distance to the true Pareto front.

    ``all_dist[i]`` is the notebook's ``best_perfs_3d[i]`` (3D euclidean
    distance of the best-of-100 generated NoC from training run ``i`` to its
    closest marginal-front reference point); ``per_axis[i]`` is
    ``best_perfs_dists[i]`` -- the [sat, pow, area] absolute deltas at that
    same closest pair.
    """
    labels = []
    all_dist = []
    per_axis = []
    for suffix, label in REWARD_COMBOS:
        perfs = np.asarray(_load_pickle(perfs_dir / f"perfs_{suffix}_norm"))
        dist, deltas = euclidean_igd_best(perfs, list_sat, list_pow_min, list_area_min)
        labels.append(label)
        all_dist.append(dist)
        per_axis.append(deltas)
    return labels, np.array(all_dist), np.array(per_axis)  # per_axis: (14, 3)


def plot_igd_bars(labels, all_dist: np.ndarray, per_axis: np.ndarray, output_dir: Path) -> Path:
    """The 'Plot Bar errors' section's ``df2_avg_t.plot.bar()`` figure -- Fig 6.20.

    One bar group per training run (multi-objective distance + per-objective
    breakdown), plus a final "IGD" group holding the plain mean of each series
    across the 14 runs (the notebook's ``df2_avg['average']`` column, i.e. the
    IGD -- eq. 6.3, formula 6.3 in the thesis).
    """
    igd_all = all_dist.mean()
    igd_axis = per_axis.mean(axis=0)

    x_labels = list(labels) + ["IGD"]
    series = {
        "Multi-objectifs": np.append(all_dist, igd_all),
        "Objectif saturation": np.append(per_axis[:, 0], igd_axis[0]),
        "Objectif puissance": np.append(per_axis[:, 1], igd_axis[1]),
        "Objectif surface": np.append(per_axis[:, 2], igd_axis[2]),
    }
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    x = np.arange(len(x_labels))
    n_series = len(series)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Combinaisons de rewards")
    ax.set_ylabel("Distance euclidienne")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "igd_compare_fig6_20.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--perfs-dir", default=None,
        help="Override the 14 perfs_<weights>_norm files' directory "
        "(default: the tracked figures/data/igd_reference/).",
    )
    parser.add_argument(
        "--dse-dir", default=None,
        help="A raw dataset_all_{X,sat,pow} directory to recompute the front from, "
        "instead of the bundled data/derived/dse_12r3c_front.npz "
        "(see figures/data/README.md).",
    )
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    perfs_dir = resolve_data_path(
        args.perfs_dir, PERFS_DIR_DEFAULT,
        what="reward-weight-sweep generated-sample perfs",
    )
    list_sat, list_pow_min, list_area_min = load_true_pareto_front(
        args.dse_dir, DSE_DIR_DEFAULT,
        what="DSE 12-router/3-class true-Pareto-front data",
    )
    labels, all_dist, per_axis = compute_distances(perfs_dir, list_sat, list_pow_min, list_area_min)

    print(f"IGD (multi-objectif) = {all_dist.mean():.4f}")
    for name, col in zip(("saturation", "puissance", "surface"), per_axis.T):
        print(f"IGD ({name}) = {col.mean():.4f}")

    out_path = plot_igd_bars(labels, all_dist, per_axis, Path(args.output_dir))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
